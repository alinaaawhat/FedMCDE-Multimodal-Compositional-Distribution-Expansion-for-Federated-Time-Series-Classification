import time
from alg.opt import *
from alg import alg, modelopera
import torch.nn.functional as F
import sys
sys.path.append('.')
from Featurenet.utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
import random
import os
import torch
import sys
from Style_conditioner.get_conditioner import conditioner
from data_load.get_domainhar import get_acthar, get_acthar_client
from torch import optim
import torch.nn as nn
from torch.nn.functional import cosine_similarity
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np
import random
import logging
from logging import handlers
from copy import deepcopy
import sys
sys.path.append('./data_load/')
from data_load.data_util.sensor_loader import SensorDataset,DataDataset
from torch.utils.data import DataLoader
import torch.utils.data as data
import pickle

def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str) 
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh) 
        self.logger.addHandler(th)

from datetime import datetime
from itertools import combinations
import math

def federated_average(models):
    """
    联邦平均算法：对多个模型参数进行平均
    """
    global_dict = {}
    
    for key in models[0].state_dict().keys():
        param_stack = torch.stack([model.state_dict()[key] for model in models], 0)
        
        # 检查参数类型，只对浮点型参数求平均，整数型参数直接使用第一个模型的值
        if param_stack.dtype.is_floating_point or param_stack.dtype.is_complex:
            global_dict[key] = param_stack.mean(0)
        else:
            # 对于整数类型参数（如batch norm的running count），使用第一个模型的值
            global_dict[key] = models[0].state_dict()[key]
    
    return global_dict

def load_client_styles(testuser, logger):
    """
    加载所有客户端的styles文件
    """
    
    styles_path = f"./client_styles/dsads_tar0_rate{testuser['remain_rate']}_seed1_client_styles.pkl"
    
    if os.path.exists(styles_path):
        logger.debug(f"Loading client styles from: {styles_path}")
        with open(styles_path, 'rb') as f:
            client_styles = pickle.load(f)
        
        logger.debug("Loaded client styles information:")
        for key, value in client_styles.items():
            if isinstance(value, torch.Tensor):
                logger.debug(f"  {key}: shape={value.shape}")
        
        return client_styles
    else:
        logger.debug(f"Client styles file not found: {styles_path}")
        return None

def sample_mixed_styles(client_styles, current_y, testuser, logger):
    """
    从多个客户端的styles中采样,混合使用
    """
    if client_styles is None:
        return None
    
    batch_size = current_y.shape[0]
    mixed_styles = []
    
    # 收集所有客户端的styles
    all_client_styles = {}
    all_client_labels = {}
    
    for key, value in client_styles.items():
        if 'style' in key:
            client_id = key.split('_')[0]
            all_client_styles[client_id] = value.to(device)
            label_key = f"{client_id}_labels"
            if label_key in client_styles:
                all_client_labels[client_id] = client_styles[label_key]
    
    logger.debug(f"Available clients for style mixing: {list(all_client_styles.keys())}")
    
    # 为当前批次的每个样本选择styles
    for i in range(batch_size):
        current_label = current_y[i].item()
        
        # 从所有客户端中随机选择一个
        available_clients = list(all_client_styles.keys())
        selected_client = random.choice(available_clients)
        
        client_style = all_client_styles[selected_client]
        client_label = all_client_labels[selected_client]
        
        # 尝试找到相同标签的样本
        same_label_indices = (client_label == current_label).nonzero(as_tuple=True)[0]
        
        if len(same_label_indices) > 0:
            selected_idx = random.choice(same_label_indices)
            selected_style = client_style[selected_idx].unsqueeze(0)
        else:
            selected_idx = random.randint(0, client_style.shape[0] - 1)
            selected_style = client_style[selected_idx].unsqueeze(0)
        
        mixed_styles.append(selected_style)
    
    mixed_styles = torch.cat(mixed_styles, dim=0)
    logger.debug(f"Generated mixed styles shape: {mixed_styles.shape}")
    
    return mixed_styles

def combine_label(y,logger, maxlen, pre_defined_weights, weighted = True):
    index_dict = {}
    for i, label in enumerate(y):
        label = label.item()
        if label not in index_dict:
            index_dict[label] = [i]
        else:
            index_dict[label].append(i)
    combination_dict = {}
    weight_dict = {}

    for label, indices in index_dict.items():
        all_combinations = [] 
        all_weights = []

        if maxlen == None:
            sum_of_all_cond_num_weights = sum([pre_defined_weights[i] for i in range(len(indices))])
            for r in range(1, len(indices) + 1):
                all_combinations.extend(combinations(indices, r))
                if weighted: 
                    cond_num_weight = pre_defined_weights[r-1] / sum_of_all_cond_num_weights
                    all_weights.extend([cond_num_weight/math.comb(len(indices), r)] * int(math.comb(len(indices), r)))
        else:
            if maxlen > len(indices): maxlen = len(indices)
            sum_of_all_cond_num_weights = sum([pre_defined_weights[i] for i in range(maxlen)])
            for r in range(1,  maxlen + 1):
                all_combinations.extend(combinations(indices, r))
                if weighted: 
                    cond_num_weight = pre_defined_weights[r-1] / sum_of_all_cond_num_weights
                    all_weights.extend([cond_num_weight/math.comb(len(indices), r)] * int(math.comb(len(indices), r)))
      
        if weighted: 
            weight_dict[label] = all_weights
        combination_dict[label] = all_combinations
    return combination_dict, weight_dict

def train_client_local(client_id, model, args, client_data, valid_loader, testuser, logger, local_epochs=50):
    """
    单个客户端本地训练
    """
    logger.debug(f"Starting local training for client {client_id}")
    
    # 为每个客户端创建独立的优化器
    algorithm = deepcopy(model)
    opto = get_optimizer(algorithm, args, nettype='step-2')
    optc = get_optimizer(algorithm, args, nettype='step-3')
    optf = get_optimizer(algorithm, args, nettype='step-1') 
    schedulera, schedulerd, scheduler, use_slr = get_slr(testuser['dataset'], testuser['target'], optf, opto, optc)
    
    # 使用传入的客户端数据创建DataLoader
    client_train_loader = client_data
    
    # 客户端本地训练
    for epoch in range(local_epochs):
        algorithm.train()
        
        if epoch % 10 == 0:
            logger.debug(f"Client {client_id} - Local epoch {epoch}")
        
        # Step 1: Fine grained
        for step in range(args.step1):
            for batch_no, minibatch in enumerate(client_train_loader, start=1):
                x = minibatch[0]
                x = x[:,:,:,-testuser['length']:]
                y = minibatch[1]
                d = minibatch[2]
                train_x, train_y, train_d = x.to(device), y.long().to(device), d.long().to(device)
                loss_result_dict = algorithm.update_ft(train_x, train_y, train_d, optf)
            schedulera.step(loss_result_dict['class'])

        # Step 2: Ori-spec
        for step in range(args.step2):
            for batch_no, minibatch in enumerate(client_train_loader, start=1):
                x = minibatch[0]
                x = x[:,:,:,-testuser['length']:]
                y = minibatch[1]
                d = minibatch[2]
                train_x, train_y, train_d = x.to(device), y.long().to(device), d.long().to(device)
                loss_result_dict = algorithm.update_os(train_x, train_y, train_d, opto)
            schedulerd.step(loss_result_dict['total'])
        
        # Step 3: Class spec
        for step in range(args.step3):
            for batch_no, minibatch in enumerate(client_train_loader, start=1):
                train_x = minibatch[0].to(device)
                train_y = minibatch[1].to(device)
                train_d = minibatch[2].to(device)
                train_x = train_x[:,:,:,-testuser['length']:]
                loss_list, y_pred, index_worse, all_z = algorithm.update_cs(train_x, train_y, optc)
            
            if use_slr:
                scheduler.step()
            else:
                scheduler.step(loss_list['total'])
    
    logger.debug(f"Client {client_id} local training completed")
    return algorithm


def train_diversity(model, args, train_loader, valid_loader, test_loader, testuser, num_clients=3):
    nowtime = datetime.now()
    timename = nowtime.strftime('%d_%m_%Y_%H_%M_%S')
    log_file_name = os.getcwd()+os.path.join('/Featurenet/logs/', testuser['name']+f"_federated_logs_{nowtime.strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)

    # 加载客户端styles文件
    client_styles = load_client_styles(testuser, logger)

    algorithm_class = alg.get_algorithm_class()
    algorithm = algorithm_class(args).cuda()
    opto = get_optimizer(algorithm, args, nettype='step-2')
    optc = get_optimizer(algorithm, args, nettype='step-3')
    optf = get_optimizer(algorithm, args, nettype='step-1') 
    schedulera, schedulerd, scheduler, use_slr = get_slr(testuser['dataset'], testuser['target'], optf, opto, optc)
    
    # 保存原始的conditioner路径
    original_conditioner = testuser['conditioner']
    testuser['conditioner'] = f"./conditioner_pth/{testuser['dataset']}_tar_{testuser['target']}_rm_{testuser['remain_data']}seed_{testuser['seed']}_global_round_9.pt"

    # 联邦学习：为每个客户端生成数据
    all_client_data = {}
    client_train_loaders = {}  # 添加这个字典来存储每个客户端的train_loader
    
    # 首先加载全局数据文件用于获取客户端数据
    data_file_path = f"./data/{testuser['dataset']}/{testuser['dataset']}_crosssubject_rawaug_rate{testuser['remain_data']}_t{testuser['target']}_seed{testuser['seed']}_scalerminmax.pkl"
    try:
        with open(data_file_path, 'rb') as f:
            global_data = pickle.load(f)
        logger.debug(f"Loaded global data from {data_file_path}")
    except Exception as e:
        logger.warning(f"Could not load global data: {e}")
        global_data = {}
    
    for client_id in range(1, num_clients + 1):
        logger.debug(f"Processing client {client_id}")
        
        # 为每个客户端设置特定的conditioner路径
        
        # 每个客户端的数据文件路径
        file_pathv_client_id = testuser['newdata'].replace('.pt', f'_client{client_id}.pt')
        diff_sample = {}
        use_mixed = True
        
        if os.path.exists(file_pathv_client_id):
            diff_sample = torch.load(file_pathv_client_id)
            logger.debug(f"Loaded existing data for client {client_id}")
        else:
            logger.debug(f"Generating new data for client {client_id}")
            k_data = -1
            
            # 检查是否有客户端特定的数据
            if 'client_raw_trs' in global_data and client_id in global_data['client_raw_trs']:
                client_raw_x = global_data['client_raw_trs'][client_id][0]  # shape: (542, 125, 45)
                client_raw_y = global_data['client_raw_trs'][client_id][1]  # shape: (542,)
                data_type = args.dataset
                target = args.target
                batch_size = 64  # 定义batch_size
                client_train_loader, client_valid_loader, target_loader, n_class = get_acthar_client(args, data_type, target, batch_size=batch_size, remain_rate=testuser['remain_data'], seed=testuser['seed'], client_id=client_id)
                train_dataset = client_train_loader.dataset
                valid_dataset = client_valid_loader.dataset
                source_loaders = data.DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
                valid_loader_client = data.DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
                current_train_loader = source_loaders
            else:
                # 如果没有客户端特定数据，使用原始train_loader
                current_train_loader = train_loader
            
            for batch_no, minibatch in enumerate(current_train_loader, start=1):
                k_data = k_data + 1
                x = minibatch[0]
                y = minibatch[1]
                x, y = x.to(device), y.long().to(device)
                length = x.shape[1]
                remainder = 0
                if x.size(1) % 64 != 0:
                    remainder = 64 - (x.size(1) % 64)
                    pad_x = F.pad(x, (0, 0, 0, remainder, 0, 0))
                if len(x.shape) == 3:
                    x = x.unsqueeze(3)
                x = x.transpose(1, 2).squeeze(3).unsqueeze(2)
                
                if k_data in diff_sample.keys():
                    pass
                else:
                    # 数据生成逻辑（保持原有逻辑）
                    if client_styles is not None:
                        current_styles = conditioner(x, y, testuser)
                        current_styles = current_styles.repeat(testuser['repeat'], 1)
                        
                        mixed_styles = sample_mixed_styles(client_styles, y, testuser, logger)
                        if mixed_styles is not None:
                            mixed_styles = mixed_styles.repeat(testuser['repeat'], 1)
                            styles = torch.cat([current_styles, mixed_styles], dim=0)
                            use_mixed = True
                            logger.debug(f"Client {client_id}: Using both current and mixed client styles for batch {k_data}")
                        else:
                            styles = current_styles
                            use_mixed = False
                            logger.debug(f"Client {client_id}: Fallback to current conditioner only for batch {k_data}")
                    else:
                        styles = conditioner(x, y, testuser)
                        styles = styles.repeat(testuser['repeat'], 1)
                        use_mixed = False
                        logger.debug(f"Client {client_id}: Using current conditioner for batch {k_data}")

                    try:
                        pad_x
                        if len(pad_x.shape) == 3:
                            pad_x = pad_x.unsqueeze(3)
                        x = pad_x.transpose(1, 2).squeeze(3).unsqueeze(2)
                        repeat_times = testuser['repeat'] * 2 if use_mixed else testuser['repeat']
                        x_aug = x.repeat(repeat_times, 1, 1, 1)
                    except:
                        repeat_times = testuser['repeat'] * 2 if use_mixed else testuser['repeat']
                        x_aug = x.repeat(repeat_times, 1, 1, 1)
                        pass
                        
                    x_ = x_aug.squeeze(2).float()
                    combination_dict, weights_dict = combine_label(y, logger, testuser['maxcond'], testuser['cond_weight'])
                    if use_mixed:
                        batch_size_gen = y.shape[0] * testuser['repeat'] * 2
                    else:
                        batch_size_gen = y.shape[0] * testuser['repeat']
                        
                    random_combinations = []
                    keys_tensor = []

                    samples_per_key = batch_size_gen // len(combination_dict)
                    remaining_samples = batch_size_gen % len(combination_dict)

                    for key in combination_dict.keys():
                        values = combination_dict[key]
                        weights = weights_dict[key]
                        random.shuffle(values)
                        sampled_values = random.sample(values, min(samples_per_key, len(values)))
                        for value in sampled_values:
                            if value not in random_combinations:
                                random_combinations.append(value)
                                keys_tensor.append(key)

                    times_sample = 0
                    while len(random_combinations) < batch_size_gen:
                        times_sample = times_sample + 1
                        if times_sample > 1500:
                            random_value = random.randint(0, len(random_combinations) - 1)
                            random_combinations.extend([random_combinations[random_value]])
                            keys_tensor.append(keys_tensor[random_value])
                        else:
                            for key, values in combination_dict.items():
                                sampled_values = random.sample(values, 1)
                                if sampled_values[0] not in random_combinations:
                                    random_combinations.extend(sampled_values)
                                    keys_tensor.append(key)
                                if len(random_combinations) == batch_size_gen:
                                    break

                    model.eval()
                    interpolate_out = model.sample(styles, random_combinations, rescaled_phi = 0.7)
                    
                    try:
                        pad_x
                        interpolate_out = interpolate_out[:,:,:, -testuser['length']:]
                    except:
                        pass

                    diff_sample[k_data] = {}
                    diff_sample[k_data]['x'] = x.float()
                    diff_sample[k_data]['y'] = torch.tensor(keys_tensor).to(device)
                    shape = y.shape

                    if use_mixed:
                        current_samples = interpolate_out[:len(interpolate_out)//2]
                        mixed_samples = interpolate_out[len(interpolate_out)//2:]
                        
                        generated_x = torch.cat([current_samples, mixed_samples], dim=0)
                        diff_sample[k_data]['x'] = torch.cat([generated_x.unsqueeze(2), diff_sample[k_data]['x']], dim=0)
                        
                        domain_current = torch.ones(current_samples.shape[0]) * 1
                        domain_mixed = torch.ones(mixed_samples.shape[0]) 
                        domain_original = torch.zeros(shape[0])
                        
                        diff_sample[k_data]['d'] = torch.cat([domain_current, domain_mixed, domain_original], dim=0).to(device)
                        diff_sample[k_data]['y'] = torch.cat([diff_sample[k_data]['y'], y], dim=0)
                        
                        logger.debug(f"Client {client_id}: Generated {current_samples.shape[0]} current client samples (d=1) and {mixed_samples.shape[0]} mixed client samples (d=2)")
                        
                    else:
                        diff_sample[k_data]['x'] = torch.cat([interpolate_out.unsqueeze(2), diff_sample[k_data]['x']], dim=0)
                        diff_sample[k_data]['y'] = torch.cat([diff_sample[k_data]['y'], y], dim=0)
                        
                        diff_sample[k_data]['d'] = torch.zeros(diff_sample[k_data]['y'].shape[0]).to(device)
                        diff_sample[k_data]['d'][:len(interpolate_out)] = 1
                        diff_sample[k_data]['d'][len(interpolate_out):] = 0
                            
                    torch.save(diff_sample, file_pathv_client_id)
                    logger.debug(f"Saved data for client {client_id} to {file_pathv_client_id}")
        
        # 记录生成数据的统计信息
        logger.debug(f"Generated data summary for client {client_id}:")
        total_generated_current = 0
        total_generated_mixed = 0
        total_original = 0
        for k_data in diff_sample.keys():
            batch_d = diff_sample[k_data]['d']
            current_count = (batch_d == 1).sum().item()
            mixed_count = (batch_d == 2).sum().item()
            original_count = (batch_d == 0).sum().item()
            total_generated_current += current_count
            total_generated_mixed += mixed_count
            total_original += original_count
            
        logger.debug(f"Client {client_id} - Total current client samples: {total_generated_current}")
        logger.debug(f"Client {client_id} - Total mixed client samples: {total_generated_mixed}")
        logger.debug(f"Client {client_id} - Total original samples: {total_original}")
        
        # 为每个客户端合并数据
        for k_data in diff_sample.keys():
            if k_data == 0:
                data_train_client = diff_sample[k_data]['x']
                label_train_client = diff_sample[k_data]['y']
                domain_train_client = diff_sample[k_data]['d']
            else:
                data_train_client = torch.cat([data_train_client, diff_sample[k_data]['x']], dim=0)
                label_train_client = torch.cat([label_train_client, diff_sample[k_data]['y']], dim=0)
                domain_train_client = torch.cat([domain_train_client, diff_sample[k_data]['d']], dim=0)
        
        # 存储客户端数据
        all_client_data[client_id] = {
            'x': data_train_client.cpu(),
            'y': label_train_client.cpu(),
            'd': domain_train_client.cpu()
        }
        
        # 为每个客户端创建train_loader
        generate_dataset = DataDataset(x=data_train_client.cpu(), label=label_train_client.cpu(), alabel=domain_train_client.cpu(), dataset=testuser['dataset'])
        client_train_loaders[client_id] = DataLoader(dataset=generate_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        
        logger.debug(f"Client {client_id} data processing completed")
    
    # 恢复原始conditioner路径
    testuser['conditioner'] = original_conditioner
    
    # 初始化全局模型
    algorithm_class = alg.get_algorithm_class()
    global_model = algorithm_class(args).cuda()
    
    # 联邦学习参数
    aggregation_rounds = 1000
    local_epochs = 30
    
    logger.debug(f"Starting federated learning with {num_clients} clients, {aggregation_rounds} rounds, {local_epochs} local epochs")
    
    best_test_acc = 0
    
    # 联邦学习主循环
    for round_idx in range(aggregation_rounds):
        logger.debug(f"\n========= Federated Round {round_idx + 1}/{aggregation_rounds} =========")
        
        # 存储客户端模型
        client_models = []
        
        # 每个客户端本地训练
        for client_id in range(1, num_clients + 1):
            logger.debug(f"Training client {client_id}")
            
            # 客户端从全局模型开始训练，使用对应客户端的数据
            client_model = train_client_local(
                client_id=client_id,
                model=global_model,
                args=args,
                client_data=client_train_loaders[client_id],  # 修复：传递正确的参数
                valid_loader=valid_loader,
                testuser=testuser,
                logger=logger,
                local_epochs=local_epochs
            )
            
            client_models.append(client_model)
        
        # 联邦平均聚合
        logger.debug("Aggregating client models...")
        global_dict = federated_average(client_models)
        global_model.load_state_dict(global_dict)
        
        # 在测试集上评估全局模型
        acc = 0
        num = 0
        global_model.eval()
        
        for batch_no, minibatch in enumerate(test_loader, start=1):
            x = minibatch[0]
            y = minibatch[1]
            x, y = x.to(device), y.long().to(device)
            if len(x.shape) == 3:
                x = x.unsqueeze(3)

            x = x.transpose(1, 2).squeeze(3).unsqueeze(2)
            y_pred_list = [] 
            y_pred, target_z = global_model.predict(x.float())
            y_prob = F.softmax(y_pred, dim=1) 
            pred1 = y_prob
            y_pred_list.append(y_prob.cpu().detach().numpy())
            y_pred_list = np.array(y_pred_list)
            class_score = np.sum(y_pred_list, axis=0)
            y_pred = np.argmax(class_score, axis=1)
            y_true = y.cpu().detach().numpy()
            acc += np.sum(y_pred == y_true)
            num += len(y_pred)
            
        test_acc = acc/num
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            
        logger.debug(f"Round {round_idx + 1} - Test Accuracy: {test_acc:.5f}, Best Accuracy: {best_test_acc:.5f}")
        
    logger.debug(f"\nFederated learning completed. Final best test accuracy: {best_test_acc:.5f}")
    logger.debug(f"Final results saved for: {testuser['name']}")