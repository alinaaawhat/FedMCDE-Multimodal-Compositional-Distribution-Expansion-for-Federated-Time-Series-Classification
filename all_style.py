import torch
import os
import numpy as np
from datetime import datetime
import argparse
import sys
sys.path.append('.')
sys.path.append('./Style_conditioner/')

# Style conditioner imports
from Style_conditioner.trainer.trainer import model_load
from Style_conditioner.models.TC import TC
from Style_conditioner.models.model import base_Model
from Style_conditioner.get_conditioner import conditioner

# Data loading
from data_load.get_domainhar import get_acthar
import torch.utils.data as data
import importlib
import pickle

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_client_data(args, data_type, target, client_id, batch_size, remain_rate, seed):
    """为指定client加载数据"""
    # 加载预处理的数据
    data_path = './data/dsads/dsads_crosssubject_rawaug_rate0.2_t0_seed1_scalerminmax.pkl'
    
    try:
        data_raw_aug = torch.load(data_path)
    except:
        data_raw_aug = np.load(data_path, allow_pickle=True)
    
    # 获取client数据
    client_raw_trs = data_raw_aug.get('client_raw_trs', {})
    client_aug_trs = data_raw_aug.get('client_aug_trs', {})
    client_raw_vas = data_raw_aug.get('client_raw_vas', {})
    client_aug_vas = data_raw_aug.get('client_aug_vas', {})
    
    if client_id not in client_raw_trs:
        print(f"Client {client_id} data not found in preprocessed data, using domain-based approach")
        # 如果没有预处理的client数据，则使用原始方法
        train_loader, valid_loader, target_loader, n_class = get_acthar(args, data_type, target, batch_size=batch_size, remain_rate=remain_rate, seed=seed, train_diff=0)
        return train_loader, valid_loader, target_loader, n_class
    
    # 构建数据集
    from data_load.data_util.sensor_loader import SensorDataset
    from torch.utils.data import DataLoader
    
    # 创建训练数据集
    train_dataset = SensorDataset(client_raw_trs[client_id], aug=False, dataset=data_type)
    valid_dataset = SensorDataset(client_raw_vas[client_id], aug=False, dataset=data_type)
    
    # 创建目标数据集（测试数据）
    raw_tet = data_raw_aug['raw_tet']
    target_dataset = SensorDataset(raw_tet, aug=False, dataset=data_type)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # 获取类别数量
    if data_type == 'pamap':
        n_class = 8
    elif data_type == 'uschad':
        n_class = 12
    elif data_type == 'dsads':
        n_class = 19
    else:
        n_class = 10  # 默认值
    
    return train_loader, valid_loader, target_loader, n_class


def generate_client_styles(args, client_id):
    """为指定client生成styles"""
    print(f"开始为Client {client_id}生成styles...")
    
    # 基本参数
    data_type = args.selected_dataset
    target = args.target
    remain_rate = args.remain_rate
    SEED = args.seed
    testuser = data_type+"_tar_"+str(target) +'_rm_'+str(remain_rate)+'seed_'+str(SEED)+f'_client{client_id}'
    batch_size = args.batch_size
    
    # 加载client数据
    train_loader, valid_loader, target_loader, n_class = load_client_data(
        args, data_type, target, client_id, batch_size, remain_rate, SEED
    )
    
    # 动态导入配置
    module_name = f'Style_conditioner.config_files.{data_type}_Configs'
    ConfigModule = importlib.import_module(module_name)
    configs = ConfigModule.Config()
    configs.batch_size = batch_size
    configs.TC.train_test = 1  # 设置为测试模式
    
    # 创建模型
    model = base_Model(configs).to(device)
    temporal_contr_model = TC(configs, device).to(device)
    
    # 构建testuser_dict用于加载conditioner
    testuser_dict = {
        'seed': SEED,
        'name': testuser,
        'conditioner': './conditioner_pth/dsads_tar_0_rm_0.2seed_1_global_round_26.pt'
    }
    
    # 检查conditioner文件是否存在
    if not os.path.exists(testuser_dict['conditioner']):
        print(f"警告: Client {client_id} 的conditioner文件不存在: {testuser_dict['conditioner']}")
        print("请先训练Style Conditioner")
        return None
    
    client_styles = []
    client_labels = []
    
    print(f"Client {client_id} 开始生成styles...")
    
    # 遍历训练数据生成styles
    model.eval()
    temporal_contr_model.eval()
    
    # 加载预训练的conditioner
    try:
        chkpoint = torch.load(testuser_dict['conditioner'], map_location=device)
        model_dict = chkpoint["model_state_dict"]
        tc_dict = chkpoint['temporal_contr_model_state_dict']
        model.load_state_dict(model_dict)
        temporal_contr_model.load_state_dict(tc_dict)
        print(f"Client {client_id} conditioner加载成功")
    except Exception as e:
        print(f"Client {client_id} conditioner加载失败: {e}")
        return None
    
    with torch.no_grad():
        for batch_idx, (x, y, _) in enumerate(train_loader):
            # 数据预处理
            if len(x.shape) == 3:
                x = x.unsqueeze(3)
            x = x.transpose(1, 2).squeeze(3)  # [batch, channel, length]
            x = x.float().to(device)
            y = y.long().to(device)
            
            # 通过模型获取特征
            predictions, features = model(x)
            features = torch.nn.functional.normalize(features, dim=1)
            
            # 通过temporal contrastive model获取context
            c_t = temporal_contr_model.context(features)
            
            # 收集styles和labels
            client_styles.append(c_t.cpu())
            client_labels.append(y.cpu())
            
            if batch_idx % 10 == 0:
                print(f"Client {client_id} 处理batch {batch_idx}/{len(train_loader)}")
    
    # 合并所有batches
    if client_styles:
        all_styles = torch.cat(client_styles, dim=0)  # [total_samples, 100]
        all_labels = torch.cat(client_labels, dim=0)  # [total_samples]
        
        print(f"Client {client_id} 生成完成:")
        print(f"  Styles shape: {all_styles.shape}")
        print(f"  Labels shape: {all_labels.shape}")
        
        return {
            f'client{client_id}_style': all_styles,
            f'client{client_id}_labels': all_labels
        }
    else:
        print(f"Client {client_id} 没有生成任何styles")
        return None


def save_all_client_styles(args):
    """生成并保存所有client的styles"""
    print("开始生成所有Client的styles...")
    
    data_type = args.selected_dataset
    target = args.target
    remain_rate = args.remain_rate
    SEED = args.seed
    
    all_client_styles = {}
    
    # 为每个client生成styles
    for client_id in range(1, args.num_clients + 1):
        print(f"\n{'='*50}")
        print(f"处理Client {client_id}")
        print(f"{'='*50}")
        
        client_styles = generate_client_styles(args, client_id)
        
        if client_styles is not None:
            all_client_styles.update(client_styles)
            print(f"Client {client_id} styles生成成功")
        else:
            print(f"Client {client_id} styles生成失败")
    
    # 保存所有client styles
    if all_client_styles:
        save_dir = './client_styles/'
        os.makedirs(save_dir, exist_ok=True)
        
        save_filename = f'{data_type}_tar{target}_rate{remain_rate}_seed{SEED}_client_styles.pkl'
        save_path = os.path.join(save_dir, save_filename)
        
        with open(save_path, 'wb') as f:
            pickle.dump(all_client_styles, f)
        
        print(f"\n{'='*60}")
        print(f"所有Client styles已保存到: {save_path}")
        print(f"包含的Client数量: {args.num_clients}")
        print("保存的数据结构:")
        for key, value in all_client_styles.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        print(f"{'='*60}")
        
        return save_path
    else:
        print("没有生成任何client styles")
        return None


def load_and_display_client_styles(save_path):
    """加载并显示client styles信息"""
    if not os.path.exists(save_path):
        print(f"文件不存在: {save_path}")
        return
    
    print(f"\n加载styles文件: {save_path}")
    
    with open(save_path, 'rb') as f:
        client_styles = pickle.load(f)
    
    print("加载的Client Styles信息:")
    for key, value in client_styles.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            if 'style' in key:
                print(f"    样本数量: {value.shape[0]}, 特征维度: {value.shape[1]}")
            elif 'labels' in key:
                print(f"    标签数量: {value.shape[0]}, 唯一标签: {torch.unique(value).tolist()}")


def main():
    parser = argparse.ArgumentParser()
    
    # 基本参数
    parser.add_argument('--seed', default=1, type=int, help='seed value')
    parser.add_argument('--selected_dataset', default='dsads', type=str, help='Dataset of choice: pamap, uschad, dsads')
    parser.add_argument('--remain_rate', default=0.2, type=float, help='Using training data ranging from 0.2 to 1.0')
    parser.add_argument('--target', default=0, type=int, help='Choose task id')
    parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
    parser.add_argument('--batch_size', default=64, type=int, help='Training batch')
    
    # Style conditioner相关参数
    parser.add_argument('--experiment_description', default='Exp1', type=str, help='Experiment Description')
    parser.add_argument('--run_description', default='run1', type=str, help='Experiment Description')
    parser.add_argument('--training_mode', default='self_supervised', type=str, help='Training mode')
    parser.add_argument('--logs_save_dir', default='./conditioner_pth/', type=str, help='saving directory')
    
    # 联邦学习参数
    parser.add_argument('--num_clients', default=3, type=int, help='number of clients')
    
    # 功能选择
    parser.add_argument('--mode', default='generate', type=str, choices=['generate', 'load'], 
                       help='generate: 生成新的styles; load: 加载并显示已有的styles')
    parser.add_argument('--styles_path', default='', type=str, help='已有styles文件的路径（mode=load时使用）')
    
    args = parser.parse_args()
    
    print(f"Client Styles生成工具")
    print(f"数据集: {args.selected_dataset}")
    print(f"目标: {args.target}")
    print(f"剩余数据比例: {args.remain_rate}")
    print(f"种子: {args.seed}")
    print(f"客户端数量: {args.num_clients}")
    print(f"模式: {args.mode}")
    
    if args.mode == 'generate':
        # 生成新的styles
        save_path = save_all_client_styles(args)
        if save_path:
            # 显示生成的结果
            load_and_display_client_styles(save_path)
    elif args.mode == 'load':
        # 加载并显示已有的styles
        if args.styles_path:
            load_and_display_client_styles(args.styles_path)
        else:
            # 使用默认路径
            data_type = args.selected_dataset
            target = args.target
            remain_rate = args.remain_rate
            SEED = args.seed
            
            save_filename = f'{data_type}_tar{target}_rate{remain_rate}_seed{SEED}_client_styles.pkl'
            default_path = os.path.join('./client_styles/', save_filename)
            load_and_display_client_styles(default_path)


if __name__ == "__main__":
    main()

    '''
    
    python generate_client_styles.py \
    --seed 1 \
    --selected_dataset 'dsads' \
    --remain_rate 0.2 \
    --target 0 \
    --num_clients 3 \
    --mode generate
    

    python generate_client_styles.py \
    --seed 1 \
    --selected_dataset 'dsads' \
    --remain_rate 0.2 \
    --target 0 \
    --num_clients 3 \
    --mode load



    client1_style: shape=torch.Size([512, 100]), dtype=torch.float32
    样本数量: 512, 特征维度: 100
  client1_labels: shape=torch.Size([512]), dtype=torch.int64
    标签数量: 512, 唯一标签: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
  client2_style: shape=torch.Size([512, 100]), dtype=torch.float32
    样本数量: 512, 特征维度: 100
  client2_labels: shape=torch.Size([512]), dtype=torch.int64
    标签数量: 512, 唯一标签: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
  client3_style: shape=torch.Size([512, 100]), dtype=torch.float32
    样本数量: 512, 特征维度: 100
  client3_labels: shape=torch.Size([512]), dtype=torch.int64
    标签数量: 512, 唯一标签: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    '''