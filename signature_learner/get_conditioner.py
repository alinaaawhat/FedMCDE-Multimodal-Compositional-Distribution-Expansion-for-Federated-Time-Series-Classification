import torch
import sys
import importlib
sys.path.append('./')
sys.path.append('./Style_conditioner2')

from models.multimodal_model import MultiModalHARModel, batch_signal_to_stft_image

# Fixed sensor-layout text per dataset (char-level tokenised, always present)
_SENSOR_TEXT = {
    'dsads': (
        "The sensor layout contains 5 body units: torso, right arm, left arm, right leg, and left leg. "
        "Each unit has 9 channels: x/y/z accelerometer, x/y/z gyroscope, and x/y/z magnetometer. "
        "Columns 1–9 correspond to torso, 10–18 to right arm, 19–27 to left arm, "
        "28–36 to right leg, and 37–45 to left leg."
    ),
    'uschad': "The sensor is placed on the right hip of the subject, recording triaxial accelerometer and gyroscope data.",
    'pamap':  "The IMU sensors are placed on the wrist, chest, and ankle, each recording temperature, accelerometer, gyroscope, and magnetometer.",
}

def _tokenize(text, text_len=320, vocab_size=256):
    ids = [min(ord(c), vocab_size - 1) for c in text[:text_len]]
    ids += [0] * (text_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def conditioner(x, y, testuser, dataset='dsads'):
    """
    Load MultiModalHARModel conditioner and return c_t [B, hidden_dim].

    x : [B, C, T]  – sensor data as produced by the diffusion trainer
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    selected_dataset = testuser['name'].split('_tar')[0]

    # Dynamically import dataset config
    module_name = f'config_files.{selected_dataset}_Configs'
    ConfigModule = importlib.import_module(module_name)
    configs = ConfigModule.Config()

    # x arrives as [B, C, T] from trainer; model expects [B, T, C]
    if x.dim() == 3:
        signal = x.transpose(1, 2).float().to(device)   # [B, T, C]
    else:
        signal = x.float().to(device)

    B = signal.shape[0]

    # Build MultiModalHARModel (same hyper-params used during training)
    model = MultiModalHARModel(
        in_channels  = configs.input_channels,
        num_classes  = configs.num_classes,
        hidden_dim   = 100,
        vocab_size   = 256,
        max_text_len = 320,
        max_lag      = 3,
        kernel_size  = configs.kernel_size,
        dropout      = configs.dropout,
    ).to(device)

    # Load checkpoint (use full model_state_dict)
    ckpt = torch.load(testuser['conditioner'], map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Prepare text tokens (fixed sensor description, always present)
    sensor_text = _SENSOR_TEXT.get(selected_dataset, _SENSOR_TEXT['dsads'])
    text_ids = _tokenize(sensor_text).unsqueeze(0).expand(B, -1).to(device)

    # Derive STFT image from signal
    image = batch_signal_to_stft_image(signal, img_h=32, img_w=32)

    # All modalities present at inference
    mask = torch.ones(B, 3, device=device)

    with torch.no_grad():
        _, c_t = model(signal, text_ids, image, mask)   # c_t: [B, 100]

    return c_t
