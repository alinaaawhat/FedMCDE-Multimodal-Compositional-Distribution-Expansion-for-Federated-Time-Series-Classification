
import os
import sys
import pickle
import argparse
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from datetime import datetime
from logging import handlers
import logging

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from models.multimodal_model import MultiModalHARModel, batch_signal_to_stft_image


# ============================================================
# Logging
# ============================================================

def _logger(log_path, level=logging.DEBUG):
    logger = logging.getLogger(log_path)
    logger.setLevel(level)
    if not logger.handlers:
        fh = handlers.TimedRotatingFileHandler(log_path, when='midnight', encoding='utf-8')
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


# ============================================================
# Argument parsing
# ============================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Train MultiModalHARModel as a style conditioner."
    )

    # Dataset
    parser.add_argument("--selected_dataset", default="dsads", type=str,
                        help="Dataset: dsads | uschad | pamap | emg")
    parser.add_argument("--target", default=0, type=int,
                        help="Target subject / domain index")
    parser.add_argument("--remain_rate", default=0.2, type=float,
                        help="Fraction of labelled training data to keep")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--data_path", default="./data", type=str,
                        help="Root data directory containing per-dataset pkl files")

    # Training
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_epoch", default=100, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--logs_save_dir",
                        default="./conditioner_pth",
                        type=str)

    # ── Missing-modality probabilities for signal and image ─────────
    # Text is always present (fixed prompt); only signal and image can be missing.
    # The three values are renormalised automatically.
    parser.add_argument("--p_full", default=1, type=float,
                        help="Fraction of training samples: signal + image both present")
    parser.add_argument("--p_missing_signal", default=0, type=float,
                        help="Fraction of training samples: signal missing")
    parser.add_argument("--p_missing_image", default=0, type=float,
                        help="Fraction of training samples: image missing")

    # ── Sensor-layout text (auto-generated, no text column in data) ──
    _DSADS_TEXT = (
        "The sensor layout contains 5 body units: torso, right arm, left arm, right leg, and left leg. Each unit has 9 channels: x/y/z accelerometer, x/y/z gyroscope, and x/y/z magnetometer. Columns 1–9 correspond to torso, 10–18 to right arm, 19–27 to left arm, 28–36 to right leg, and 37–45 to left leg."
    )
    parser.add_argument("--sensor_text", default=_DSADS_TEXT, type=str,
                        help="Fixed sensor layout description used as the text modality "
                             "(char-level tokenised; always present, mask[:,1]=1).")

    # ── Multimodal model hyper-parameters ───────────────────────────
    parser.add_argument("--hidden_dim", default=100, type=int,
                        help="Hidden dimension of MultiModalHARModel")
    parser.add_argument("--text_len", default=320, type=int,
                        help="Text token sequence length (clipped/padded from sensor_text)")
    parser.add_argument("--img_h", default=32, type=int,
                        help="Pseudo-image height")
    parser.add_argument("--img_w", default=32, type=int,
                        help="Pseudo-image width")
    parser.add_argument("--vocab_size", default=256, type=int,
                        help="Vocab size for TextEncoder (dummy tokens)")
    parser.add_argument("--max_lag", default=3, type=int,
                        help="Max lag for DynamicAlignment")
    parser.add_argument("--ctx_mode", default="signal+cond", type=str,
                        choices=["orig", "signal+cond", "signal_only"],
                        help="Context vector mode: "
                             "orig=alignment only (A), "
                             "signal+cond=signal backbone + alignment (B, recommended), "
                             "signal_only=signal backbone only (C)")

    return parser.parse_args()


# ============================================================
# Mask table: [signal, text, image]
# Text column is always 1 (text is never missing).
# ============================================================

_MASK_TABLE = torch.tensor([
    [1., 1., 1.],  # full
    [0., 1., 1.],  # signal missing
    [1., 1., 0.],  # image missing
])


def _sample_masks(B: int, device,
                  p_full: float, p_missing_signal: float, p_missing_image: float
                  ) -> torch.Tensor:
    """
    Sample a [B, 3] mask tensor. Text column (col 1) is always 1.
    The three input probabilities are renormalised to sum to 1.
    """
    probs = torch.tensor([p_full, p_missing_signal, p_missing_image], dtype=torch.float)
    probs = probs / probs.sum()
    pids  = torch.multinomial(probs.unsqueeze(0).expand(B, -1), num_samples=1).squeeze(1)
    return _MASK_TABLE[pids].to(device)  # [B, 3]


# ============================================================
# Dataset helpers
# ============================================================

def build_dataloaders(args, configs):
    """
    Load the raw HAR dataset directly from the pkl file.
    pkl layout: dict with keys 'raw_trs', 'raw_vas', 'raw_tet'
    each value is [X (N, T, C) float64, Y (N,) int64].

    Returns (train_loader, val_loader, test_loader).
    Signal batches are [B, T, C] float32 tensors.
    STFT images are precomputed once: [N, 3, img_h, img_w] float32 tensors.
    """
    pkl_path = os.path.join(
        args.data_path,
        args.selected_dataset,
        f"{args.selected_dataset}_crosssubject_rawaug"
        f"_rate{args.remain_rate}_t{args.target}_seed{args.seed}_scalernorm.pkl",
    )
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    def _make_loader(key, shuffle, drop_last):
        X = torch.from_numpy(raw[key][0].astype(np.float32))  # [N, T, C]
        Y = torch.from_numpy(raw[key][1].astype(np.int64))    # [N]

        # Pre-compute STFT images once (saves GPU time during training)
        print(f"  Pre-computing STFT images for {key} ({len(X)} samples)…")
        imgs = batch_signal_to_stft_image(X, img_h=args.img_h, img_w=args.img_w)
        # imgs: [N, 3, H, W]

        from torch.utils.data import TensorDataset, DataLoader
        ds = TensorDataset(X, Y, imgs)
        return DataLoader(
            ds,
            batch_size=configs.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    train_loader = _make_loader("raw_trs", shuffle=True,  drop_last=True)
    val_loader   = _make_loader("raw_vas", shuffle=False, drop_last=False)
    test_loader  = _make_loader("raw_tet", shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


def _tokenize_text(text: str, text_len: int, vocab_size: int) -> torch.Tensor:
    """Char-level tokenisation, clipped / zero-padded to text_len."""
    ids = [min(ord(c), vocab_size - 1) for c in text[:text_len]]
    ids += [0] * (text_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# ============================================================
# Train / evaluate loops
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, sensor_text_ids,
                    p_full, p_missing_signal, p_missing_image, aux_weight=0.1):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for batch in loader:
        signal = batch[0].float().to(device)   # [B, T, C]
        labels = batch[1].long().to(device)
        image  = batch[2].float().to(device)   # [B, 3, H, W]  precomputed STFT

        B = signal.size(0)
        # Fixed sensor-layout text (always present)
        text_ids = sensor_text_ids.unsqueeze(0).expand(B, -1)
        # Random missing-modality mask (text col always 1)
        mask = _sample_masks(B, device, p_full, p_missing_signal, p_missing_image)

        optimizer.zero_grad()
        logits, _, aux = model(signal, text_ids, image, mask, return_aux=True)
        aux_loss = (aux['content_consistency']
                    + aux['orthogonality']
                    + 0.1 * aux['style_diversity'])
        loss = criterion(logits, labels) + aux_weight * aux_loss
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += (logits.argmax(1) == labels).float().sum().item()
        n += bs

    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, sensor_text_ids):
    """
    Evaluate on a loader that returns (signal, label, image).
    At eval time all modalities are marked present (mask = [1,1,1]).
    Text tokens come from the fixed sensor-layout description.
    """
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for batch in loader:
        signal = batch[0].float().to(device)
        labels = batch[1].long().to(device)
        image  = batch[2].float().to(device)

        B = signal.size(0)
        text_ids = sensor_text_ids.unsqueeze(0).expand(B, -1)
        mask = torch.ones(B, 3, device=device)

        logits, _, aux = model(signal, text_ids, image, mask, return_aux=True)
        aux_loss = (aux['content_consistency']
                    + aux['orthogonality']
                    + 0.1 * aux['style_diversity'])
        loss = criterion(logits, labels) + 0.1 * aux_loss

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += (logits.argmax(1) == labels).float().sum().item()
        n += bs

    return total_loss / n, total_acc / n


# ============================================================
# Main
# ============================================================

def main():
    args = get_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True   # safe: input shapes are fixed within a run

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Dataset config ───────────────────────────────────────────────
    ConfigModule = importlib.import_module(
        f"config_files.{args.selected_dataset}_Configs"
    )
    configs = ConfigModule.Config()
    configs.batch_size = args.batch_size

    # ── Logging ─────────────────────────────────────────────────────
    # Derive modality suffix from missing-modality probabilities
    _pm_sig = args.p_missing_signal
    _pm_img = args.p_missing_image
    if _pm_img >= 1.0 and _pm_sig == 0:
        _mod_tag = "noimage"
    elif _pm_sig >= 1.0 and _pm_img == 0:
        _mod_tag = "notext"
    elif _pm_sig == 0 and _pm_img == 0:
        _mod_tag = "full"
    else:
        _mod_tag = f"pmS{_pm_sig}_pmI{_pm_img}"

    run_tag = (
        f"{args.selected_dataset}_tar{args.target}_rm{args.remain_rate}"
        f"_seed{args.seed}_mm_{_mod_tag}"
    )
    log_dir = os.path.join(args.logs_save_dir, run_tag)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"log_{datetime.now().strftime('%d%m%Y_%H%M%S')}.log"
    )
    logger = _logger(log_path)

    logger.debug("=" * 55)
    logger.debug(f"Dataset : {args.selected_dataset}  target={args.target}")
    logger.debug(f"Missing-modality distribution (text always present):")
    logger.debug(f"  p_full           = {args.p_full:.2f}")
    logger.debug(f"  p_missing_signal = {args.p_missing_signal:.2f}")
    logger.debug(f"  p_missing_image  = {args.p_missing_image:.2f}")
    logger.debug("=" * 55)

    # ── Sensor-layout text tokens (fixed, always present) ───────────
    sensor_text_ids = _tokenize_text(
        args.sensor_text, args.text_len, args.vocab_size
    ).to(device)   # [text_len]

    # ── Data ─────────────────────────────────────────────────────────
    print("Building data loaders…")
    train_loader, val_loader, test_loader = build_dataloaders(args, configs)

    # ── Model ────────────────────────────────────────────────────────
    model = MultiModalHARModel(
        in_channels  = configs.input_channels,
        num_classes  = configs.num_classes,
        hidden_dim   = args.hidden_dim,
        vocab_size   = args.vocab_size,
        max_text_len = args.text_len,
        max_lag      = args.max_lag,
        kernel_size  = configs.kernel_size,
        dropout      = configs.dropout,
        ctx_mode     = args.ctx_mode,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )

    # ── Training loop ────────────────────────────────────────────────
    best_val_acc = 0.0
    ckpt_path = os.path.join(args.logs_save_dir, f"{run_tag}_best.pt")

    for epoch in range(1, args.num_epoch + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, sensor_text_ids,
            args.p_full, args.p_missing_signal, args.p_missing_image,
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, sensor_text_ids,
        )
        scheduler.step(val_loss)

        logger.debug(
            f"Epoch {epoch:3d} | "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        print(
            f"Epoch {epoch:3d} | "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # Build classifier-free conditioner state (everything except classifier)
            conditioner_state = {
                k: v for k, v in model.state_dict().items()
                if not k.startswith("classifier")
            }

            torch.save(
                {"model_state_dict": model.state_dict(),
                 "conditioner_state_dict": conditioner_state,
                 "epoch": epoch, "val_acc": val_acc, "args": vars(args)},
                ckpt_path,
            )
            logger.debug(f"  ✓ Saved best checkpoint (val_acc={val_acc:.4f})")

    # ── Final evaluation on best checkpoint ─────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device, sensor_text_ids,
    )
    logger.debug(f"Test acc (best ckpt): {test_acc:.4f}")
    print(f"\nTest acc (best ckpt): {test_acc:.4f}")
    print(f"Checkpoint saved to : {ckpt_path}")


if __name__ == "__main__":
    main()
