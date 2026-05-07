"""
MissingModalityDataset
======================
Wraps any HAR dataset and randomly applies per-sample modality-missing masks
according to configurable probabilities.

Training distribution (defaults: 25% each):
    p_full           – signal + text + image all present  → mask [1, 1, 1]
    p_missing_signal – signal missing                     → mask [0, 1, 1]
    p_missing_text   – text missing                       → mask [1, 0, 1]
    p_missing_image  – image missing                      → mask [1, 1, 0]

The input/output interface is unchanged:
    signal   : [T, C]          real sensor time-series (always passed through)
    text_ids : [text_len]      token ids (zeros – no real text corpus needed)
    image    : [3, img_h, img_w] pseudo-image derived from signal via resize
    mask     : [3] float       1=present, 0=missing
    label    : scalar long
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MissingModalityDataset(Dataset):
    """
    Args:
        base_dataset         : dataset whose __getitem__ returns (x [T,C], y, ...)
        p_full               : fraction of samples with all modalities present
        p_missing_signal     : fraction with signal masked
        p_missing_text       : fraction with text  masked
        p_missing_image      : fraction with image masked
        text_len             : length of dummy text token sequence
        img_h, img_w         : spatial size of the generated pseudo-image
        seed                 : RNG seed for reproducible pattern assignment
        resample_each_epoch  : if True, re-draw patterns on every __getitem__
                               call (stochastic); if False, patterns are fixed
                               at construction time (deterministic, default)
    """

    # Pattern index → mask [signal, text, image]
    _MASKS = {
        0: [1., 1., 1.],  # full
        1: [0., 1., 1.],  # missing signal
        2: [1., 0., 1.],  # missing text
        3: [1., 1., 0.],  # missing image
    }

    def __init__(
        self,
        base_dataset: Dataset,
        p_full: float = 0.25,
        p_missing_signal: float = 0.25,
        p_missing_text: float = 0.25,
        p_missing_image: float = 0.25,
        text_len: int = 32,
        img_h: int = 32,
        img_w: int = 32,
        seed: int = 42,
        resample_each_epoch: bool = False,
    ):
        self.base = base_dataset
        self.text_len = text_len
        self.img_h = img_h
        self.img_w = img_w
        self.resample_each_epoch = resample_each_epoch
        self.rng = np.random.default_rng(seed)

        probs = np.array(
            [p_full, p_missing_signal, p_missing_text, p_missing_image],
            dtype=np.float64,
        )
        assert abs(probs.sum() - 1.0) < 1e-5, (
            f"Missing-modality probabilities must sum to 1, got {probs.sum():.4f}"
        )
        self.probs = probs

        if not resample_each_epoch:
            self._pattern_ids = self.rng.choice(4, size=len(base_dataset), p=probs)
        else:
            self._pattern_ids = None

    # ------------------------------------------------------------------
    def _signal_to_image(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Convert [T, C] sensor signal to a [3, H, W] pseudo-image.

        Strategy: treat [T, C] as a 2D spatial feature map → [1, 1, T, C],
        bilinear-resize to [1, H, W], replicate to 3 channels, normalize to [0, 1].
        """
        x = signal.float().unsqueeze(0).unsqueeze(0)  # [1, 1, T, C]
        img = F.interpolate(
            x, size=(self.img_h, self.img_w),
            mode="bilinear", align_corners=False,
        ).squeeze(0)  # [1, H, W]
        img = img.repeat(3, 1, 1)  # [3, H, W]
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        return img

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.base)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        item = self.base[idx]
        x = item[0].float()  # [T, C]
        y = item[1]

        # ── Dummy text tokens (zeros; mask controls whether model uses them)
        text_ids = torch.zeros(self.text_len, dtype=torch.long)

        # ── Pseudo-image derived from signal
        image = self._signal_to_image(x)  # [3, H, W]

        # ── Missing-modality mask
        if self.resample_each_epoch:
            pid = int(self.rng.choice(4, p=self.probs))
        else:
            pid = int(self._pattern_ids[idx])

        mask = torch.tensor(self._MASKS[pid], dtype=torch.float)

        return x, text_ids, image, mask, y
