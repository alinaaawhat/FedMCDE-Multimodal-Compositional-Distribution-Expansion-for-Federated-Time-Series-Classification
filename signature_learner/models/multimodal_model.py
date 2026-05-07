"""
Multimodal Human Activity Recognition Model

Inputs:
  signal:   [B, T, C]      multivariate time-series sensor data
  text_ids: [B, L]         tokenized metadata description (long tensor)
  image:    [B, 3, H, W]   STFT time-frequency image
  mask:     [B, 3] float   modality availability: 1=present, 0=missing
                            mask[:,0]=signal, mask[:,1]=text, mask[:,2]=image

Outputs:
  logits: [B, num_classes]
  c_t:    [B, hidden_dim]  conditioning context vector
                           (same interface as the original conditioner)
  aux:    dict             auxiliary losses (only when return_aux=True)
                             content_consistency, orthogonality, style_diversity

Design:
  - Signal is always the primary modality.
  - Missing text / image → learnable null token (no recovery).
  - Modality mask embedding tells the model which are missing.
  - Content / style disentanglement via two-layer MLP projection heads.
  - Modality-specific (c_m, s_m) fused into global content c and style s.
  - Gated fusion across modalities for content and style.
  - Reliability-aware gating scales each modality's token sequence;
    reliability scores for missing modalities are hard-zeroed.
  - Cross-attention: signal tokens as Q only, cond tokens as K/V.
    cond_tokens = [text_tokens | image_tokens | c_token | s_token | mask_embed]
  - Token-level lag-aware DynamicAlignment finalises the context vector.
  - Auxiliary losses: content consistency, content-style orthogonality,
    style diversity (returned only when return_aux=True).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Signal → STFT image utility
# ============================================================
def batch_signal_to_stft_image(
    signal: torch.Tensor,   # [B, T, C]
    img_h: int = 64,
    img_w: int = 64,
    n_fft: int = 32,
    hop_length: int = 16,
    win_length: int = 32,
) -> torch.Tensor:
    """
    Convert a batch of multivariate time-series signals to 3-channel
    STFT spectrogram images.

    signal: [B, T, C]
    return: [B, 3, img_h, img_w]
    """
    B, T, C = signal.shape
    signal = signal.float()
    sig_bct = signal.transpose(1, 2)          # [B, C, T]

    window = torch.hann_window(win_length, device=signal.device)

    x = sig_bct.reshape(B * C, T)            # [B*C, T]
    stft = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True,
        pad_mode="reflect",
    )                                          # [B*C, F, TT]

    spec = torch.log1p(stft.abs())            # [B*C, F, TT]

    Freq, Time = spec.shape[-2], spec.shape[-1]
    spec = spec.reshape(B, C, Freq, Time)     # [B, C, F, TT]

    ch_mean = spec.mean(dim=1)                # [B, F, TT]
    ch_max  = spec.max(dim=1).values          # [B, F, TT]
    ch_std  = spec.std(dim=1, unbiased=False) # [B, F, TT]

    image = torch.stack([ch_mean, ch_max, ch_std], dim=1)  # [B, 3, F, TT]

    image = F.interpolate(
        image, size=(img_h, img_w),
        mode="bilinear", align_corners=False,
    )                                          # [B, 3, img_h, img_w]

    vmin = image.amin(dim=(1, 2, 3), keepdim=True)
    vmax = image.amax(dim=(1, 2, 3), keepdim=True)
    image = (image - vmin) / (vmax - vmin + 1e-6)

    return image   # [B, 3, img_h, img_w]


# ============================================================
# SignalEncoder
# ============================================================
class SignalEncoder(nn.Module):
    """
    Three-block 1-D convolutional encoder for multivariate time-series.

    Input:  x [B, C, T]
    Output: tokens [B, N_x, D]   N_x ≈ T / 8  (three 2× pooling steps)
    """
    def __init__(self, in_channels: int, hidden_dim: int,
                 kernel_size: int = 9, dropout: float = 0.35):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size, stride=1,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=1),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, hidden_dim, 8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=1),
        )

    def forward(self, x):
        # x: [B, T, C] → [B, C, T] for Conv1d
        x = x.transpose(1, 2)
        x = self.conv1(x)  # [B, 32, T/2]
        x = self.conv2(x)  # [B, 64, T/4]
        x = self.conv3(x)  # [B, D,  T/8]
        return x.transpose(1, 2)  # [B, N_x, D]


# ============================================================
# TextEncoder
# ============================================================
class TextEncoder(nn.Module):
    """
    Embedding + lightweight Transformer for tokenised text metadata.

    Input:  ids [B, L]  long tensor
    Output: tokens [B, L, D]
    """
    def __init__(self, vocab_size: int, hidden_dim: int,
                 max_len: int = 128, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, ids):
        # ids: [B, L]
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0)  # [1, L]
        x   = self.embed(ids) + self.pos_embed(pos)            # [B, L, D]
        return self.transformer(x)                              # [B, L, D]


# ============================================================
# ImageEncoder
# ============================================================
class ImageEncoder(nn.Module):
    """
    Three-stage 2-D CNN for STFT time-frequency images.

    Input:  img [B, 3, H, W]
    Output: tokens [B, N_v, D]   N_v = (H/8) * (W/8)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),   # H/2, W/2
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # H/4, W/4
            nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1), nn.ReLU(),  # H/8, W/8
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, img):
        # img: [B, 3, H, W]
        feat = self.cnn(img)                   # [B, D, H/8, W/8]
        B, D, h, w = feat.shape
        feat = feat.flatten(2).transpose(1, 2) # [B, N_v, D]
        return self.proj(feat)                 # [B, N_v, D]


# ============================================================
# ContentStyleDisentangler
# ============================================================
class ContentStyleDisentangler(nn.Module):
    """
    Split a modality summary vector into content and style halves via MLP heads.
    Two-layer MLP (feat_dim → feat_dim → feat_dim//2) with ReLU non-linearity.

    Input:  h [B, D]
    Output: (h_c [B, D//2], h_s [B, D//2])
    """
    def __init__(self, feat_dim: int):
        super().__init__()
        half = feat_dim // 2
        self.content_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, half),
        )
        self.style_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, half),
        )

    def forward(self, h):
        return self.content_head(h), self.style_head(h)


# ============================================================
# GatedFusion
# ============================================================
class GatedFusion(nn.Module):
    """
    Fuse a list of M vectors [B, D] with learned softmax gates.

    Output: [B, D]
    """
    def __init__(self, feat_dim: int, num_modalities: int):
        super().__init__()
        self.gate = nn.Linear(feat_dim * num_modalities, num_modalities)

    def forward(self, features):
        # features: list of M tensors, each [B, D]
        concat  = torch.cat(features, dim=-1)             # [B, D*M]
        gates   = torch.softmax(self.gate(concat), dim=-1) # [B, M]
        stacked = torch.stack(features, dim=1)             # [B, M, D]
        return (stacked * gates.unsqueeze(-1)).sum(dim=1)  # [B, D]


# ============================================================
# ReliabilityGate
# ============================================================
class ReliabilityGate(nn.Module):
    """
    Estimate a reliability scalar ρ ∈ (0, 1) for a modality.

    Input:  h [B, D]
    Output: ρ [B, 1]
    """
    def __init__(self, feat_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2), nn.ReLU(),
            nn.Linear(feat_dim // 2, 1), nn.Sigmoid(),
        )

    def forward(self, h):
        return self.fc(h)  # [B, 1]


# ============================================================
# UnifiedFusion
# ============================================================
class UnifiedFusion(nn.Module):
    """
    Cross-attention fusion: signal tokens as Query, condition tokens as K/V.

    Input:
      signal_tokens: [B, N_x, D]
      cond_tokens:   [B, N_c, D]
    Output:
      fused: [B, N_x, D]
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, signal_tokens, cond_tokens):
        attn_out, _ = self.cross_attn(
            query=signal_tokens,
            key=cond_tokens,
            value=cond_tokens,
        )
        return self.norm(signal_tokens + attn_out)  # [B, N_x, D]


# ============================================================
# DynamicAlignment
# ============================================================
class DynamicAlignment(nn.Module):
    """
    Token-level lag-aware soft alignment.

    For each lag l in [-max_lag, ..., max_lag]:
      1. Circularly shift cond_tokens by l steps along the sequence axis.
      2. Compute mean token-level dot-product similarity with signal_tokens.
    Softmax over lag scores → scalar weights per lag.
    Aggregate shifted condition tokens into H_bar, return H_bar.mean(1).

    Input:
      signal_tokens: [B, N_x, D]
      cond_tokens:   [B, N_c, D]
    Output:
      aligned: [B, D]
    """
    def __init__(self, hidden_dim: int, max_lag: int = 3):
        super().__init__()
        self.max_lag = max_lag
        self.scale   = hidden_dim ** 0.5

    def forward(self, signal_tokens, cond_tokens):
        N_x = signal_tokens.shape[1]
        N_c = cond_tokens.shape[1]
        N   = min(N_x, N_c)
        sig_q = signal_tokens[:, :N, :]   # [B, N, D]  – query anchor

        lag_scores    = []
        shifted_conds = []

        for l in range(-self.max_lag, self.max_lag + 1):
            shifted = torch.roll(cond_tokens, l, dims=1)           # [B, N_c, D]
            # mean token-level dot product between sig_q and first N shifted tokens
            sim = (sig_q * shifted[:, :N, :]).sum(-1).mean(-1)     # [B]
            lag_scores.append(sim / self.scale)
            shifted_conds.append(shifted)

        lag_scores = torch.stack(lag_scores, dim=1)                # [B, num_lags]
        weights    = torch.softmax(lag_scores, dim=1)              # [B, num_lags]

        # Weighted sum of shifted condition token sequences
        shifted_stack = torch.stack(shifted_conds, dim=1)          # [B, num_lags, N_c, D]
        H_bar = (shifted_stack * weights[:, :, None, None]).sum(1) # [B, N_c, D]

        return H_bar.mean(1)  # [B, D]


# ============================================================
# ClassifierHead
# ============================================================
class ClassifierHead(nn.Module):
    """Linear classifier on top of the context vector."""
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)  # [B, num_classes]


# ============================================================
# MultiModalHARModel
# ============================================================
class MultiModalHARModel(nn.Module):
    """
    Full multimodal HAR model with content/style disentanglement,
    reliability-aware gating (missing modalities hard-zeroed),
    unified cross-attention fusion (signal Q, cond K/V only),
    and token-level lag-aware dynamic alignment.

    Interface (same as the original conditioner):
      forward(signal [B,T,C], text_ids, image, mask, return_aux=False)
        → (logits [B, num_classes], c_t [B, hidden_dim])
        → (logits, c_t, aux_dict) when return_aux=True
    """

    NUM_MODALITIES = 3  # signal, text, image

    def __init__(
        self,
        in_channels:  int = 45,
        num_classes:  int = 19,
        hidden_dim:   int = 100,
        vocab_size:   int = 1000,
        max_text_len: int = 64,
        max_lag:      int = 3,
        kernel_size:  int = 9,
        dropout:      float = 0.35,
        ctx_mode:     str = "signal+cond",  # "orig" | "signal+cond" | "signal_only"
    ):
        super().__init__()
        D  = hidden_dim
        M  = self.NUM_MODALITIES

        # ── Modality encoders ──────────────────────────────────────────
        self.signal_encoder = SignalEncoder(in_channels, D, kernel_size, dropout)
        self.text_encoder   = TextEncoder(vocab_size, D, max_len=max_text_len)
        self.image_encoder  = ImageEncoder(D)

        # ── Learnable null tokens for missing signal / text / image ────
        # shape [1, 1, D] – broadcast over batch and sequence
        self.null_signal_token = nn.Parameter(torch.randn(1, 1, D))
        self.null_text_token   = nn.Parameter(torch.randn(1, 1, D))
        self.null_image_token  = nn.Parameter(torch.randn(1, 1, D))

        # ── Modality mask embedding: 2^3 = 8 possible combinations ─────
        self.mask_embed = nn.Embedding(2 ** M, D)

        # ── Content / style disentanglers (one per modality) ──────────
        self.disentanglers = nn.ModuleDict({
            'signal': ContentStyleDisentangler(D),
            'text':   ContentStyleDisentangler(D),
            'image':  ContentStyleDisentangler(D),
        })

        # ── Gated fusion for content and style across modalities ───────
        half_D = D // 2
        self.content_fusion = GatedFusion(half_D, M)
        self.style_fusion   = GatedFusion(half_D, M)

        # ── Project fused content / style back to D ────────────────────
        self.content_proj = nn.Linear(half_D, D)
        self.style_proj   = nn.Linear(half_D, D)

        # ── Per-modality reliability gates ─────────────────────────────
        self.rel_gate_signal = ReliabilityGate(D)
        self.rel_gate_text   = ReliabilityGate(D)
        self.rel_gate_image  = ReliabilityGate(D)

        # ── Unified cross-attention fusion ─────────────────────────────
        self.fusion = UnifiedFusion(D)

        # ── Lag-aware dynamic alignment ────────────────────────────────
        self.alignment = DynamicAlignment(D, max_lag=max_lag)

        # ── Final projection to context vector ─────────────────────────
        self.ctx_proj = nn.Linear(D, D)
        self.ctx_mode = ctx_mode

        # ── Classifier ─────────────────────────────────────────────────
        self.classifier = ClassifierHead(D, num_classes)

    # ------------------------------------------------------------------
    def _mask_to_idx(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert [B, 3] binary mask → [B] integer index (0–7)."""
        return (mask[:, 0].long() * 4
                + mask[:, 1].long() * 2
                + mask[:, 2].long())

    # ------------------------------------------------------------------
    def compute_aux_losses(
        self,
        sig_c, sig_s,
        txt_c, txt_s,
        img_c, img_s,
        mask: torch.Tensor,
    ) -> dict:
        """
        Three auxiliary losses for content/style disentanglement.

        mask: [B, 3]  mask[:,0]=signal, mask[:,1]=text, mask[:,2]=image

        Returns a dict with keys:
          'content_consistency'  – present-modality content vectors should align
          'orthogonality'        – content ⊥ style within each modality
          'style_diversity'      – style vectors should differ across modalities
        """
        dev = sig_c.device

        def _weighted_mean(vals, weights):
            """Weighted mean of a [B] tensor, ignoring zero-weight samples."""
            denom = weights.sum() + 1e-8
            return (vals * weights).sum() / denom

        # ── Content consistency ─────────────────────────────────────────
        cc = torch.tensor(0.0, device=dev)
        if mask[:, 1].sum() > 0:
            sim = F.cosine_similarity(sig_c, txt_c, dim=-1)   # [B]
            cc  = cc + _weighted_mean(1.0 - sim, mask[:, 1])
        if mask[:, 2].sum() > 0:
            sim = F.cosine_similarity(sig_c, img_c, dim=-1)   # [B]
            cc  = cc + _weighted_mean(1.0 - sim, mask[:, 2])

        # ── Orthogonality: (ĉ · ŝ)² ────────────────────────────────────
        def _orth(c, s):
            return (F.normalize(c, dim=-1) * F.normalize(s, dim=-1)).sum(-1).pow(2)

        orth = _orth(sig_c, sig_s).mean()                       # signal always present
        if mask[:, 1].sum() > 0:
            orth = orth + _weighted_mean(_orth(txt_c, txt_s), mask[:, 1])
        if mask[:, 2].sum() > 0:
            orth = orth + _weighted_mean(_orth(img_c, img_s), mask[:, 2])

        # ── Style diversity: styles across modalities should differ ─────
        sd = torch.tensor(0.0, device=dev)
        if mask[:, 1].sum() > 0:
            sim = F.cosine_similarity(sig_s, txt_s, dim=-1).abs()
            sd  = sd + _weighted_mean(sim, mask[:, 1])
        if mask[:, 2].sum() > 0:
            sim = F.cosine_similarity(sig_s, img_s, dim=-1).abs()
            sd  = sd + _weighted_mean(sim, mask[:, 2])

        return {
            'content_consistency': cc,
            'orthogonality':       orth,
            'style_diversity':     sd,
        }

    # ------------------------------------------------------------------
    def forward(
        self,
        signal:   torch.Tensor,  # [B, T, C]
        text_ids: torch.Tensor,  # [B, L]
        image:    torch.Tensor,  # [B, 3, H, W]
        mask:     torch.Tensor,  # [B, 3]  float, 1=present 0=missing
        return_aux: bool = False,
    ):
        B = signal.shape[0]
        D = self.null_text_token.shape[-1]

        # ── Signal tokens (replace with null when missing) ────────────
        Hx_full     = self.signal_encoder(signal)                    # [B, N_x, D]
        null_sig    = self.null_signal_token.expand(B, Hx_full.shape[1], D)
        sig_avail   = mask[:, 0].view(B, 1, 1)                      # [B, 1, 1]
        Hx          = torch.where(sig_avail > 0.5, Hx_full, null_sig)
        sig_summary = Hx.mean(dim=1)                                 # [B, D]

        # ── Text tokens (replace with null when missing) ───────────────
        Hd_full    = self.text_encoder(text_ids)             # [B, L, D]
        null_text  = self.null_text_token.expand(B, Hd_full.shape[1], D)
        text_avail = mask[:, 1].view(B, 1, 1)               # [B, 1, 1]
        Hd         = torch.where(text_avail > 0.5, Hd_full, null_text)
        txt_summary = Hd.mean(dim=1)                         # [B, D]

        # ── Image tokens (replace with null when missing) ──────────────
        Hv_full   = self.image_encoder(image)                # [B, N_v, D]
        null_img  = self.null_image_token.expand(B, Hv_full.shape[1], D)
        img_avail = mask[:, 2].view(B, 1, 1)                # [B, 1, 1]
        Hv        = torch.where(img_avail > 0.5, Hv_full, null_img)
        img_summary = Hv.mean(dim=1)                         # [B, D]

        # ── Reliability scores (masked by availability) ────────────────
        rho_x = self.rel_gate_signal(sig_summary) * mask[:, 0:1]    # [B, 1]  0 if missing
        rho_d = self.rel_gate_text(txt_summary)   * mask[:, 1:2]    # [B, 1]  0 if missing
        rho_v = self.rel_gate_image(img_summary)  * mask[:, 2:3]    # [B, 1]  0 if missing

        # Scale token sequences by reliability
        Hx_s = Hx * rho_x.unsqueeze(1)   # [B, N_x, D]
        Hd_s = Hd * rho_d.unsqueeze(1)   # [B, L,   D]
        Hv_s = Hv * rho_v.unsqueeze(1)   # [B, N_v, D]

        # ── Content / style disentanglement ───────────────────────────
        sig_c, sig_s = self.disentanglers['signal'](sig_summary)  # [B, D//2]
        txt_c, txt_s = self.disentanglers['text'](txt_summary)    # [B, D//2]
        img_c, img_s = self.disentanglers['image'](img_summary)   # [B, D//2]

        c_fused = self.content_fusion([sig_c, txt_c, img_c])  # [B, D//2]
        s_fused = self.style_fusion([sig_s, txt_s, img_s])    # [B, D//2]

        c_token = self.content_proj(c_fused).unsqueeze(1)  # [B, 1, D]
        s_token = self.style_proj(s_fused).unsqueeze(1)    # [B, 1, D]

        # ── Modality mask embedding ────────────────────────────────────
        mask_idx = self._mask_to_idx(mask)           # [B]
        e_m      = self.mask_embed(mask_idx).unsqueeze(1)  # [B, 1, D]

        # ── Condition tokens: [ρ_d·Hd | ρ_v·Hv | c | s | e_m]
        # Signal tokens are Query only – NOT included in K/V.
        cond_tokens = torch.cat([
            Hd_s,    # [B, L,   D]
            Hv_s,    # [B, N_v, D]
            c_token, # [B, 1,   D]
            s_token, # [B, 1,   D]
            e_m,     # [B, 1,   D]
        ], dim=1)    # [B, N_c, D]

        # ── Unified fusion: signal tokens as Q, cond as K/V ───────────
        fused = self.fusion(Hx_s, cond_tokens)  # [B, N_x, D]

        # ── Dynamic alignment → cond-side summary ──────────────────────
        aligned_cond = self.alignment(fused, cond_tokens)  # [B, D]

        # ── Context vector: three ablation modes ───────────────────────
        # A "orig":        only alignment output (original)
        # B "signal+cond": signal backbone + alignment (recommended)
        # C "signal_only": only signal backbone
        if self.ctx_mode == "signal+cond":
            c_t = self.ctx_proj(fused.mean(dim=1) + aligned_cond)
        elif self.ctx_mode == "signal_only":
            c_t = self.ctx_proj(fused.mean(dim=1))
        else:  # "orig"
            c_t = self.ctx_proj(aligned_cond)

        # ── Classification ─────────────────────────────────────────────
        logits = self.classifier(c_t)  # [B, num_classes]

        if return_aux:
            aux = self.compute_aux_losses(
                sig_c, sig_s, txt_c, txt_s, img_c, img_s, mask
            )
            return logits, c_t, aux

        return logits, c_t


# ============================================================
# Quick forward-pass test
# ============================================================
if __name__ == '__main__':
    torch.manual_seed(42)

    B, T, C       = 4, 125, 45   # batch, time steps, signal channels
    L             = 32            # text token length
    H, W          = 64, 64        # STFT image size
    num_classes   = 19
    hidden_dim    = 100

    model = MultiModalHARModel(
        in_channels  = C,
        num_classes  = num_classes,
        hidden_dim   = hidden_dim,
        vocab_size   = 1000,
        max_text_len = L,
    )
    model.eval()

    signal   = torch.randn(B, T, C)           # [B, T, C]
    text_ids = torch.randint(0, 1000, (B, L))
    # derive image from signal via STFT
    image    = batch_signal_to_stft_image(signal, img_h=H, img_w=W)  # [B, 3, H, W]

    # Four different missing-modality scenarios
    mask = torch.tensor([
        [1, 1, 1],   # all present
        [1, 0, 1],   # text missing
        [1, 1, 0],   # image missing
        [1, 0, 0],   # only signal
    ], dtype=torch.float)

    # ── Baseline forward (no aux losses) ───────────────────────────────
    with torch.no_grad():
        logits, c_t = model(signal, text_ids, image, mask)

    print(f"logits : {logits.shape}")   # [4, 19]
    print(f"c_t    : {c_t.shape}")      # [4, 100]
    print("Forward pass OK!")

    # ── Forward with auxiliary losses (training mode) ──────────────────
    model.train()
    logits, c_t, aux = model(signal, text_ids, image, mask, return_aux=True)
    total_aux = (aux['content_consistency']
                 + aux['orthogonality']
                 + 0.1 * aux['style_diversity'])
    print(f"\nAux losses:")
    for k, v in aux.items():
        print(f"  {k}: {v.item():.4f}")
    print(f"  total_aux: {total_aux.item():.4f}")

    # ── Backward check ─────────────────────────────────────────────────
    ce_loss = torch.nn.CrossEntropyLoss()(
        logits, torch.randint(0, num_classes, (B,))
    )
    (ce_loss + total_aux).backward()
    print("\nBackward pass OK!")

    # ── Parameter count ────────────────────────────────────────────────
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
