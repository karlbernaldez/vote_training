from __future__ import annotations

"""
ml/transformer.py
-----------------
CNN-Transformer hybrid for spatio-temporal wind-field data.

Architecture
============

                        Input  (N, C, T, H, W)
                           │
              ┌────────────▼────────────┐
              │   CNN Patch Embedder    │   Conv3d stem  →  (N, d_model, T', H', W')
              └────────────┬────────────┘   T'=T  H'=H/4  W'=W/4
                           │  flatten + transpose
                        tokens  (N, L, d_model)      L = T' * H' * W'
                           │
              ┌────────────▼────────────┐
              │  Positional Encoding    │   learnable 3-D factorised
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Transformer Encoder    │   N_layers × (MHSA + FFN)
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  CNN Patch Decoder      │   ConvTranspose3d stem  →  (N, C, T, H, W)
              └─────────────────────────┘

Pretraining objective — Masked Autoencoding (MAE)
==================================================
A fraction *mask_ratio* of spatial tokens is randomly masked before the
Transformer.  The decoder reconstructs the full field.  MSE is computed
only on masked positions (like the original MAE paper) or on all positions
depending on the *mask_only_loss* flag.

The masking operates on **spatial** patches only (H'/W' axes) so the model
always sees the full temporal context — appropriate for meteorological data
where the time axis carries strong physical correlations.

Intended training workflow
==========================
1. Pretrain with MaskedAutoencoderWrapper  (train_transformer.py)
2. Strip the decoder; fine-tune the encoder + a lightweight head for a
   downstream task (e.g. wind-speed forecasting, anomaly detection).
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBNGELU3D(nn.Sequential):
    """Conv3d → BatchNorm3d → GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__(
            nn.Conv3d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
        )


class ResidualBlock3D(nn.Module):
    """Two-conv residual block with spatial dropout."""

    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


# ---------------------------------------------------------------------------
# CNN stem  (shared encoder front-end)
# ---------------------------------------------------------------------------


class CNNStem(nn.Module):
    """
    Progressively down-samples (H, W) by 4× via two stride-2 conv stages
    while projecting to *d_model* channels.  T is preserved throughout.

    Input : (N, in_channels, T, H, W)
    Output: (N, d_model,     T, H/4, W/4)   [ceil arithmetic]
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        mid = max(d_model // 2, 16)

        self.net = nn.Sequential(
            # stage 1: in_channels → mid,  H/2
            ConvBNGELU3D(in_channels, mid),
            ResidualBlock3D(mid, dropout),
            nn.MaxPool3d(kernel_size=(1, 2, 2), ceil_mode=True),

            # stage 2: mid → d_model,  H/4
            ConvBNGELU3D(mid, d_model),
            ResidualBlock3D(d_model, dropout),
            nn.MaxPool3d(kernel_size=(1, 2, 2), ceil_mode=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# CNN decoder  (reconstruction head)
# ---------------------------------------------------------------------------


class CNNDecoder(nn.Module):
    """
    Mirrors CNNStem: up-samples (H, W) back to original resolution.

    Input : (N, d_model, T, H/4, W/4)
    Output: (N, in_channels, T, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        mid = max(d_model // 2, 16)

        self.net = nn.Sequential(
            # upsample 1: d_model → mid,  H/2
            nn.ConvTranspose3d(d_model, mid, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(mid),
            nn.GELU(),
            ResidualBlock3D(mid, dropout),

            # upsample 2: mid → mid,  H
            nn.ConvTranspose3d(mid, mid, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(mid),
            nn.GELU(),
            ResidualBlock3D(mid, dropout),

            # final projection back to input channels
            nn.Conv3d(mid, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
        out = self.net(z)
        return out[..., : target_shape[-3], : target_shape[-2], : target_shape[-1]]


# ---------------------------------------------------------------------------
# Positional encoding — learnable factorised 3-D
# ---------------------------------------------------------------------------


class FactorisedLearnablePositionalEncoding(nn.Module):
    """
    Learnable positional embedding factorised along T, H', W' independently,
    then summed.  Factorisation keeps the parameter count small while still
    providing full 3-D positional information.

    Args:
        d_model:  Embedding dimension.
        max_t:    Maximum temporal length.
        max_h:    Maximum down-sampled height.
        max_w:    Maximum down-sampled width.
    """

    def __init__(
        self,
        d_model: int,
        max_t: int = 64,
        max_h: int = 64,
        max_w: int = 128,
    ) -> None:
        super().__init__()
        self.emb_t = nn.Embedding(max_t, d_model)
        self.emb_h = nn.Embedding(max_h, d_model)
        self.emb_w = nn.Embedding(max_w, d_model)

    def forward(self, t: int, h: int, w: int) -> torch.Tensor:
        """
        Returns positional embeddings of shape (1, T*H*W, d_model) ready to
        be added to the token sequence.
        """
        device = self.emb_t.weight.device
        idx_t = torch.arange(t, device=device)
        idx_h = torch.arange(h, device=device)
        idx_w = torch.arange(w, device=device)

        # (T, H, W, d_model) by broadcasting
        pe = (
            self.emb_t(idx_t)[:, None, None, :]   # (T, 1, 1, d)
            + self.emb_h(idx_h)[None, :, None, :]  # (1, H, 1, d)
            + self.emb_w(idx_w)[None, None, :, :]  # (1, 1, W, d)
        )
        return pe.reshape(1, t * h * w, -1)  # (1, L, d_model)


# ---------------------------------------------------------------------------
# Transformer encoder
# ---------------------------------------------------------------------------


class TransformerEncoderLayer(nn.Module):
    """
    Pre-LN Transformer encoder layer (more stable than post-LN for pretraining).

    Pre-LN:  x = x + Attn(LN(x))
             x = x + FFN(LN(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # self-attention (pre-LN)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        # FFN (pre-LN)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of *n_layers* pre-LN Transformer encoder layers."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Core CNN-Transformer model
# ---------------------------------------------------------------------------


class CNNTransformer(nn.Module):
    """
    CNN-Transformer encoder for spatio-temporal gridded data.

    The CNN stem tokenises the input field into a sequence of patch embeddings.
    The Transformer encoder contextualises the tokens globally.

    This class is the *encoder only* — it is reused both during MAE pretraining
    (inside MaskedAutoencoderWrapper) and for downstream fine-tuning.

    Args:
        in_channels:  Number of input variables (e.g. 5 wind components).
        d_model:      Transformer embedding dimension.
        n_heads:      Number of attention heads (must divide d_model).
        n_layers:     Number of Transformer encoder layers.
        ffn_dim:      Hidden size of the FFN sub-layer.
        dropout:      Dropout probability throughout.
        max_t:        Maximum temporal size (for positional encodings).
        max_h:        Maximum down-sampled height.
        max_w:        Maximum down-sampled width.
    """

    def __init__(
        self,
        in_channels: int = 5,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_t: int = 64,
        max_h: int = 64,
        max_w: int = 128,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        self.cnn_stem = CNNStem(in_channels, d_model, dropout)
        self.pos_enc = FactorisedLearnablePositionalEncoding(d_model, max_t, max_h, max_w)
        self.dropout = nn.Dropout(dropout)
        self.transformer = TransformerEncoder(d_model, n_heads, n_layers, ffn_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """
        Args:
            x:    (N, C, T, H, W)
            mask: (N, L) boolean tensor — True = masked (token is zeroed out).
                  L = T * ceil(H/4) * ceil(W/4).

        Returns:
            tokens:       (N, L, d_model)  contextualised token sequence.
            grid_shape:   (T, H', W')  spatial grid dimensions after the CNN stem,
                          needed by the decoder to reshape tokens back to a volume.
        """
        # CNN stem: (N, C, T, H, W) → (N, d_model, T, H', W')
        feat = self.cnn_stem(x)                         # (N, d_model, T, H', W')
        N, D, T, Hp, Wp = feat.shape

        # Flatten spatial+temporal → token sequence
        tokens = feat.permute(0, 2, 3, 4, 1)           # (N, T, H', W', d_model)
        tokens = tokens.reshape(N, T * Hp * Wp, D)     # (N, L, d_model)

        # Add positional encoding
        tokens = tokens + self.pos_enc(T, Hp, Wp)
        tokens = self.dropout(tokens)

        # Apply mask: zero out masked tokens (visible-only attention not used
        # here so the decoder can still receive full-sequence output)
        if mask is not None:
            tokens = tokens.masked_fill(mask.unsqueeze(-1), 0.0)

        # Transformer
        tokens = self.transformer(tokens)

        return tokens, (T, Hp, Wp)


# ---------------------------------------------------------------------------
# MAE wrapper
# ---------------------------------------------------------------------------


class MaskedAutoencoderWrapper(nn.Module):
    """
    Wraps CNNTransformer with a CNN decoder for Masked Autoencoder pretraining.

    Masking strategy
    ----------------
    *Spatial* patches are masked uniformly at random — the same set of (H', W')
    positions is masked across all T timesteps for each sample.  This ensures
    the model cannot trivially infer a masked spatial location from adjacent
    timesteps (which are otherwise very highly correlated).

    Loss
    ----
    MSE in normalised pixel space.  When *mask_only_loss=True* (default) the
    loss is averaged over masked tokens only, following the original MAE paper.
    When False, loss is computed over all positions.

    Args:
        in_channels:      Input channels.
        d_model:          Transformer hidden dim.
        n_heads:          Attention heads.
        n_layers:         Transformer depth.
        ffn_dim:          FFN hidden size.
        dropout:          Dropout probability.
        mask_ratio:       Fraction of spatial patches to mask (e.g. 0.75).
        mask_only_loss:   Compute loss on masked tokens only.
        max_t / max_h / max_w: Positional encoding bounds.
    """

    def __init__(
        self,
        in_channels: int = 5,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        mask_ratio: float = 0.75,
        mask_only_loss: bool = True,
        max_t: int = 64,
        max_h: int = 64,
        max_w: int = 128,
    ) -> None:
        super().__init__()

        self.mask_ratio = mask_ratio
        self.mask_only_loss = mask_only_loss

        self.encoder = CNNTransformer(
            in_channels=in_channels,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_t=max_t,
            max_h=max_h,
            max_w=max_w,
        )

        self.cnn_decoder = CNNDecoder(in_channels, d_model, dropout)

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def _make_spatial_mask(
        self,
        N: int,
        T: int,
        Hp: int,
        Wp: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample a random spatial mask and broadcast over T.

        Returns a boolean tensor of shape (N, T*Hp*Wp) where True = masked.
        The same (Hp, Wp) positions are masked for every timestep.
        """
        n_spatial = Hp * Wp
        n_mask = int(math.floor(n_spatial * self.mask_ratio))

        # Random spatial indices per sample
        noise = torch.rand(N, n_spatial, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)          # ascending
        ids_mask = ids_shuffle[:, :n_mask]                 # first n_mask → masked

        spatial_mask = torch.zeros(N, n_spatial, dtype=torch.bool, device=device)
        spatial_mask.scatter_(1, ids_mask, True)           # (N, Hp*Wp)

        # Broadcast across T: (N, Hp*Wp) → (N, T, Hp, Wp) → (N, T*Hp*Wp)
        spatial_mask = spatial_mask.unsqueeze(1).expand(N, T, Hp * Wp)
        return spatial_mask.reshape(N, T * Hp * Wp)        # (N, L)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Full MAE forward pass.

        Args:
            x: (N, C, T, H, W) normalised input.

        Returns a dict with keys:
            ``loss``          — scalar pretraining loss.
            ``reconstruction``— (N, C, T, H, W) full reconstruction.
            ``mask``          — (N, L) bool mask (True = was masked).
        """
        N = x.shape[0]

        # 1. Encode (CNN stem gives us the grid shape)
        #    We run the stem once to get T, Hp, Wp, then generate the mask.
        feat = self.encoder.cnn_stem(x)
        _, _, T, Hp, Wp = feat.shape

        mask = self._make_spatial_mask(N, T, Hp, Wp, x.device)

        # 2. Full encoder forward with mask
        tokens, grid_shape = self.encoder(x, mask=mask)

        # 3. Reshape tokens back to volume: (N, L, d_model) → (N, d_model, T, H', W')
        T_, Hp_, Wp_ = grid_shape
        z = tokens.reshape(N, T_, Hp_, Wp_, self.encoder.d_model)
        z = z.permute(0, 4, 1, 2, 3).contiguous()         # (N, d_model, T, H', W')

        # 4. CNN decoder
        recon = self.cnn_decoder(z, x.shape)               # (N, C, T, H, W)

        # 5. Loss — on masked tokens (compared in pixel/voxel space)
        loss = self._compute_loss(x, recon, mask, grid_shape)

        return {
            "loss": loss,
            "reconstruction": recon,
            "mask": mask,
        }

    def _compute_loss(
        self,
        target: torch.Tensor,
        recon: torch.Tensor,
        mask: torch.Tensor,
        grid_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        MSE loss.  If *mask_only_loss* is True, average over masked voxels only.
        """
        if not self.mask_only_loss:
            return F.mse_loss(recon, target)

        T, Hp, Wp = grid_shape
        N, C, _, H, W = target.shape

        # Upsample mask from token resolution to pixel resolution for loss.
        # mask: (N, T*Hp*Wp) → (N, T, Hp, Wp) → (N, 1, T, H, W) via nearest interp
        mask_vol = mask.reshape(N, T, Hp, Wp).float()      # (N, T, Hp, Wp)
        mask_vol = mask_vol.unsqueeze(1)                    # (N, 1, T, Hp, Wp)
        mask_vol = F.interpolate(
            mask_vol,
            size=(T, H, W),
            mode="nearest",
        )                                                   # (N, 1, T, H, W)

        diff_sq = (recon - target) ** 2                    # (N, C, T, H, W)
        masked_loss = (diff_sq * mask_vol).sum()
        n_masked = mask_vol.sum() * C

        return masked_loss / (n_masked + 1e-8)

    # ------------------------------------------------------------------
    # Convenience: encoder-only forward for downstream use
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run encoder only (no masking).  Returns mean-pooled token embedding
        of shape (N, d_model) — suitable as a global feature vector.
        """
        tokens, _ = self.encoder(x, mask=None)
        return tokens.mean(dim=1)