from __future__ import annotations

import torch
from torch import nn


class ConvBNGELU3D(nn.Sequential):
    """Conv3d → BatchNorm3d → GELU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )


class ResidualBlock3D(nn.Module):
    """Pre-activation residual block with dropout."""

    def __init__(
        self,
        channels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class Conv3dAutoencoder(nn.Module):
    """
    3-D convolutional autoencoder for spatio-temporal data.

    Spatial dimensions (H, W) are halved twice in the encoder and
    restored in the decoder. The temporal dimension (T) is left
    untouched so temporal resolution is fully preserved.

    Args:
        in_channels:     Number of input channels (e.g. wind components).
        latent_channels: Channel width of the bottleneck representation.
        dropout:         Dropout probability used in residual blocks.
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            # --- stage 1: in_channels → 16 ---
            ConvBNGELU3D(in_channels, 16),
            ResidualBlock3D(16, dropout),
            nn.MaxPool3d(kernel_size=(1, 2, 2), ceil_mode=True),   # H/2, W/2

            # --- stage 2: 16 → 32 ---
            ConvBNGELU3D(16, 32),
            ResidualBlock3D(32, dropout),
            nn.MaxPool3d(kernel_size=(1, 2, 2), ceil_mode=True),   # H/4, W/4

            # --- bottleneck: 32 → latent_channels ---
            ConvBNGELU3D(32, latent_channels),
            nn.Dropout3d(dropout),
        )

        self.decoder = nn.Sequential(
            # --- stage 1: latent_channels → 32 ---
            nn.ConvTranspose3d(latent_channels, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.GELU(),
            ResidualBlock3D(32, dropout),

            # --- stage 2: 32 → 16 ---
            nn.ConvTranspose3d(32, 16, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(16),
            nn.GELU(),
            ResidualBlock3D(16, dropout),

            # --- output projection: 16 → in_channels ---
            nn.Conv3d(16, in_channels, kernel_size=3, padding=1),
        )

        self.apply(self._initialize_weights)

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
        out = self.decoder(z)
        # Trim any extra pixels introduced by ceil_mode / ConvTranspose padding.
        return out[..., : target_shape[-3], : target_shape[-2], : target_shape[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x), x.shape)