from __future__ import annotations

import torch
from torch import nn


class Conv3dAutoencoder(nn.Module):
    """Small 3-layer Conv3D autoencoder for gridded wind pretraining.

    Input/output shape: batch x channels x time x height x width.
    """

    def __init__(self, in_channels: int, latent_channels: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(24, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_channels, 24, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(24, 16, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded[..., : x.shape[-3], : x.shape[-2], : x.shape[-1]]


class Conv3dPredictor(nn.Module):
    """Predictor that reuses a Conv3D encoder and predicts scalar/vector targets."""

    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        latent_channels: int = 32,
        pretrained_autoencoder: Conv3dAutoencoder | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        if pretrained_autoencoder is None:
            self.encoder = Conv3dAutoencoder(in_channels, latent_channels).encoder
        else:
            self.encoder = pretrained_autoencoder.encoder

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(latent_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.head(encoded)
