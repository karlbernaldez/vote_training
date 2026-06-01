from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ml.models import Conv3dAutoencoder


# -----------------------------------------------------
# Data
# -----------------------------------------------------

def load_inputs(dataset_dir: Path) -> torch.Tensor:
    X = np.load(
        dataset_dir / "X.npy"
    ).astype(np.float32)

    return torch.from_numpy(X)


def load_targets(
    csv_path: Path,
    columns: list[str],
) -> torch.Tensor:

    df = pd.read_csv(csv_path)

    y = (
        df[columns]
        .to_numpy(dtype=np.float32)
        .copy()
    )

    return torch.from_numpy(y)


# -----------------------------------------------------
# Positional Encoding
# -----------------------------------------------------

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
    ):
        super().__init__()

        pe = torch.zeros(
            max_len,
            d_model,
        )

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(self, x):

        seq_len = x.size(1)

        return x + self.pe[:, :seq_len]


# -----------------------------------------------------
# Transformer Predictor
# -----------------------------------------------------

class TransformerPredictor(nn.Module):

    def __init__(
        self,
        encoder,
        latent_channels,
        output_dim=1,
        num_layers=2,
        nhead=4,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder = encoder

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.positional_encoding = PositionalEncoding(
            latent_channels
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.Linear(
                latent_channels,
                64,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                64,
                output_dim,
            ),
        )

    def forward(self, x):

        z = self.encoder(x)

        # B,C,T,H,W

        z = z.mean(dim=(3, 4))

        # B,C,T

        z = z.permute(0, 2, 1)

        # B,T,C

        z = self.positional_encoding(z)

        z = self.transformer(z)

        pooled = z.mean(dim=1)

        return self.head(pooled)


# -----------------------------------------------------
# Autoencoder
# -----------------------------------------------------

def load_encoder(
    checkpoint_path,
    run_meta_path,
    device,
):
    with open(run_meta_path, "r") as f:
        meta = json.load(f)

    latent_channels = meta["config"]["latent_channels"]

    model = Conv3dAutoencoder(
        in_channels=5,
        latent_channels=latent_channels,
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(state_dict)

    return (
        model.encoder,
        latent_channels,
    )


# -----------------------------------------------------
# Metrics
# -----------------------------------------------------

def compute_metrics(
    predictions,
    targets,
):
    mae = np.mean(
        np.abs(predictions - targets)
    )

    mse = np.mean(
        (predictions - targets) ** 2
    )

    rmse = np.sqrt(mse)

    mape = np.mean(
        np.abs(
            (targets - predictions)
            / (targets + 1e-6)
        )
    ) * 100

    ss_res = np.sum(
        (targets - predictions) ** 2
    )

    ss_tot = np.sum(
        (targets - np.mean(targets)) ** 2
    )

    r2 = 1.0 - (
        ss_res / (ss_tot + 1e-12)
    )

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
    }


# -----------------------------------------------------
# Prediction
# -----------------------------------------------------

@torch.no_grad()
def predict(
    model,
    X,
    device,
):
    model.eval()

    X = X.to(device)

    preds = model(X)

    return preds.cpu().numpy()


# -----------------------------------------------------
# Main
# -----------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
    )

    parser.add_argument(
        "--target-csv",
        required=True,
    )

    parser.add_argument(
        "--target-columns",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--autoencoder",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    X = load_inputs(
        Path(args.dataset_dir)
    )

    y = load_targets(
        Path(args.target_csv),
        args.target_columns,
    )

    encoder, latent_channels = load_encoder(
        Path(args.autoencoder),
        Path(args.autoencoder).parent
        / "run_meta.json",
        device,
    )

    model = TransformerPredictor(
        encoder=encoder,
        latent_channels=latent_channels,
        output_dim=y.shape[1],
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    predictions = predict(
        model,
        X,
        device,
    )

    targets = y.numpy()

    np.save(
        output_dir / "predictions.npy",
        predictions,
    )

    np.save(
        output_dir / "targets.npy",
        targets,
    )

    metrics = compute_metrics(
        predictions,
        targets,
    )

    with open(
        output_dir / "metrics.json",
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    print("\nEvaluation")
    print("-" * 40)

    for k, v in metrics.items():
        print(
            f"{k:<10} {v:.6f}"
        )


if __name__ == "__main__":
    main()