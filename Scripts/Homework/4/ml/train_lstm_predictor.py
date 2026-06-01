# ml/train_lstm_predictor.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from ml.models import Conv3dAutoencoder


# -----------------------------------------------------
# Data loading
# -----------------------------------------------------

def load_inputs(dataset_dir: Path) -> torch.Tensor:
    x_path = dataset_dir / "X.npy"

    if not x_path.exists():
        raise FileNotFoundError(x_path)

    X = np.load(x_path).astype(np.float32)

    if X.ndim != 5:
        raise ValueError(
            f"Expected N,C,T,H,W tensor. Got {X.shape}"
        )

    return torch.from_numpy(X)


def load_targets(csv_path: Path, columns: list[str]) -> torch.Tensor:
    df = pd.read_csv(csv_path)

    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing target columns: {missing}"
        )

    y = df[columns].to_numpy(
        dtype=np.float32
    ).copy()

    return torch.from_numpy(y)


# -----------------------------------------------------
# LSTM Predictor
# -----------------------------------------------------

class LSTMPredictor(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        latent_channels: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.encoder = encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=latent_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):

        z = self.encoder(x)

        # z:
        # B,C,T,H,W

        z = z.mean(dim=(3, 4))

        # B,C,T

        z = z.permute(0, 2, 1)

        # B,T,C

        output, (hidden, _) = self.lstm(z)

        final_state = hidden[-1]

        return self.head(final_state)


# -----------------------------------------------------
# Autoencoder loader
# -----------------------------------------------------

def load_encoder(
    checkpoint_path,
    run_meta_path,
    device,
):
    import json

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
# Training
# -----------------------------------------------------

def train(
    X,
    y,
    model,
    output_dir,
    epochs,
    batch_size,
    lr,
    validation_fraction,
    device,
):

    dataset = TensorDataset(X, y)

    val_size = int(
        len(dataset) * validation_fraction
    )

    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    criterion = nn.MSELoss()

    best_val = float("inf")

    history = []

    for epoch in range(1, epochs + 1):

        model.train()

        train_losses = []

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)

            loss = criterion(pred, yb)

            loss.backward()

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        train_loss = np.mean(train_losses)

        model.eval()

        val_losses = []

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)

                loss = criterion(pred, yb)

                val_losses.append(
                    loss.item()
                )

        val_loss = np.mean(val_losses)

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        })

        print(
            f"Epoch {epoch:03d} "
            f"train={train_loss:.6f} "
            f"val={val_loss:.6f}"
        )

        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "history":
                        history,
                },
                output_dir / "best_lstm.pt",
            )

    with open(
        output_dir / "history.json",
        "w",
    ) as f:
        json.dump(history, f, indent=2)


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
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    device = args.device

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
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
        Path(args.autoencoder).parent / "run_meta.json",
        device,
    )

    model = LSTMPredictor(
        encoder=encoder,
        latent_channels=latent_channels,
        output_dim=y.shape[1],
    ).to(device)

    train(
        X,
        y,
        model,
        output_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.validation_fraction,
        device,
    )


if __name__ == "__main__":
    main()