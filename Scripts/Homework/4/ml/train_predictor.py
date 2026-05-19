from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from ml.models import Conv3dAutoencoder, Conv3dPredictor


def load_inputs(dataset_dir: Path) -> torch.Tensor:
    x_path = dataset_dir / "X.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing tensor file: {x_path}")
    X = np.load(x_path).astype(np.float32)
    if X.ndim != 5:
        raise ValueError(f"Conv3D predictor expects X with 5 dimensions N x C x T x H x W; got {X.shape}")
    return torch.from_numpy(X)


def load_targets(target_csv: Path, target_columns: list[str]) -> torch.Tensor:
    if not target_csv.exists():
        raise FileNotFoundError(f"Missing target CSV: {target_csv}")
    targets = pd.read_csv(target_csv)
    missing = set(target_columns) - set(targets.columns)
    if missing:
        raise ValueError(f"Target CSV is missing target columns: {sorted(missing)}")
    y = targets[target_columns].to_numpy(dtype=np.float32)
    if np.isnan(y).any():
        raise ValueError("Target CSV contains NaN values in target columns.")
    return torch.from_numpy(y)


def load_pretrained_autoencoder(path: Path, in_channels: int, latent_channels: int, device: str) -> Conv3dAutoencoder:
    checkpoint = torch.load(path, map_location=device)
    model = Conv3dAutoencoder(
        in_channels=int(checkpoint.get("in_channels", in_channels)),
        latent_channels=int(checkpoint.get("latent_channels", latent_channels)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def train_predictor(
    X: torch.Tensor,
    y: torch.Tensor,
    output_dir: Path,
    target_columns: list[str],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    latent_channels: int,
    pretrained_autoencoder_path: Path | None,
    freeze_encoder: bool,
    seed: int,
    device: str,
) -> dict:
    if len(X) != len(y):
        raise ValueError(f"X and target row count mismatch: X has {len(X)} samples, y has {len(y)} rows")

    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TensorDataset(X, y)
    validation_size = int(len(dataset) * validation_fraction)
    train_size = len(dataset) - validation_size
    if train_size <= 0:
        raise ValueError("Training set is empty. Reduce --validation-fraction or add more samples.")

    generator = torch.Generator().manual_seed(seed)
    if validation_size > 0:
        train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=generator)
    else:
        train_dataset, validation_dataset = dataset, None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size) if validation_dataset is not None else None

    pretrained = None
    if pretrained_autoencoder_path is not None:
        pretrained = load_pretrained_autoencoder(pretrained_autoencoder_path, X.shape[1], latent_channels, device)
        latent_channels = int(pretrained.encoder[-2].out_channels) if hasattr(pretrained.encoder[-2], "out_channels") else latent_channels

    model = Conv3dPredictor(
        in_channels=int(X.shape[1]),
        output_dim=int(y.shape[1]),
        latent_channels=latent_channels,
        pretrained_autoencoder=pretrained,
        freeze_encoder=freeze_encoder,
    ).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.MSELoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses))}
        if validation_loader is not None:
            model.eval()
            validation_losses = []
            with torch.no_grad():
                for batch_x, batch_y in validation_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    validation_losses.append(float(criterion(model(batch_x), batch_y).detach().cpu()))
            row["validation_loss"] = float(np.mean(validation_losses))
        history.append(row)
        print(row)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "in_channels": int(X.shape[1]),
        "latent_channels": latent_channels,
        "output_dim": int(y.shape[1]),
        "target_columns": target_columns,
        "input_shape": list(X.shape),
        "history": history,
    }
    torch.save(checkpoint, output_dir / "predictor.pt")

    metrics = {
        "task": "conv3d_predictor_training",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "validation_fraction": validation_fraction,
        "device": device,
        "input_shape": list(X.shape),
        "target_shape": list(y.shape),
        "target_columns": target_columns,
        "used_pretrained_autoencoder": pretrained_autoencoder_path is not None,
        "freeze_encoder": freeze_encoder,
        "history": history,
    }
    with (output_dir / "predictor_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Conv3D predictor from gridded wind tensors.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing Conv3D X.npy.")
    parser.add_argument("--target-csv", required=True, help="CSV with one target row per X sample.")
    parser.add_argument("--target-columns", nargs="+", required=True, help="Target column names in target CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoint and metrics.")
    parser.add_argument("--pretrained-autoencoder", help="Optional autoencoder.pt checkpoint to initialize the encoder.")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze pretrained encoder weights.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if not 0 <= args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be in [0, 1)")

    X = load_inputs(Path(args.dataset_dir))
    y = load_targets(Path(args.target_csv), args.target_columns)
    metrics = train_predictor(
        X=X,
        y=y,
        output_dir=Path(args.output_dir),
        target_columns=args.target_columns,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        latent_channels=args.latent_channels,
        pretrained_autoencoder_path=Path(args.pretrained_autoencoder) if args.pretrained_autoencoder else None,
        freeze_encoder=args.freeze_encoder,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
