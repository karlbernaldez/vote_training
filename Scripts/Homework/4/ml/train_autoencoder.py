from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from ml.models import Conv3dAutoencoder


def load_tensor(dataset_dir: Path) -> torch.Tensor:
    x_path = dataset_dir / "X.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing tensor file: {x_path}")

    x = np.load(x_path).astype(np.float32)
    if x.ndim != 5:
        raise ValueError(f"Conv3D autoencoder expects X with 5 dimensions N x C x T x H x W; got {x.shape}")
    return torch.from_numpy(x)


def train_autoencoder(
    X: torch.Tensor,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    latent_channels: int,
    seed: int,
    device: str,
) -> dict:
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TensorDataset(X)
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

    model = Conv3dAutoencoder(in_channels=X.shape[1], latent_channels=latent_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses))}

        if validation_loader is not None:
            model.eval()
            validation_losses = []
            with torch.no_grad():
                for (batch,) in validation_loader:
                    batch = batch.to(device)
                    reconstructed = model(batch)
                    validation_losses.append(float(criterion(reconstructed, batch).detach().cpu()))
            row["validation_loss"] = float(np.mean(validation_losses))

        history.append(row)
        print(row)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "in_channels": int(X.shape[1]),
        "latent_channels": latent_channels,
        "input_shape": list(X.shape),
        "history": history,
    }
    torch.save(checkpoint, output_dir / "autoencoder.pt")

    metrics = {
        "task": "conv3d_autoencoder_pretraining",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "validation_fraction": validation_fraction,
        "device": device,
        "input_shape": list(X.shape),
        "history": history,
    }
    with (output_dir / "autoencoder_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a 3-layer Conv3D autoencoder on gridded wind tensors.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing Conv3D X.npy.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoint and metrics.")
    parser.add_argument("--epochs", type=int, default=10)
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

    X = load_tensor(Path(args.dataset_dir))
    metrics = train_autoencoder(
        X=X,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        latent_channels=args.latent_channels,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
