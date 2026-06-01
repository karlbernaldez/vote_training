from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from ml.models import Conv3dAutoencoder


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_tensor(dataset_dir: Path) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Load ``X.npy`` from *dataset_dir*, validate its shape, z-score normalise,
    and return the tensor together with the normalisation statistics so they
    can be saved alongside the model weights.

    Returns:
        x:     Float32 tensor of shape (N, C, T, H, W).
        stats: Dict with keys ``mean`` and ``std``.
    """
    x_path = dataset_dir / "X.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing tensor file: {x_path}")

    x = np.load(x_path).astype(np.float32)

    if x.ndim != 5:
        raise ValueError(
            f"Expected array of shape (N, C, T, H, W). Got shape {x.shape}."
        )

    mean = float(x.mean())
    std = float(x.std())
    x = (x - mean) / (std + 1e-8)

    return torch.from_numpy(x), {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    validation_fraction: float = 0.2
    latent_channels: int = 64
    dropout: float = 0.1
    # LR scheduler
    lr_patience: int = 10
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    # Early stopping (0 = disabled)
    early_stopping_patience: int = 20
    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """
    Stops training when the monitored metric has not improved for
    *patience* consecutive epochs.

    Args:
        patience:  Number of epochs to wait without improvement.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best: float = float("inf")
        self._counter: int = 0
        self.triggered: bool = False

    def step(self, value: float) -> bool:
        """Call each epoch. Returns True when training should stop."""
        if value < self._best - self.min_delta:
            self._best = value
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.triggered = True
        return self.triggered

    @property
    def counter(self) -> int:
        return self._counter


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
) -> float:
    """
    Run a single training or validation epoch.

    Pass ``optimizer=None`` for evaluation (no gradients, no parameter update).
    Returns the mean loss over all batches.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for (batch,) in pbar:
            batch = batch.to(device)

            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------


def plot_history(history: dict[str, list], output_dir: Path) -> None:
    """Save a two-panel loss + LR curve to *output_dir/loss_curve.png*."""
    epochs = range(1, len(history["train_loss"]) + 1)
    train_losses = history["train_loss"]
    val_losses = history["val_loss"]
    lrs = history["lr"]

    has_val = any(v is not None for v in val_losses)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4), constrained_layout=True
    )

    # --- panel 1: loss ---
    ax = axes[0]
    ax.plot(epochs, train_losses, label="Train", linewidth=1.8, color="#2563eb")
    if has_val:
        clean_val = [v if v is not None else float("nan") for v in val_losses]
        ax.plot(epochs, clean_val, label="Val", linewidth=1.8, color="#dc2626", linestyle="--")
        best_epoch = int(np.nanargmin(clean_val)) + 1
        best_val = np.nanmin(clean_val)
        ax.axvline(best_epoch, color="#16a34a", linewidth=1, linestyle=":", label=f"Best val (ep {best_epoch})")
        ax.scatter([best_epoch], [best_val], color="#16a34a", zorder=5, s=50)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # --- panel 2: learning rate ---
    ax = axes[1]
    ax.plot(epochs, lrs, linewidth=1.8, color="#7c3aed")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    out_path = output_dir / "loss_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Loss curve saved → {out_path}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def train_autoencoder(
    X: torch.Tensor,
    norm_stats: dict[str, float],
    output_dir: Path,
    cfg: TrainConfig,
) -> dict:
    """
    Full training loop with validation, LR scheduling, and best-model
    checkpointing.

    Args:
        X:           Normalised input tensor (N, C, T, H, W).
        norm_stats:  Normalisation statistics to persist with the checkpoint.
        output_dir:  Directory for model weights, metrics, and config.
        cfg:         Hyper-parameter configuration.

    Returns:
        history: Dict containing per-epoch train/val losses and LR.
    """
    # ---- Setup ----------------------------------------------------------------
    print("=" * 72)
    print("TRAINING START")
    print("=" * 72)

    device = torch.device(cfg.device)
    seed_everything(cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    n, c, t, h, w = X.shape
    size_gb = X.numel() * X.element_size() / 1024 ** 3
    print(f"  Device  : {device}  (CUDA available: {torch.cuda.is_available()})")
    print(f"  Shape   : {tuple(X.shape)}")
    print(f"  Memory  : {size_gb:.3f} GB")

    # ---- Dataset split --------------------------------------------------------
    dataset = TensorDataset(X)
    val_size = max(1, int(len(dataset) * cfg.validation_fraction)) if cfg.validation_fraction > 0 else 0
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(cfg.seed)

    if val_size > 0:
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_ds, val_ds = dataset, None

    print(f"\n  Samples : {len(dataset)}  (train={train_size}, val={val_size})")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        if val_ds is not None
        else None
    )

    # ---- Model ----------------------------------------------------------------
    model = Conv3dAutoencoder(
        in_channels=c,
        latent_channels=cfg.latent_channels,
        dropout=cfg.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters: {total_params:,}")

    # Smoke-test forward pass
    with torch.no_grad():
        dummy = next(iter(train_loader))[0].to(device)
        out = model(dummy)
    assert out.shape == dummy.shape, f"Shape mismatch: {out.shape} vs {dummy.shape}"
    print(f"  Forward pass OK — output shape: {tuple(out.shape)}")

    # ---- Optimiser & scheduler ------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg.lr_patience,
        factor=cfg.lr_factor,
        min_lr=cfg.min_lr,
    )
    criterion = nn.MSELoss()

    # ---- Epoch loop -----------------------------------------------------------
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_epoch = 0

    early_stopper = (
        EarlyStopping(patience=cfg.early_stopping_patience)
        if cfg.early_stopping_patience > 0 and val_loader is not None
        else None
    )

    print("\n" + "=" * 72)

    for epoch in range(1, cfg.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = _run_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"Epoch {epoch:>4}/{cfg.epochs} [train]",
        )

        val_loss: float | None = None
        if val_loader is not None:
            val_loss = _run_epoch(
                model, val_loader, criterion, None, device,
                desc=f"Epoch {epoch:>4}/{cfg.epochs} [val]  ",
            )
            scheduler.step(val_loss)

            # Best-model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), output_dir / "best_autoencoder.pt")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        es_str = ""
        if early_stopper is not None and val_loss is not None:
            if early_stopper.step(val_loss):
                val_str = f"  val={val_loss:.6f}" if val_loss is not None else ""
                print(
                    f"Epoch {epoch:>4}/{cfg.epochs} — "
                    f"train={train_loss:.6f}{val_str}  "
                    f"lr={current_lr:.2e}"
                )
                print(
                    f"\nEarly stopping triggered — "
                    f"no improvement for {cfg.early_stopping_patience} epochs."
                )
                break
            es_str = f"  [no-improve {early_stopper.counter}/{cfg.early_stopping_patience}]"

        val_str = f"  val={val_loss:.6f}" if val_loss is not None else ""
        print(
            f"Epoch {epoch:>4}/{cfg.epochs} — "
            f"train={train_loss:.6f}{val_str}  "
            f"lr={current_lr:.2e}"
            f"{es_str}"
        )

    # ---- Save artefacts -------------------------------------------------------
    # Final weights (always)
    torch.save(model.state_dict(), output_dir / "last_autoencoder.pt")

    # Training history
    with open(output_dir / "history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    # Config + normalisation stats
    run_meta = {
        "config": asdict(cfg),
        "norm_stats": norm_stats,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss if val_loader else None,
    }
    with open(output_dir / "run_meta.json", "w") as fh:
        json.dump(run_meta, fh, indent=2)

    # Loss curve plot
    plot_history(history, output_dir)

    print("\n" + "=" * 72)
    if val_loader:
        print(f"Best val loss : {best_val_loss:.6f}  (epoch {best_epoch})")
    print(f"Artefacts saved → {output_dir}")
    print("=" * 72)

    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 3-D convolutional autoencoder on gridded wind data."
    )
    parser.add_argument("--dataset-dir", required=True, help="Directory containing X.npy")
    parser.add_argument("--output-dir", required=True, help="Directory for model outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=20,
        help="Epochs without val-loss improvement before stopping. 0 = disabled.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        latent_channels=args.latent_channels,
        dropout=args.dropout,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        min_lr=args.min_lr,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        device=args.device,
    )

    X, norm_stats = load_tensor(Path(args.dataset_dir))

    train_autoencoder(
        X=X,
        norm_stats=norm_stats,
        output_dir=Path(args.output_dir),
        cfg=cfg,
    )


if __name__ == "__main__":
    main()