"""
ml/pretrain_transformer.py
--------------------------
Pretrain the CNN-Transformer using Masked Autoencoding (MAE).

Usage
-----
python -m ml.pretrain_transformer \
    --dataset-dir  data/ml/gridded_wind_conv3d_many \
    --output-dir   data/ml/runs/cnn_transformer_pretrain_v1 \
    --epochs       200 \
    --batch-size   2 \
    --d-model      256 \
    --n-heads      8 \
    --n-layers     6 \
    --mask-ratio   0.75 \
    --device       cuda
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from ml.transformer import MaskedAutoencoderWrapper


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
# Data
# ---------------------------------------------------------------------------


def load_tensor(dataset_dir: Path) -> tuple[torch.Tensor, dict[str, float]]:
    x_path = dataset_dir / "X.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing tensor file: {x_path}")

    x = np.load(x_path).astype(np.float32)
    if x.ndim != 5:
        raise ValueError(f"Expected (N, C, T, H, W). Got {x.shape}.")

    mean = float(x.mean())
    std = float(x.std())
    x = (x - mean) / (std + 1e-8)

    return torch.from_numpy(x), {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PretrainConfig:
    # Architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ffn_dim: int = 1024
    dropout: float = 0.1
    # MAE
    mask_ratio: float = 0.75
    mask_only_loss: bool = True
    # Training
    epochs: int = 200
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 20
    validation_fraction: float = 0.2
    early_stopping_patience: int = 30
    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Warmup + cosine LR schedule
# ---------------------------------------------------------------------------


class WarmupCosineScheduler:
    """
    Linear warmup for *warmup_epochs*, then cosine decay to *min_lr*.
    Called once per epoch (not per step).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch: int) -> float:
        """Set LR for *epoch* (1-indexed). Returns current LR."""
        if epoch <= self.warmup_epochs:
            lr = self.base_lr * epoch / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("inf")
        self._counter = 0
        self.triggered = False

    def step(self, value: float) -> bool:
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
# Epoch helpers
# ---------------------------------------------------------------------------


def _run_epoch(
    model: MaskedAutoencoderWrapper,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for (batch,) in pbar:
            batch = batch.to(device)
            out = model(batch)
            loss = out["loss"]

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # gradient clipping — important for Transformers
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------


def plot_history(history: dict, output_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # Loss panel
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train", linewidth=1.8, color="#2563eb")
    val = [v if v is not None else float("nan") for v in history["val_loss"]]
    if any(not np.isnan(v) for v in val):
        ax.plot(epochs, val, label="Val", linewidth=1.8, color="#dc2626", linestyle="--")
        best_ep = int(np.nanargmin(val)) + 1
        ax.axvline(best_ep, color="#16a34a", linewidth=1, linestyle=":", label=f"Best (ep {best_ep})")
        ax.scatter([best_ep], [np.nanmin(val)], color="#16a34a", zorder=5, s=50)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE Loss (masked MSE)")
    ax.set_title("Pretraining Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # LR panel
    ax = axes[1]
    ax.plot(epochs, history["lr"], linewidth=1.8, color="#7c3aed")
    ax.axvline(history.get("warmup_epochs", 0), color="#f59e0b", linewidth=1,
               linestyle=":", label="End warmup")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Warmup + Cosine LR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    out_path = output_dir / "pretrain_loss_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Loss curve saved → {out_path}")


# ---------------------------------------------------------------------------
# Main pretraining routine
# ---------------------------------------------------------------------------


def pretrain(
    X: torch.Tensor,
    norm_stats: dict,
    output_dir: Path,
    cfg: PretrainConfig,
) -> dict:
    print("=" * 72)
    print("MAE PRETRAINING — CNN-Transformer")
    print("=" * 72)

    device = torch.device(cfg.device)
    seed_everything(cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    N, C, T, H, W = X.shape
    size_gb = X.numel() * X.element_size() / 1024 ** 3
    print(f"  Device  : {device}  (CUDA: {torch.cuda.is_available()})")
    print(f"  Shape   : {tuple(X.shape)}")
    print(f"  Memory  : {size_gb:.3f} GB")
    print(f"  Mask ratio : {cfg.mask_ratio:.0%}")

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
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                   num_workers=0, pin_memory=device.type == "cuda")
        if val_ds is not None else None
    )

    # ---- Model ----------------------------------------------------------------
    # Compute positional encoding bounds from actual data
    import math
    max_hp = math.ceil(math.ceil(H / 2) / 2)
    max_wp = math.ceil(math.ceil(W / 2) / 2)

    model = MaskedAutoencoderWrapper(
        in_channels=C,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        ffn_dim=cfg.ffn_dim,
        dropout=cfg.dropout,
        mask_ratio=cfg.mask_ratio,
        mask_only_loss=cfg.mask_only_loss,
        max_t=T + 4,       # small buffer above actual size
        max_h=max_hp + 4,
        max_w=max_wp + 4,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"\n  Total parameters    : {total_params:,}")
    print(f"  Encoder parameters  : {enc_params:,}")

    # Smoke test
    with torch.no_grad():
        dummy = next(iter(train_loader))[0].to(device)
        out = model(dummy)
    assert out["reconstruction"].shape == dummy.shape
    print(f"  Forward pass OK — recon shape: {tuple(out['reconstruction'].shape)}")
    print(f"  Sample MAE loss: {out['loss'].item():.6f}")

    # ---- Optimiser & scheduler ------------------------------------------------
    # Use separate LR for Transformer vs CNN — Transformer benefits from lower LR
    cnn_params = list(model.encoder.cnn_stem.parameters()) + list(model.cnn_decoder.parameters())
    transformer_params = [
        p for p in model.parameters()
        if not any(p is cp for cp in cnn_params)
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": cnn_params,         "lr": cfg.learning_rate * 2.0},
            {"params": transformer_params, "lr": cfg.learning_rate},
        ],
        weight_decay=cfg.weight_decay,
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg.warmup_epochs,
        total_epochs=cfg.epochs,
        base_lr=cfg.learning_rate,
    )

    early_stopper = (
        EarlyStopping(cfg.early_stopping_patience)
        if cfg.early_stopping_patience > 0 and val_loader is not None
        else None
    )

    # ---- Epoch loop -----------------------------------------------------------
    history: dict[str, list] = {
        "train_loss": [], "val_loss": [], "lr": [],
        "warmup_epochs": cfg.warmup_epochs,
    }
    best_val_loss = float("inf")
    best_epoch = 0

    print("\n" + "=" * 72)

    for epoch in range(1, cfg.epochs + 1):
        current_lr = scheduler.step(epoch)

        train_loss = _run_epoch(
            model, train_loader, optimizer, device,
            desc=f"Epoch {epoch:>4}/{cfg.epochs} [train]",
        )

        val_loss: float | None = None
        if val_loader is not None:
            val_loss = _run_epoch(
                model, val_loader, None, device,
                desc=f"Epoch {epoch:>4}/{cfg.epochs} [val]  ",
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), output_dir / "best_pretrain.pt")
                # Also save encoder weights separately for fine-tuning
                torch.save(model.encoder.state_dict(), output_dir / "best_encoder.pt")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        es_str = ""
        if early_stopper is not None and val_loss is not None:
            if early_stopper.step(val_loss):
                val_str = f"  val={val_loss:.6f}" if val_loss is not None else ""
                print(
                    f"Epoch {epoch:>4}/{cfg.epochs} — "
                    f"train={train_loss:.6f}{val_str}  lr={current_lr:.2e}"
                )
                print(f"\nEarly stopping — no improvement for {cfg.early_stopping_patience} epochs.")
                break
            es_str = f"  [no-improve {early_stopper.counter}/{cfg.early_stopping_patience}]"

        val_str = f"  val={val_loss:.6f}" if val_loss is not None else ""
        print(
            f"Epoch {epoch:>4}/{cfg.epochs} — "
            f"train={train_loss:.6f}{val_str}  lr={current_lr:.2e}{es_str}"
        )

    # ---- Save artefacts -------------------------------------------------------
    torch.save(model.state_dict(), output_dir / "last_pretrain.pt")
    torch.save(model.encoder.state_dict(), output_dir / "last_encoder.pt")

    with open(output_dir / "pretrain_history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    run_meta = {
        "config": asdict(cfg),
        "norm_stats": norm_stats,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss if val_loader else None,
        "data_shape": list(X.shape),
    }
    with open(output_dir / "pretrain_meta.json", "w") as fh:
        json.dump(run_meta, fh, indent=2)

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
        description="Pretrain CNN-Transformer via Masked Autoencoding."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    # Architecture
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    # MAE
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--mask-only-loss", action="store_true", default=True)
    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = PretrainConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
        mask_only_loss=args.mask_only_loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        validation_fraction=args.validation_fraction,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        device=args.device,
    )

    X, norm_stats = load_tensor(Path(args.dataset_dir))

    pretrain(
        X=X,
        norm_stats=norm_stats,
        output_dir=Path(args.output_dir),
        cfg=cfg,
    )


if __name__ == "__main__":
    main()