"""
ml/encode.py
------------
Inference script for a trained Conv3dAutoencoder.

Three modes
-----------
encode      Run only the encoder; save latent representations to latents.npy
decode      Run only the decoder on a pre-saved latents.npy
reconstruct Full encode → decode pass; save reconstructed output and per-sample
            MSE; optionally denormalise back to the original data scale.

Usage examples
--------------
# Encode every sample and save latents
python -m ml.encode encode \
    --dataset-dir  data/ml/gridded_wind_conv3d_many \
    --run-dir      data/ml/runs/autoencoder_bucket_v1 \
    --output-dir   data/ml/inference/v1

# Reconstruct and compare against original
python -m ml.encode reconstruct \
    --dataset-dir  data/ml/gridded_wind_conv3d_many \
    --run-dir      data/ml/runs/autoencoder_bucket_v1 \
    --output-dir   data/ml/inference/v1 \
    --denormalise

# Decode a previously saved latents.npy (e.g. after editing latent space)
python -m ml.encode decode \
    --latents-path data/ml/inference/v1/latents.npy \
    --run-dir      data/ml/runs/autoencoder_bucket_v1 \
    --output-dir   data/ml/inference/v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ml.models import Conv3dAutoencoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_run_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_meta.json not found in {run_dir}")
    with open(meta_path) as fh:
        return json.load(fh)


def _load_model(run_dir: Path, device: torch.device) -> tuple[Conv3dAutoencoder, dict]:
    """
    Load the best checkpoint (falls back to last) from *run_dir*.
    Returns the model in eval mode and the run metadata dict.
    """
    meta = _load_run_meta(run_dir)
    cfg = meta["config"]

    # Resolve checkpoint: prefer best, fall back to last
    best_path = run_dir / "best_autoencoder.pt"
    last_path = run_dir / "last_autoencoder.pt"
    ckpt_path = best_path if best_path.exists() else last_path

    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")

    print(f"  Loading checkpoint : {ckpt_path.name}")

    # in_channels is inferred from the state-dict key shape
    state = torch.load(ckpt_path, map_location=device)
    first_weight = next(iter(state.values()))
    # encoder.0.0.weight → (out_ch, in_ch, kD, kH, kW)
    in_channels = first_weight.shape[1]

    model = Conv3dAutoencoder(
        in_channels=in_channels,
        latent_channels=cfg["latent_channels"],
        dropout=0.0,          # disable dropout at inference
    ).to(device)

    model.load_state_dict(state)
    model.eval()

    return model, meta


def _load_and_normalise(dataset_dir: Path, norm_stats: dict) -> torch.Tensor:
    """Load X.npy and apply the same z-score normalisation used during training."""
    x_path = dataset_dir / "X.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing tensor file: {x_path}")

    x = np.load(x_path).astype(np.float32)
    mean = norm_stats["mean"]
    std = norm_stats["std"]
    x = (x - mean) / (std + 1e-8)
    return torch.from_numpy(x)


def _denormalise(x: np.ndarray, norm_stats: dict) -> np.ndarray:
    return x * (norm_stats["std"] + 1e-8) + norm_stats["mean"]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run_encode(
    X: torch.Tensor,
    model: Conv3dAutoencoder,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
) -> np.ndarray:
    """Encode every sample in X and save to *output_dir/latents.npy*."""
    latents = []
    loader = tqdm(
        torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X),
            batch_size=batch_size,
            shuffle=False,
        ),
        desc="Encoding",
        unit="batch",
    )
    with torch.no_grad():
        for (batch,) in loader:
            z = model.encode(batch.to(device))
            latents.append(z.cpu().numpy())

    latents_np = np.concatenate(latents, axis=0)
    out_path = output_dir / "latents.npy"
    np.save(out_path, latents_np)
    print(f"  Latents shape : {latents_np.shape}")
    print(f"  Saved         → {out_path}")
    return latents_np


def run_decode(
    latents: np.ndarray,
    model: Conv3dAutoencoder,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    denormalise: bool,
    norm_stats: dict,
) -> np.ndarray:
    """Decode a latents array and save to *output_dir/decoded.npy*."""
    Z = torch.from_numpy(latents)
    decoded = []

    loader = tqdm(
        torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Z),
            batch_size=batch_size,
            shuffle=False,
        ),
        desc="Decoding",
        unit="batch",
    )
    # We need the original spatial size to trim ConvTranspose padding.
    # Compute it from the latent shape: latent H = ceil(ceil(H/2)/2) → H ≈ latent_h * 4
    # The model crops to the exact input shape, so we pass a generous upper bound
    # and let the crop handle it. We reconstruct via the full forward pass instead.
    with torch.no_grad():
        for (batch,) in loader:
            # decode needs a target_shape hint; estimate from latent spatial dims
            z = batch.to(device)
            # rough target — will be over-estimated, crop is safe
            t_size = z.shape[-3]
            h_size = z.shape[-2] * 4
            w_size = z.shape[-1] * 4
            target_shape = (1, 1, t_size, h_size, w_size)
            out = model.decode(z, target_shape)
            decoded.append(out.cpu().numpy())

    decoded_np = np.concatenate(decoded, axis=0)
    if denormalise:
        decoded_np = _denormalise(decoded_np, norm_stats)

    out_path = output_dir / "decoded.npy"
    np.save(out_path, decoded_np)
    print(f"  Decoded shape : {decoded_np.shape}")
    print(f"  Saved         → {out_path}")
    return decoded_np


def run_reconstruct(
    X: torch.Tensor,
    model: Conv3dAutoencoder,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    denormalise: bool,
    norm_stats: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full encode → decode pass.

    Saves:
        reconstructions.npy   shape (N, C, T, H, W)
        mse_per_sample.npy    shape (N,)  per-sample MSE in normalised space
    """
    reconstructions = []
    mse_per_sample = []

    loader = tqdm(
        torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X),
            batch_size=batch_size,
            shuffle=False,
        ),
        desc="Reconstructing",
        unit="batch",
    )
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=(1, 2, 3, 4))
            reconstructions.append(recon.cpu().numpy())
            mse_per_sample.append(mse.cpu().numpy())

    recon_np = np.concatenate(reconstructions, axis=0)
    mse_np = np.concatenate(mse_per_sample, axis=0)

    if denormalise:
        recon_np = _denormalise(recon_np, norm_stats)

    recon_path = output_dir / "reconstructions.npy"
    mse_path = output_dir / "mse_per_sample.npy"
    np.save(recon_path, recon_np)
    np.save(mse_path, mse_np)

    print(f"  Reconstructions shape : {recon_np.shape}")
    print(f"  Mean MSE (normalised) : {mse_np.mean():.6f}  ±  {mse_np.std():.6f}")
    for i, v in enumerate(mse_np):
        print(f"    sample {i:>3} : {v:.6f}")
    print(f"  Saved → {recon_path}")
    print(f"  Saved → {mse_path}")
    return recon_np, mse_np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference with a trained Conv3dAutoencoder.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # shared args
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--run-dir", required=True, help="Directory containing checkpoint + run_meta.json")
    shared.add_argument("--output-dir", required=True, help="Directory to write outputs")
    shared.add_argument("--batch-size", type=int, default=4)
    shared.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # encode
    p_enc = sub.add_parser("encode", parents=[shared], help="Encode X.npy → latents.npy")
    p_enc.add_argument("--dataset-dir", required=True)

    # decode
    p_dec = sub.add_parser("decode", parents=[shared], help="Decode latents.npy → decoded.npy")
    p_dec.add_argument("--latents-path", required=True, help="Path to a latents.npy produced by encode mode")
    p_dec.add_argument("--denormalise", action="store_true")

    # reconstruct
    p_rec = sub.add_parser("reconstruct", parents=[shared], help="Encode + decode X.npy → reconstructions.npy")
    p_rec.add_argument("--dataset-dir", required=True)
    p_rec.add_argument("--denormalise", action="store_true", help="Invert z-score normalisation in the output")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"MODE   : {args.mode.upper()}")
    print(f"Device : {device}")

    model, meta = _load_model(run_dir, device)
    norm_stats = meta["norm_stats"]

    if args.mode == "encode":
        X = _load_and_normalise(Path(args.dataset_dir), norm_stats)
        print(f"Input  : {tuple(X.shape)}")
        run_encode(X, model, device, output_dir, args.batch_size)

    elif args.mode == "decode":
        latents = np.load(args.latents_path).astype(np.float32)
        print(f"Latents: {latents.shape}")
        run_decode(latents, model, device, output_dir, args.batch_size, args.denormalise, norm_stats)

    elif args.mode == "reconstruct":
        X = _load_and_normalise(Path(args.dataset_dir), norm_stats)
        print(f"Input  : {tuple(X.shape)}")
        run_reconstruct(X, model, device, output_dir, args.batch_size, args.denormalise, norm_stats)

    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()