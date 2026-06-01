# ml/predict_lstm.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ml.models import Conv3dAutoencoder


class LSTMPredictor(nn.Module):

    def __init__(
        self,
        encoder,
        latent_channels,
        hidden_size=128,
        num_layers=2,
        output_dim=1,
    ):
        super().__init__()

        self.encoder = encoder

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

        # B,C,T,H,W
        z = z.mean(dim=(3, 4))

        # B,C,T
        z = z.permute(0, 2, 1)

        # B,T,C
        _, (hidden, _) = self.lstm(z)

        return self.head(hidden[-1])


def load_autoencoder_encoder(
    checkpoint_path: Path,
    run_meta_path: Path,
    device: str,
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

    model.eval()

    return model.encoder, latent_channels


def load_lstm_model(
    lstm_checkpoint: Path,
    encoder,
    latent_channels,
    output_dim,
    device,
):

    model = LSTMPredictor(
        encoder=encoder,
        latent_channels=latent_channels,
        output_dim=output_dim,
    )

    ckpt = torch.load(
        lstm_checkpoint,
        map_location=device,
    )

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(
            ckpt["model_state_dict"]
        )
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    return model


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
        "--lstm-model",
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

    device = args.device

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------
    # Load X
    # ------------------------

    X = np.load(
        Path(args.dataset_dir) / "X.npy"
    ).astype(np.float32)

    X = torch.from_numpy(X)

    # ------------------------
    # Load targets
    # ------------------------

    df = pd.read_csv(args.target_csv)

    y = (
        df[args.target_columns]
        .to_numpy(dtype=np.float32)
        .copy()
    )

    # ------------------------
    # Load encoder
    # ------------------------

    encoder, latent_channels = (
        load_autoencoder_encoder(
            Path(args.autoencoder),
            Path(args.autoencoder).parent
            / "run_meta.json",
            device,
        )
    )

    # ------------------------
    # Load LSTM
    # ------------------------

    model = load_lstm_model(
        Path(args.lstm_model),
        encoder,
        latent_channels,
        y.shape[1],
        device,
    )

    # ------------------------
    # Predict
    # ------------------------

    preds = []

    with torch.no_grad():

        for sample in X:

            sample = (
                sample.unsqueeze(0)
                .to(device)
            )

            pred = model(sample)

            preds.append(
                pred.cpu().numpy()
            )

    preds = np.concatenate(
        preds,
        axis=0,
    )

    np.save(
        output_dir / "predictions.npy",
        preds,
    )

    np.save(
        output_dir / "targets.npy",
        y,
    )

    print(
        f"Predictions shape: {preds.shape}"
    )

    print(
        f"Saved -> {output_dir/'predictions.npy'}"
    )

    print(
        f"Saved -> {output_dir/'targets.npy'}"
    )


if __name__ == "__main__":
    main()