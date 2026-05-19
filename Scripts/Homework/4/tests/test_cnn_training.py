import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

import torch

from ml.models import Conv3dAutoencoder, Conv3dPredictor
from ml.train_autoencoder import load_tensor, train_autoencoder
from ml.train_predictor import load_targets, train_predictor


def tiny_conv3d_tensor():
    return torch.from_numpy(np.random.default_rng(42).normal(size=(2, 5, 2, 8, 8)).astype(np.float32))


def test_conv3d_autoencoder_preserves_shape():
    X = tiny_conv3d_tensor()
    model = Conv3dAutoencoder(in_channels=5, latent_channels=8)

    output = model(X)

    assert output.shape == X.shape


def test_conv3d_predictor_outputs_target_dimension():
    X = tiny_conv3d_tensor()
    model = Conv3dPredictor(in_channels=5, output_dim=3, latent_channels=8)

    output = model(X)

    assert output.shape == (2, 3)


def test_load_tensor_requires_5d_input(tmp_path):
    np.save(tmp_path / "X.npy", np.zeros((2, 5, 8, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="5 dimensions"):
        load_tensor(tmp_path)


def test_train_autoencoder_writes_checkpoint_and_metrics(tmp_path):
    X = tiny_conv3d_tensor()
    output_dir = tmp_path / "out"

    metrics = train_autoencoder(
        X=X,
        output_dir=output_dir,
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        validation_fraction=0.0,
        latent_channels=8,
        seed=42,
        device="cpu",
    )

    assert (output_dir / "autoencoder.pt").exists()
    assert (output_dir / "autoencoder_metrics.json").exists()
    assert metrics["task"] == "conv3d_autoencoder_pretraining"

    with (output_dir / "autoencoder_metrics.json").open() as f:
        assert json.load(f)["input_shape"] == [2, 5, 2, 8, 8]


def test_load_targets_reads_selected_columns(tmp_path):
    target_csv = tmp_path / "targets.csv"
    pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}).to_csv(target_csv, index=False)

    y = load_targets(target_csv, ["a", "b"])

    assert tuple(y.shape) == (2, 2)


def test_train_predictor_writes_checkpoint_and_metrics(tmp_path):
    X = tiny_conv3d_tensor()
    y = torch.from_numpy(np.array([[1.0], [2.0]], dtype=np.float32))
    output_dir = tmp_path / "predictor"

    metrics = train_predictor(
        X=X,
        y=y,
        output_dir=output_dir,
        target_columns=["target"],
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        validation_fraction=0.0,
        latent_channels=8,
        pretrained_autoencoder_path=None,
        freeze_encoder=False,
        seed=42,
        device="cpu",
    )

    assert (output_dir / "predictor.pt").exists()
    assert (output_dir / "predictor_metrics.json").exists()
    assert metrics["task"] == "conv3d_predictor_training"
