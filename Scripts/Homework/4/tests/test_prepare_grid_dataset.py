import json

import numpy as np
import pytest
import xarray as xr

from ml.prepare_grid_dataset import build_grid_cnn_dataset, save_dataset


def sample_gridded_dataset():
    coords = {
        "step": np.array([np.timedelta64(0, "h"), np.timedelta64(3, "h")]),
        "latitude": np.array([14.0, 15.0]),
        "longitude": np.array([120.0, 121.0, 122.0]),
    }
    shape = (2, 2, 3)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return xr.Dataset(
        data_vars={
            "u10_ms": (("step", "latitude", "longitude"), base + 1),
            "v10_ms": (("step", "latitude", "longitude"), base + 2),
            "wind_speed_kph": (("step", "latitude", "longitude"), base + 3),
            "wind_dir_sin": (("step", "latitude", "longitude"), np.full(shape, 0.5, dtype=np.float32)),
            "wind_dir_cos": (("step", "latitude", "longitude"), np.full(shape, 0.8, dtype=np.float32)),
        },
        coords=coords,
    )


def test_build_grid_cnn_dataset_conv2d_shape_and_manifest():
    X, metadata, manifest = build_grid_cnn_dataset(sample_gridded_dataset(), layout="conv2d")

    assert X.shape == (2, 5, 2, 3)
    assert metadata["forecast_hour"].tolist() == [0, 3]
    assert manifest["dataset_type"] == "gridded_wind_conv2d"
    assert manifest["axis_order"] == ["sample", "channel", "latitude", "longitude"]
    assert manifest["tensor_shape"] == {"N": 2, "C": 5, "H": 2, "W": 3}


def test_build_grid_cnn_dataset_conv3d_shape_and_manifest():
    X, metadata, manifest = build_grid_cnn_dataset(sample_gridded_dataset(), layout="conv3d")

    assert X.shape == (1, 5, 2, 2, 3)
    assert metadata["sample_index"].tolist() == [0]
    assert manifest["dataset_type"] == "gridded_wind_conv3d"
    assert manifest["axis_order"] == ["sample", "channel", "forecast_step", "latitude", "longitude"]
    assert manifest["tensor_shape"] == {"N": 1, "C": 5, "T": 2, "H": 2, "W": 3}


def test_build_grid_cnn_dataset_rejects_missing_variable():
    ds = sample_gridded_dataset().drop_vars("wind_dir_cos")

    with pytest.raises(ValueError, match="missing feature variables"):
        build_grid_cnn_dataset(ds, layout="conv2d")


def test_build_grid_cnn_dataset_rejects_nan_values():
    ds = sample_gridded_dataset()
    ds["u10_ms"].values[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        build_grid_cnn_dataset(ds, layout="conv2d")


def test_save_grid_dataset_outputs_files(tmp_path):
    X, metadata, manifest = build_grid_cnn_dataset(sample_gridded_dataset(), layout="conv2d")

    save_dataset(X, metadata, manifest, tmp_path)

    assert (tmp_path / "X.npy").exists()
    assert (tmp_path / "metadata.csv").exists()
    assert (tmp_path / "dataset_manifest.json").exists()
    assert np.load(tmp_path / "X.npy").shape == (2, 5, 2, 3)

    with (tmp_path / "dataset_manifest.json").open() as f:
        assert json.load(f)["dataset_type"] == "gridded_wind_conv2d"
