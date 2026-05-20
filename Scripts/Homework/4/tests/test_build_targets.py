import json

import numpy as np
import pandas as pd
import pytest

from ml.build_targets import build_targets_from_tensor, load_dataset_manifest, load_tensor, save_targets


def conv3d_manifest():
    return {
        "dataset_type": "gridded_wind_conv3d",
        "axis_order": ["sample", "channel", "forecast_step", "latitude", "longitude"],
        "feature_variables": ["u10_ms", "v10_ms", "wind_speed_kph", "wind_dir_sin", "wind_dir_cos"],
        "forecast_hours": [0, 3],
    }


def conv2d_manifest():
    return {
        "dataset_type": "gridded_wind_conv2d",
        "axis_order": ["sample", "channel", "latitude", "longitude"],
        "feature_variables": ["u10_ms", "v10_ms", "wind_speed_kph", "wind_dir_sin", "wind_dir_cos"],
    }


def station_manifest():
    return {
        "dataset_type": "station_wind_cnn_1d",
        "axis_order": ["sample", "channel", "forecast_step"],
        "feature_columns": ["u10_ms", "v10_ms", "wind_speed_kph", "wind_dir_sin", "wind_dir_cos"],
        "forecast_hours": [0, 3, 6],
    }


def test_build_targets_from_conv3d_tensor():
    X = np.zeros((1, 5, 2, 2, 2), dtype=np.float32)
    X[0, 2] = np.array(
        [
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]],
        ],
        dtype=np.float32,
    )

    targets = build_targets_from_tensor(
        X,
        conv3d_manifest(),
        targets=[
            "max_wind_speed_kph",
            "mean_wind_speed_kph",
            "p95_wind_speed_kph",
            "strong_wind_event",
            "final_step_mean_wind_speed_kph",
            "final_step_max_wind_speed_kph",
        ],
        strong_wind_threshold_kph=39,
    )

    row = targets.iloc[0]
    assert row["max_wind_speed_kph"] == 80
    assert row["mean_wind_speed_kph"] == 45
    assert row["strong_wind_event"] == 1
    assert row["final_step_mean_wind_speed_kph"] == 65
    assert row["final_step_max_wind_speed_kph"] == 80


def test_build_targets_from_conv2d_tensor():
    X = np.zeros((2, 5, 2, 2), dtype=np.float32)
    X[0, 2] = np.array([[10, 20], [30, 40]], dtype=np.float32)
    X[1, 2] = np.array([[1, 2], [3, 4]], dtype=np.float32)

    targets = build_targets_from_tensor(X, conv2d_manifest(), strong_wind_threshold_kph=39)

    assert targets["max_wind_speed_kph"].tolist() == [40, 4]
    assert targets["strong_wind_event"].tolist() == [1, 0]
    assert "marine_wind_hazard_score" in targets.columns


def test_build_targets_from_station_tensor():
    X = np.zeros((1, 5, 3), dtype=np.float32)
    X[0, 2] = np.array([10, 20, 30], dtype=np.float32)

    targets = build_targets_from_tensor(
        X,
        station_manifest(),
        targets=["final_step_mean_wind_speed_kph", "final_step_max_wind_speed_kph"],
    )

    assert targets["final_step_mean_wind_speed_kph"].tolist() == [30]
    assert targets["final_step_max_wind_speed_kph"].tolist() == [30]


def test_build_targets_generates_marine_wind_hazard_metrics():
    X = np.zeros((1, 5, 2, 2, 2), dtype=np.float32)
    X[0, 2] = np.array(
        [
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]],
        ],
        dtype=np.float32,
    )

    targets = build_targets_from_tensor(
        X,
        conv3d_manifest(),
        targets=[
            "max_step_p95_wind_speed_kph",
            "max_spatial_coverage_strong_wind_pct",
            "sustained_strong_wind_steps",
            "sustained_strong_wind_hours",
            "marine_wind_hazard_score",
            "marine_wind_hazard_level",
        ],
        strong_wind_threshold_kph=39,
    )

    row = targets.iloc[0]
    assert row["max_step_p95_wind_speed_kph"] > 70
    assert row["max_spatial_coverage_strong_wind_pct"] == 100
    assert row["sustained_strong_wind_steps"] == 1
    assert row["sustained_strong_wind_hours"] == 3
    assert 0 <= row["marine_wind_hazard_score"] <= 100
    assert row["marine_wind_hazard_level"] in {0, 1, 2, 3}


def test_build_targets_rejects_unknown_target():
    X = np.zeros((1, 5, 2, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="Unknown target"):
        build_targets_from_tensor(X, conv3d_manifest(), targets=["unknown"])


def test_build_targets_rejects_final_step_for_conv2d():
    X = np.zeros((1, 5, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="requires a time"):
        build_targets_from_tensor(X, conv2d_manifest(), targets=["final_step_max_wind_speed_kph"])


def test_load_and_save_targets(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    X = np.zeros((1, 5, 2, 2, 2), dtype=np.float32)
    np.save(dataset_dir / "X.npy", X)
    with (dataset_dir / "dataset_manifest.json").open("w") as f:
        json.dump(conv3d_manifest(), f)

    loaded_x = load_tensor(dataset_dir)
    manifest = load_dataset_manifest(dataset_dir)
    targets = build_targets_from_tensor(loaded_x, manifest)
    output_csv = tmp_path / "targets.csv"
    save_targets(targets, output_csv)

    loaded_targets = pd.read_csv(output_csv)
    assert loaded_targets.shape[0] == 1
    assert "max_wind_speed_kph" in loaded_targets.columns
    assert "marine_wind_hazard_score" in loaded_targets.columns
