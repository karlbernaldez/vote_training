import json

import numpy as np
import pandas as pd
import pytest

from ml.prepare_station_dataset import (
    build_station_sequence_dataset,
    read_station_ml_csvs,
    save_dataset,
)


def sample_station_ml_dataframe():
    return pd.DataFrame(
        [
            {
                "run_time": "2026051900",
                "lat": 14.6,
                "lon": 121.0,
                "forecast_hour": 0,
                "u10_ms": 1.0,
                "v10_ms": 2.0,
                "wind_speed_kph": 8,
                "wind_dir_sin": 0.1,
                "wind_dir_cos": 0.9,
            },
            {
                "run_time": "2026051900",
                "lat": 14.6,
                "lon": 121.0,
                "forecast_hour": 3,
                "u10_ms": 1.5,
                "v10_ms": 2.5,
                "wind_speed_kph": 10,
                "wind_dir_sin": 0.2,
                "wind_dir_cos": 0.8,
            },
            {
                "run_time": "2026051900",
                "lat": 15.0,
                "lon": 120.5,
                "forecast_hour": 0,
                "u10_ms": 3.0,
                "v10_ms": 4.0,
                "wind_speed_kph": 18,
                "wind_dir_sin": 0.3,
                "wind_dir_cos": 0.7,
            },
            {
                "run_time": "2026051900",
                "lat": 15.0,
                "lon": 120.5,
                "forecast_hour": 3,
                "u10_ms": 3.5,
                "v10_ms": 4.5,
                "wind_speed_kph": 20,
                "wind_dir_sin": 0.4,
                "wind_dir_cos": 0.6,
            },
        ]
    )


def test_build_station_sequence_dataset_shape_and_manifest():
    X, metadata, manifest = build_station_sequence_dataset(sample_station_ml_dataframe())

    assert X.shape == (2, 5, 2)
    assert metadata.shape[0] == 2
    assert manifest["dataset_type"] == "station_wind_cnn_1d"
    assert manifest["axis_order"] == ["sample", "channel", "forecast_step"]
    assert manifest["forecast_hours"] == [0, 3]


def test_build_station_sequence_dataset_values_are_channel_first():
    X, _, _ = build_station_sequence_dataset(sample_station_ml_dataframe())

    np.testing.assert_array_equal(
        X[0],
        np.array(
            [
                [1.0, 1.5],
                [2.0, 2.5],
                [8.0, 10.0],
                [0.1, 0.2],
                [0.9, 0.8],
            ],
            dtype=np.float32,
        ),
    )


def test_build_station_sequence_dataset_rejects_missing_required_column():
    df = sample_station_ml_dataframe().drop(columns=["wind_dir_cos"])

    with pytest.raises(ValueError, match="missing required columns"):
        build_station_sequence_dataset(df)


def test_build_station_sequence_dataset_rejects_incomplete_forecast_sequence():
    df = sample_station_ml_dataframe()
    df = df[~((df["lat"] == 15.0) & (df["forecast_hour"] == 3))]

    with pytest.raises(ValueError, match="Incomplete forecast sequence"):
        build_station_sequence_dataset(df)


def test_save_station_dataset_outputs_files(tmp_path):
    X, metadata, manifest = build_station_sequence_dataset(sample_station_ml_dataframe())

    save_dataset(X, metadata, manifest, tmp_path)

    assert (tmp_path / "X.npy").exists()
    assert (tmp_path / "metadata.csv").exists()
    assert (tmp_path / "dataset_manifest.json").exists()
    assert np.load(tmp_path / "X.npy").shape == (2, 5, 2)

    with (tmp_path / "dataset_manifest.json").open() as f:
        assert json.load(f)["dataset_type"] == "station_wind_cnn_1d"


def test_read_station_ml_csvs_combines_files(tmp_path):
    df = sample_station_ml_dataframe()
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    df.iloc[:2].to_csv(first, index=False)
    df.iloc[2:].to_csv(second, index=False)

    combined = read_station_ml_csvs([first, second])

    assert len(combined) == 4
    assert "source_file" in combined.columns
