from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "run_time",
    "forecast_hour",
    "lat",
    "lon",
    "u10_ms",
    "v10_ms",
    "wind_speed_kph",
    "wind_dir_sin",
    "wind_dir_cos",
}

DEFAULT_FEATURE_COLUMNS = [
    "u10_ms",
    "v10_ms",
    "wind_speed_kph",
    "wind_dir_sin",
    "wind_dir_cos",
]

DEFAULT_ID_COLUMNS = [
    "run_time",
    "lat",
    "lon",
]


def read_station_ml_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Station ML CSV not found: {path}")
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)

    if not frames:
        raise ValueError("No station ML CSV files were provided.")

    return pd.concat(frames, ignore_index=True)


def validate_station_ml_dataframe(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Station ML CSV is missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Station ML CSV is empty.")

    numeric_columns = [
        "forecast_hour",
        "lat",
        "lon",
        "u10_ms",
        "v10_ms",
        "wind_speed_kph",
        "wind_dir_sin",
        "wind_dir_cos",
    ]
    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column must be numeric: {column}")


def build_station_sequence_dataset(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    id_columns: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Convert long-format station rows into a Conv1d-ready tensor.

    Output shape is N x C x T, where:
    - N is one station/run sample
    - C is feature channels
    - T is ordered forecast hours
    """
    validate_station_ml_dataframe(df)

    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    id_columns = id_columns or DEFAULT_ID_COLUMNS

    missing_features = set(feature_columns) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns: {sorted(missing_features)}")

    missing_ids = set(id_columns) - set(df.columns)
    if missing_ids:
        raise ValueError(f"Missing id columns: {sorted(missing_ids)}")

    work = df.copy()
    work["run_time"] = work["run_time"].astype(str)
    work = work.sort_values(id_columns + ["forecast_hour"])

    forecast_hours = sorted(work["forecast_hour"].dropna().unique().tolist())
    if not forecast_hours:
        raise ValueError("No forecast hours found.")

    samples = []
    metadata_rows = []

    for sample_id, group in work.groupby(id_columns, dropna=False):
        if not isinstance(sample_id, tuple):
            sample_id = (sample_id,)

        pivoted = (
            group.pivot_table(index="forecast_hour", values=feature_columns, aggfunc="first")
            .reindex(forecast_hours)
        )

        if pivoted.isna().any().any():
            missing_hours = pivoted[pivoted.isna().any(axis=1)].index.tolist()
            raise ValueError(
                f"Incomplete forecast sequence for {dict(zip(id_columns, sample_id))}; "
                f"missing or invalid forecast hours: {missing_hours}"
            )

        # pivot_table gives T x C. PyTorch Conv1d expects C x T per sample.
        samples.append(pivoted[feature_columns].to_numpy(dtype=np.float32).T)
        metadata_rows.append(dict(zip(id_columns, sample_id)))

    X = np.stack(samples, axis=0)
    metadata = pd.DataFrame(metadata_rows)
    manifest = {
        "format_version": "1.0",
        "dataset_type": "station_wind_cnn_1d",
        "tensor_file": "X.npy",
        "metadata_file": "metadata.csv",
        "tensor_shape": {
            "N": int(X.shape[0]),
            "C": int(X.shape[1]),
            "T": int(X.shape[2]),
        },
        "axis_order": ["sample", "channel", "forecast_step"],
        "feature_columns": feature_columns,
        "id_columns": id_columns,
        "forecast_hours": [int(hour) for hour in forecast_hours],
        "target_columns": [],
        "notes": [
            "Prepared for PyTorch Conv1d as batch x channels x sequence_length.",
            "Targets are intentionally not included until observed labels are defined.",
        ],
    }
    return X, metadata, manifest


def save_dataset(X: np.ndarray, metadata: pd.DataFrame, manifest: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X.npy", X)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    with (output_dir / "dataset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Conv1d station wind dataset from station_wind_ml CSV files.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more station_wind_ml CSV files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for X.npy and metadata.")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURE_COLUMNS, help="Feature columns to use as channels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_station_ml_csvs([Path(path) for path in args.input])
    X, metadata, manifest = build_station_sequence_dataset(df, feature_columns=args.features)
    save_dataset(X, metadata, manifest, Path(args.output_dir))
    print(f"Wrote station CNN dataset to: {args.output_dir}")
    print(f"X shape: {X.shape}")


if __name__ == "__main__":
    main()
