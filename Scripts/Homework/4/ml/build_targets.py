from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TARGETS = [
    "max_wind_speed_kph",
    "mean_wind_speed_kph",
    "p95_wind_speed_kph",
    "strong_wind_event",
]


def load_dataset_manifest(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as f:
        return json.load(f)


def load_tensor(dataset_dir: Path) -> np.ndarray:
    x_path = dataset_dir / "X.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing tensor file: {x_path}")
    return np.load(x_path).astype(np.float32)


def wind_speed_channel_index(manifest: dict) -> int:
    feature_names = manifest.get("feature_variables") or manifest.get("feature_columns")
    if not feature_names:
        raise ValueError("Dataset manifest must contain feature_variables or feature_columns.")
    if "wind_speed_kph" not in feature_names:
        raise ValueError("Dataset features must include wind_speed_kph to build wind-speed targets.")
    return feature_names.index("wind_speed_kph")


def validate_supported_layout(X: np.ndarray, manifest: dict) -> str:
    dataset_type = manifest.get("dataset_type", "")
    axis_order = manifest.get("axis_order", [])

    if X.ndim == 5 and axis_order == ["sample", "channel", "forecast_step", "latitude", "longitude"]:
        return "conv3d"
    if X.ndim == 4 and axis_order == ["sample", "channel", "latitude", "longitude"]:
        return "conv2d"
    if X.ndim == 3 and axis_order == ["sample", "channel", "forecast_step"]:
        return "station_conv1d"

    raise ValueError(
        f"Unsupported dataset layout for target generation: dataset_type={dataset_type!r}, "
        f"shape={X.shape}, axis_order={axis_order}"
    )


def build_targets_from_tensor(
    X: np.ndarray,
    manifest: dict,
    targets: list[str] | None = None,
    strong_wind_threshold_kph: float = 39.0,
) -> pd.DataFrame:
    """Build simple target columns from prepared ML tensors.

    These are model-derived proxy targets useful for smoke tests and baseline
    experiments. Production supervised training should replace or join these
    with observed labels when available.
    """
    targets = targets or DEFAULT_TARGETS
    layout = validate_supported_layout(X, manifest)
    speed_index = wind_speed_channel_index(manifest)
    wind_speed = X[:, speed_index]

    rows = []
    for sample_index in range(X.shape[0]):
        sample_speed = wind_speed[sample_index]
        row = {"sample_index": sample_index, "layout": layout}

        if "max_wind_speed_kph" in targets:
            row["max_wind_speed_kph"] = float(np.max(sample_speed))
        if "mean_wind_speed_kph" in targets:
            row["mean_wind_speed_kph"] = float(np.mean(sample_speed))
        if "p95_wind_speed_kph" in targets:
            row["p95_wind_speed_kph"] = float(np.percentile(sample_speed, 95))
        if "strong_wind_event" in targets:
            row["strong_wind_event"] = int(np.max(sample_speed) >= strong_wind_threshold_kph)
        if "final_step_mean_wind_speed_kph" in targets:
            if layout not in {"conv3d", "station_conv1d"}:
                raise ValueError("final_step_mean_wind_speed_kph requires a time/forecast dimension.")
            row["final_step_mean_wind_speed_kph"] = float(np.mean(sample_speed[-1]))
        if "final_step_max_wind_speed_kph" in targets:
            if layout not in {"conv3d", "station_conv1d"}:
                raise ValueError("final_step_max_wind_speed_kph requires a time/forecast dimension.")
            row["final_step_max_wind_speed_kph"] = float(np.max(sample_speed[-1]))

        unknown = set(targets) - set(row) - {"final_step_mean_wind_speed_kph", "final_step_max_wind_speed_kph"}
        if unknown:
            raise ValueError(f"Unknown target definition(s): {sorted(unknown)}")

        rows.append(row)

    return pd.DataFrame(rows)


def save_targets(targets: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build target CSVs from prepared VOTE ML tensors.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing X.npy and dataset_manifest.json.")
    parser.add_argument("--output-csv", required=True, help="Target CSV path to write.")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="Target definitions to generate.",
    )
    parser.add_argument(
        "--strong-wind-threshold-kph",
        type=float,
        default=39.0,
        help="Threshold for strong_wind_event target.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    manifest = load_dataset_manifest(dataset_dir)
    X = load_tensor(dataset_dir)
    targets = build_targets_from_tensor(
        X,
        manifest,
        targets=args.targets,
        strong_wind_threshold_kph=args.strong_wind_threshold_kph,
    )
    save_targets(targets, Path(args.output_csv))
    print(f"Wrote targets: {args.output_csv}")
    print(targets.head())


if __name__ == "__main__":
    main()
