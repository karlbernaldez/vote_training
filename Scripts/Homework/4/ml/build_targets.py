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
    "max_spatial_coverage_strong_wind_pct",
    "sustained_strong_wind_hours",
    "marine_wind_hazard_score",
    "marine_wind_hazard_level",
]

SUPPORTED_TARGETS = {
    "max_wind_speed_kph",
    "mean_wind_speed_kph",
    "p95_wind_speed_kph",
    "strong_wind_event",
    "final_step_mean_wind_speed_kph",
    "final_step_max_wind_speed_kph",
    "max_spatial_coverage_strong_wind_pct",
    "sustained_strong_wind_steps",
    "sustained_strong_wind_hours",
    "max_step_p95_wind_speed_kph",
    "marine_wind_hazard_score",
    "marine_wind_hazard_level",
}


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


def forecast_step_hours(manifest: dict) -> float:
    forecast_hours = manifest.get("forecast_hours") or []
    if len(forecast_hours) >= 2:
        deltas = np.diff(np.array(forecast_hours, dtype=float))
        positive_deltas = deltas[deltas > 0]
        if len(positive_deltas):
            return float(np.median(positive_deltas))
    return 3.0


def longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values.astype(bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def time_series_for_hazard(sample_speed: np.ndarray, layout: str) -> tuple[np.ndarray, np.ndarray]:
    """Return step-level p95 wind and strong-wind coverage fractions.

    For time-aware layouts, each value represents one forecast step. For Conv2d,
    there is no time axis left, so the single sample is treated as one step.
    """
    if layout == "conv3d":
        step_p95 = np.percentile(sample_speed, 95, axis=(1, 2))
        # coverage by forecast step: fraction of grid cells above threshold is filled later.
        return step_p95.astype(np.float32), sample_speed
    if layout == "station_conv1d":
        # station Conv1d sample_speed is T.
        return sample_speed.astype(np.float32), sample_speed
    return np.array([np.percentile(sample_speed, 95)], dtype=np.float32), sample_speed


def marine_hazard_level(score: float) -> int:
    """Bucket a 0-100 hazard score into 0 low, 1 moderate, 2 high, 3 severe."""
    if score >= 75:
        return 3
    if score >= 50:
        return 2
    if score >= 25:
        return 1
    return 0


def marine_hazard_metrics(
    sample_speed: np.ndarray,
    layout: str,
    step_hours: float,
    strong_wind_threshold_kph: float,
) -> dict[str, float | int]:
    """Compute explainable operational marine wind-hazard proxy metrics.

    This is not real wave height. It is a wind-driven hazard proxy based on:
    - intensity: strongest step-level p95 wind speed
    - coverage: largest fraction of area/stations above the strong-wind threshold
    - duration: longest consecutive strong-wind period
    """
    step_p95, speed_for_coverage = time_series_for_hazard(sample_speed, layout)

    if layout == "conv3d":
        coverage_by_step = np.mean(speed_for_coverage >= strong_wind_threshold_kph, axis=(1, 2))
        strong_steps = step_p95 >= strong_wind_threshold_kph
    elif layout == "station_conv1d":
        coverage_by_step = (speed_for_coverage >= strong_wind_threshold_kph).astype(float)
        strong_steps = speed_for_coverage >= strong_wind_threshold_kph
    else:
        coverage_by_step = np.array([float(np.mean(speed_for_coverage >= strong_wind_threshold_kph))])
        strong_steps = np.array([np.max(speed_for_coverage) >= strong_wind_threshold_kph])

    sustained_steps = longest_true_run(strong_steps)
    sustained_hours = sustained_steps * step_hours
    max_step_p95 = float(np.max(step_p95))
    max_coverage_pct = float(np.max(coverage_by_step) * 100.0)

    # 0-100 score. Threshold-relative intensity dominates; coverage and duration
    # make the target more operationally useful than a single max grid point.
    intensity_component = min(max_step_p95 / strong_wind_threshold_kph, 2.0) / 2.0 * 50.0
    coverage_component = min(max_coverage_pct, 100.0) / 100.0 * 30.0
    duration_component = min(sustained_hours / 24.0, 1.0) * 20.0
    score = float(min(100.0, intensity_component + coverage_component + duration_component))

    return {
        "max_step_p95_wind_speed_kph": max_step_p95,
        "max_spatial_coverage_strong_wind_pct": max_coverage_pct,
        "sustained_strong_wind_steps": int(sustained_steps),
        "sustained_strong_wind_hours": float(sustained_hours),
        "marine_wind_hazard_score": score,
        "marine_wind_hazard_level": marine_hazard_level(score),
    }


def build_targets_from_tensor(
    X: np.ndarray,
    manifest: dict,
    targets: list[str] | None = None,
    strong_wind_threshold_kph: float = 39.0,
) -> pd.DataFrame:
    """Build target columns from prepared ML tensors.

    Targets are derived from model input data. They are useful for smoke tests,
    baseline experiments, and operational wind-hazard proxy modeling. Production
    wave-height training should replace or join these with observed/WW3 labels.
    """
    targets = targets or DEFAULT_TARGETS
    unknown = set(targets) - SUPPORTED_TARGETS
    if unknown:
        raise ValueError(f"Unknown target definition(s): {sorted(unknown)}")

    layout = validate_supported_layout(X, manifest)
    speed_index = wind_speed_channel_index(manifest)
    wind_speed = X[:, speed_index]
    step_hours = forecast_step_hours(manifest)

    rows = []
    for sample_index in range(X.shape[0]):
        sample_speed = wind_speed[sample_index]
        row = {"sample_index": sample_index, "layout": layout}
        hazard_metrics = marine_hazard_metrics(
            sample_speed,
            layout=layout,
            step_hours=step_hours,
            strong_wind_threshold_kph=strong_wind_threshold_kph,
        )

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

        for name in (
            "max_spatial_coverage_strong_wind_pct",
            "sustained_strong_wind_steps",
            "sustained_strong_wind_hours",
            "max_step_p95_wind_speed_kph",
            "marine_wind_hazard_score",
            "marine_wind_hazard_level",
        ):
            if name in targets:
                row[name] = hazard_metrics[name]

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
        help="Threshold for strong-wind and marine wind-hazard targets.",
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
