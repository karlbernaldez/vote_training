from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_FEATURE_VARIABLES = [
    "u10_ms",
    "v10_ms",
    "wind_speed_kph",
    "wind_dir_sin",
    "wind_dir_cos",
]

SUPPORTED_LAYOUTS = {"conv2d", "conv3d"}


def read_gridded_wind_datasets(paths: Iterable[Path]) -> xr.Dataset:
    datasets = []
    source_files = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Gridded wind NetCDF not found: {path}")
        datasets.append(xr.open_dataset(path))
        source_files.append(str(path))

    if not datasets:
        raise ValueError("No gridded wind NetCDF files were provided.")

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = xr.concat(
            datasets,
            dim="sample",
            data_vars="all",
            coords="minimal",
            compat="override",
            join="exact",
            combine_attrs="override",
        )

    dataset.attrs["source_files"] = json.dumps(source_files)
    return dataset


def validate_gridded_dataset(ds: xr.Dataset, feature_variables: list[str]) -> None:
    missing = set(feature_variables) - set(ds.data_vars)
    if missing:
        raise ValueError(f"Gridded dataset is missing feature variables: {sorted(missing)}")

    required_dims = {"step", "latitude", "longitude"}
    missing_dims = required_dims - set(ds.sizes)
    if missing_dims:
        raise ValueError(f"Gridded dataset is missing required dimensions: {sorted(missing_dims)}")

    for name in feature_variables:
        variable_dims = set(ds[name].dims)
        if not required_dims.issubset(variable_dims):
            raise ValueError(f"Variable {name} must include dimensions step, latitude, and longitude")


def forecast_hours_from_step(step_values: np.ndarray) -> list[int]:
    hours = []
    for value in np.atleast_1d(step_values):
        if np.issubdtype(np.asarray(value).dtype, np.timedelta64):
            hours.append(int(value / np.timedelta64(1, "h")))
        else:
            hours.append(int(value))
    return hours


def build_grid_cnn_dataset(
    ds: xr.Dataset,
    layout: str,
    feature_variables: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Convert gridded wind data into CNN-ready tensors.

    Supported layouts:
    - conv2d: X shape is N x C x H x W, one forecast hour per sample
    - conv3d: X shape is N x C x T x H x W, one run sequence per sample
    """
    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(f"Unsupported layout: {layout}. Supported layouts are: {sorted(SUPPORTED_LAYOUTS)}")

    feature_variables = feature_variables or DEFAULT_FEATURE_VARIABLES
    validate_gridded_dataset(ds, feature_variables)

    ordered = ds[feature_variables].transpose(..., "step", "latitude", "longitude")
    stacked = ordered.to_array(dim="channel").transpose("channel", "step", "latitude", "longitude")
    values = stacked.to_numpy().astype(np.float32)

    if np.isnan(values).any():
        raise ValueError("Gridded dataset contains NaN values in selected feature variables.")

    forecast_hours = forecast_hours_from_step(ds["step"].values)
    latitude_count = int(ds.sizes["latitude"])
    longitude_count = int(ds.sizes["longitude"])

    if layout == "conv2d":
        # C x T x H x W -> T x C x H x W
        X = np.moveaxis(values, 1, 0)
        metadata = pd.DataFrame({"forecast_hour": forecast_hours})
        axis_order = ["sample", "channel", "latitude", "longitude"]
        tensor_shape = {
            "N": int(X.shape[0]),
            "C": int(X.shape[1]),
            "H": int(X.shape[2]),
            "W": int(X.shape[3]),
        }
        notes = ["Prepared for PyTorch Conv2d as batch x channels x height x width."]
    else:
        # Single run sequence: C x T x H x W -> N x C x T x H x W.
        X = values[np.newaxis, ...]
        metadata = pd.DataFrame({"sample_index": [0]})
        axis_order = ["sample", "channel", "forecast_step", "latitude", "longitude"]
        tensor_shape = {
            "N": int(X.shape[0]),
            "C": int(X.shape[1]),
            "T": int(X.shape[2]),
            "H": int(X.shape[3]),
            "W": int(X.shape[4]),
        }
        notes = ["Prepared for PyTorch Conv3d as batch x channels x depth/time x height x width."]

    manifest = {
        "format_version": "1.0",
        "dataset_type": f"gridded_wind_{layout}",
        "layout": layout,
        "tensor_file": "X.npy",
        "metadata_file": "metadata.csv",
        "tensor_shape": tensor_shape,
        "axis_order": axis_order,
        "feature_variables": feature_variables,
        "forecast_hours": forecast_hours,
        "latitude_count": latitude_count,
        "longitude_count": longitude_count,
        "target_columns": [],
        "source_files": json.loads(ds.attrs.get("source_files", "[]")),
        "notes": notes + ["Targets are intentionally not included until observed labels are defined."],
    }
    return X, metadata, manifest


def save_dataset(X: np.ndarray, metadata: pd.DataFrame, manifest: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X.npy", X)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    with (output_dir / "dataset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Conv2d/Conv3d gridded wind dataset from NetCDF files.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more gridded_wind NetCDF files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for X.npy and metadata.")
    parser.add_argument("--layout", choices=sorted(SUPPORTED_LAYOUTS), required=True, help="CNN layout to generate.")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURE_VARIABLES, help="NetCDF variables to use as channels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = read_gridded_wind_datasets([Path(path) for path in args.input])
    try:
        X, metadata, manifest = build_grid_cnn_dataset(ds, layout=args.layout, feature_variables=args.features)
        save_dataset(X, metadata, manifest, Path(args.output_dir))
    finally:
        ds.close()
    print(f"Wrote gridded CNN dataset to: {args.output_dir}")
    print(f"X shape: {X.shape}")


if __name__ == "__main__":
    main()
