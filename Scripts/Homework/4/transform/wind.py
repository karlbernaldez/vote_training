from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


DIRECTIONS_16 = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
]


def validate_station_columns(stations: pd.DataFrame) -> None:
    required_columns = {"lat", "lon"}
    missing_columns = required_columns - set(stations.columns)
    if missing_columns:
        raise ValueError(f"Stations CSV is missing required columns: {sorted(missing_columns)}")


def safe_open_grib(file_path: Path, backend_kwargs: dict[str, Any]) -> xr.Dataset | None:
    """Open one GRIB2 file and return None when cfgrib cannot read it."""
    try:
        return xr.open_dataset(
            file_path,
            engine="cfgrib",
            backend_kwargs=backend_kwargs,
            errors="ignore",
        )
    except Exception as exc:
        print(f"Failed to open {file_path.name}: {exc}")
        return None


def load_wind_dataset(local_paths: list[Path]) -> xr.Dataset:
    """Load GFS 10m U/V wind files and concatenate them by forecast step."""
    backend_kwargs = {
        "filter_by_keys": {
            "cfVarName": ["u10", "v10"],
            "stepType": "instant",
        },
        "indexpath": "",
    }

    datasets = []
    for local_path in local_paths:
        ds = safe_open_grib(local_path, backend_kwargs)
        if ds is not None:
            datasets.append(ds)

    if not datasets:
        raise ValueError("No readable GFS wind GRIB2 files were found for station wind transform.")

    return xr.concat(
        datasets,
        dim="step",
        coords="minimal",
        compat="override",
    )


def wind_direction_labels(direction_deg: np.ndarray) -> list[str | None]:
    """Convert wind direction degrees into 16-point compass labels."""
    labels = []
    for direction in np.atleast_1d(direction_deg):
        if np.isnan(direction):
            labels.append(None)
        else:
            labels.append(DIRECTIONS_16[int((direction + 11.25) / 22.5) % 16])
    return labels


def extract_wind_features(ds: xr.Dataset, lat: float, lon: float) -> list[dict[str, Any]]:
    """Extract ML-friendly wind features for one station point."""
    point = ds.interp(latitude=lat, longitude=lon, method="linear")

    u_values = np.atleast_1d(point.u10.values).astype(float)
    v_values = np.atleast_1d(point.v10.values).astype(float)
    speed_kph = np.hypot(u_values, v_values) * 3.6
    direction_deg = (270 - np.degrees(np.arctan2(v_values, u_values))) % 360
    forecast_hours = np.atleast_1d(
        (point.step.dt.total_seconds().values / 3600).astype(int)
    )
    direction_labels = wind_direction_labels(direction_deg)

    records = []
    for index, forecast_hour in enumerate(forecast_hours):
        degree = direction_deg[index]
        degree_rad = np.deg2rad(degree) if not np.isnan(degree) else np.nan
        records.append({
            "forecast_hour": int(forecast_hour),
            "forecast_hour_label": f"f{int(forecast_hour):03d}",
            "u10_ms": round(float(u_values[index]), 6) if not np.isnan(u_values[index]) else np.nan,
            "v10_ms": round(float(v_values[index]), 6) if not np.isnan(v_values[index]) else np.nan,
            "wind_speed_kph": int(round(float(speed_kph[index]))) if not np.isnan(speed_kph[index]) else np.nan,
            "wind_dir_deg": round(float(degree), 6) if not np.isnan(degree) else np.nan,
            "wind_dir_sin": round(float(np.sin(degree_rad)), 6) if not np.isnan(degree_rad) else np.nan,
            "wind_dir_cos": round(float(np.cos(degree_rad)), 6) if not np.isnan(degree_rad) else np.nan,
            "wind_direction": direction_labels[index],
        })

    return records


def extract_wind(ds: xr.Dataset, lat: float, lon: float) -> tuple[list[int], list[int | float], list[str | None]]:
    """Extract forecast hour, wind speed in kph, and wind direction for one point."""
    records = extract_wind_features(ds, lat, lon)
    return (
        [record["forecast_hour"] for record in records],
        [record["wind_speed_kph"] for record in records],
        [record["wind_direction"] for record in records],
    )


def build_station_wind_dataframe(ds: xr.Dataset, stations_csv: Path) -> pd.DataFrame:
    """Build a human-readable station-level forecast table."""
    stations = pd.read_csv(stations_csv)
    validate_station_columns(stations)

    rows = []
    for _, station in stations.iterrows():
        forecast_hours, speed, direction = extract_wind(
            ds,
            lat=float(station["lat"]),
            lon=float(station["lon"]),
        )

        output_row = station.to_dict()
        for index, hour in enumerate(forecast_hours):
            output_row[f"f{hour:03d}"] = f"{direction[index]} @ {speed[index]} kph"

        rows.append(output_row)

    return pd.DataFrame(rows)


def build_station_wind_ml_dataframe(
    ds: xr.Dataset,
    stations_csv: Path,
    run_time: str | None = None,
) -> pd.DataFrame:
    """Build a long-format, numeric station wind table for ML.

    Each output row represents one station and one forecast hour. This format is
    easier for scikit-learn than wide columns like f000, f003, and f006.
    """
    stations = pd.read_csv(stations_csv)
    validate_station_columns(stations)

    rows = []
    for _, station in stations.iterrows():
        station_data = station.to_dict()
        wind_records = extract_wind_features(
            ds,
            lat=float(station["lat"]),
            lon=float(station["lon"]),
        )

        for wind_record in wind_records:
            output_row = dict(station_data)
            if run_time is not None:
                output_row["run_time"] = run_time
            output_row.update(wind_record)
            rows.append(output_row)

    return pd.DataFrame(rows)
