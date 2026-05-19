from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ingest import daterange, normalize_storage_backend, sha256sum, storage_uri, upload_json
from transform.config import TransformConfig, load_transform_config
from transform.paths import merge_output_paths, source_object_path, station_wind_output_paths
from transform.storage import download_object, upload_binary

import numpy as np
import pandas as pd
import xarray as xr


DIRECTIONS_16 = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
]


# Backward-compatible alias for older imports/tests that used output_paths().
output_paths = merge_output_paths


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
    except Exception as e:
        print(f"Failed to open {file_path.name}: {e}")
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

    return xr.concat(datasets, dim="step")


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


def download_forecast_files(
    storage_backend: str,
    bucket_name: str,
    prefix: str,
    run_dt: datetime,
    run_hour: str,
    forecast_start: int,
    forecast_step: int,
    forecast_max: int,
    tmpdir_path: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Download bronze GRIB2 forecast files and return local paths plus source metadata."""
    backend = normalize_storage_backend(storage_backend)
    local_paths = []
    source_records = []

    for forecast_hour in range(forecast_start, forecast_max + 1, forecast_step):
        object_path = source_object_path(prefix, run_dt, run_hour, forecast_hour)
        local_path = tmpdir_path / Path(object_path).name
        source_uri = storage_uri(backend, bucket_name, object_path)

        print(f"Downloading {source_uri}")
        download_object(backend, bucket_name, object_path, local_path)
        local_paths.append(local_path)

        source_records.append({
            "forecast_hour": f"{forecast_hour:03d}",
            "source_uri": source_uri,
            "file_size": local_path.stat().st_size,
            "checksum_sha256": sha256sum(local_path),
        })

    return local_paths, source_records


def merge_grib_files(local_paths: list[Path], output_path: Path) -> None:
    """Concatenate downloaded GRIB2 files into one merged binary file."""
    with open(output_path, "wb") as merged_file:
        for local_path in local_paths:
            with open(local_path, "rb") as input_file:
                for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                    merged_file.write(chunk)


def build_base_manifest(
    transform_name: str,
    storage_backend: str,
    bucket_name: str,
    prefix: str,
    run_dt: datetime,
    run_hour: str,
    forecast_start: int,
    forecast_step: int,
    forecast_max: int,
    source_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build fields common to all transform manifests."""
    return {
        "manifest_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "GFS",
        "model": "WW3",
        "transform": transform_name,
        "storage_backend": normalize_storage_backend(storage_backend),
        "bucket": bucket_name,
        "storage_prefix": prefix,
        "run_date": run_dt.strftime("%Y-%m-%d"),
        "run_hour": run_hour,
        "run_time": f"{run_dt:%Y%m%d}{run_hour}",
        "forecast_start": forecast_start,
        "forecast_step": forecast_step,
        "forecast_max": forecast_max,
        "source_count": len(source_records),
        "sources": source_records,
    }


def merge_forecast_hours(
    storage_backend: str,
    bucket_name: str,
    prefix: str,
    run_dt: datetime,
    run_hour: str,
    forecast_start: int,
    forecast_step: int,
    forecast_max: int,
) -> dict[str, Any]:
    """Merge bronze forecast-hour GRIB2 files into one silver GRIB2 file."""
    backend = normalize_storage_backend(storage_backend)
    ymdh = f"{run_dt:%Y%m%d}{run_hour}"
    merged_object_path, manifest_object_path = merge_output_paths(
        prefix,
        run_dt,
        run_hour,
        forecast_start,
        forecast_max,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        merged_path = tmpdir_path / f"{ymdh}_f{forecast_start:03d}-f{forecast_max:03d}_merged.grib2"
        local_paths, source_records = download_forecast_files(
            backend,
            bucket_name,
            prefix,
            run_dt,
            run_hour,
            forecast_start,
            forecast_step,
            forecast_max,
            tmpdir_path,
        )

        merge_grib_files(local_paths, merged_path)
        merged_uri = upload_binary(backend, bucket_name, merged_path, merged_object_path)

        manifest = build_base_manifest(
            "merge_forecast_hours_grib2",
            backend,
            bucket_name,
            prefix,
            run_dt,
            run_hour,
            forecast_start,
            forecast_step,
            forecast_max,
            source_records,
        )
        manifest.update({
            "merged_uri": merged_uri,
            "merged_object_path": merged_object_path,
            "merged_file_size": merged_path.stat().st_size,
            "merged_checksum_sha256": hashlib.sha256(merged_path.read_bytes()).hexdigest(),
        })

        manifest_uri = upload_json(backend, bucket_name, manifest, manifest_object_path)
        manifest["manifest_uri"] = manifest_uri
        print(f"Uploaded merged file: {merged_uri}")
        print(f"Uploaded manifest: {manifest_uri}")
        return manifest


def transform_station_wind(
    storage_backend: str,
    bucket_name: str,
    prefix: str,
    run_dt: datetime,
    run_hour: str,
    forecast_start: int,
    forecast_step: int,
    forecast_max: int,
    stations_csv: Path,
) -> dict[str, Any]:
    """Create human-readable and ML-ready station wind CSV outputs from bronze GRIB2 files."""
    backend = normalize_storage_backend(storage_backend)
    ymdh = f"{run_dt:%Y%m%d}{run_hour}"
    human_csv_object_path, ml_csv_object_path, manifest_object_path = station_wind_output_paths(
        prefix,
        run_dt,
        run_hour,
        forecast_start,
        forecast_max,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        local_paths, source_records = download_forecast_files(
            backend,
            bucket_name,
            prefix,
            run_dt,
            run_hour,
            forecast_start,
            forecast_step,
            forecast_max,
            tmpdir_path,
        )

        ds = load_wind_dataset(local_paths)
        try:
            station_wind = build_station_wind_dataframe(ds, stations_csv)
            station_wind_ml = build_station_wind_ml_dataframe(ds, stations_csv, run_time=ymdh)

            human_csv_path = tmpdir_path / f"{ymdh}_station_wind_f{forecast_start:03d}-f{forecast_max:03d}.csv"
            ml_csv_path = tmpdir_path / f"{ymdh}_station_wind_ml_f{forecast_start:03d}-f{forecast_max:03d}.csv"
            station_wind.to_csv(human_csv_path, index=False)
            station_wind_ml.to_csv(ml_csv_path, index=False)
        finally:
            ds.close()

        human_csv_uri = upload_binary(
            backend,
            bucket_name,
            human_csv_path,
            human_csv_object_path,
            content_type="text/csv",
        )
        ml_csv_uri = upload_binary(
            backend,
            bucket_name,
            ml_csv_path,
            ml_csv_object_path,
            content_type="text/csv",
        )

        manifest = build_base_manifest(
            "station_wind_csv",
            backend,
            bucket_name,
            prefix,
            run_dt,
            run_hour,
            forecast_start,
            forecast_step,
            forecast_max,
            source_records,
        )
        manifest.update({
            "station_count": len(station_wind),
            "ml_record_count": len(station_wind_ml),
            "stations_csv": str(stations_csv),
            "human_csv_uri": human_csv_uri,
            "human_csv_object_path": human_csv_object_path,
            "human_csv_file_size": human_csv_path.stat().st_size,
            "human_csv_checksum_sha256": sha256sum(human_csv_path),
            "ml_csv_uri": ml_csv_uri,
            "ml_csv_object_path": ml_csv_object_path,
            "ml_csv_file_size": ml_csv_path.stat().st_size,
            "ml_csv_checksum_sha256": sha256sum(ml_csv_path),
            "ml_columns": list(station_wind_ml.columns),
        })

        manifest_uri = upload_json(backend, bucket_name, manifest, manifest_object_path)
        manifest["manifest_uri"] = manifest_uri
        print(f"Uploaded station wind CSV: {human_csv_uri}")
        print(f"Uploaded station wind ML CSV: {ml_csv_uri}")
        print(f"Uploaded manifest: {manifest_uri}")
        return manifest


def run_transform(config: TransformConfig) -> list[dict[str, Any]]:
    """Run the configured transform for every requested date."""
    manifests = []
    for run_dt in daterange(config.start_date, config.end_date):
        if config.transform_mode == "station_wind":
            if config.stations_csv is None:
                raise ValueError("stations_csv is required for station_wind transform")
            manifest = transform_station_wind(
                config.storage_backend,
                config.bucket_name,
                config.prefix,
                run_dt,
                config.run_hour,
                config.forecast_start,
                config.forecast_step,
                config.forecast_max,
                config.stations_csv,
            )
        else:
            manifest = merge_forecast_hours(
                config.storage_backend,
                config.bucket_name,
                config.prefix,
                run_dt,
                config.run_hour,
                config.forecast_start,
                config.forecast_step,
                config.forecast_max,
            )
        manifests.append(manifest)
    return manifests


def main() -> None:
    """Load transform config from environment and run the selected transform."""
    load_dotenv()
    run_transform(load_transform_config())


if __name__ == "__main__":
    main()
