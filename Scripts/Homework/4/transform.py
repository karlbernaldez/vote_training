import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv

from ingest import (
    bucket_name_for_backend,
    daterange,
    env_value,
    gcs_client,
    normalize_storage_backend,
    parse_date,
    s3_client,
    sha256sum,
    storage_uri,
    upload_json,
)


DIRECTIONS_16 = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
]


def download_object(storage_backend: str, bucket_name: str, object_path: str, local_path: Path) -> None:
    backend = normalize_storage_backend(storage_backend)
    if backend == "gcs":
        gcs_client().bucket(bucket_name).blob(object_path).download_to_filename(str(local_path))
        return

    s3_client().download_file(bucket_name, object_path, str(local_path))


def upload_binary(storage_backend: str, bucket_name: str, local_path: Path, object_path: str) -> str:
    backend = normalize_storage_backend(storage_backend)
    if backend == "gcs":
        blob = gcs_client().bucket(bucket_name).blob(object_path)
        blob.upload_from_filename(str(local_path), content_type="application/octet-stream")
        return storage_uri(backend, bucket_name, object_path)

    s3_client().upload_file(str(local_path), bucket_name, object_path)
    return storage_uri(backend, bucket_name, object_path)


def source_object_path(prefix: str, run_dt: datetime, run_hour: str, forecast_hour: int) -> str:
    yyyy = run_dt.strftime("%Y")
    mm = run_dt.strftime("%m")
    dd = run_dt.strftime("%d")
    fhr = f"{forecast_hour:03d}"
    filename = f"{yyyy}{mm}{dd}{run_hour}_f{fhr}.grib2"
    return f"{prefix}/bronze/GFS/{yyyy}/{mm}/{dd}/{run_hour}/{fhr}/{filename}"


def output_paths(prefix: str, run_dt: datetime, run_hour: str, start_hour: int, end_hour: int) -> tuple[str, str]:
    yyyy = run_dt.strftime("%Y")
    mm = run_dt.strftime("%m")
    dd = run_dt.strftime("%d")
    ymdh = f"{yyyy}{mm}{dd}{run_hour}"
    hour_range = f"f{start_hour:03d}-f{end_hour:03d}"
    base_path = f"{prefix}/silver/GFS/{yyyy}/{mm}/{dd}/{run_hour}/{hour_range}"
    return (
        f"{base_path}/{ymdh}_{hour_range}_merged.grib2",
        f"{base_path}/transform_manifest.json",
    )


def station_wind_output_paths(
    prefix: str,
    run_dt: datetime,
    run_hour: str,
    start_hour: int,
    end_hour: int,
) -> tuple[str, str, str]:
    yyyy = run_dt.strftime("%Y")
    mm = run_dt.strftime("%m")
    dd = run_dt.strftime("%d")
    ymdh = f"{yyyy}{mm}{dd}{run_hour}"
    hour_range = f"f{start_hour:03d}-f{end_hour:03d}"
    base_path = f"{prefix}/silver/GFS/{yyyy}/{mm}/{dd}/{run_hour}/station_wind/{hour_range}"
    return (
        f"{base_path}/{ymdh}_station_wind_{hour_range}.csv",
        f"{base_path}/{ymdh}_station_wind_ml_{hour_range}.csv",
        f"{base_path}/transform_manifest.json",
    )


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


def wind_direction_labels(direction_deg: np.ndarray) -> list[str]:
    """Convert wind direction degrees into 16-point compass labels."""
    labels = []
    for direction in np.atleast_1d(direction_deg):
        if np.isnan(direction):
            labels.append(None)
        else:
            labels.append(DIRECTIONS_16[int((direction + 11.25) / 22.5) % 16])
    return labels


def extract_wind_features(ds: xr.Dataset, lat: float, lon: float) -> list[dict[str, Any]]:
    """Extract ML-friendly wind features for one station point.

    Wind direction is represented three ways:
    - wind_direction: human-readable 16-point label
    - wind_dir_deg: direction in degrees
    - wind_dir_sin / wind_dir_cos: circular features for ML models
    """
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


def extract_wind(ds: xr.Dataset, lat: float, lon: float) -> tuple[list[int], list[int], list[str]]:
    """Extract forecast hour, wind speed in kph, and wind direction for one point."""
    records = extract_wind_features(ds, lat, lon)
    return (
        [record["forecast_hour"] for record in records],
        [record["wind_speed_kph"] for record in records],
        [record["wind_direction"] for record in records],
    )


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


def build_station_wind_dataframe(ds: xr.Dataset, stations_csv: Path) -> pd.DataFrame:
    """Build a human-readable station-level forecast table."""
    stations = pd.read_csv(stations_csv)
    required_columns = {"lat", "lon"}
    missing_columns = required_columns - set(stations.columns)
    if missing_columns:
        raise ValueError(f"Stations CSV is missing required columns: {sorted(missing_columns)}")

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
    required_columns = {"lat", "lon"}
    missing_columns = required_columns - set(stations.columns)
    if missing_columns:
        raise ValueError(f"Stations CSV is missing required columns: {sorted(missing_columns)}")

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
    backend = normalize_storage_backend(storage_backend)
    ymdh = f"{run_dt:%Y%m%d}{run_hour}"
    merged_object_path, manifest_object_path = output_paths(prefix, run_dt, run_hour, forecast_start, forecast_max)
    source_records = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        merged_path = tmpdir_path / f"{ymdh}_f{forecast_start:03d}-f{forecast_max:03d}_merged.grib2"

        with open(merged_path, "wb") as merged_file:
            for forecast_hour in range(forecast_start, forecast_max + 1, forecast_step):
                object_path = source_object_path(prefix, run_dt, run_hour, forecast_hour)
                local_path = tmpdir_path / Path(object_path).name
                print(f"Downloading {storage_uri(backend, bucket_name, object_path)}")
                download_object(backend, bucket_name, object_path, local_path)

                size = local_path.stat().st_size
                checksum = sha256sum(local_path)
                with open(local_path, "rb") as input_file:
                    for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                        merged_file.write(chunk)

                source_records.append({
                    "forecast_hour": f"{forecast_hour:03d}",
                    "source_uri": storage_uri(backend, bucket_name, object_path),
                    "file_size": size,
                    "checksum_sha256": checksum,
                })

        merged_uri = upload_binary(backend, bucket_name, merged_path, merged_object_path)
        manifest = {
            "manifest_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "GFS",
            "model": "WW3",
            "transform": "merge_forecast_hours_grib2",
            "storage_backend": backend,
            "bucket": bucket_name,
            "storage_prefix": prefix,
            "run_date": run_dt.strftime("%Y-%m-%d"),
            "run_hour": run_hour,
            "run_time": ymdh,
            "forecast_start": forecast_start,
            "forecast_step": forecast_step,
            "forecast_max": forecast_max,
            "source_count": len(source_records),
            "merged_uri": merged_uri,
            "merged_object_path": merged_object_path,
            "merged_file_size": merged_path.stat().st_size,
            "merged_checksum_sha256": hashlib.sha256(merged_path.read_bytes()).hexdigest(),
            "sources": source_records,
        }
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
    """Create human-readable and ML-ready station wind CSV outputs from bronze GFS GRIB2 files."""
    backend = normalize_storage_backend(storage_backend)
    ymdh = f"{run_dt:%Y%m%d}{run_hour}"
    human_csv_object_path, ml_csv_object_path, manifest_object_path = station_wind_output_paths(
        prefix,
        run_dt,
        run_hour,
        forecast_start,
        forecast_max,
    )
    source_records = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        local_grib_paths = []

        for forecast_hour in range(forecast_start, forecast_max + 1, forecast_step):
            object_path = source_object_path(prefix, run_dt, run_hour, forecast_hour)
            local_path = tmpdir_path / Path(object_path).name
            print(f"Downloading {storage_uri(backend, bucket_name, object_path)}")
            download_object(backend, bucket_name, object_path, local_path)
            local_grib_paths.append(local_path)

            source_records.append({
                "forecast_hour": f"{forecast_hour:03d}",
                "source_uri": storage_uri(backend, bucket_name, object_path),
                "file_size": local_path.stat().st_size,
                "checksum_sha256": sha256sum(local_path),
            })

        ds = load_wind_dataset(local_grib_paths)
        try:
            station_wind = build_station_wind_dataframe(ds, stations_csv)
            station_wind_ml = build_station_wind_ml_dataframe(ds, stations_csv, run_time=ymdh)

            human_csv_path = tmpdir_path / f"{ymdh}_station_wind_f{forecast_start:03d}-f{forecast_max:03d}.csv"
            ml_csv_path = tmpdir_path / f"{ymdh}_station_wind_ml_f{forecast_start:03d}-f{forecast_max:03d}.csv"
            station_wind.to_csv(human_csv_path, index=False)
            station_wind_ml.to_csv(ml_csv_path, index=False)
        finally:
            ds.close()

        human_csv_uri = upload_binary(backend, bucket_name, human_csv_path, human_csv_object_path)
        ml_csv_uri = upload_binary(backend, bucket_name, ml_csv_path, ml_csv_object_path)
        manifest = {
            "manifest_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "GFS",
            "model": "WW3",
            "transform": "station_wind_csv",
            "storage_backend": backend,
            "bucket": bucket_name,
            "storage_prefix": prefix,
            "run_date": run_dt.strftime("%Y-%m-%d"),
            "run_hour": run_hour,
            "run_time": ymdh,
            "forecast_start": forecast_start,
            "forecast_step": forecast_step,
            "forecast_max": forecast_max,
            "station_count": len(station_wind),
            "ml_record_count": len(station_wind_ml),
            "source_count": len(source_records),
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
            "sources": source_records,
        }
        manifest_uri = upload_json(backend, bucket_name, manifest, manifest_object_path)
        manifest["manifest_uri"] = manifest_uri
        print(f"Uploaded station wind CSV: {human_csv_uri}")
        print(f"Uploaded station wind ML CSV: {ml_csv_uri}")
        print(f"Uploaded manifest: {manifest_uri}")
        return manifest


def main():
    load_dotenv()

    storage_backend = normalize_storage_backend(env_value("STORAGE_BACKEND") or "gcs")
    bucket_name = bucket_name_for_backend(storage_backend)
    prefix = env_value("STORAGE_PREFIX") or env_value("GCS_PREFIX") or "vote"
    run_hour = env_value("RUN_HOUR") or "00"
    forecast_start = int(env_value("FORECAST_START") or "0")
    forecast_step = int(env_value("FORECAST_STEP") or "3")
    forecast_max = int(env_value("FORECAST_MAX") or "72")
    start_date = env_value("START_DATE")
    end_date = env_value("END_DATE")
    transform_mode = (env_value("TRANSFORM_MODE") or "merge").strip().lower()
    stations_csv_value = env_value("STATIONS_CSV")

    if not bucket_name:
        bucket_env_var = "GCS_BUCKET_NAME" if storage_backend == "gcs" else "S3_BUCKET_NAME"
        raise ValueError(f"{bucket_env_var} is required when STORAGE_BACKEND={storage_backend}")
    if not start_date or not end_date:
        raise ValueError("START_DATE and END_DATE are required")
    if forecast_step <= 0:
        raise ValueError("FORECAST_STEP must be greater than zero")
    if forecast_max < forecast_start:
        raise ValueError("FORECAST_MAX must be greater than or equal to FORECAST_START")
    if transform_mode not in {"merge", "station_wind"}:
        raise ValueError("TRANSFORM_MODE must be either 'merge' or 'station_wind'")
    if transform_mode == "station_wind" and not stations_csv_value:
        raise ValueError("STATIONS_CSV is required when TRANSFORM_MODE=station_wind")

    stations_csv = Path(stations_csv_value) if stations_csv_value else None
    if stations_csv and not stations_csv.exists():
        raise FileNotFoundError(f"Stations CSV does not exist: {stations_csv}")

    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)
    if end_dt < start_dt:
        raise ValueError("END_DATE must be on or after START_DATE")

    for run_dt in daterange(start_dt, end_dt):
        if transform_mode == "station_wind":
            transform_station_wind(
                storage_backend,
                bucket_name,
                prefix,
                run_dt,
                run_hour,
                forecast_start,
                forecast_step,
                forecast_max,
                stations_csv,
            )
        else:
            merge_forecast_hours(
                storage_backend,
                bucket_name,
                prefix,
                run_dt,
                run_hour,
                forecast_start,
                forecast_step,
                forecast_max,
            )


if __name__ == "__main__":
    # Keep transform.py importable for function-level tests without requiring .env values.
    # Uncomment main() only when running the full storage-backed transform workflow.
    # main()
    print("transform.py loaded. Import and call transform functions from tests or the pipeline.")
