from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ingest import bucket_name_for_backend, env_value, normalize_storage_backend, parse_date


@dataclass(frozen=True)
class TransformConfig:
    storage_backend: str
    bucket_name: str
    prefix: str
    run_hour: str
    forecast_start: int
    forecast_step: int
    forecast_max: int
    start_date: datetime
    end_date: datetime
    transform_mode: str
    stations_csv: Path | None


def load_transform_config() -> TransformConfig:
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

    return TransformConfig(
        storage_backend=storage_backend,
        bucket_name=bucket_name,
        prefix=prefix,
        run_hour=run_hour,
        forecast_start=forecast_start,
        forecast_step=forecast_step,
        forecast_max=forecast_max,
        start_date=start_dt,
        end_date=end_dt,
        transform_mode=transform_mode,
        stations_csv=stations_csv,
    )
