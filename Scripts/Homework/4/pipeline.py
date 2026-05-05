import os
import sys
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from ingest import (
    build_download_url,
    bucket_name_for_backend,
    env_value,
    normalize_storage_backend,
    run_cycle,
)
from transform import merge_forecast_hours


def current_run_date() -> datetime:
    configured_date = env_value("RUN_DATE")
    if configured_date:
        from ingest import parse_date

        return parse_date(configured_date)
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def source_is_available(run_dt: datetime, run_hour: str, forecast_hour: int, timeout: int = 30) -> bool:
    yyyy = run_dt.strftime("%Y")
    mm = run_dt.strftime("%m")
    dd = run_dt.strftime("%d")
    fhr = f"{forecast_hour:03d}"
    url = build_download_url(yyyy, mm, dd, run_hour, fhr)

    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code == 405:
            response = requests.get(url, timeout=timeout, stream=True)
        return response.status_code == 200
    except requests.RequestException as exc:
        print(f"Source availability check failed: {exc}")
        return False


def main() -> int:
    load_dotenv()

    storage_backend = normalize_storage_backend(env_value("STORAGE_BACKEND") or "gcs")
    bucket_name = bucket_name_for_backend(storage_backend)
    prefix = env_value("STORAGE_PREFIX") or env_value("GCS_PREFIX") or "vote"
    run_hour = env_value("RUN_HOUR") or "00"
    forecast_start = int(env_value("FORECAST_START") or "0")
    forecast_step = int(env_value("FORECAST_STEP") or "3")
    forecast_max = int(env_value("FORECAST_MAX") or "72")
    availability_forecast_hour = int(env_value("AVAILABILITY_FORECAST_HOUR") or str(forecast_max))
    run_dt = current_run_date()

    if not bucket_name:
        bucket_env_var = "GCS_BUCKET_NAME" if storage_backend == "gcs" else "S3_BUCKET_NAME"
        raise ValueError(f"{bucket_env_var} is required when STORAGE_BACKEND={storage_backend}")
    if forecast_step <= 0:
        raise ValueError("FORECAST_STEP must be greater than zero")
    if forecast_max < forecast_start:
        raise ValueError("FORECAST_MAX must be greater than or equal to FORECAST_START")

    print(
        "Pipeline config:",
        {
            "storage_backend": storage_backend,
            "bucket": bucket_name,
            "prefix": prefix,
            "run_date": run_dt.strftime("%Y-%m-%d"),
            "run_hour": run_hour,
            "forecast_start": forecast_start,
            "forecast_step": forecast_step,
            "forecast_max": forecast_max,
            "availability_forecast_hour": availability_forecast_hour,
        },
    )

    if not source_is_available(run_dt, run_hour, availability_forecast_hour):
        print(
            "Source data is not available yet; skipping ingest and transform for",
            f"run_date={run_dt:%Y-%m-%d}",
            f"run_hour={run_hour}",
            f"forecast_hour={availability_forecast_hour:03d}",
        )
        return 0

    print("Source data is available. Running ingest...")
    ingest_log = run_cycle(
        bucket_name,
        prefix,
        run_dt,
        run_hour,
        forecast_step,
        forecast_max,
        storage_backend,
    )

    failed = [item for item in ingest_log if item.get("status") == "FAILED"]
    if failed:
        print(f"Ingest finished with {len(failed)} failed file(s). Transform will not run.")
        return 1

    print("Ingest completed successfully. Running transform...")
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
    print("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
