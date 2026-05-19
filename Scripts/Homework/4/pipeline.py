import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from ingest import (
    build_download_url,
    bucket_name_for_backend,
    env_value,
    normalize_storage_backend,
    run_cycle,
)
from transform import merge_forecast_hours, transform_station_wind


def current_run_date() -> datetime:
    configured_date = env_value("RUN_DATE")
    if configured_date:
        from ingest import parse_date

        return parse_date(configured_date)
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def selected_storage_backends() -> list[str]:
    """Return the concrete storage backends this pipeline should run.

    Supported forms:
    - STORAGE_BACKEND=gcs
    - STORAGE_BACKEND=s3
    - STORAGE_BACKEND=both
    - STORAGE_BACKENDS=gcs,s3
    """
    explicit_backends = env_value("STORAGE_BACKENDS")
    if explicit_backends:
        raw_backends = [item.strip() for item in explicit_backends.split(",") if item.strip()]
    else:
        raw_backend = (env_value("STORAGE_BACKEND") or "gcs").strip().lower()
        raw_backends = ["gcs", "s3"] if raw_backend == "both" else [raw_backend]

    backends = []
    for backend in raw_backends:
        normalized_backend = normalize_storage_backend(backend)
        if normalized_backend not in backends:
            backends.append(normalized_backend)
    return backends


def selected_transform_mode() -> str:
    transform_mode = (env_value("TRANSFORM_MODE") or "merge").strip().lower()
    if transform_mode not in {"merge", "station_wind"}:
        raise ValueError("TRANSFORM_MODE must be either 'merge' or 'station_wind'")
    return transform_mode


def prefix_for_backend(storage_backend: str) -> str:
    if storage_backend == "gcs":
        return env_value("GCS_PREFIX") or env_value("STORAGE_PREFIX") or "vote"
    return env_value("S3_PREFIX") or env_value("STORAGE_PREFIX") or "vote"


def validate_backend_config(storage_backends: list[str]) -> None:
    """Fail early with actionable config errors before any ingest work starts."""
    if "gcs" in storage_backends:
        credentials_path = env_value("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS is required when running the GCS backend. "
                "When using Docker, mount the key to /app/gcp-key.json and set "
                "GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-key.json."
            )
        if not Path(credentials_path).is_file():
            raise FileNotFoundError(
                f"GOOGLE_APPLICATION_CREDENTIALS points to '{credentials_path}', but that file does not exist "
                "inside the container. Pass GCP_KEY_PATH=/host/path/key.json to run-pipeline.sh so it can mount "
                "the key as /app/gcp-key.json."
            )
        if not env_value("GCS_BUCKET_NAME"):
            raise ValueError("GCS_BUCKET_NAME is required when running the GCS backend.")

    if "s3" in storage_backends:
        if not env_value("S3_BUCKET_NAME"):
            raise ValueError("S3_BUCKET_NAME is required when running the S3 backend.")
        has_ceph_credentials = bool(env_value("CEPH_ACCESS_KEY") and env_value("CEPH_SECRET_KEY"))
        has_aws_credentials = bool(env_value("AWS_ACCESS_KEY_ID") and env_value("AWS_SECRET_ACCESS_KEY"))
        if not has_ceph_credentials and not has_aws_credentials and not env_value("S3_ALLOW_ANONYMOUS"):
            raise ValueError(
                "S3 credentials are required when running the S3 backend. Set CEPH_ACCESS_KEY and "
                "CEPH_SECRET_KEY, or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
            )


def validate_transform_config(transform_mode: str) -> None:
    if transform_mode == "station_wind":
        stations_csv = env_value("STATIONS_CSV")
        if not stations_csv:
            raise ValueError("STATIONS_CSV is required when TRANSFORM_MODE=station_wind")
        if not Path(stations_csv).is_file():
            raise FileNotFoundError(f"STATIONS_CSV does not exist inside this runtime: {stations_csv}")


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


def run_transform_stage(
    transform_mode: str,
    storage_backend: str,
    bucket_name: str,
    prefix: str,
    run_dt: datetime,
    run_hour: str,
    forecast_start: int,
    forecast_step: int,
    forecast_max: int,
) -> None:
    if transform_mode == "station_wind":
        stations_csv = env_value("STATIONS_CSV")
        if not stations_csv:
            raise ValueError("STATIONS_CSV is required when TRANSFORM_MODE=station_wind")
        transform_station_wind(
            storage_backend,
            bucket_name,
            prefix,
            run_dt,
            run_hour,
            forecast_start,
            forecast_step,
            forecast_max,
            Path(stations_csv),
        )
        return

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


def run_backend_pipeline(
    storage_backend: str,
    transform_mode: str,
    run_dt: datetime,
    run_hour: str,
    forecast_start: int,
    forecast_step: int,
    forecast_max: int,
) -> bool:
    bucket_name = bucket_name_for_backend(storage_backend)
    prefix = prefix_for_backend(storage_backend)

    if not bucket_name:
        bucket_env_var = "GCS_BUCKET_NAME" if storage_backend == "gcs" else "S3_BUCKET_NAME"
        raise ValueError(f"{bucket_env_var} is required when STORAGE_BACKEND={storage_backend}")

    print(
        "Running backend pipeline:",
        {
            "storage_backend": storage_backend,
            "bucket": bucket_name,
            "prefix": prefix,
            "run_date": run_dt.strftime("%Y-%m-%d"),
            "run_hour": run_hour,
            "forecast_start": forecast_start,
            "forecast_step": forecast_step,
            "forecast_max": forecast_max,
            "transform_mode": transform_mode,
        },
    )

    print(f"Running ingest for backend={storage_backend}...")
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
        print(
            f"Ingest finished with {len(failed)} failed file(s) for backend={storage_backend}. "
            "Transform will not run for this backend."
        )
        return False

    print(f"Ingest completed successfully for backend={storage_backend}. Running transform mode={transform_mode}...")
    run_transform_stage(
        transform_mode,
        storage_backend,
        bucket_name,
        prefix,
        run_dt,
        run_hour,
        forecast_start,
        forecast_step,
        forecast_max,
    )
    print(f"Backend pipeline completed successfully for backend={storage_backend}.")
    return True


def main() -> int:
    load_dotenv()

    storage_backends = selected_storage_backends()
    transform_mode = selected_transform_mode()
    validate_backend_config(storage_backends)
    validate_transform_config(transform_mode)

    run_hour = env_value("RUN_HOUR") or "00"
    forecast_start = int(env_value("FORECAST_START") or "0")
    forecast_step = int(env_value("FORECAST_STEP") or "3")
    forecast_max = int(env_value("FORECAST_MAX") or "72")
    availability_forecast_hour = int(env_value("AVAILABILITY_FORECAST_HOUR") or str(forecast_max))
    run_dt = current_run_date()

    if forecast_step <= 0:
        raise ValueError("FORECAST_STEP must be greater than zero")
    if forecast_max < forecast_start:
        raise ValueError("FORECAST_MAX must be greater than or equal to FORECAST_START")

    print(
        "Pipeline config:",
        {
            "storage_backends": storage_backends,
            "transform_mode": transform_mode,
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

    print("Source data is available. Running configured storage backends...")
    results = []
    for storage_backend in storage_backends:
        results.append(
            run_backend_pipeline(
                storage_backend,
                transform_mode,
                run_dt,
                run_hour,
                forecast_start,
                forecast_step,
                forecast_max,
            )
        )

    if all(results):
        print("Pipeline completed successfully for all configured backends.")
        return 0

    print("Pipeline completed with one or more backend failures.")
    return 1


if __name__ == "__main__":
    sys.exit(main())