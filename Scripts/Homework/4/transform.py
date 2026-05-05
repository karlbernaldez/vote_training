import hashlib
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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

    if not bucket_name:
        bucket_env_var = "GCS_BUCKET_NAME" if storage_backend == "gcs" else "S3_BUCKET_NAME"
        raise ValueError(f"{bucket_env_var} is required when STORAGE_BACKEND={storage_backend}")
    if not start_date or not end_date:
        raise ValueError("START_DATE and END_DATE are required")
    if forecast_step <= 0:
        raise ValueError("FORECAST_STEP must be greater than zero")
    if forecast_max < forecast_start:
        raise ValueError("FORECAST_MAX must be greater than or equal to FORECAST_START")

    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)
    if end_dt < start_dt:
        raise ValueError("END_DATE must be on or after START_DATE")

    for run_dt in daterange(start_dt, end_dt):
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
    main()
