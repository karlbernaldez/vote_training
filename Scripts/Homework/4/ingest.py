import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import requests
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from google.cloud import storage

SUPPORTED_STORAGE_BACKENDS = {"gcs", "s3"}
TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def sha256sum(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_storage_backend(storage_backend: str) -> str:
    backend = storage_backend.strip().lower()
    if backend not in SUPPORTED_STORAGE_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_STORAGE_BACKENDS))
        raise ValueError(f"Unsupported STORAGE_BACKEND '{storage_backend}'. Use one of: {supported}.")
    return backend


def split_s3_bucket_config(raw_bucket_config: str) -> tuple[str | None, str]:
    """Return (endpoint_url, bucket_name) from S3_BUCKET_NAME.

    Preferred config is S3_ENDPOINT_URL=http://host:port and S3_BUCKET_NAME=bucket.
    For convenience, S3_BUCKET_NAME=http://host:port/bucket is also accepted.
    """
    value = raw_bucket_config.strip()
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        bucket_name = parsed.path.strip("/")
        if not bucket_name or "/" in bucket_name:
            raise ValueError(
                "When S3_BUCKET_NAME is a URL, it must be in the form "
                "http://host:port/bucket-name."
            )
        endpoint_url = f"{parsed.scheme}://{parsed.netloc}"
        return endpoint_url, bucket_name
    return None, value


def s3_endpoint_url() -> str | None:
    explicit_endpoint = env_value("S3_ENDPOINT_URL")
    if explicit_endpoint:
        return explicit_endpoint.rstrip("/")

    raw_bucket_config = env_value("S3_BUCKET_NAME")
    if not raw_bucket_config:
        return None

    endpoint_url, _ = split_s3_bucket_config(raw_bucket_config)
    return endpoint_url


def s3_credentials() -> dict[str, str]:
    access_key = env_value("CEPH_ACCESS_KEY") or env_value("AWS_ACCESS_KEY_ID")
    secret_key = env_value("CEPH_SECRET_KEY") or env_value("AWS_SECRET_ACCESS_KEY")

    if bool(access_key) != bool(secret_key):
        raise ValueError(
            "Both access key and secret key are required for S3 authentication. "
            "Set CEPH_ACCESS_KEY and CEPH_SECRET_KEY, or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
        )
    if not access_key:
        if s3_endpoint_url() and not env_flag("S3_ALLOW_ANONYMOUS"):
            raise ValueError(
                "S3 credentials are required when using a custom S3/Ceph endpoint. "
                "Set CEPH_ACCESS_KEY and CEPH_SECRET_KEY, or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY. "
                "Set S3_ALLOW_ANONYMOUS=true only if this bucket intentionally allows anonymous writes."
            )
        return {}

    return {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
    }


def storage_uri(storage_backend: str, bucket_name: str, object_path: str) -> str:
    backend = normalize_storage_backend(storage_backend)
    scheme = "gs" if backend == "gcs" else "s3"
    return f"{scheme}://{bucket_name}/{object_path}"


def with_storage_uri(record: dict[str, Any], storage_backend: str, bucket_name: str, object_path: str) -> dict[str, Any]:
    uri = storage_uri(storage_backend, bucket_name, object_path)
    record["storage_backend"] = normalize_storage_backend(storage_backend)
    record["storage_uri"] = uri

    # Keep the provider-specific URI fields for backward compatibility with older logs.
    if record["storage_backend"] == "gcs":
        record["gcs_uri"] = uri
    else:
        record["s3_uri"] = uri

    return record


def build_download_url(yyyy: str, mm: str, dd: str, hh: str, fhr: str) -> str:
    return (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        f"?dir=%2Fgfs.{yyyy}{mm}{dd}%2F{hh}%2Fatmos"
        f"&file=gfs.t{hh}z.pgrb2.0p25.f{fhr}"
        "&var_UGRD=on"
        "&var_VGRD=on"
        "&lev_10_m_above_ground=on"
        "&subregion="
        "&toplat=50"
        "&leftlon=100"
        "&rightlon=180"
        "&bottomlat=-5"
    )


def gcs_client() -> storage.Client:
    return storage.Client()


def s3_client():
    client_kwargs = {
        "config": Config(
            signature_version=env_value("S3_SIGNATURE_VERSION") or "s3v4",
            s3={"addressing_style": env_value("S3_ADDRESSING_STYLE") or "path"},
        )
    }
    endpoint_url = s3_endpoint_url()
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    client_kwargs.update(s3_credentials())
    return boto3.client("s3", **client_kwargs)


def object_exists(storage_backend: str, bucket_name: str, object_path: str) -> bool:
    backend = normalize_storage_backend(storage_backend)
    if backend == "gcs":
        client = gcs_client()
        return client.bucket(bucket_name).blob(object_path).exists(client)

    if env_flag("S3_SKIP_EXISTS_CHECK"):
        return False

    try:
        s3_client().head_object(Bucket=bucket_name, Key=object_path)
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return False
        if error_code in {"403", "AccessDenied"}:
            raise PermissionError(
                "Ceph/S3 denied HeadObject while checking whether the object already exists. "
                "Verify the configured S3 credentials. If the key can write objects but cannot head/read them, "
                "set S3_SKIP_EXISTS_CHECK=true to upload without the pre-check."
            ) from e
        raise


def upload_file(storage_backend: str, bucket_name: str, local_file: Path, object_path: str) -> str:
    backend = normalize_storage_backend(storage_backend)
    if backend == "gcs":
        client = gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_path)
        blob.upload_from_filename(str(local_file))
        return storage_uri(backend, bucket_name, object_path)

    s3_client().upload_file(str(local_file), bucket_name, object_path)
    return storage_uri(backend, bucket_name, object_path)


def upload_json(storage_backend: str, bucket_name: str, data: Any, object_path: str) -> str:
    backend = normalize_storage_backend(storage_backend)
    body = json.dumps(data, indent=2)
    if backend == "gcs":
        client = gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_path)
        blob.upload_from_string(body, content_type="application/json")
        return storage_uri(backend, bucket_name, object_path)

    s3_client().put_object(
        Bucket=bucket_name,
        Key=object_path,
        Body=body.encode("utf-8"),
        ContentType="application/json",
    )
    return storage_uri(backend, bucket_name, object_path)


# Backward-compatible wrappers for callers that imported the old GCS-specific helpers.
def upload_to_gcs(bucket_name: str, local_file: Path, object_path: str) -> str:
    return upload_file("gcs", bucket_name, local_file, object_path)


def upload_json_to_gcs(bucket_name: str, data: Any, object_path: str) -> str:
    return upload_json("gcs", bucket_name, data, object_path)


def download_file(url: str, local_path: Path, timeout: int = 120) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def parse_date(value: str) -> datetime:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Invalid date '{value}'. Use YYYYMMDD or YYYY-MM-DD.")


def daterange(start: datetime, end: datetime):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_manifest(
    bucket_name: str,
    gcs_prefix: str,
    run_dt: datetime,
    run_hour: str,
    forecast_step: int,
    forecast_max: int,
    ingest_log: list[dict[str, Any]],
    storage_backend: str = "gcs",
) -> dict[str, Any]:
    backend = normalize_storage_backend(storage_backend)
    yyyy = run_dt.strftime("%Y")
    mm = run_dt.strftime("%m")
    dd = run_dt.strftime("%d")
    ymdh = f"{yyyy}{mm}{dd}{run_hour}"

    return {
        "manifest_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "GFS",
        "model": "WW3",
        "storage_backend": backend,
        "bucket": bucket_name,
        "gcs_prefix": gcs_prefix,
        "storage_prefix": gcs_prefix,
        "run_date": run_dt.strftime("%Y-%m-%d"),
        "run_hour": run_hour,
        "run_time": ymdh,
        "forecast_step": forecast_step,
        "forecast_max": forecast_max,
        "record_count": len(ingest_log),
        "downloaded": sum(1 for item in ingest_log if item["status"] == "DOWNLOADED_AND_UPLOADED"),
        "skipped": sum(1 for item in ingest_log if item["status"] == "SKIPPED_ALREADY_EXISTS"),
        "failed": sum(1 for item in ingest_log if item["status"] == "FAILED"),
        "files": ingest_log,
    }


def run_cycle(
    bucket_name: str,
    gcs_prefix: str,
    run_dt: datetime,
    run_hour: str,
    forecast_step: int,
    forecast_max: int,
    storage_backend: str = "gcs",
) -> list:
    backend = normalize_storage_backend(storage_backend)
    yyyy = run_dt.strftime("%Y")
    mm = run_dt.strftime("%m")
    dd = run_dt.strftime("%d")
    hh = run_hour
    ymdh = f"{yyyy}{mm}{dd}{hh}"

    ingest_log = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for n in range(0, forecast_max + 1, forecast_step):
            fhr = f"{n:03d}"
            filename = f"{ymdh}_f{fhr}.grib2"
            object_path = (
                f"{gcs_prefix}/bronze/GFS/"
                f"{yyyy}/{mm}/{dd}/{hh}/{fhr}/{filename}"
            )
            download_url = build_download_url(yyyy, mm, dd, hh, fhr)

            if object_exists(backend, bucket_name, object_path):
                ingest_log.append(with_storage_uri({
                    "source": "GFS",
                    "model": "WW3",
                    "run_time": ymdh,
                    "forecast_hour": fhr,
                    "format": "grib2",
                    "object_path": object_path,
                    "status": "SKIPPED_ALREADY_EXISTS",
                }, backend, bucket_name, object_path))
                print(f"Skip existing: {storage_uri(backend, bucket_name, object_path)}")
                continue

            local_path = tmpdir_path / filename
            print(f"Downloading run={ymdh} forecast_hour={fhr}")
            try:
                download_file(download_url, local_path)
                checksum = sha256sum(local_path)
                size = local_path.stat().st_size
                object_uri = upload_file(backend, bucket_name, local_path, object_path)
                ingest_log.append(with_storage_uri({
                    "source": "GFS",
                    "model": "WW3",
                    "run_time": ymdh,
                    "forecast_hour": fhr,
                    "format": "grib2",
                    "download_url": download_url,
                    "object_path": object_path,
                    "file_name": filename,
                    "file_size": size,
                    "checksum_sha256": checksum,
                    "status": "DOWNLOADED_AND_UPLOADED",
                }, backend, bucket_name, object_path))
                print(f"Uploaded: {object_uri}")
            except Exception as e:
                ingest_log.append({
                    "source": "GFS",
                    "model": "WW3",
                    "storage_backend": backend,
                    "run_time": ymdh,
                    "forecast_hour": fhr,
                    "format": "grib2",
                    "download_url": download_url,
                    "file_name": filename,
                    "object_path": object_path,
                    "status": "FAILED",
                    "error": str(e),
                })
                print(f"Failed: {filename} -> {e}")

    base_object_path = f"{gcs_prefix}/bronze/GFS/{yyyy}/{mm}/{dd}/{hh}"
    log_object_path = f"{base_object_path}/ingest_log.json"
    manifest_object_path = f"{base_object_path}/manifest.json"

    upload_json(backend, bucket_name, ingest_log, log_object_path)
    manifest = build_manifest(bucket_name, gcs_prefix, run_dt, run_hour, forecast_step, forecast_max, ingest_log, backend)
    upload_json(backend, bucket_name, manifest, manifest_object_path)
    return ingest_log


def bucket_name_for_backend(storage_backend: str) -> str | None:
    backend = normalize_storage_backend(storage_backend)
    if backend == "gcs":
        return env_value("GCS_BUCKET_NAME")

    raw_bucket_config = env_value("S3_BUCKET_NAME")
    if not raw_bucket_config:
        return None
    _, bucket_name = split_s3_bucket_config(raw_bucket_config)
    return bucket_name


def main():
    load_dotenv()

    storage_backend = normalize_storage_backend(env_value("STORAGE_BACKEND") or "gcs")
    bucket_name = bucket_name_for_backend(storage_backend)
    gcs_prefix = env_value("STORAGE_PREFIX") or env_value("GCS_PREFIX") or "vote"
    run_hour = env_value("RUN_HOUR") or "00"
    forecast_step = int(env_value("FORECAST_STEP") or "3")
    forecast_max = int(env_value("FORECAST_MAX") or "72")
    start_date = env_value("START_DATE")
    end_date = env_value("END_DATE")

    if not bucket_name:
        bucket_env_var = "GCS_BUCKET_NAME" if storage_backend == "gcs" else "S3_BUCKET_NAME"
        raise ValueError(f"{bucket_env_var} is required in .env when STORAGE_BACKEND={storage_backend}")
    if not start_date or not end_date:
        raise ValueError("START_DATE and END_DATE are required in .env")

    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)
    if end_dt < start_dt:
        raise ValueError("END_DATE must be on or after START_DATE")

    summary = []
    for run_dt in daterange(start_dt, end_dt):
        cycle_log = run_cycle(bucket_name, gcs_prefix, run_dt, run_hour, forecast_step, forecast_max, storage_backend)
        summary.append({
            "storage_backend": storage_backend,
            "run_date": run_dt.strftime("%Y-%m-%d"),
            "run_hour": run_hour,
            "records": len(cycle_log),
            "downloaded": sum(1 for x in cycle_log if x["status"] == "DOWNLOADED_AND_UPLOADED"),
            "skipped": sum(1 for x in cycle_log if x["status"] == "SKIPPED_ALREADY_EXISTS"),
            "failed": sum(1 for x in cycle_log if x["status"] == "FAILED"),
        })

    print("\nSummary")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
