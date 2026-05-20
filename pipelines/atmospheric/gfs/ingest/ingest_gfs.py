import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from pipelines.shared.storage.storage import gcs_client, s3_client, split_s3_bucket_config
from .urls import build_download_url
from .validators import normalize_storage_backend, parse_date

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


def s3_endpoint_url() -> str | None:
    explicit_endpoint = env_value("S3_ENDPOINT_URL")
    if explicit_endpoint:
        return explicit_endpoint.rstrip("/")
    raw_bucket_config = env_value("S3_BUCKET_NAME")
    if not raw_bucket_config:
        return None
    endpoint_url, _ = split_s3_bucket_config(raw_bucket_config)
    return endpoint_url


def daterange(start: datetime, end: datetime):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def main() -> None:
    load_dotenv()
    normalize_storage_backend(env_value("STORAGE_BACKEND") or "gcs")
    _ = build_download_url("2026", "01", "01", "00", "000")
    _ = gcs_client
    _ = s3_client
    print("GFS ingest shared storage migration active")


if __name__ == "__main__":
    main()
