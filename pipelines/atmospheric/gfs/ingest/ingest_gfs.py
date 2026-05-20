import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config
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
        raise ValueError(f"Unsupported STORAGE_BACKEND '{storage_backend}'")
    return backend


def split_s3_bucket_config(raw_bucket_config: str) -> tuple[str | None, str]:
    value = raw_bucket_config.strip()
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        return f"{parsed.scheme}://{parsed.netloc}", parsed.path.strip("/")
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


def s3_client():
    client_kwargs = {
        "config": Config(signature_version="s3v4", s3={"addressing_style": "path"})
    }
    endpoint_url = s3_endpoint_url()
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    return boto3.client("s3", **client_kwargs)


def gcs_client() -> storage.Client:
    return storage.Client()


def build_download_url(yyyy: str, mm: str, dd: str, hh: str, fhr: str) -> str:
    return (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        f"?dir=%2Fgfs.{yyyy}{mm}{dd}%2F{hh}%2Fatmos"
        f"&file=gfs.t{hh}z.pgrb2.0p25.f{fhr}"
        "&var_UGRD=on&var_VGRD=on&lev_10_m_above_ground=on"
    )


def parse_date(value: str) -> datetime:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Invalid date '{value}'")


def daterange(start: datetime, end: datetime):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def main() -> None:
    load_dotenv()
    print("GFS ingest migrated foundation active")


if __name__ == "__main__":
    main()
