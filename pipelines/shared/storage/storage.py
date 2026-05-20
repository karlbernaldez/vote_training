from urllib.parse import urlparse

import boto3
from botocore.config import Config
from google.cloud import storage


def split_s3_bucket_config(raw_bucket_config: str) -> tuple[str | None, str]:
    value = raw_bucket_config.strip()
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        return f"{parsed.scheme}://{parsed.netloc}", parsed.path.strip("/")
    return None, value


def s3_client(endpoint_url: str | None = None):
    client_kwargs = {
        "config": Config(signature_version="s3v4", s3={"addressing_style": "path"})
    }
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    return boto3.client("s3", **client_kwargs)


def gcs_client() -> storage.Client:
    return storage.Client()
