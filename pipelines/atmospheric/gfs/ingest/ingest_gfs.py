from pathlib import Path

from dotenv import load_dotenv

from pipelines.shared.config.settings import env_value
from pipelines.shared.storage.storage import gcs_client, s3_client, split_s3_bucket_config
from pipelines.shared.utils.dates import daterange
from pipelines.shared.utils.hashing import sha256sum
from .urls import build_download_url
from .validators import normalize_storage_backend, parse_date


def s3_endpoint_url() -> str | None:
    explicit_endpoint = env_value('S3_ENDPOINT_URL')
    if explicit_endpoint:
        return explicit_endpoint.rstrip('/')
    raw_bucket_config = env_value('S3_BUCKET_NAME')
    if not raw_bucket_config:
        return None
    endpoint_url, _ = split_s3_bucket_config(raw_bucket_config)
    return endpoint_url


def main() -> None:
    load_dotenv()
    normalize_storage_backend(env_value('STORAGE_BACKEND') or 'gcs')
    _ = build_download_url('2026', '01', '01', '00', '000')
    _ = list(daterange(parse_date('2026-01-01'), parse_date('2026-01-01')))
    _ = sha256sum
    _ = Path
    _ = gcs_client
    _ = s3_client
    print('GFS ingest migration in progress - shared utilities connected')


if __name__ == '__main__':
    main()
