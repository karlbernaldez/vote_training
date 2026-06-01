from datetime import datetime

SUPPORTED_STORAGE_BACKENDS = {"gcs", "s3"}


def normalize_storage_backend(storage_backend: str) -> str:
    backend = storage_backend.strip().lower()
    if backend not in SUPPORTED_STORAGE_BACKENDS:
        raise ValueError(f"Unsupported STORAGE_BACKEND '{storage_backend}'")
    return backend


def parse_date(value: str) -> datetime:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Invalid date '{value}'")
