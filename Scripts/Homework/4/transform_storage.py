from __future__ import annotations

from pathlib import Path

from ingest import gcs_client, normalize_storage_backend, s3_client, storage_uri


def download_object(storage_backend: str, bucket_name: str, object_path: str, local_path: Path) -> None:
    backend = normalize_storage_backend(storage_backend)
    if backend == "gcs":
        gcs_client().bucket(bucket_name).blob(object_path).download_to_filename(str(local_path))
        return

    s3_client().download_file(bucket_name, object_path, str(local_path))


def upload_binary(
    storage_backend: str,
    bucket_name: str,
    local_path: Path,
    object_path: str,
    content_type: str = "application/octet-stream",
) -> str:
    backend = normalize_storage_backend(storage_backend)
    if backend == "gcs":
        blob = gcs_client().bucket(bucket_name).blob(object_path)
        blob.upload_from_filename(str(local_path), content_type=content_type)
        return storage_uri(backend, bucket_name, object_path)

    s3_client().upload_file(str(local_path), bucket_name, object_path)
    return storage_uri(backend, bucket_name, object_path)
