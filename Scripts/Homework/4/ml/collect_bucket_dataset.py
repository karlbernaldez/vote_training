from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from ingest import (
    bucket_name_for_backend,
    env_value,
    gcs_client,
    normalize_storage_backend,
    s3_client,
    storage_uri,
)
from transform.storage import download_object


DEFAULT_GCS_PREFIX = "vote/silver/GFS/"
DEFAULT_OUTPUT_DIR = Path("data/ml/bucket_gridded_wind")


def prefix_for_backend(storage_backend: str) -> str:
    if storage_backend == "gcs":
        return env_value("GCS_PREFIX") or env_value("STORAGE_PREFIX") or "vote"
    return env_value("S3_PREFIX") or env_value("STORAGE_PREFIX") or "vote"


def default_gridded_prefix(storage_backend: str) -> str:
    prefix = prefix_for_backend(storage_backend).strip("/")
    return f"{prefix}/silver/GFS/"


def is_gridded_wind_netcdf(object_name: str) -> bool:
    return "/gridded_wind/" in object_name and object_name.endswith(".nc")


def list_gcs_objects(bucket_name: str, prefix: str) -> list[str]:
    bucket = gcs_client().bucket(bucket_name)
    return [blob.name for blob in bucket.list_blobs(prefix=prefix) if is_gridded_wind_netcdf(blob.name)]


def list_s3_objects(bucket_name: str, prefix: str) -> list[str]:
    client = s3_client()
    paginator = client.get_paginator("list_objects_v2")
    object_names = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if is_gridded_wind_netcdf(key):
                object_names.append(key)
    return object_names


def list_bucket_gridded_wind_objects(storage_backend: str, bucket_name: str, prefix: str) -> list[str]:
    backend = normalize_storage_backend(storage_backend)
    if backend == "gcs":
        return sorted(list_gcs_objects(bucket_name, prefix))
    return sorted(list_s3_objects(bucket_name, prefix))


def local_name_for_object(object_name: str) -> str:
    parts = object_name.split("/")
    filename = parts[-1]

    # Expected shape:
    # <prefix>/silver/GFS/YYYY/MM/DD/HH/gridded_wind/f000-f072/file.nc
    try:
        gfs_index = parts.index("GFS")
        yyyy = parts[gfs_index + 1]
        mm = parts[gfs_index + 2]
        dd = parts[gfs_index + 3]
        hh = parts[gfs_index + 4]
        transform_name = parts[gfs_index + 5]
        hour_range = parts[gfs_index + 6]
    except (ValueError, IndexError):
        return filename

    if transform_name != "gridded_wind":
        return filename

    return f"{yyyy}{mm}{dd}{hh}_{hour_range}_{filename}"


def download_bucket_objects(
    storage_backend: str,
    bucket_name: str,
    object_names: list[str],
    output_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for object_name in object_names:
        local_path = output_dir / local_name_for_object(object_name)
        print(f"Downloading {storage_uri(storage_backend, bucket_name, object_name)} -> {local_path}")
        download_object(storage_backend, bucket_name, object_name, local_path)
        records.append(
            {
                "storage_backend": storage_backend,
                "bucket": bucket_name,
                "object_name": object_name,
                "uri": storage_uri(storage_backend, bucket_name, object_name),
                "local_path": str(local_path),
                "file_size": local_path.stat().st_size,
            }
        )
    return records


def write_collection_manifest(records: list[dict], output_dir: Path, prefix: str) -> Path:
    manifest_path = output_dir / "collection_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "collection_type": "bucket_gridded_wind_netcdf",
                "prefix": prefix,
                "file_count": len(records),
                "files": records,
            },
            f,
            indent=2,
        )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download gridded_wind NetCDF outputs from GCS/S3 for ML training.")
    parser.add_argument("--storage-backend", choices=["gcs", "s3"], default=None, help="Storage backend. Defaults to STORAGE_BACKEND or gcs.")
    parser.add_argument("--bucket", default=None, help="Bucket name. Defaults to GCS_BUCKET_NAME or S3_BUCKET_NAME.")
    parser.add_argument("--prefix", default=None, help="Object prefix to search. Defaults to <prefix>/silver/GFS/.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Local directory for downloaded NetCDF files.")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of NetCDF files to download.")
    parser.add_argument("--dry-run", action="store_true", help="List matching objects without downloading them.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    storage_backend = normalize_storage_backend(args.storage_backend or env_value("STORAGE_BACKEND") or "gcs")
    bucket_name = args.bucket or bucket_name_for_backend(storage_backend)
    if not bucket_name:
        bucket_env = "GCS_BUCKET_NAME" if storage_backend == "gcs" else "S3_BUCKET_NAME"
        raise ValueError(f"Bucket is required. Pass --bucket or set {bucket_env}.")

    prefix = args.prefix or default_gridded_prefix(storage_backend)
    object_names = list_bucket_gridded_wind_objects(storage_backend, bucket_name, prefix)
    if args.max_files is not None:
        object_names = object_names[: args.max_files]

    print(f"Found {len(object_names)} gridded_wind NetCDF file(s) under {storage_backend}://{bucket_name}/{prefix}")
    for object_name in object_names:
        print(storage_uri(storage_backend, bucket_name, object_name))

    if args.dry_run:
        return
    if not object_names:
        raise ValueError("No gridded_wind NetCDF files found. Run the gridded_wind transform first or adjust --prefix.")

    output_dir = Path(args.output_dir)
    records = download_bucket_objects(storage_backend, bucket_name, object_names, output_dir)
    manifest_path = write_collection_manifest(records, output_dir, prefix)
    print(f"Wrote collection manifest: {manifest_path}")


if __name__ == "__main__":
    main()
