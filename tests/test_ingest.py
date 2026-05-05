import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


INGEST_PATH = Path(__file__).resolve().parents[1] / "Scripts" / "Homework" / "4" / "ingest.py"


def load_ingest_module():
    spec = importlib.util.spec_from_file_location("ingest", INGEST_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ingest():
    return load_ingest_module()


def test_build_manifest_summarizes_ingest_log(ingest, monkeypatch):
    fixed_now = datetime(2026, 5, 5, 12, 30, tzinfo=timezone.utc)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(ingest, "datetime", FixedDatetime)
    ingest_log = [
        {
            "source": "GFS",
            "model": "WW3",
            "storage_backend": "gcs",
            "run_time": "2026050500",
            "forecast_hour": "000",
            "format": "grib2",
            "object_path": "vote/bronze/GFS/2026/05/05/00/000/2026050500_f000.grib2",
            "storage_uri": "gs://bucket/vote/bronze/GFS/2026/05/05/00/000/2026050500_f000.grib2",
            "gcs_uri": "gs://bucket/vote/bronze/GFS/2026/05/05/00/000/2026050500_f000.grib2",
            "status": "DOWNLOADED_AND_UPLOADED",
        },
        {
            "source": "GFS",
            "model": "WW3",
            "storage_backend": "gcs",
            "run_time": "2026050500",
            "forecast_hour": "003",
            "format": "grib2",
            "object_path": "vote/bronze/GFS/2026/05/05/00/003/2026050500_f003.grib2",
            "storage_uri": "gs://bucket/vote/bronze/GFS/2026/05/05/00/003/2026050500_f003.grib2",
            "gcs_uri": "gs://bucket/vote/bronze/GFS/2026/05/05/00/003/2026050500_f003.grib2",
            "status": "SKIPPED_ALREADY_EXISTS",
        },
        {
            "source": "GFS",
            "model": "WW3",
            "storage_backend": "gcs",
            "run_time": "2026050500",
            "forecast_hour": "006",
            "format": "grib2",
            "object_path": "vote/bronze/GFS/2026/05/05/00/006/2026050500_f006.grib2",
            "status": "FAILED",
            "error": "network timeout",
        },
    ]

    manifest = ingest.build_manifest(
        bucket_name="bucket",
        gcs_prefix="vote",
        run_dt=datetime(2026, 5, 5),
        run_hour="00",
        forecast_step=3,
        forecast_max=6,
        ingest_log=ingest_log,
        storage_backend="gcs",
    )

    assert manifest["manifest_version"] == "1.0"
    assert manifest["generated_at"] == fixed_now.isoformat()
    assert manifest["source"] == "GFS"
    assert manifest["model"] == "WW3"
    assert manifest["storage_backend"] == "gcs"
    assert manifest["bucket"] == "bucket"
    assert manifest["gcs_prefix"] == "vote"
    assert manifest["storage_prefix"] == "vote"
    assert manifest["run_date"] == "2026-05-05"
    assert manifest["run_hour"] == "00"
    assert manifest["run_time"] == "2026050500"
    assert manifest["forecast_step"] == 3
    assert manifest["forecast_max"] == 6
    assert manifest["record_count"] == 3
    assert manifest["downloaded"] == 1
    assert manifest["skipped"] == 1
    assert manifest["failed"] == 1
    assert manifest["files"] == ingest_log


@pytest.mark.parametrize(
    ("storage_backend", "expected_scheme", "provider_uri_key"),
    [
        ("gcs", "gs", "gcs_uri"),
        ("s3", "s3", "s3_uri"),
    ],
)
def test_run_cycle_uploads_ingest_log_and_manifest(ingest, monkeypatch, storage_backend, expected_scheme, provider_uri_key):
    uploaded_json = {}

    monkeypatch.setattr(ingest, "object_exists", lambda backend, bucket_name, object_path: False)

    def fake_download_file(url, local_path, timeout=120):
        local_path.write_bytes(b"grib-data")

    monkeypatch.setattr(ingest, "download_file", fake_download_file)
    monkeypatch.setattr(
        ingest,
        "upload_file",
        lambda backend, bucket_name, local_file, object_path: f"{expected_scheme}://{bucket_name}/{object_path}",
    )

    def fake_upload_json(backend, bucket_name, data, object_path):
        uploaded_json[object_path] = json.loads(json.dumps(data))
        return f"{expected_scheme}://{bucket_name}/{object_path}"

    monkeypatch.setattr(ingest, "upload_json", fake_upload_json)

    cycle_log = ingest.run_cycle(
        bucket_name="bucket",
        gcs_prefix="vote",
        run_dt=datetime(2026, 5, 5),
        run_hour="00",
        forecast_step=3,
        forecast_max=3,
        storage_backend=storage_backend,
    )

    log_path = "vote/bronze/GFS/2026/05/05/00/ingest_log.json"
    manifest_path = "vote/bronze/GFS/2026/05/05/00/manifest.json"

    assert len(cycle_log) == 2
    assert cycle_log[0]["storage_backend"] == storage_backend
    assert cycle_log[0]["storage_uri"].startswith(f"{expected_scheme}://bucket/")
    assert cycle_log[0][provider_uri_key] == cycle_log[0]["storage_uri"]
    assert uploaded_json[log_path] == cycle_log
    assert uploaded_json[manifest_path]["storage_backend"] == storage_backend
    assert uploaded_json[manifest_path]["record_count"] == 2
    assert uploaded_json[manifest_path]["downloaded"] == 2
    assert uploaded_json[manifest_path]["skipped"] == 0
    assert uploaded_json[manifest_path]["failed"] == 0
    assert uploaded_json[manifest_path]["files"] == cycle_log


def test_run_cycle_marks_existing_s3_objects_as_skipped(ingest, monkeypatch):
    uploaded_json = {}

    monkeypatch.setattr(ingest, "object_exists", lambda backend, bucket_name, object_path: True)
    monkeypatch.setattr(ingest, "download_file", lambda *args, **kwargs: pytest.fail("download should not run"))
    monkeypatch.setattr(ingest, "upload_file", lambda *args, **kwargs: pytest.fail("upload should not run"))

    def fake_upload_json(backend, bucket_name, data, object_path):
        uploaded_json[object_path] = json.loads(json.dumps(data))
        return f"s3://{bucket_name}/{object_path}"

    monkeypatch.setattr(ingest, "upload_json", fake_upload_json)

    cycle_log = ingest.run_cycle(
        bucket_name="bucket",
        gcs_prefix="vote",
        run_dt=datetime(2026, 5, 5),
        run_hour="00",
        forecast_step=3,
        forecast_max=0,
        storage_backend="s3",
    )

    assert cycle_log == [
        {
            "source": "GFS",
            "model": "WW3",
            "run_time": "2026050500",
            "forecast_hour": "000",
            "format": "grib2",
            "object_path": "vote/bronze/GFS/2026/05/05/00/000/2026050500_f000.grib2",
            "status": "SKIPPED_ALREADY_EXISTS",
            "storage_backend": "s3",
            "storage_uri": "s3://bucket/vote/bronze/GFS/2026/05/05/00/000/2026050500_f000.grib2",
            "s3_uri": "s3://bucket/vote/bronze/GFS/2026/05/05/00/000/2026050500_f000.grib2",
        }
    ]
    manifest_path = "vote/bronze/GFS/2026/05/05/00/manifest.json"
    assert uploaded_json[manifest_path]["downloaded"] == 0
    assert uploaded_json[manifest_path]["skipped"] == 1
    assert uploaded_json[manifest_path]["failed"] == 0


def test_storage_backend_validation_rejects_unknown_backend(ingest):
    with pytest.raises(ValueError, match="Unsupported STORAGE_BACKEND"):
        ingest.normalize_storage_backend("azure")


def test_bucket_name_for_backend_uses_backend_specific_env_var(ingest, monkeypatch):
    monkeypatch.setenv("GCS_BUCKET_NAME", "gcs-bucket")
    monkeypatch.setenv("S3_BUCKET_NAME", "s3-bucket")

    assert ingest.bucket_name_for_backend("gcs") == "gcs-bucket"
    assert ingest.bucket_name_for_backend("s3") == "s3-bucket"


def test_s3_bucket_name_can_include_endpoint_url(ingest, monkeypatch):
    monkeypatch.setenv("S3_BUCKET_NAME", "http://10.11.1.171:30080/raw-ingest")

    assert ingest.bucket_name_for_backend("s3") == "raw-ingest"
    assert ingest.s3_endpoint_url() == "http://10.11.1.171:30080"


def test_s3_endpoint_url_env_takes_precedence(ingest, monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://10.11.1.171:30080/")
    monkeypatch.setenv("S3_BUCKET_NAME", "raw-ingest")

    assert ingest.bucket_name_for_backend("s3") == "raw-ingest"
    assert ingest.s3_endpoint_url() == "http://10.11.1.171:30080"


def test_s3_bucket_url_must_point_to_one_bucket(ingest):
    with pytest.raises(ValueError, match="http://host:port/bucket-name"):
        ingest.split_s3_bucket_config("http://10.11.1.171:30080/path/to/raw-ingest")
