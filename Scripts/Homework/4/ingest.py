import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
from google.cloud import storage


def sha256sum(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def object_exists(bucket_name: str, object_path: str) -> bool:
    client = gcs_client()
    return client.bucket(bucket_name).blob(object_path).exists(client)


def upload_to_gcs(bucket_name: str, local_file: Path, object_path: str) -> str:
    client = gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_path)
    blob.upload_from_filename(str(local_file))
    return f"gs://{bucket_name}/{object_path}"


def upload_json_to_gcs(bucket_name: str, data: list, object_path: str) -> str:
    client = gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_path)
    blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")
    return f"gs://{bucket_name}/{object_path}"


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


def run_cycle(bucket_name: str, gcs_prefix: str, run_dt: datetime, run_hour: str, forecast_step: int, forecast_max: int) -> list:
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

            if object_exists(bucket_name, object_path):
                ingest_log.append({
                    "source": "GFS",
                    "run_time": ymdh,
                    "forecast_hour": fhr,
                    "format": "grib2",
                    "object_path": object_path,
                    "gcs_uri": f"gs://{bucket_name}/{object_path}",
                    "status": "SKIPPED_ALREADY_EXISTS",
                })
                print(f"Skip existing: gs://{bucket_name}/{object_path}")
                continue

            local_path = tmpdir_path / filename
            print(f"Downloading run={ymdh} forecast_hour={fhr}")
            try:
                download_file(download_url, local_path)
                checksum = sha256sum(local_path)
                size = local_path.stat().st_size
                gcs_uri = upload_to_gcs(bucket_name, local_path, object_path)
                ingest_log.append({
                    "source": "GFS",
                    "model": "WW3",
                    "run_time": ymdh,
                    "forecast_hour": fhr,
                    "format": "grib2",
                    "download_url": download_url,
                    "object_path": object_path,
                    "gcs_uri": gcs_uri,
                    "file_name": filename,
                    "file_size": size,
                    "checksum_sha256": checksum,
                    "status": "DOWNLOADED_AND_UPLOADED",
                })
                print(f"Uploaded: {gcs_uri}")
            except Exception as e:
                ingest_log.append({
                    "source": "GFS",
                    "model": "WW3",
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

    log_object_path = (
        f"{gcs_prefix}/bronze/GFS/"
        f"{yyyy}/{mm}/{dd}/{hh}/ingest_log.json"
    )
    upload_json_to_gcs(bucket_name, ingest_log, log_object_path)
    return ingest_log


def main():
    load_dotenv()

    bucket_name = os.getenv("GCS_BUCKET_NAME")
    gcs_prefix = os.getenv("GCS_PREFIX", "vote")
    run_hour = os.getenv("RUN_HOUR", "00")
    forecast_step = int(os.getenv("FORECAST_STEP", "3"))
    forecast_max = int(os.getenv("FORECAST_MAX", "72"))
    start_date = os.getenv("START_DATE")
    end_date = os.getenv("END_DATE")

    if not bucket_name:
        raise ValueError("GCS_BUCKET_NAME is required in .env")
    if not start_date or not end_date:
        raise ValueError("START_DATE and END_DATE are required in .env")

    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)
    if end_dt < start_dt:
        raise ValueError("END_DATE must be on or after START_DATE")

    summary = []
    for run_dt in daterange(start_dt, end_dt):
        cycle_log = run_cycle(bucket_name, gcs_prefix, run_dt, run_hour, forecast_step, forecast_max)
        summary.append({
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
