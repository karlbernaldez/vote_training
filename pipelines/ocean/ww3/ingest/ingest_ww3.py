import hashlib
import tempfile
from datetime import datetime
from pathlib import Path

from pipelines.ocean.ww3.ingest.downloader import download_file
from pipelines.ocean.ww3.ingest.urls import build_gfs_wind_url
from pipelines.shared.storage.storage import object_exists, upload_file, upload_json
from pipelines.shared.utils.dates import daterange


def sha256sum(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run_cycle(bucket_name: str, prefix: str, start_dt: datetime, run_hour: str = '00', forecast_step: int = 3, forecast_max: int = 72, storage_backend: str = 'gcs'):
    ingest_log = []
    yyyy = start_dt.strftime('%Y')
    mm = start_dt.strftime('%m')
    dd = start_dt.strftime('%d')
    ymdh = f'{yyyy}{mm}{dd}{run_hour}'

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for n in range(0, forecast_max + 1, forecast_step):
            fhr = f'{n:03d}'
            filename = f'{ymdh}_f{fhr}.grib2'
            object_path = f'{prefix}/bronze/GFS/{yyyy}/{mm}/{dd}/{run_hour}/{fhr}/{filename}'

            if object_exists(storage_backend, bucket_name, object_path):
                continue

            url = build_gfs_wind_url(start_dt, run_hour, fhr)
            local_path = tmpdir_path / filename
            download_file(url, local_path)
            checksum = sha256sum(local_path)
            upload_file(storage_backend, bucket_name, local_path, object_path)

            ingest_log.append({
                'run_time': ymdh,
                'forecast_hour': fhr,
                'checksum_sha256': checksum,
                'object_path': object_path,
                'download_url': url,
            })

    upload_json(storage_backend, bucket_name, ingest_log, f'{prefix}/bronze/GFS/{yyyy}/{mm}/{dd}/{run_hour}/ingest_log.json')
    return ingest_log


def main(start_date: str, end_date: str, bucket_name: str, prefix: str, storage_backend: str = 'gcs'):
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    for run_dt in daterange(start_dt, end_dt):
        run_cycle(bucket_name, prefix, run_dt, storage_backend=storage_backend)
