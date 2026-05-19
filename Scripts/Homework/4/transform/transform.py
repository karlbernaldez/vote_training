from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ingest import daterange, normalize_storage_backend, sha256sum, storage_uri, upload_json
from transform.config import TransformConfig, load_transform_config
from transform.paths import (
    gridded_wind_output_paths,
    merge_output_paths,
    source_object_path,
    station_wind_output_paths,
)
from transform.storage import download_object, upload_binary
from transform.wind import (
    build_gridded_wind_dataset,
    build_station_wind_dataframe,
    build_station_wind_ml_dataframe,
    extract_wind,
    extract_wind_features,
    load_wind_dataset,
)


# Backward-compatible alias for older imports/tests that used output_paths().
output_paths = merge_output_paths


def download_forecast_files(
    storage_backend: str,
    bucket_name: str,
    prefix: str,
    run_dt: datetime,
    run_hour: str,
    forecast_start: int,
    forecast_step: int,
    forecast_max: int,
    tmpdir_path: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Download bronze GRIB2 forecast files and return local paths plus source metadata."""
    backend = normalize_storage_backend(storage_backend)
    local_paths = []
    source_records = []

    for forecast_hour in range(forecast_start, forecast_max + 1, forecast_step):
        object_path = source_object_path(prefix, run_dt, run_hour, forecast_hour)
        local_path = tmpdir_path / Path(object_path).name
        source_uri = storage_uri(backend, bucket_name, object_path)

        print(f"Downloading {source_uri}")
        download_object(backend, bucket_name, object_path, local_path)
        local_paths.append(local_path)

        source_records.append({
            "forecast_hour": f"{forecast_hour:03d}",
            "source_uri": source_uri,
            "file_size": local_path.stat().st_size,
            "checksum_sha256": sha256sum(local_path),
        })

    return local_paths, source_records


def merge_grib_files(local_paths: list[Path], output_path: Path) -> None:
    """Concatenate downloaded GRIB2 files into one merged binary file."""
    with open(output_path, "wb") as merged_file:
        for local_path in local_paths:
            with open(local_path, "rb") as input_file:
                for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                    merged_file.write(chunk)


def build_base_manifest(
    transform_name: str,
    storage_backend: str,
    bucket_name: str,
    prefix: str,
    run_dt: datetime,
    run_hour: str,
    forecast_start: int,
    forecast_step: int,
    forecast_max: int,
    source_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build fields common to all transform manifests."""
    return {
        "manifest_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "GFS",
        "model": "WW3",
        "transform": transform_name,
        "storage_backend": normalize_storage_backend(storage_backend),
        "bucket": bucket_name,
        "storage_prefix": prefix,
        "run_date": run_dt.strftime("%Y-%m-%d"),
        "run_hour": run_hour,
        "run_time": f"{run_dt:%Y%m%d}{run_hour}",
        "forecast_start": forecast_start,
        "forecast_step": forecast_step,
        "forecast_max": forecast_max,
        "source_count": len(source_records),
        "sources": source_records,
    }


def merge_forecast_hours(
    storage_backend: str,
    bucket_name: str,
    prefix: str,
    run_dt: datetime,
    run_hour: str,
    forecast_start: int,
    forecast_step: int,
    forecast_max: int,
) -> dict[str, Any]:
    """Merge bronze forecast-hour GRIB2 files into one silver GRIB2 file."""
    backend = normalize_storage_backend(storage_backend)
    ymdh = f"{run_dt:%Y%m%d}{run_hour}"
    merged_object_path, manifest_object_path = merge_output_paths(
        prefix,
        run_dt,
        run_hour,
        forecast_start,
        forecast_max,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        merged_path = tmpdir_path / f"{ymdh}_f{forecast_start:03d}-f{forecast_max:03d}_merged.grib2"
        local_paths, source_records = download_forecast_files(
            backend,
            bucket_name,
            prefix,
            run_dt,
            run_hour,
            forecast_start,
            forecast_step,
            forecast_max,
            tmpdir_path,
        )

        merge_grib_files(local_paths, merged_path)
        merged_uri = upload_binary(backend, bucket_name, merged_path, merged_object_path)

        manifest = build_base_manifest(
            "merge_forecast_hours_grib2",
            backend,
            bucket_name,
            prefix,
            run_dt,
            run_hour,
            forecast_start,
            forecast_step,
            forecast_max,
            source_records,
        )
        manifest.update({
            "merged_uri": merged_uri,
            "merged_object_path": merged_object_path,
            "merged_file_size": merged_path.stat().st_size,
            "merged_checksum_sha256": hashlib.sha256(merged_path.read_bytes()).hexdigest(),
        })

        manifest_uri = upload_json(backend, bucket_name, manifest, manifest_object_path)
        manifest["manifest_uri"] = manifest_uri
        print(f"Uploaded merged file: {merged_uri}")
        print(f"Uploaded manifest: {manifest_uri}")
        return manifest


def transform_station_wind(
    storage_backend: str,
    bucket_name: str,
    prefix: str,
    run_dt: datetime,
    run_hour: str,
    forecast_start: int,
    forecast_step: int,
    forecast_max: int,
    stations_csv: Path,
) -> dict[str, Any]:
    """Create human-readable and ML-ready station wind CSV outputs from bronze GRIB2 files."""
    backend = normalize_storage_backend(storage_backend)
    ymdh = f"{run_dt:%Y%m%d}{run_hour}"
    human_csv_object_path, ml_csv_object_path, manifest_object_path = station_wind_output_paths(
        prefix,
        run_dt,
        run_hour,
        forecast_start,
        forecast_max,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        local_paths, source_records = download_forecast_files(
            backend,
            bucket_name,
            prefix,
            run_dt,
            run_hour,
            forecast_start,
            forecast_step,
            forecast_max,
            tmpdir_path,
        )

        ds = load_wind_dataset(local_paths)
        try:
            station_wind = build_station_wind_dataframe(ds, stations_csv)
            station_wind_ml = build_station_wind_ml_dataframe(ds, stations_csv, run_time=ymdh)

            human_csv_path = tmpdir_path / f"{ymdh}_station_wind_f{forecast_start:03d}-f{forecast_max:03d}.csv"
            ml_csv_path = tmpdir_path / f"{ymdh}_station_wind_ml_f{forecast_start:03d}-f{forecast_max:03d}.csv"
            station_wind.to_csv(human_csv_path, index=False)
            station_wind_ml.to_csv(ml_csv_path, index=False)
        finally:
            ds.close()

        human_csv_uri = upload_binary(
            backend,
            bucket_name,
            human_csv_path,
            human_csv_object_path,
            content_type="text/csv",
        )
        ml_csv_uri = upload_binary(
            backend,
            bucket_name,
            ml_csv_path,
            ml_csv_object_path,
            content_type="text/csv",
        )

        manifest = build_base_manifest(
            "station_wind_csv",
            backend,
            bucket_name,
            prefix,
            run_dt,
            run_hour,
            forecast_start,
            forecast_step,
            forecast_max,
            source_records,
        )
        manifest.update({
            "station_count": len(station_wind),
            "ml_record_count": len(station_wind_ml),
            "stations_csv": str(stations_csv),
            "human_csv_uri": human_csv_uri,
            "human_csv_object_path": human_csv_object_path,
            "human_csv_file_size": human_csv_path.stat().st_size,
            "human_csv_checksum_sha256": sha256sum(human_csv_path),
            "ml_csv_uri": ml_csv_uri,
            "ml_csv_object_path": ml_csv_object_path,
            "ml_csv_file_size": ml_csv_path.stat().st_size,
            "ml_csv_checksum_sha256": sha256sum(ml_csv_path),
            "ml_columns": list(station_wind_ml.columns),
        })

        manifest_uri = upload_json(backend, bucket_name, manifest, manifest_object_path)
        manifest["manifest_uri"] = manifest_uri
        print(f"Uploaded station wind CSV: {human_csv_uri}")
        print(f"Uploaded station wind ML CSV: {ml_csv_uri}")
        print(f"Uploaded manifest: {manifest_uri}")
        return manifest


def transform_gridded_wind(
    storage_backend: str,
    bucket_name: str,
    prefix: str,
    run_dt: datetime,
    run_hour: str,
    forecast_start: int,
    forecast_step: int,
    forecast_max: int,
) -> dict[str, Any]:
    """Create a grid-native NetCDF wind dataset for 2D/3D CNN preparation."""
    backend = normalize_storage_backend(storage_backend)
    ymdh = f"{run_dt:%Y%m%d}{run_hour}"
    netcdf_object_path, manifest_object_path = gridded_wind_output_paths(
        prefix,
        run_dt,
        run_hour,
        forecast_start,
        forecast_max,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        local_paths, source_records = download_forecast_files(
            backend,
            bucket_name,
            prefix,
            run_dt,
            run_hour,
            forecast_start,
            forecast_step,
            forecast_max,
            tmpdir_path,
        )

        ds = load_wind_dataset(local_paths)
        try:
            gridded_wind = build_gridded_wind_dataset(ds, run_time=ymdh)
            netcdf_path = tmpdir_path / f"{ymdh}_gridded_wind_f{forecast_start:03d}-f{forecast_max:03d}.nc"
            gridded_wind.to_netcdf(netcdf_path)
        finally:
            ds.close()
            if "gridded_wind" in locals():
                gridded_wind.close()

        netcdf_uri = upload_binary(
            backend,
            bucket_name,
            netcdf_path,
            netcdf_object_path,
            content_type="application/x-netcdf",
        )

        manifest = build_base_manifest(
            "gridded_wind_netcdf",
            backend,
            bucket_name,
            prefix,
            run_dt,
            run_hour,
            forecast_start,
            forecast_step,
            forecast_max,
            source_records,
        )
        manifest.update({
            "netcdf_uri": netcdf_uri,
            "netcdf_object_path": netcdf_object_path,
            "netcdf_file_size": netcdf_path.stat().st_size,
            "netcdf_checksum_sha256": sha256sum(netcdf_path),
            "grid_dims": dict(gridded_wind.sizes),
            "grid_variables": list(gridded_wind.data_vars),
            "cnn_layout_options": {
                "conv2d": "N x C x H x W, one forecast hour per sample",
                "conv3d": "N x C x T x H x W, one run sequence per sample",
            },
        })

        manifest_uri = upload_json(backend, bucket_name, manifest, manifest_object_path)
        manifest["manifest_uri"] = manifest_uri
        print(f"Uploaded gridded wind NetCDF: {netcdf_uri}")
        print(f"Uploaded manifest: {manifest_uri}")
        return manifest


def run_one_transform_mode(config: TransformConfig, run_dt: datetime, transform_mode: str) -> dict[str, Any]:
    if transform_mode == "station_wind":
        if config.stations_csv is None:
            raise ValueError("stations_csv is required for station_wind transform")
        return transform_station_wind(
            config.storage_backend,
            config.bucket_name,
            config.prefix,
            run_dt,
            config.run_hour,
            config.forecast_start,
            config.forecast_step,
            config.forecast_max,
            config.stations_csv,
        )

    if transform_mode == "gridded_wind":
        return transform_gridded_wind(
            config.storage_backend,
            config.bucket_name,
            config.prefix,
            run_dt,
            config.run_hour,
            config.forecast_start,
            config.forecast_step,
            config.forecast_max,
        )

    return merge_forecast_hours(
        config.storage_backend,
        config.bucket_name,
        config.prefix,
        run_dt,
        config.run_hour,
        config.forecast_start,
        config.forecast_step,
        config.forecast_max,
    )


def run_transform(config: TransformConfig) -> list[dict[str, Any]]:
    """Run the configured transform mode(s) for every requested date."""
    manifests = []
    for run_dt in daterange(config.start_date, config.end_date):
        for transform_mode in config.transform_modes:
            manifests.append(run_one_transform_mode(config, run_dt, transform_mode))
    return manifests


def main() -> None:
    """Load transform config from environment and run the selected transform."""
    load_dotenv()
    run_transform(load_transform_config())


if __name__ == "__main__":
    main()
