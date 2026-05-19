from __future__ import annotations

from datetime import datetime


def hour_range_label(start_hour: int, end_hour: int) -> str:
    return f"f{start_hour:03d}-f{end_hour:03d}"


def source_object_path(prefix: str, run_dt: datetime, run_hour: str, forecast_hour: int) -> str:
    yyyy = run_dt.strftime("%Y")
    mm = run_dt.strftime("%m")
    dd = run_dt.strftime("%d")
    fhr = f"{forecast_hour:03d}"
    filename = f"{yyyy}{mm}{dd}{run_hour}_f{fhr}.grib2"
    return f"{prefix}/bronze/GFS/{yyyy}/{mm}/{dd}/{run_hour}/{fhr}/{filename}"


def merge_output_paths(
    prefix: str,
    run_dt: datetime,
    run_hour: str,
    start_hour: int,
    end_hour: int,
) -> tuple[str, str]:
    yyyy = run_dt.strftime("%Y")
    mm = run_dt.strftime("%m")
    dd = run_dt.strftime("%d")
    ymdh = f"{yyyy}{mm}{dd}{run_hour}"
    hour_range = hour_range_label(start_hour, end_hour)
    base_path = f"{prefix}/silver/GFS/{yyyy}/{mm}/{dd}/{run_hour}/{hour_range}"
    return (
        f"{base_path}/{ymdh}_{hour_range}_merged.grib2",
        f"{base_path}/transform_manifest.json",
    )


def station_wind_output_paths(
    prefix: str,
    run_dt: datetime,
    run_hour: str,
    start_hour: int,
    end_hour: int,
) -> tuple[str, str, str]:
    yyyy = run_dt.strftime("%Y")
    mm = run_dt.strftime("%m")
    dd = run_dt.strftime("%d")
    ymdh = f"{yyyy}{mm}{dd}{run_hour}"
    hour_range = hour_range_label(start_hour, end_hour)
    base_path = f"{prefix}/silver/GFS/{yyyy}/{mm}/{dd}/{run_hour}/station_wind/{hour_range}"
    return (
        f"{base_path}/{ymdh}_station_wind_{hour_range}.csv",
        f"{base_path}/{ymdh}_station_wind_ml_{hour_range}.csv",
        f"{base_path}/transform_manifest.json",
    )
