"""GFS transform orchestration for migrated platform architecture."""

from pipelines.atmospheric.gfs.transform.gridded_wind import transform_gridded_wind
from pipelines.atmospheric.gfs.transform.station_wind import transform_station_wind


def merge_forecast_transforms() -> None:
    """Run all GFS transform stages."""
    transform_gridded_wind()
    transform_station_wind()
    print("GFS transform orchestration active")
