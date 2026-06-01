from pipelines.atmospheric.gfs.transform.gridded_wind import (
    transform_gridded_wind,
)
from pipelines.atmospheric.gfs.transform.station_wind import (
    transform_station_wind,
)


def merge_forecast_transforms(
    input_path: str,
    netcdf_out: str,
    station_csv: str,
    stations: list[dict],
):
    ds = transform_gridded_wind(
        input_path,
        netcdf_out,
    )

    return transform_station_wind(
        ds,
        station_csv,
        stations,
    )