from pathlib import Path
import xarray as xr


def transform_gridded_wind(
    input_path: str,
    netcdf_out: str,
) -> xr.Dataset:

    input_dir = Path(input_path)

    nc_files = sorted(input_dir.glob("*.nc"))

    if not nc_files:
        raise FileNotFoundError(
            f"No NetCDF files found in {input_dir}"
        )

    datasets = [xr.open_dataset(f) for f in nc_files]

    ds = xr.concat(datasets, dim="forecast_cycle")

    Path(netcdf_out).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ds.to_netcdf(netcdf_out)

    return ds