import xarray as xr


def transform_wave_direction(input_path: str, output_path: str):
    ds = xr.open_dataset(input_path, engine='cfgrib')
    wd = ds['mwd'] if 'mwd' in ds else ds[list(ds.data_vars)[0]]
    out = xr.Dataset({'mean_wave_direction': wd})
    out.to_netcdf(output_path)
    return out
