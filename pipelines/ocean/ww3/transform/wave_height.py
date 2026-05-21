import xarray as xr


def transform_wave_height(input_path: str, output_path: str):
    ds = xr.open_dataset(input_path, engine='cfgrib')
    hs = ds['swh'] if 'swh' in ds else ds[list(ds.data_vars)[0]]
    out = xr.Dataset({'significant_wave_height': hs})
    out.to_netcdf(output_path)
    return out
