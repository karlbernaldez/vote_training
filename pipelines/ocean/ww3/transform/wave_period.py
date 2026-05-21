import xarray as xr


def transform_wave_period(input_path: str, output_path: str):
    ds = xr.open_dataset(input_path, engine='cfgrib')
    tp = ds['mwp'] if 'mwp' in ds else ds[list(ds.data_vars)[0]]
    out = xr.Dataset({'mean_wave_period': tp})
    out.to_netcdf(output_path)
    return out
