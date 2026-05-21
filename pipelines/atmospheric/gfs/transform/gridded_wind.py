import xarray as xr
import numpy as np


def transform_gridded_wind(input_path: str, output_path: str) -> xr.Dataset:
    ds = xr.open_dataset(input_path, engine='cfgrib')

    u = ds['u10'] if 'u10' in ds else ds['u']
    v = ds['v10'] if 'v10' in ds else ds['v']

    wind_speed = np.sqrt(u ** 2 + v ** 2)
    wind_direction = (270 - np.degrees(np.arctan2(v, u))) % 360

    out = xr.Dataset({
        'u10': u,
        'v10': v,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
    })

    out.to_netcdf(output_path)
    return out
