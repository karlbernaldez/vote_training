import pandas as pd
from pipelines.atmospheric.gfs.transform.gridded_wind import transform_gridded_wind


def transform_station_wind(input_path: str, output_csv: str, stations: list[dict]) -> pd.DataFrame:
    ds = transform_gridded_wind(input_path, '/tmp/gridded.nc')
    rows = []

    for station in stations:
        lat = station['lat']
        lon = station['lon']
        point = ds.sel(latitude=lat, longitude=lon, method='nearest')
        rows.append({
            'station': station['name'],
            'lat': lat,
            'lon': lon,
            'wind_speed': float(point['wind_speed'].values),
            'wind_direction': float(point['wind_direction'].values),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df
