import pandas as pd
import xarray as xr


def transform_station_wind(
    ds: xr.Dataset,
    output_csv: str,
    stations: list[dict],
) -> pd.DataFrame:

    rows = []

    for station in stations:
        lat = float(station["lat"])
        lon = float(station["lon"])

        point = ds.sel(
            latitude=lat,
            longitude=lon,
            method="nearest",
        )

        rows.append(
            {
                "station": station.get(
                    "stnName",
                    station.get("name", "unknown"),
                ),
                "lat": lat,
                "lon": lon,
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        output_csv,
        index=False,
    )

    return df