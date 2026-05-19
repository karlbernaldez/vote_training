from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from transform.paths import merge_output_paths, source_object_path, station_wind_output_paths
from transform.wind import build_station_wind_dataframe, build_station_wind_ml_dataframe


def sample_wind_dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "u10": (
                ("step", "latitude", "longitude"),
                np.array([
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]),
            ),
            "v10": (
                ("step", "latitude", "longitude"),
                np.array([
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[1.0, 1.0], [1.0, 1.0]],
                ]),
            ),
        },
        coords={
            "step": pd.to_timedelta([0, 3], unit="h"),
            "latitude": [13.5, 14.5],
            "longitude": [120.5, 121.5],
        },
    )


def test_transform_paths() -> None:
    run_dt = datetime(2026, 5, 1)

    assert source_object_path("WaveWatchIII", run_dt, "00", 3) == (
        "WaveWatchIII/bronze/GFS/2026/05/01/00/003/2026050100_f003.grib2"
    )

    merged_path, merged_manifest = merge_output_paths("WaveWatchIII", run_dt, "00", 0, 6)
    assert merged_path == "WaveWatchIII/silver/GFS/2026/05/01/00/f000-f006/2026050100_f000-f006_merged.grib2"
    assert merged_manifest == "WaveWatchIII/silver/GFS/2026/05/01/00/f000-f006/transform_manifest.json"

    human_path, ml_path, station_manifest = station_wind_output_paths("WaveWatchIII", run_dt, "00", 0, 6)
    assert human_path == (
        "WaveWatchIII/silver/GFS/2026/05/01/00/station_wind/f000-f006/"
        "2026050100_station_wind_f000-f006.csv"
    )
    assert ml_path == (
        "WaveWatchIII/silver/GFS/2026/05/01/00/station_wind/f000-f006/"
        "2026050100_station_wind_ml_f000-f006.csv"
    )
    assert station_manifest == "WaveWatchIII/silver/GFS/2026/05/01/00/station_wind/f000-f006/transform_manifest.json"


def test_station_wind_human_and_ml_outputs(tmp_path: Path) -> None:
    stations_csv = tmp_path / "stations.csv"
    stations_csv.write_text("name,lat,lon\nTest Station,14.0,121.0\n", encoding="utf-8")

    ds = sample_wind_dataset()
    human_df = build_station_wind_dataframe(ds, stations_csv)
    ml_df = build_station_wind_ml_dataframe(ds, stations_csv, run_time="2026050100")

    assert human_df.loc[0, "f000"] == "W @ 4 kph"
    assert human_df.loc[0, "f003"] == "S @ 4 kph"

    assert len(ml_df) == 2
    assert set([
        "run_time",
        "forecast_hour",
        "forecast_hour_label",
        "u10_ms",
        "v10_ms",
        "wind_speed_kph",
        "wind_dir_deg",
        "wind_dir_sin",
        "wind_dir_cos",
        "wind_direction",
    ]).issubset(ml_df.columns)
    assert ml_df.loc[0, "forecast_hour"] == 0
    assert ml_df.loc[1, "forecast_hour"] == 3
    assert ml_df.loc[0, "wind_speed_kph"] == 4
    assert ml_df.loc[1, "wind_speed_kph"] == 4
