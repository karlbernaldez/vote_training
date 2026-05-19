from ml.collect_bucket_dataset import is_gridded_wind_netcdf, local_name_for_object


def test_is_gridded_wind_netcdf_matches_expected_outputs():
    assert is_gridded_wind_netcdf(
        "vote/silver/GFS/2026/05/19/00/gridded_wind/f000-f072/2026051900_gridded_wind_f000-f072.nc"
    )


def test_is_gridded_wind_netcdf_rejects_non_grid_outputs():
    assert not is_gridded_wind_netcdf(
        "vote/silver/GFS/2026/05/19/00/station_wind/f000-f072/2026051900_station_wind_ml_f000-f072.csv"
    )
    assert not is_gridded_wind_netcdf(
        "vote/silver/GFS/2026/05/19/00/gridded_wind/f000-f072/transform_manifest.json"
    )


def test_local_name_for_object_keeps_date_hour_and_range_context():
    object_name = "vote/silver/GFS/2026/05/19/00/gridded_wind/f000-f072/2026051900_gridded_wind_f000-f072.nc"

    assert local_name_for_object(object_name) == "2026051900_f000-f072_2026051900_gridded_wind_f000-f072.nc"
