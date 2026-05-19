import numpy as np
import xarray as xr

from transform.wind import build_gridded_wind_dataset


def test_build_gridded_wind_dataset_preserves_grid_and_adds_features():
    ds = xr.Dataset(
        data_vars={
            "u10": (("step", "latitude", "longitude"), np.array([[[1.0, 0.0], [0.0, -1.0]]], dtype=np.float32)),
            "v10": (("step", "latitude", "longitude"), np.array([[[0.0, 1.0], [-1.0, 0.0]]], dtype=np.float32)),
        },
        coords={
            "step": np.array([np.timedelta64(0, "h")]),
            "latitude": np.array([14.0, 15.0]),
            "longitude": np.array([120.0, 121.0]),
        },
    )

    grid = build_gridded_wind_dataset(ds, run_time="2026051900")

    assert set(grid.data_vars) == {
        "u10_ms",
        "v10_ms",
        "wind_speed_kph",
        "wind_dir_deg",
        "wind_dir_sin",
        "wind_dir_cos",
    }
    assert grid.sizes == ds.sizes
    assert grid.attrs["dataset_type"] == "gridded_wind"
    assert grid.attrs["run_time"] == "2026051900"

    np.testing.assert_allclose(grid["u10_ms"].values, ds["u10"].values)
    np.testing.assert_allclose(grid["v10_ms"].values, ds["v10"].values)
    np.testing.assert_allclose(grid["wind_speed_kph"].values, np.hypot(ds["u10"], ds["v10"]).values * 3.6)


def test_build_gridded_wind_dataset_rejects_missing_variables():
    ds = xr.Dataset({"u10": (("step",), np.array([1.0], dtype=np.float32))})

    try:
        build_gridded_wind_dataset(ds)
    except ValueError as exc:
        assert "u10 and v10" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
