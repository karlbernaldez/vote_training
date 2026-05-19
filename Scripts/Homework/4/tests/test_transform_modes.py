import pytest

from transform.modes import format_transform_modes, parse_transform_modes


def test_parse_transform_modes_defaults_to_merge():
    assert parse_transform_modes(None) == ["merge"]
    assert parse_transform_modes("") == ["merge"]


def test_parse_transform_modes_single_mode():
    assert parse_transform_modes("station_wind") == ["station_wind"]
    assert parse_transform_modes("gridded_wind") == ["gridded_wind"]


def test_parse_transform_modes_multiple_modes():
    assert parse_transform_modes("station_wind,gridded_wind") == ["station_wind", "gridded_wind"]


def test_parse_transform_modes_both_alias():
    assert parse_transform_modes("both") == ["station_wind", "gridded_wind"]


def test_parse_transform_modes_deduplicates_preserving_order():
    assert parse_transform_modes("station_wind,gridded_wind,station_wind") == [
        "station_wind",
        "gridded_wind",
    ]


def test_parse_transform_modes_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown transform mode"):
        parse_transform_modes("station_wind,unknown")


def test_format_transform_modes():
    assert format_transform_modes(["station_wind", "gridded_wind"]) == "station_wind,gridded_wind"
