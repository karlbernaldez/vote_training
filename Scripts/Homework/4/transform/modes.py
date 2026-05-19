from __future__ import annotations

ALLOWED_TRANSFORM_MODES = {"merge", "station_wind", "gridded_wind"}
DEFAULT_TRANSFORM_MODES = ["merge"]
BOTH_TRANSFORM_MODES = ["station_wind", "gridded_wind"]


def parse_transform_modes(value: str | None) -> list[str]:
    """Parse TRANSFORM_MODE into ordered, de-duplicated concrete transform modes.

    Supported forms:
    - TRANSFORM_MODE=station_wind
    - TRANSFORM_MODE=gridded_wind
    - TRANSFORM_MODE=station_wind,gridded_wind
    - TRANSFORM_MODE=both

    The legacy merge mode remains available for the old merged-GRIB output.
    """
    raw_value = (value or "").strip().lower()
    if not raw_value:
        return list(DEFAULT_TRANSFORM_MODES)

    raw_modes = BOTH_TRANSFORM_MODES if raw_value == "both" else [
        item.strip() for item in raw_value.split(",") if item.strip()
    ]

    if not raw_modes:
        raise ValueError("At least one transform mode is required.")

    modes = []
    for mode in raw_modes:
        if mode not in ALLOWED_TRANSFORM_MODES:
            raise ValueError(
                f"Unknown transform mode: {mode}. "
                f"Allowed modes are: {sorted(ALLOWED_TRANSFORM_MODES | {'both'})}"
            )
        if mode not in modes:
            modes.append(mode)

    return modes


def format_transform_modes(modes: list[str]) -> str:
    """Return a stable, human-readable transform mode label for logs/manifests."""
    return ",".join(modes)
