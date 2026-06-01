from pathlib import Path
import csv

from pipelines.atmospheric.gfs.ingest.ingest_gfs import main as ingest_main
from pipelines.atmospheric.gfs.transform.merge_forecast import (
    merge_forecast_transforms,
)

BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = BASE_DIR / "pipelines" / "data"

STATIONS_CSV = DATA_DIR / "pagasa_stations.csv"
RAW_DATA_DIR = DATA_DIR / "ml" / "raw"
PROCESSED_DIR = DATA_DIR / "ml" / "processed"
NETCDF_OUTPUT = PROCESSED_DIR / "gfs.nc"    


def load_stations(path: Path) -> list[dict]:
    """
    Load PAGASA station metadata.

    Returns:
        List[Dict]: station records from CSV.
    """
    with path.open(
        mode="r",
        encoding="utf-8",
        newline=""
    ) as file:
        return list(csv.DictReader(file))


def validate_paths() -> None:
    """
    Ensure required directories/files exist.
    """
    if not STATIONS_CSV.exists():
        raise FileNotFoundError(
            f"Station file not found: {STATIONS_CSV}"
        )

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw GFS directory not found: {RAW_DATA_DIR}"
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def run_pipeline() -> None:
    """
    Execute end-to-end GFS workflow.
    """

    print("Starting GFS pipeline...")

    validate_paths()

    stations = load_stations(STATIONS_CSV)

    print(
        f"Loaded {len(stations)} PAGASA stations."
    )

    ingest_main()

    merge_forecast_transforms(
        input_path=str(RAW_DATA_DIR),
        netcdf_out=str(NETCDF_OUTPUT),
        station_csv=str(STATIONS_CSV),
        stations=stations,
    )

    print(
        f"GFS pipeline completed successfully.\n"
        f"Output: {NETCDF_OUTPUT}"
    )


if __name__ == "__main__":
    run_pipeline()