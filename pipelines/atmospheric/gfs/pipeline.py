"""GFS pipeline orchestrator migrated from legacy Scripts/Homework/4/pipeline.py."""

from pipelines.atmospheric.gfs.ingest.ingest_gfs import main as ingest_main
from pipelines.atmospheric.gfs.transform.merge_forecast import merge_forecast_transforms


def run_pipeline() -> None:
    """Run migrated GFS pipeline stages."""
    ingest_main()
    merge_forecast_transforms()
    print("GFS pipeline orchestrator migration active")


if __name__ == "__main__":
    run_pipeline()
