"""GFS pipeline orchestrator migrated from legacy Scripts/Homework/4/pipeline.py."""

from pipelines.atmospheric.gfs.ingest.ingest_gfs import main as ingest_main


def run_pipeline() -> None:
    """Run migrated GFS pipeline stages."""
    ingest_main()
    print("GFS pipeline orchestrator migration active")


if __name__ == "__main__":
    run_pipeline()
