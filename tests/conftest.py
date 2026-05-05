import sys
import types


# Keep ingestion tests importable in lightweight CI/local environments before optional
# runtime dependencies are installed or configured.
if "dotenv" not in sys.modules:
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_module

if "google" not in sys.modules:
    google_module = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    storage_module = types.ModuleType("google.cloud.storage")

    class _StorageClient:
        pass

    storage_module.Client = _StorageClient
    cloud_module.storage = storage_module
    google_module.cloud = cloud_module

    sys.modules["google"] = google_module
    sys.modules["google.cloud"] = cloud_module
    sys.modules["google.cloud.storage"] = storage_module
