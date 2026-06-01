import os

TRUE_VALUES = {'1', 'true', 'yes', 'y', 'on'}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES



def env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None



def load_settings() -> dict:
    return {
        'storage_backend': env_value('STORAGE_BACKEND') or 'gcs',
        'forecast_max_hours': env_value('FORECAST_MAX_HOURS'),
        'run_hour': env_value('RUN_HOUR'),
    }
