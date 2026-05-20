from pathlib import Path

import requests


def download_file(url: str, local_path: Path, timeout: int = 120) -> None:
    """Download one NOAA GFS file to local storage."""
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
