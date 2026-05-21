import requests
from pathlib import Path

def download_file(url: str, local_path: Path, timeout: int = 120):
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
