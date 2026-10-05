"""Download CUAD v1 (CC BY 4.0, The Atticus Project) into data/."""

import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for name in ("CUADv1.json", "test.json"):
        (DATA / name).write_bytes(archive.read(name))
        print(f"wrote data/{name}")


if __name__ == "__main__":
    main()
