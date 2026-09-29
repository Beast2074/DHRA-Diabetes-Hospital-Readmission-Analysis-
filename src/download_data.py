import io
import zipfile

import requests

from config import DATA_RAW, DATA_URL


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    target = DATA_RAW / "diabetic_data.csv"

    # skip if we already have it
    if target.exists() and (DATA_RAW / "IDS_mapping.csv").exists():
        print("raw data already downloaded")
        return

    print("downloading dataset from UCI...")
    resp = requests.get(DATA_URL, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall(DATA_RAW)

    print("saved files:", [p.name for p in DATA_RAW.iterdir()])


if __name__ == "__main__":
    main()
