"""Download the E0 datasets into data/<name>/.

Usage: python scripts/fetch_data.py [name ...]   (no names = all)
Roboflow needs ROBOFLOW_API_KEY (env or .env); CatFLW is public.
"""

import getpass
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from dotenv import load_dotenv, set_key

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ENV = ROOT / ".env"
ROBOFLOW = {
    "cat-emotions-3": ("cat-emotion-classification", "cat-emotions-cgrxv"),
    "cat-emotions-7": ("cats-xofvm", "cat-emotions"),
}
KAGGLE = {"catflw": "georgemartvel/catflw"}


def roboflow_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("ROBOFLOW_API_KEY not set (app.roboflow.com -> Settings -> API).")
        key = getpass.getpass("Paste it (saved to .env): ").strip()
        set_key(ENV, "ROBOFLOW_API_KEY", key)
        os.environ["ROBOFLOW_API_KEY"] = key
    return key


def get_json(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url) as r:
        return r.status, json.load(r)


def fetch_zip(url: str, dest: Path) -> None:
    part = dest.with_suffix(".zip.part")
    with urllib.request.urlopen(url) as r, part.open("wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(part) as z:
        z.extractall(dest)
    part.unlink()


def roboflow_zip_url(workspace: str, project: str) -> str:
    key = roboflow_key()
    api = f"https://api.roboflow.com/{workspace}/{project}"
    _, info = get_json(f"{api}?api_key={key}")
    version = info["versions"][0]["id"].rsplit("/", 1)[1]  # newest first
    while True:
        status, export = get_json(f"{api}/{version}/folder?api_key={key}&nocache=true")
        if status == 200 and "export" in export:
            return export["export"]["link"]
        print(f"  export generating: {export.get('progress', 0):.0%}")
        time.sleep(5)


def main(*names: str) -> None:
    load_dotenv(ENV)
    for name in names or [*ROBOFLOW, *KAGGLE]:
        dest = DATA / name
        if dest.exists() and any(dest.iterdir()):
            print(f"{name}: ok")
            continue
        print(f"{name}: downloading")
        if name in ROBOFLOW:
            url = roboflow_zip_url(*ROBOFLOW[name])
        elif name in KAGGLE:
            url = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE[name]}"
        else:
            sys.exit(
                f"unknown dataset {name!r}; known: {', '.join([*ROBOFLOW, *KAGGLE])}"
            )
        dest.mkdir(parents=True, exist_ok=True)
        fetch_zip(url, dest)
        print(f"{name}: done")


if __name__ == "__main__":
    main(*sys.argv[1:])
