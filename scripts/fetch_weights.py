"""Download every model listed in models/manifest.json and verify its sha256.

Usage: python scripts/fetch_weights.py [path/to/manifest.json]
Files land next to the manifest. Already-present files with a matching hash are skipped.
"""

import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main(manifest_path: str | Path = ROOT / "models" / "manifest.json") -> None:
    manifest = Path(manifest_path)
    for name, entry in json.loads(manifest.read_text()).items():
        dest = manifest.parent / entry["file"]
        if dest.exists() and sha256(dest) == entry["sha256"]:
            print(f"{name}: ok")
            continue
        print(f"{name}: downloading {entry['source']}")
        part = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(entry["source"]) as r, part.open("wb") as f:
            shutil.copyfileobj(r, f)
        got = sha256(part)
        if got != entry["sha256"]:
            part.unlink()
            sys.exit(f"{name}: sha256 mismatch\n  want {entry['sha256']}\n  got  {got}")
        part.rename(dest)
        print(f"{name}: verified")


if __name__ == "__main__":
    main(*sys.argv[1:])
