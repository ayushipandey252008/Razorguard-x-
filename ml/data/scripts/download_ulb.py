#!/usr/bin/env python3
"""Download the ULB credit-card CSV into ml/data/raw/ (gitignored)."""

from __future__ import annotations

import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.ulb.constants import DOWNLOAD_URL, EXPECTED_FILENAME, LEGACY_CSV, RAW_CSV, RAW_DIR  # noqa: E402


def download() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists() and RAW_CSV.stat().st_size > 1_000_000:
        print(f"Already present: {RAW_CSV}")
        return RAW_CSV
    if LEGACY_CSV.exists() and LEGACY_CSV.stat().st_size > 1_000_000:
        import shutil

        shutil.copy2(LEGACY_CSV, RAW_CSV)
        print(f"Copied legacy {LEGACY_CSV} → {RAW_CSV}")
        return RAW_CSV
    print(f"Downloading {DOWNLOAD_URL}")
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(DOWNLOAD_URL, context=ctx, timeout=180) as resp, RAW_CSV.open("wb") as out:
            out.write(resp.read())
    except Exception as exc:
        print(f"urllib failed ({exc}); trying curl")
        subprocess.run(["curl", "-L", "--fail", "-o", str(RAW_CSV), DOWNLOAD_URL], check=True)
    if not RAW_CSV.exists() or RAW_CSV.stat().st_size < 1_000_000:
        if RAW_CSV.exists():
            RAW_CSV.unlink()
        raise RuntimeError(f"Download failed. Place {EXPECTED_FILENAME} at {RAW_CSV} manually.")
    print(f"Wrote {RAW_CSV} ({RAW_CSV.stat().st_size} bytes)")
    return RAW_CSV


if __name__ == "__main__":
    download()
