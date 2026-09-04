#!/usr/bin/env python3
"""Run ULB cleaning only (no model training)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.ulb.adapter import ULBFraudDatasetAdapter  # noqa: E402


def main() -> int:
    adapter = ULBFraudDatasetAdapter()
    adapter.preprocess()
    print(json.dumps(adapter._clean_stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
