#!/usr/bin/env python3
"""Validate the ULB CSV. Exits non-zero on schema/target failures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.ulb.adapter import ULBFraudDatasetAdapter  # noqa: E402
from ml.ulb.errors import DatasetValidationError  # noqa: E402


def main() -> int:
    adapter = ULBFraudDatasetAdapter()
    try:
        adapter.load()
        report = adapter.validate()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    except DatasetValidationError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(json.dumps({k: report[k] for k in ("ok", "n_rows", "class_counts", "duplicate_rows", "missing_total")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
