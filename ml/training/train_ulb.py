#!/usr/bin/env python3
"""Train the ULB offline model. Does not overwrite the synthetic product model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.ulb.adapter import ULBFraudDatasetAdapter  # noqa: E402


def main() -> int:
    adapter = ULBFraudDatasetAdapter()
    payload = adapter.run_full()
    keys = ("track", "model_version", "pr_auc", "roc_auc", "precision", "recall", "f1", "label")
    print(json.dumps({k: payload.get(k) for k in keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
