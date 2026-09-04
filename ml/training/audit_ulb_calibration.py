#!/usr/bin/env python3
"""Calibration robustness audit. Does not score test or rewrite Phase 2 artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.ulb.adapter import ULBFraudDatasetAdapter  # noqa: E402
from ml.ulb.robustness_audit import run_calibration_robustness_audit  # noqa: E402


def main() -> int:
    adapter = ULBFraudDatasetAdapter()
    payload = run_calibration_robustness_audit(adapter)
    rec = payload["recommendation"]
    print(
        json.dumps(
            {
                "recommended_calibration_method": rec.get("recommended_calibration_method"),
                "inconclusive": rec.get("inconclusive"),
                "test_scored": payload["methodology"]["test_scored"],
                "protected_artifacts_unchanged": payload.get("protected_artifacts_unchanged"),
                "nested_isotonic_minus_sigmoid_brier": payload["nested_holdout_validation"][
                    "paired_differences"
                ]["brier_isotonic_minus_sigmoid"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
