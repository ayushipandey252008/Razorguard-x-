#!/usr/bin/env python3
"""Fit ULB probability calibrators on validation; evaluate once on chronological test.

Does not overwrite the synthetic product model or change ULB preprocessing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.ulb.adapter import ULBFraudDatasetAdapter  # noqa: E402
from ml.ulb.calibration_pipeline import run_ulb_calibration  # noqa: E402


def main() -> int:
    adapter = ULBFraudDatasetAdapter()
    payload = run_ulb_calibration(adapter)
    summary = {
        "selected_method": payload.get("selected_method"),
        "calibrated_model_version": payload.get("calibrated_model_version"),
        "prototype_operating_thresholds": payload.get("prototype_operating_thresholds"),
        "test_brier_raw": payload["test_evaluation"]["raw_probability"]["brier"],
        "test_brier_calibrated": payload["test_evaluation"]["calibrated_probability"]["brier"],
        "test_pr_auc": payload["test_evaluation"]["metrics_at_0_5"]["pr_auc"],
        "test_used_for_fit": payload["methodology"]["test_used_for_calibrator_fit"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
