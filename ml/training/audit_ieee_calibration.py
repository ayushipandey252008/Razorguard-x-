#!/usr/bin/env python3
"""IEEE-CIS calibration robustness audit. Does not retrain or activate models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT))

from app.ml.ieee.calibration_robustness_audit import run_calibration_robustness_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="IEEE-CIS calibration robustness audit (pretest only)")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--n-nested", type=int, default=40)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    payload = run_calibration_robustness_audit(
        data_dir=args.data_dir,
        eval_dir=args.eval_dir,
        model_dir=args.model_dir,
        n_boot=args.n_boot,
        n_nested=args.n_nested,
        n_folds=args.n_folds,
        seed=args.seed,
        write=True,
    )
    rec = payload.get("recommendation") or {}
    print(
        json.dumps(
            {
                "decision": rec.get("decision"),
                "recommended_calibration_method": rec.get("recommended_calibration_method"),
                "test_used_for_decision": rec.get("test_used_for_decision"),
                "xgboost_refit": (payload.get("methodology") or {}).get("xgboost_refit"),
                "test_scored": (payload.get("methodology") or {}).get("test_scored"),
                "protected_artifacts_unchanged": (payload.get("integrity") or {}).get("protected_artifacts_unchanged"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
