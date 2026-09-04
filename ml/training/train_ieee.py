#!/usr/bin/env python3
"""Train IEEE-CIS offline candidates. Does not overwrite the live or ULB models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT))

from app.ml.ieee.pipeline import run_ieee_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="IEEE-CIS offline training (candidate artifacts only)")
    parser.add_argument("--data-dir", default=None, help="Directory with train_transaction.csv and train_identity.csv")
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--fixture", action="store_true", help="Use the synthetic fixture (NOT IEEE-CIS results)")
    parser.add_argument("--n-estimators", type=int, default=None)
    args = parser.parse_args()
    payload = run_ieee_pipeline(
        data_dir=args.data_dir,
        eval_dir=args.eval_dir,
        model_dir=args.model_dir,
        max_rows=args.max_rows,
        allow_fixture=args.fixture,
        n_estimators=args.n_estimators,
    )
    keys = (
        "source",
        "dataset_available",
        "active_live_model",
        "ieee_status",
        "runtime_seconds",
        "setup_message",
        "disclaimer",
    )
    print(json.dumps({k: payload.get(k) for k in keys if k in payload}, indent=2))
    if payload.get("frozen_test_metrics") and payload.get("source") == "IEEE_CIS_CSV":
        m = payload["frozen_test_metrics"]
        print(json.dumps({"pr_auc": m.get("pr_auc"), "roc_auc": m.get("roc_auc")}, indent=2))
    elif payload.get("source") == "SYNTHETIC_FIXTURE_NOT_IEEE_CIS":
        print("Fixture metrics are not IEEE-CIS public-dataset results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
