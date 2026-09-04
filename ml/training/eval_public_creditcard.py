"""Thin CLI that delegates to ULBFraudDatasetAdapter.

Kept so older docs/commands still work. Does not train the synthetic product model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.ulb.adapter import ULBFraudDatasetAdapter  # noqa: E402


def evaluate(path=None) -> dict:
    adapter = ULBFraudDatasetAdapter(csv_path=Path(path) if path else None)
    try:
        return adapter.run_full()
    except FileNotFoundError as exc:
        skipped = {
            "track": "REAL_DATASET",
            "skipped": True,
            "reason": str(exc),
            "note": "Do not invent REAL_DATASET metrics.",
        }
        eval_dir = ROOT / "ml" / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "public_metrics.json").write_text(json.dumps(skipped, indent=2))
        (eval_dir / "ulb_metrics.json").write_text(json.dumps(skipped, indent=2))
        (eval_dir / "public_report.md").write_text(
            "# REAL_DATASET — skipped\n\nThe public credit-card CSV was not available. No metrics were fabricated.\n"
        )
        return skipped


if __name__ == "__main__":
    out = evaluate()
    print(json.dumps({k: out.get(k) for k in ("track", "skipped", "pr_auc", "roc_auc", "f1", "precision", "recall", "model_version")}, indent=2))
