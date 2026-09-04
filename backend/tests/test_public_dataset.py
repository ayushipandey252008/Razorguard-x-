"""Skip cleanly when the public CSV / metrics are absent. Never invent scores."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "ml" / "evaluation" / "public_metrics.json"


def test_public_dataset_metrics_are_isolated():
    if not PUBLIC.exists():
        pytest.skip("REAL_DATASET metrics not generated yet")
    import json

    data = json.loads(PUBLIC.read_text())
    assert data.get("track") == "REAL_DATASET"
    if data.get("skipped"):
        pytest.skip(data.get("reason") or "REAL_DATASET skipped")
    for key in ("pr_auc", "roc_auc", "precision", "recall", "f1", "false_positive_rate", "confusion_matrix"):
        assert key in data
    assert data.get("incompatible_with_product_pipeline") is True
    assert data["track"] != "SYNTHETIC_DATASET"
