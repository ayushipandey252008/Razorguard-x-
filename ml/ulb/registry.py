"""Model identities. Keep the synthetic product model and ULB offline models distinct."""

from __future__ import annotations

from ml.ulb.constants import (
    CALIBRATED_MODEL_VERSION,
    DATASET_ID,
    HGB_MODEL_VERSION,
    MODEL_VERSION,
    SYNTHETIC_MODEL_VERSION,
    TRACK,
)

MODEL_REGISTRY = [
    {
        "id": SYNTHETIC_MODEL_VERSION,
        "track": "SYNTHETIC_DATASET",
        "role": "live_product_scoring",
        "kind": "ensemble_risk_score",
        "signals": ["model_probability", "behavior_score", "rule_score", "graph_score"],
        "final_output": "final_risk_score",
        "final_output_is_probability": False,
        "deployed_to_live_pipeline": True,
        "note": "Live RazorGuard X scoring. Final risk is a weighted combination, not P(fraud).",
    },
    {
        "id": MODEL_VERSION,
        "track": TRACK,
        "dataset_id": DATASET_ID,
        "role": "offline_supervised_raw",
        "kind": "raw_booster_probability",
        "final_output": "raw_probability",
        "final_output_is_probability": False,
        "note": "Uncalibrated XGBoost P(class=1) on ULB PCA features. Offline only.",
        "deployed_to_live_pipeline": False,
    },
    {
        "id": HGB_MODEL_VERSION,
        "track": TRACK,
        "dataset_id": DATASET_ID,
        "role": "offline_supervised_raw_fallback",
        "kind": "raw_booster_probability",
        "final_output": "raw_probability",
        "final_output_is_probability": False,
        "deployed_to_live_pipeline": False,
        "note": "HistGBM fallback if XGBoost is unavailable. Offline only.",
    },
    {
        "id": CALIBRATED_MODEL_VERSION,
        "track": TRACK,
        "dataset_id": DATASET_ID,
        "role": "offline_supervised_calibrated",
        "kind": "calibrated_probability",
        "final_output": "calibrated_probability",
        "final_output_is_probability": True,
        "risk_score_alias": "calibrated_probability * 100 is a scaled risk score, not a second probability",
        "deployed_to_live_pipeline": False,
        "note": (
            "Calibrated estimate of P(Class=1 | ULB features) for offline evaluation. "
            "Not the product final_risk_score. Not mixed with behavior/rule/graph scores."
        ),
    },
]


def get_model_record(model_id: str) -> dict | None:
    for row in MODEL_REGISTRY:
        if row["id"] == model_id:
            return dict(row)
    return None
