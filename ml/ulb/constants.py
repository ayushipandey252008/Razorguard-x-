from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET_ID = "ULB_CREDIT_CARD_FRAUD"
DATASET_NAME = "ULB Credit Card Fraud Detection"
TRACK = "REAL_DATASET"
MODEL_VERSION = "ulb-xgb-v1"
HGB_MODEL_VERSION = "ulb-hgb-v1"
CALIBRATED_MODEL_VERSION = "ulb-xgb-v1-calibrated"
SYNTHETIC_MODEL_VERSION = "xgb-iforest-v1-calibrated"

# Prototype cost parameters — not industry-standard. Tune via this config only.
COST_SCENARIOS = {
    "A": {
        "id": "A",
        "label": "Scenario A",
        "false_negative_cost": 100.0,
        "false_positive_cost": 5.0,
        "review_cost": 2.0,
    },
    "B": {
        "id": "B",
        "label": "Scenario B",
        "false_negative_cost": 50.0,
        "false_positive_cost": 10.0,
        "review_cost": 3.0,
    },
}
DEFAULT_COST_SCENARIO = "A"
# Ops-capacity cap used for the documented prototype (fraction not auto-approved).
PROTOTYPE_CAPACITY_CAP = 0.05
RANDOM_SEED = 42
DOWNLOAD_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
EXPECTED_FILENAME = "creditcard.csv"

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
REQUIRED_COLUMNS = ["Time", *PCA_COLUMNS, "Amount", "Class"]
RAW_FEATURE_COLUMNS = ["Time", *PCA_COLUMNS, "Amount"]
DERIVED_COLUMNS = ["log_amount", "time_of_day_proxy", "transaction_time_bucket"]
SCALE_COLUMNS = ["Time", "Amount", *DERIVED_COLUMNS]
TARGET_COLUMN = "Class"

RAW_DIR = REPO_ROOT / "ml" / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "ml" / "data" / "processed"
EVAL_DIR = REPO_ROOT / "ml" / "evaluation"
ULB_MODEL_DIR = REPO_ROOT / "ml" / "models" / "ulb"
LEGACY_CSV = REPO_ROOT / "ml" / "data" / EXPECTED_FILENAME
RAW_CSV = RAW_DIR / EXPECTED_FILENAME
