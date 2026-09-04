"""IEEE-CIS Fraud Detection offline track. Isolated from live scoring and ULB."""

from __future__ import annotations

from pathlib import Path

from app.config import REPO_ROOT

DATASET_ID = "IEEE_CIS_FRAUD"
DATASET_NAME = "IEEE-CIS Fraud Detection"
TRACK = "IEEE_CIS_OFFLINE"
TARGET_COLUMN = "isFraud"
JOIN_KEY = "TransactionID"
TIME_COLUMN = "TransactionDT"
RANDOM_SEED = 42

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

LIVE_MODEL_VERSION = "xgb-iforest-v1-calibrated"
BASELINE_VERSION = "ieee-xgb-baseline-v1"
COMBINED_VERSION = "ieee-xgb-combined-v1"
GRAPH_VERSION = "ieee-xgb-graph-v1"

DEFAULT_DATA_DIR = REPO_ROOT / "ml" / "data" / "ieee"
TXN_FILENAME = "train_transaction.csv"
ID_FILENAME = "train_identity.csv"
EVAL_DIR = REPO_ROOT / "ml" / "evaluation"
IEEE_MODEL_DIR = REPO_ROOT / "ml" / "models" / "ieee"
LIVE_MODEL_DIR = REPO_ROOT / "ml" / "models"
ULB_MODEL_DIR = REPO_ROOT / "ml" / "models" / "ulb"
ULB_METRICS_PATH = EVAL_DIR / "ulb_metrics.json"

SETUP_MESSAGE = (
    "IEEE-CIS files were not found. Place Kaggle `train_transaction.csv` and "
    "`train_identity.csv` in IEEE_DATA_DIR (default ml/data/ieee/). "
    "This prototype does not download the dataset. Raw CSVs must not be committed. "
    "Tests use a tiny synthetic fixture and must not be reported as IEEE-CIS results."
)

# Official-ish columns we understand. Extra columns are kept in audit but not
# silently dropped from the raw frame; they are excluded from models with a reason.
TRANSACTION_CORE = [
    JOIN_KEY,
    TARGET_COLUMN,
    TIME_COLUMN,
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "P_emaildomain",
    "R_emaildomain",
    "C1",
    "C2",
    "D1",
    "M1",
    "M2",
]
IDENTITY_CORE = [
    JOIN_KEY,
    "id_01",
    "id_02",
    "id_12",
    "id_30",
    "id_31",
    "DeviceType",
    "DeviceInfo",
]

# IEEE-CIS identity dtypes (Kaggle: id_01–id_11 numeric; DeviceType/DeviceInfo and id_12–id_38 categorical).
# Do not force all id_* to float32: id_12/id_16/id_27/id_29 use Found/NotFound sentinels.
IDENTITY_NUMERIC_COLUMNS = tuple(f"id_{i:02d}" for i in range(1, 12))
IDENTITY_CATEGORICAL_COLUMNS = (
    "DeviceType",
    "DeviceInfo",
    *tuple(f"id_{i:02d}" for i in range(12, 39)),
)

# Transaction columns that are safe as float32 (C/D/V are numbered contest fields, not Device*).
TRANSACTION_FLOAT32_COLUMNS = frozenset({"TransactionAmt", "dist1"})

FORBIDDEN_FEATURE_REASONS = {
    JOIN_KEY: "Join key / transaction identifier. Using it as a feature is ID leakage.",
    TARGET_COLUMN: "Training target. Never a covariate.",
    TIME_COLUMN: (
        "Monotonic contest clock. Raw TransactionDT leaks split position "
        "(later rows are always in val/test). Hour-of-day proxy is used instead."
    ),
}

COST_FALSE_NEGATIVE = 100.0
COST_FALSE_POSITIVE = 5.0
COST_REVIEW = 2.0
