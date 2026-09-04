"""Shared feature definitions for the supervised fraud model.

Feature engineering is kept here so training and inference cannot drift.
No raw card numbers or authentication secrets are used.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "amount",
    "account_age_days",
    "failed_attempts",
    "transaction_velocity",
    "previous_transaction_count",
    "previous_average_amount",
    "current_device_known",
    "current_location_known",
    "amount_vs_avg_ratio",
    "hour_of_day",
    "day_of_week",
    "payment_method_code",
    "merchant_category_code",
]

PAYMENT_METHODS = ["UPI", "NETBANKING", "WALLET", "CARD_TOKEN"]
MERCHANT_CATEGORIES = [
    "GROCERY",
    "ELECTRONICS",
    "TRAVEL",
    "GAMING",
    "UTILITIES",
    "FASHION",
    "DIGITAL_GOODS",
    "MONEY_TRANSFER",
    "CRYPTO_ONRAMP",
]


def payment_method_code(method: str) -> int:
    try:
        return PAYMENT_METHODS.index(method)
    except ValueError:
        return 0


def merchant_category_code(category: str) -> int:
    try:
        return MERCHANT_CATEGORIES.index(category)
    except ValueError:
        return 0


def amount_vs_avg_ratio(amount: float, previous_average: float) -> float:
    if previous_average is None or previous_average <= 0:
        return 1.0
    return float(amount / previous_average)


def row_to_features(row: dict) -> dict:
    ts = row.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if ts is None:
        ts = datetime.utcnow()
    prev_avg = float(row.get("previous_average_amount") or 0)
    amount = float(row["amount"])
    return {
        "amount": amount,
        "account_age_days": int(row.get("account_age_days") or 0),
        "failed_attempts": int(row.get("failed_attempts") or 0),
        "transaction_velocity": int(row.get("transaction_velocity") or 1),
        "previous_transaction_count": int(row.get("previous_transaction_count") or 0),
        "previous_average_amount": prev_avg,
        "current_device_known": int(bool(row.get("current_device_known", True))),
        "current_location_known": int(bool(row.get("current_location_known", True))),
        "amount_vs_avg_ratio": amount_vs_avg_ratio(amount, prev_avg),
        "hour_of_day": int(ts.hour),
        "day_of_week": int(ts.weekday()),
        "payment_method_code": payment_method_code(row.get("payment_method") or "UPI"),
        "merchant_category_code": merchant_category_code(row.get("merchant_category") or "GROCERY"),
    }


def features_to_array(features: dict) -> np.ndarray:
    return np.array([[features[c] for c in FEATURE_COLUMNS]], dtype=float)


def dataframe_from_records(records: list[dict]) -> pd.DataFrame:
    rows = [row_to_features(r) for r in records]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
