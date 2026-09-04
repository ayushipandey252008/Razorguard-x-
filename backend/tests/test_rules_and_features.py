from __future__ import annotations

from datetime import datetime, timezone

from app.ml.features import FEATURE_COLUMNS, row_to_features
from app.rules.catalog import engine
from app.services.risk_engine import combine_scores
from app.services.synthetic import generate_labeled_dataset, scenario_transactions


def test_feature_vector_complete():
    feats = row_to_features(
        {
            "amount": 100.0,
            "account_age_days": 30,
            "failed_attempts": 0,
            "transaction_velocity": 1,
            "previous_transaction_count": 4,
            "previous_average_amount": 80.0,
            "current_device_known": True,
            "current_location_known": True,
            "timestamp": datetime.now(timezone.utc),
            "payment_method": "UPI",
            "merchant_category": "GROCERY",
        }
    )
    assert list(feats.keys()) == FEATURE_COLUMNS


def test_labeled_dataset_has_both_classes():
    rows = generate_labeled_dataset(200, seed=1)
    labels = {r["is_fraud"] for r in rows}
    assert labels == {0, 1}


def test_rules_fire_on_high_amount_new_device():
    txn = {
        "amount": 80000,
        "previous_average_amount": 400,
        "current_device_known": False,
        "current_location_known": False,
        "transaction_velocity": 12,
        "failed_attempts": 4,
        "account_age_days": 2,
        "device_id": "d1",
        "ip_address": "1.1.1.1",
        "merchant_id": "m",
        "location": "Dubai",
    }
    ctx = {
        "typical_amount": 400,
        "device_user_count": 4,
        "ip_user_count": 4,
        "merchant_watchlisted": True,
        "anomalies": [{"code": "amount_deviation"}, {"code": "unknown_device"}],
    }
    fired = engine.evaluate(txn, ctx)
    ids = {r.rule_id for r in fired}
    assert "HIGH_AMOUNT" in ids
    assert "NEW_DEVICE" in ids
    assert engine.aggregate_score(fired) > 40


def test_risk_thresholds_prototype():
    low = combine_scores(10, 10, 5, 0, 0)
    assert low["decision"] == "APPROVE"
    mid = combine_scores(50, 50, 50, 50, 2)
    assert mid["decision"] in {"REVIEW", "BLOCK"}
    high = combine_scores(90, 90, 90, 90, 5)
    assert high["decision"] == "BLOCK"


def test_scenario_generator_device_farm_shares_device():
    rows = scenario_transactions("device_farm", 6)
    devices = {r["device_id"] for r in rows}
    users = {r["user_id"] for r in rows}
    assert devices == {"dev_farm_01"}
    assert len(users) >= 2
