"""Controlled synthetic fraud scenarios for evaluation only.

Labels here are generator ground truth, not ULB Class and not real-world fraud.
Transaction scenario_tag is prefixed with eval_ so they never enter feedback training.
"""

from __future__ import annotations

import random
from typing import Any

from app.ml.feedback_dataset import EVAL_SCENARIO_PREFIX
from app.services.synthetic import MERCHANTS, SEED_USERS, _ip, _pack

SCENARIO_NAMES = (
    "normal_payment",
    "stolen_account",
    "card_testing",
    "high_velocity",
    "unusual_amount",
    "new_device",
    "shared_device",
    "shared_ip",
    "device_farm",
    "fraud_ring",
)

GRAPH_SCENARIOS = ("shared_device", "shared_ip", "device_farm", "fraud_ring")
HIGH_RISK_SCENARIOS = ("stolen_account", "card_testing", "device_farm", "fraud_ring")

GROUND_TRUTH = {
    "normal_payment": 0,
    "stolen_account": 1,
    "card_testing": 1,
    "high_velocity": 1,
    "unusual_amount": 1,
    "new_device": 1,
    "shared_device": 1,
    "shared_ip": 1,
    "device_farm": 1,
    "fraud_ring": 1,
}


def eval_tag(name: str) -> str:
    return f"{EVAL_SCENARIO_PREFIX}{name}"


def _legit_users() -> list[dict]:
    return SEED_USERS[:5]


def _ring_users() -> list[dict]:
    return SEED_USERS[5:8]


def generate_scenario(name: str, count: int, seed: int = 42) -> list[dict[str, Any]]:
    if name not in GROUND_TRUTH:
        raise ValueError(f"Unknown scenario '{name}'. Choose from {SCENARIO_NAMES}")
    rng = random.Random(seed)
    builders = {
        "normal_payment": _normal_payment,
        "stolen_account": _stolen_account,
        "card_testing": _card_testing,
        "high_velocity": _high_velocity,
        "unusual_amount": _unusual_amount,
        "new_device": _new_device,
        "shared_device": _shared_device,
        "shared_ip": _shared_ip,
        "device_farm": _device_farm,
        "fraud_ring": _fraud_ring,
    }
    rows = [builders[name](rng, i) for i in range(count)]
    for row in rows:
        row["expected_fraud"] = GROUND_TRUTH[name]
        row["scenario"] = name
        row["scenario_tag"] = eval_tag(name)
        row["is_fraud"] = GROUND_TRUTH[name]
        row["track"] = "SYNTHETIC_SCENARIO_EVALUATION"
    return rows


def generate_bundle(counts: dict[str, int], seed: int = 42) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, (name, n) in enumerate(counts.items()):
        if n <= 0:
            continue
        out.extend(generate_scenario(name, n, seed=seed + i * 17))
    return out


def _normal_payment(rng: random.Random, i: int) -> dict:
    user = _legit_users()[i % 5]
    merchant = MERCHANTS[i % 7]
    amount = max(40.0, rng.gauss(user["typical_amount"], user["typical_amount"] * 0.15))
    return _pack(
        rng,
        user_id=user["user_id"],
        merchant=merchant,
        amount=amount,
        hour=user["typical_hour"],
        device=user["known_devices"][0],
        ip=_ip(rng, 10, 10, 1, 10 + (i % 40)),
        location=user["home_location"],
        age=user["account_age_days"],
        known_dev=True,
        known_loc=True,
        velocity=1,
        fails=0,
        prev_count=40,
        prev_avg=user["typical_amount"],
        is_fraud=0,
        tag=eval_tag("normal_payment"),
    )


def _stolen_account(rng: random.Random, i: int) -> dict:
    user = _legit_users()[0]
    return _pack(
        rng,
        user_id=user["user_id"],
        merchant=MERCHANTS[1],
        amount=rng.uniform(18000, 42000),
        hour=3,
        device=f"dev_eval_stolen_{i}",
        ip=_ip(rng, 203, 0, 113, 40 + i % 50),
        location="Dubai",
        age=user["account_age_days"],
        known_dev=False,
        known_loc=False,
        velocity=4,
        fails=2,
        prev_count=55,
        prev_avg=user["typical_amount"],
        is_fraud=1,
        tag=eval_tag("stolen_account"),
    )


def _card_testing(rng: random.Random, i: int) -> dict:
    user = _legit_users()[3]
    return _pack(
        rng,
        user_id=user["user_id"],
        merchant=MERCHANTS[6],
        amount=rng.uniform(1.5, 18),
        hour=2,
        device=f"dev_eval_probe_{i}",
        ip=_ip(rng, 198, 51, 100, 10 + i % 80),
        location="Singapore",
        age=user["account_age_days"],
        known_dev=False,
        known_loc=False,
        velocity=9 + (i % 6),
        fails=5 + (i % 4),
        prev_count=i % 4,
        prev_avg=user["typical_amount"],
        is_fraud=1,
        tag=eval_tag("card_testing"),
    )


def _high_velocity(rng: random.Random, i: int) -> dict:
    user = _legit_users()[2]
    return _pack(
        rng,
        user_id=user["user_id"],
        merchant=MERCHANTS[3],
        amount=rng.uniform(120, 380),
        hour=user["typical_hour"],
        device=user["known_devices"][0],
        ip="10.10.1.22",
        location=user["home_location"],
        age=user["account_age_days"],
        known_dev=True,
        known_loc=True,
        velocity=10 + i,
        fails=0,
        prev_count=30,
        prev_avg=user["typical_amount"],
        is_fraud=1,
        tag=eval_tag("high_velocity"),
    )


def _unusual_amount(rng: random.Random, i: int) -> dict:
    user = _legit_users()[1]
    return _pack(
        rng,
        user_id=user["user_id"],
        merchant=MERCHANTS[8],
        amount=user["typical_amount"] * rng.uniform(9, 14),
        hour=2,
        device=user["known_devices"][0],
        ip=_ip(rng, 10, 20, 8, 20 + i),
        location=user["home_location"],
        age=user["account_age_days"],
        known_dev=True,
        known_loc=True,
        velocity=2,
        fails=0,
        prev_count=70,
        prev_avg=user["typical_amount"],
        is_fraud=1,
        tag=eval_tag("unusual_amount"),
    )


def _new_device(rng: random.Random, i: int) -> dict:
    user = _legit_users()[4]
    return _pack(
        rng,
        user_id=user["user_id"],
        merchant=MERCHANTS[7],
        amount=user["typical_amount"] * rng.uniform(2.5, 5),
        hour=1,
        device=f"dev_eval_new_{i}",
        ip=_ip(rng, 45, 12, 8, 30 + i),
        location="Singapore",
        age=user["account_age_days"],
        known_dev=False,
        known_loc=False,
        velocity=3,
        fails=3,
        prev_count=40,
        prev_avg=user["typical_amount"],
        is_fraud=1,
        tag=eval_tag("new_device"),
    )


def _shared_device(rng: random.Random, i: int) -> dict:
    users = _ring_users()
    user = users[i % 3]
    return _pack(
        rng,
        user_id=user["user_id"],
        merchant=MERCHANTS[9],
        amount=rng.uniform(800, 2400),
        hour=3,
        device="dev_eval_shared",
        ip=_ip(rng, 10, 30, 1, 40 + i),
        location=user["home_location"],
        age=user["account_age_days"],
        known_dev=True,
        known_loc=True,
        velocity=4,
        fails=1,
        prev_count=4,
        prev_avg=user["typical_amount"],
        is_fraud=1,
        tag=eval_tag("shared_device"),
    )


def _shared_ip(rng: random.Random, i: int) -> dict:
    users = _ring_users()
    user = users[i % 3]
    return _pack(
        rng,
        user_id=user["user_id"],
        merchant=MERCHANTS[4],
        amount=rng.uniform(500, 1600),
        hour=4,
        device=user["known_devices"][0],
        ip="203.0.113.88",
        location="Dubai",
        age=user["account_age_days"],
        known_dev=True,
        known_loc=False,
        velocity=3,
        fails=0,
        prev_count=3,
        prev_avg=user["typical_amount"],
        is_fraud=1,
        tag=eval_tag("shared_ip"),
    )


def _device_farm(rng: random.Random, i: int) -> dict:
    users = _ring_users()
    user = users[i % 3]
    return _pack(
        rng,
        user_id=user["user_id"],
        merchant=MERCHANTS[9],
        amount=rng.uniform(900, 4200),
        hour=3,
        device="dev_farm_01",
        ip="203.0.113.200",
        location="Dubai",
        age=user["account_age_days"],
        known_dev=True,
        known_loc=False,
        velocity=5,
        fails=1,
        prev_count=4,
        prev_avg=user["typical_amount"],
        is_fraud=1,
        tag=eval_tag("device_farm"),
    )


def _fraud_ring(rng: random.Random, i: int) -> dict:
    users = _ring_users()
    user = users[i % 3]
    return _pack(
        rng,
        user_id=user["user_id"],
        merchant=MERCHANTS[9],
        amount=rng.uniform(1100, 3800),
        hour=2,
        device="dev_farm_01",
        ip="203.0.113.200",
        location="Dubai",
        age=user["account_age_days"],
        known_dev=True,
        known_loc=False,
        velocity=6,
        fails=2,
        prev_count=5,
        prev_avg=user["typical_amount"],
        is_fraud=1,
        tag=eval_tag("fraud_ring"),
    )
