"""Synthetic transaction generator for training, seeding, and simulation.

All identifiers are fake. No real PANs, CVVs, or PII are produced.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from app.ml.features import MERCHANT_CATEGORIES, PAYMENT_METHODS

LOCATIONS = [
    "Bengaluru",
    "Mumbai",
    "Delhi",
    "Hyderabad",
    "Chennai",
    "Pune",
    "Kolkata",
    "Jaipur",
    "Ahmedabad",
    "Kochi",
    "Singapore",
    "Dubai",
]

MERCHANTS = [
    ("m_groc_01", "Nimbus Mart", "GROCERY", "Bengaluru"),
    ("m_elec_01", "Volt & Co", "ELECTRONICS", "Mumbai"),
    ("m_trav_01", "Skyline Bookings", "TRAVEL", "Delhi"),
    ("m_game_01", "Pixel Arena", "GAMING", "Hyderabad"),
    ("m_util_01", "GridPay Utilities", "UTILITIES", "Chennai"),
    ("m_fash_01", "Loom Street", "FASHION", "Pune"),
    ("m_digi_01", "ByteCart", "DIGITAL_GOODS", "Bengaluru"),
    ("m_xfer_01", "Relay Transfer", "MONEY_TRANSFER", "Mumbai"),
    ("m_crypto_01", "Onramp Desk", "CRYPTO_ONRAMP", "Singapore"),
    ("m_watch_01", "Watchlist Outlet", "DIGITAL_GOODS", "Dubai"),
]

WATCHLISTED = {"m_watch_01"}

SEED_USERS = [
    {
        "user_id": "usr_ananya",
        "account_age_days": 820,
        "home_location": "Bengaluru",
        "typical_amount": 640.0,
        "typical_hour": 13,
        "known_devices": ["dev_ananya_phone"],
        "known_locations": ["Bengaluru", "Mysuru"],
    },
    {
        "user_id": "usr_kabir",
        "account_age_days": 410,
        "home_location": "Mumbai",
        "typical_amount": 1800.0,
        "typical_hour": 19,
        "known_devices": ["dev_kabir_laptop"],
        "known_locations": ["Mumbai", "Pune"],
    },
    {
        "user_id": "usr_meera",
        "account_age_days": 95,
        "home_location": "Delhi",
        "typical_amount": 420.0,
        "typical_hour": 11,
        "known_devices": ["dev_meera_phone"],
        "known_locations": ["Delhi"],
    },
    {
        "user_id": "usr_ravi",
        "account_age_days": 12,
        "home_location": "Hyderabad",
        "typical_amount": 250.0,
        "typical_hour": 21,
        "known_devices": ["dev_ravi_phone"],
        "known_locations": ["Hyderabad"],
    },
    {
        "user_id": "usr_leila",
        "account_age_days": 540,
        "home_location": "Chennai",
        "typical_amount": 980.0,
        "typical_hour": 16,
        "known_devices": ["dev_leila_tablet"],
        "known_locations": ["Chennai", "Bengaluru"],
    },
    {
        "user_id": "usr_ring_a",
        "account_age_days": 18,
        "home_location": "Jaipur",
        "typical_amount": 300.0,
        "typical_hour": 2,
        "known_devices": ["dev_farm_01"],
        "known_locations": ["Jaipur"],
    },
    {
        "user_id": "usr_ring_b",
        "account_age_days": 9,
        "home_location": "Ahmedabad",
        "typical_amount": 280.0,
        "typical_hour": 3,
        "known_devices": ["dev_farm_01"],
        "known_locations": ["Ahmedabad"],
    },
    {
        "user_id": "usr_ring_c",
        "account_age_days": 6,
        "home_location": "Kochi",
        "typical_amount": 310.0,
        "typical_hour": 4,
        "known_devices": ["dev_farm_01"],
        "known_locations": ["Kochi"],
    },
]


def _ip(rng: random.Random, *octets: int) -> str:
    if octets:
        return ".".join(str(o) for o in octets)
    return ".".join(str(rng.randint(10, 220)) for _ in range(4))


def _pay_id(rng: random.Random) -> str:
    return f"pay_{rng.randbytes(6).hex()}"


def _ts(rng: random.Random, hour: int | None = None) -> datetime:
    now = datetime.now(timezone.utc)
    delta = timedelta(days=rng.randint(0, 14), minutes=rng.randint(0, 1400))
    t = now - delta
    if hour is not None:
        t = t.replace(hour=hour % 24, minute=rng.randint(0, 59))
    return t


def generate_labeled_dataset(n: int = 8000, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        # Inject a controlled mix of patterns (~12% fraud-like) plus overlap.
        roll = rng.random()
        if roll < 0.06:
            rows.append(_card_testing(rng, i))
        elif roll < 0.09:
            rows.append(_account_takeover(rng, i))
        elif roll < 0.11:
            rows.append(_stolen_account(rng, i))
        elif roll < 0.12:
            rows.append(_velocity_attack(rng, i))
        else:
            rows.append(_normal(rng, i))
        if rng.random() < 0.03:
            rows[-1]["is_fraud"] = 1 - int(rows[-1]["is_fraud"])
    return rows


def _base_user(rng: random.Random, i: int) -> tuple[str, int, float, str, list[str], list[str]]:
    user_id = f"usr_syn_{i % 400:04d}"
    age = rng.randint(20, 1400)
    typical = rng.uniform(200, 2500)
    home = rng.choice(LOCATIONS[:10])
    devices = [f"dev_{user_id}_a"]
    locations = [home]
    return user_id, age, typical, home, devices, locations


def _normal(rng: random.Random, i: int) -> dict:
    user_id, age, typical, home, devices, locations = _base_user(rng, i)
    merchant = rng.choice(MERCHANTS[:7])
    amount = max(40.0, rng.gauss(typical, typical * 0.25))
    hour = rng.choice([10, 11, 12, 13, 14, 18, 19, 20])
    return _pack(
        rng,
        user_id=user_id,
        merchant=merchant,
        amount=amount,
        hour=hour,
        device=devices[0],
        ip=_ip(rng, 10, 20, rng.randint(1, 40), rng.randint(1, 200)),
        location=home,
        age=age,
        known_dev=True,
        known_loc=True,
        velocity=rng.randint(1, 3),
        fails=0,
        prev_count=rng.randint(8, 80),
        prev_avg=typical,
        is_fraud=0,
        tag="normal",
    )


def _card_testing(rng: random.Random, i: int) -> dict:
    user_id, age, typical, home, *_ = _base_user(rng, i)
    merchant = MERCHANTS[6]  # digital goods
    return _pack(
        rng,
        user_id=user_id,
        merchant=merchant,
        amount=rng.uniform(1, 25),
        hour=rng.randint(0, 5),
        device=f"dev_test_{rng.randint(1, 30)}",
        ip=_ip(rng, 185, rng.randint(1, 200), rng.randint(1, 200), rng.randint(1, 200)),
        location=rng.choice(["Singapore", "Dubai"]),
        age=rng.randint(0, 6),
        known_dev=False,
        known_loc=False,
        velocity=rng.randint(8, 18),
        fails=rng.randint(4, 12),
        prev_count=rng.randint(0, 3),
        prev_avg=typical,
        is_fraud=1,
        tag="card_testing",
    )


def _account_takeover(rng: random.Random, i: int) -> dict:
    user_id, age, typical, home, *_ = _base_user(rng, i)
    merchant = rng.choice(MERCHANTS[7:])
    return _pack(
        rng,
        user_id=user_id,
        merchant=merchant,
        amount=typical * rng.uniform(3.5, 8),
        hour=rng.choice([1, 2, 3, 4]),
        device=f"dev_ato_{rng.randint(1, 80)}",
        ip=_ip(rng, 45, rng.randint(1, 200), rng.randint(1, 200), rng.randint(1, 200)),
        location=rng.choice(["Dubai", "Singapore", "Kolkata"]),
        age=age,
        known_dev=False,
        known_loc=False,
        velocity=rng.randint(4, 9),
        fails=rng.randint(2, 6),
        prev_count=rng.randint(20, 90),
        prev_avg=typical,
        is_fraud=1,
        tag="account_takeover",
    )


def _stolen_account(rng: random.Random, i: int) -> dict:
    user_id, age, typical, home, *_ = _base_user(rng, i)
    merchant = MERCHANTS[1]
    return _pack(
        rng,
        user_id=user_id,
        merchant=merchant,
        amount=rng.uniform(8000, 28000),
        hour=rng.randint(0, 23),
        device=f"dev_stolen_{i}",
        ip=_ip(rng),
        location=rng.choice(LOCATIONS[8:]),
        age=age,
        known_dev=False,
        known_loc=False,
        velocity=rng.randint(2, 5),
        fails=1,
        prev_count=rng.randint(10, 40),
        prev_avg=typical,
        is_fraud=1,
        tag="stolen_account",
    )


def _velocity_attack(rng: random.Random, i: int) -> dict:
    user_id, age, typical, home, devices, _ = _base_user(rng, i)
    merchant = rng.choice(MERCHANTS)
    return _pack(
        rng,
        user_id=user_id,
        merchant=merchant,
        amount=rng.uniform(80, 400),
        hour=rng.randint(0, 23),
        device=devices[0],
        ip=_ip(rng, 10, 20, 1, rng.randint(1, 200)),
        location=home,
        age=age,
        known_dev=True,
        known_loc=True,
        velocity=rng.randint(6, 14),
        fails=0,
        prev_count=rng.randint(15, 50),
        prev_avg=typical,
        is_fraud=1,
        tag="velocity_attack",
    )


def _pack(rng, **kw) -> dict:
    merchant = kw["merchant"]
    ts = _ts(rng, kw["hour"])
    return {
        "user_id": kw["user_id"],
        "merchant_id": merchant[0],
        "merchant_category": merchant[2],
        "amount": round(float(kw["amount"]), 2),
        "currency": "INR",
        "timestamp": ts,
        "device_id": kw["device"],
        "ip_address": kw["ip"],
        "location": kw["location"],
        "payment_method": rng.choice(PAYMENT_METHODS),
        "account_age_days": kw["age"],
        "failed_attempts": kw["fails"],
        "transaction_velocity": kw["velocity"],
        "previous_transaction_count": kw["prev_count"],
        "previous_average_amount": round(float(kw["prev_avg"]), 2),
        "current_device_known": kw["known_dev"],
        "current_location_known": kw["known_loc"],
        "payment_identifier": _pay_id(rng),
        "is_fraud": kw["is_fraud"],
        "scenario_tag": kw["tag"],
    }


def scenario_transactions(scenario: str, count: int, rng: random.Random | None = None) -> list[dict]:
    rng = rng or random.Random()
    builders = {
        "normal": lambda i: _normal_seeded(rng, i),
        "stolen_account": lambda i: _stolen_from_seed(rng, i),
        "card_testing": lambda i: _card_from_seed(rng, i),
        "account_takeover": lambda i: _ato_from_seed(rng, i),
        "device_farm": lambda i: _device_farm(rng, i),
        "fraud_ring": lambda i: _fraud_ring(rng, i),
        "velocity_attack": lambda i: _velocity_from_seed(rng, i),
    }
    fn = builders.get(scenario, builders["normal"])
    return [fn(i) for i in range(count)]


def _user_lookup(user_id: str) -> dict:
    for u in SEED_USERS:
        if u["user_id"] == user_id:
            return u
    return SEED_USERS[0]


def _normal_seeded(rng: random.Random, i: int) -> dict:
    user = rng.choice(SEED_USERS[:5])
    merchant = rng.choice(MERCHANTS[:7])
    amount = max(40.0, rng.gauss(user["typical_amount"], user["typical_amount"] * 0.2))
    return _pack(
        rng,
        user_id=user["user_id"],
        merchant=merchant,
        amount=amount,
        hour=user["typical_hour"],
        device=user["known_devices"][0],
        ip=_ip(rng, 10, 10, 1, rng.randint(2, 40)),
        location=user["home_location"],
        age=user["account_age_days"],
        known_dev=True,
        known_loc=True,
        velocity=rng.randint(1, 2),
        fails=0,
        prev_count=40,
        prev_avg=user["typical_amount"],
        is_fraud=0,
        tag="normal",
    )


def _stolen_from_seed(rng: random.Random, i: int) -> dict:
    user = SEED_USERS[0]
    return _pack(
        rng,
        user_id=user["user_id"],
        merchant=MERCHANTS[1],
        amount=rng.uniform(22000, 54000),
        hour=3,
        device="dev_unknown_stolen",
        ip="203.0.113.44",
        location="Dubai",
        age=user["account_age_days"],
        known_dev=False,
        known_loc=False,
        velocity=3,
        fails=2,
        prev_count=55,
        prev_avg=user["typical_amount"],
        is_fraud=1,
        tag="stolen_account",
    )


def _card_from_seed(rng: random.Random, i: int) -> dict:
    user = SEED_USERS[3]
    return _pack(
        rng,
        user_id=user["user_id"],
        merchant=MERCHANTS[6],
        amount=rng.uniform(2, 19),
        hour=2,
        device=f"dev_probe_{i}",
        ip=f"198.51.100.{10 + i}",
        location="Singapore",
        age=user["account_age_days"],
        known_dev=False,
        known_loc=False,
        velocity=8 + i,
        fails=5 + i,
        prev_count=i,
        prev_avg=user["typical_amount"],
        is_fraud=1,
        tag="card_testing",
    )


def _ato_from_seed(rng: random.Random, i: int) -> dict:
    user = SEED_USERS[1]
    return _pack(
        rng,
        user_id=user["user_id"],
        merchant=MERCHANTS[8],
        amount=user["typical_amount"] * rng.uniform(8, 12),
        hour=2,
        device="dev_ato_session",
        ip="203.0.113.90",
        location="Singapore",
        age=user["account_age_days"],
        known_dev=False,
        known_loc=False,
        velocity=6,
        fails=4,
        prev_count=70,
        prev_avg=user["typical_amount"],
        is_fraud=1,
        tag="account_takeover",
    )


def _device_farm(rng: random.Random, i: int) -> dict:
    user = SEED_USERS[5 + (i % 3)]
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
        tag="device_farm",
    )


def _fraud_ring(rng: random.Random, i: int) -> dict:
    return _device_farm(rng, i)


def _velocity_from_seed(rng: random.Random, i: int) -> dict:
    user = SEED_USERS[2]
    return _pack(
        rng,
        user_id=user["user_id"],
        merchant=MERCHANTS[3],
        amount=rng.uniform(180, 420),
        hour=user["typical_hour"],
        device=user["known_devices"][0],
        ip="10.10.1.22",
        location=user["home_location"],
        age=user["account_age_days"],
        known_dev=True,
        known_loc=True,
        velocity=12 + i,
        fails=0,
        prev_count=30,
        prev_avg=user["typical_amount"],
        is_fraud=1,
        tag="velocity_attack",
    )
