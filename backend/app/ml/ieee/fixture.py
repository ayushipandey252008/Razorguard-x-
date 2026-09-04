"""Tiny synthetic tables that mimic IEEE-CIS *shape*, not its statistics.

Metrics from this fixture are NOT IEEE-CIS public-dataset results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.ieee.constants import JOIN_KEY, TARGET_COLUMN, TIME_COLUMN


def make_ieee_fixture(n: int = 90, n_identity: int = 55, n_fraud: int = 14, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    txn_ids = np.arange(1, n + 1, dtype=np.int32)
    dt = np.arange(10, 10 + n, dtype=np.int32)
    fraud = np.zeros(n, dtype=np.int8)
    # Later-period fraud cluster sharing a device/card — graph should see prior labels only.
    fraud_idx = np.linspace(n // 3, n - 1, n_fraud, dtype=int)
    fraud[fraud_idx] = 1
    cards = rng.integers(1000, 1012, size=n)
    cards[fraud_idx[:6]] = 1001
    devices = np.array([f"dev_{i % 7}" for i in range(n)], dtype=object)
    devices[fraud_idx[:6]] = "dev_shared"
    emails = np.array([f"user{i % 9}@mail.test" for i in range(n)], dtype=object)
    product = rng.choice(["W", "C", "H", "R", "S"], size=n)
    amt = rng.uniform(5, 400, size=n)
    amt[fraud_idx] *= rng.uniform(2.5, 6.0, size=n_fraud)

    txn = pd.DataFrame(
        {
            JOIN_KEY: txn_ids,
            TARGET_COLUMN: fraud,
            TIME_COLUMN: dt,
            "TransactionAmt": amt.astype(np.float32),
            "ProductCD": product,
            "card1": cards.astype(np.int32),
            "card2": rng.choice([100, 200, 300, np.nan], size=n),
            "card3": rng.choice([150, 185, np.nan], size=n),
            "card4": rng.choice(["visa", "mastercard", "discover", None], size=n),
            "card5": rng.choice([166, 224, np.nan], size=n),
            "card6": rng.choice(["debit", "credit", None], size=n),
            "addr1": rng.choice([123, 456, 789, np.nan], size=n),
            "addr2": rng.choice([87, 60, np.nan], size=n),
            "dist1": rng.choice([0, 5, 12, np.nan], size=n),
            "P_emaildomain": emails,
            "R_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], size=n),
            "C1": rng.integers(1, 8, size=n).astype(np.float32),
            "C2": rng.integers(0, 5, size=n).astype(np.float32),
            "D1": rng.choice([0, 1, 7, 30, np.nan], size=n),
            "M1": rng.choice(["T", "F", None], size=n),
            "M2": rng.choice(["T", "F", None], size=n),
        }
    )
    # Duplicate one legitimate row to exercise duplicate-ID audit.
    dup = txn.iloc[[2]].copy()
    txn = pd.concat([txn, dup], ignore_index=True)

    id_ids = txn_ids[:n_identity]
    ident = pd.DataFrame(
        {
            JOIN_KEY: id_ids,
            "id_01": rng.normal(0, 1, size=n_identity).astype(np.float32),
            "id_02": rng.integers(1000, 9000, size=n_identity).astype(np.float32),
            "id_12": rng.choice(["Found", "NotFound", None], size=n_identity),
            "id_30": rng.choice(["iOS", "Android", "Windows", None], size=n_identity),
            "id_31": rng.choice(["chrome", "safari", "samsung", None], size=n_identity),
            "DeviceType": rng.choice(["mobile", "desktop", None], size=n_identity),
            "DeviceInfo": devices[:n_identity],
        }
    )
    return txn, ident
