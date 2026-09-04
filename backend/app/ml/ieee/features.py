"""IEEE-CIS feature families. Behavioral features use only prior transactions."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from app.ml.ieee.constants import JOIN_KEY, TARGET_COLUMN, TIME_COLUMN

TRANSACTION_FEATURES = ["TransactionAmt", "ProductCD", "hour_of_day_proxy", "C1", "C2", "D1", "M1", "M2"]
CARD_FEATURES = ["card1", "card2", "card3", "card4", "card5", "card6"]
ADDRESS_FEATURES = ["addr1", "addr2", "dist1"]
IDENTITY_FEATURES = [
    "identity_present",
    "id_01",
    "id_02",
    "id_12",
    "id_30",
    "id_31",
    "DeviceType",
    "DeviceInfo",
]
EMAIL_FEATURES = ["P_emaildomain", "R_emaildomain"]
BEHAVIORAL_FEATURES = [
    "card_prior_count",
    "card_prior_amt_mean",
    "card_amt_ratio",
    "card_seconds_since_prev",
    "email_prior_count",
    "device_prior_count",
]
GRAPH_FEATURES = [
    "graph_card_degree",
    "graph_device_txn_count",
    "graph_device_account_count",
    "graph_addr_account_count",
    "graph_card_prior_fraud_count",
    "graph_device_prior_fraud_count",
    "graph_card_cluster_size",
    "graph_suspicious_cluster",
    "graph_ip_entity_unavailable",
]

FAMILY_MAP = {
    "transaction": TRANSACTION_FEATURES,
    "card": CARD_FEATURES,
    "address": ADDRESS_FEATURES,
    "identity": IDENTITY_FEATURES,
    "email": EMAIL_FEATURES,
    "behavioral": BEHAVIORAL_FEATURES,
    "graph": GRAPH_FEATURES,
}

EXPERIMENTS = {
    "A_transaction_only": ["transaction"],
    "B_transaction_card": ["transaction", "card"],
    "C_transaction_identity": ["transaction", "identity"],
    "D_transaction_behavioral": ["transaction", "behavioral"],
    "E_transaction_graph": ["transaction", "graph"],
    "F_combined": ["transaction", "card", "address", "identity", "email", "behavioral", "graph"],
    "ablation_no_graph": ["transaction", "card", "address", "identity", "email", "behavioral"],
}

CATEGORICAL_COLUMNS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "id_12",
    "id_30",
    "id_31",
    "DeviceType",
    "DeviceInfo",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
]


def columns_for_families(families: list[str], available: list[str] | None = None) -> list[str]:
    cols: list[str] = []
    avail = set(available) if available is not None else None
    for fam in families:
        for col in FAMILY_MAP[fam]:
            if avail is None or col in avail:
                if col not in cols:
                    cols.append(col)
    return cols


def add_transaction_timing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_of_day_proxy"] = (out[TIME_COLUMN].astype("int64") % 86400) / 3600.0
    return out


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Entity history strictly before the current TransactionDT."""
    ordered = df.sort_values(TIME_COLUMN, kind="mergesort")
    card_n: dict = defaultdict(int)
    card_sum: dict = defaultdict(float)
    card_last: dict = {}
    email_n: dict = defaultdict(int)
    device_n: dict = defaultdict(int)

    card_prior_count = np.zeros(len(ordered), dtype=np.int32)
    card_prior_amt_mean = np.full(len(ordered), np.nan, dtype=np.float32)
    card_amt_ratio = np.full(len(ordered), np.nan, dtype=np.float32)
    card_dt = np.full(len(ordered), np.nan, dtype=np.float32)
    email_prior = np.zeros(len(ordered), dtype=np.int32)
    device_prior = np.zeros(len(ordered), dtype=np.int32)

    positions = list(ordered.index)
    for i, idx in enumerate(positions):
        row = ordered.iloc[i]
        card = row.get("card1")
        email = row.get("P_emaildomain")
        device = row.get("DeviceInfo")
        amt = float(row["TransactionAmt"]) if pd.notna(row.get("TransactionAmt")) else 0.0
        t = int(row[TIME_COLUMN])

        if pd.notna(card):
            card_prior_count[i] = card_n[card]
            if card_n[card]:
                mean = card_sum[card] / card_n[card]
                card_prior_amt_mean[i] = mean
                card_amt_ratio[i] = amt / mean if mean else np.nan
            if card in card_last:
                card_dt[i] = t - card_last[card]
            card_n[card] += 1
            card_sum[card] += amt
            card_last[card] = t
        if pd.notna(email):
            email_prior[i] = email_n[email]
            email_n[email] += 1
        if pd.notna(device):
            device_prior[i] = device_n[device]
            device_n[device] += 1

    out = df.copy()
    out.loc[positions, "card_prior_count"] = card_prior_count
    out.loc[positions, "card_prior_amt_mean"] = card_prior_amt_mean
    out.loc[positions, "card_amt_ratio"] = card_amt_ratio
    out.loc[positions, "card_seconds_since_prev"] = card_dt
    out.loc[positions, "email_prior_count"] = email_prior
    out.loc[positions, "device_prior_count"] = device_prior
    return out


def unused_raw_columns(df: pd.DataFrame) -> list[str]:
    used = set(FAMILY_MAP["transaction"] + FAMILY_MAP["card"] + FAMILY_MAP["address"] + FAMILY_MAP["identity"] + FAMILY_MAP["email"])
    used |= {JOIN_KEY, TARGET_COLUMN, TIME_COLUMN, "identity_present", "hour_of_day_proxy"}
    used |= set(BEHAVIORAL_FEATURES)
    used |= set(GRAPH_FEATURES)
    return [c for c in df.columns if c not in used]
