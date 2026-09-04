"""Time-aware graph-derived features. Not a replacement for live NetworkX/Neo4j."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from app.ml.ieee.constants import TARGET_COLUMN, TIME_COLUMN
from app.ml.ieee.features import GRAPH_FEATURES

GRAPH_CONSTRUCTION_NOTES = [
    "Offline IEEE graph features are built by scanning transactions in TransactionDT order.",
    "A row at time T only sees entity statistics from transactions with time < T.",
    "Current-row isFraud is never written into that row's graph features.",
    "Prior-label neighbor counts use historical isFraud from earlier rows only.",
    "IEEE-CIS identity has no raw IP; graph_ip_entity_unavailable=1 and IP counts are not fabricated.",
    "addr1 is a coarse billing/geo entity, not an IP address.",
    "These features do not replace the live GraphBackend (NetworkX / optional Neo4j).",
]


def add_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(TIME_COLUMN, kind="mergesort")
    n = len(ordered)
    card_degree: dict = defaultdict(int)
    card_fraud: dict = defaultdict(int)
    device_txn: dict = defaultdict(int)
    device_fraud: dict = defaultdict(int)
    device_accounts: dict = defaultdict(set)
    addr_accounts: dict = defaultdict(set)
    card_devices: dict = defaultdict(set)

    cols = {name: np.zeros(n, dtype=np.float32) for name in GRAPH_FEATURES}
    positions = list(ordered.index)
    for i, idx in enumerate(positions):
        row = ordered.iloc[i]
        card = row.get("card1")
        device = row.get("DeviceInfo")
        addr = row.get("addr1")
        label = int(row[TARGET_COLUMN]) if TARGET_COLUMN in row and pd.notna(row[TARGET_COLUMN]) else 0

        cols["graph_card_degree"][i] = card_degree[card] if pd.notna(card) else 0
        cols["graph_device_txn_count"][i] = device_txn[device] if pd.notna(device) else 0
        cols["graph_device_account_count"][i] = len(device_accounts[device]) if pd.notna(device) else 0
        cols["graph_addr_account_count"][i] = len(addr_accounts[addr]) if pd.notna(addr) else 0
        cols["graph_card_prior_fraud_count"][i] = card_fraud[card] if pd.notna(card) else 0
        cols["graph_device_prior_fraud_count"][i] = device_fraud[device] if pd.notna(device) else 0
        cluster = 0
        if pd.notna(card):
            cluster = card_degree[card] + len(card_devices[card])
        cols["graph_card_cluster_size"][i] = cluster
        prior_fraud = 0
        if pd.notna(card):
            prior_fraud += card_fraud[card]
        if pd.notna(device):
            prior_fraud += device_fraud[device]
        cols["graph_suspicious_cluster"][i] = 1.0 if prior_fraud >= 1 else 0.0
        cols["graph_ip_entity_unavailable"][i] = 1.0

        if pd.notna(card):
            card_degree[card] += 1
            card_fraud[card] += label
            if pd.notna(device):
                card_devices[card].add(device)
        if pd.notna(device):
            device_txn[device] += 1
            device_fraud[device] += label
            if pd.notna(card):
                device_accounts[device].add(card)
        if pd.notna(addr) and pd.notna(card):
            addr_accounts[addr].add(card)

    out = df.copy()
    for name, arr in cols.items():
        out.loc[positions, name] = arr
    return out


def graph_feature_uses_future(df: pd.DataFrame, sample_index: int) -> bool:
    """Recompute one row from strict past; True if stored features saw the future."""
    if sample_index not in df.index:
        return False
    t = int(df.loc[sample_index, TIME_COLUMN])
    past = df[df[TIME_COLUMN] < t]
    row = df.loc[sample_index]
    card = row.get("card1")
    expected = int((past["card1"] == card).sum()) if pd.notna(card) else 0
    stored = int(row.get("graph_card_degree") or 0)
    return stored != expected
