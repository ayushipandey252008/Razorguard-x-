"""Load, audit, and join IEEE-CIS transaction + identity tables."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.ieee.constants import (
    DEFAULT_DATA_DIR,
    EVAL_DIR,
    ID_FILENAME,
    IDENTITY_CATEGORICAL_COLUMNS,
    IDENTITY_CORE,
    IDENTITY_NUMERIC_COLUMNS,
    JOIN_KEY,
    SETUP_MESSAGE,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRANSACTION_CORE,
    TRANSACTION_FLOAT32_COLUMNS,
    TXN_FILENAME,
)
from app.ml.ieee.errors import IeeeDatasetError


def resolve_data_dir(data_dir: Path | str | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    env = os.environ.get("IEEE_DATA_DIR")
    if env:
        return Path(env)
    return DEFAULT_DATA_DIR


def ieee_files_present(data_dir: Path | str | None = None) -> bool:
    root = resolve_data_dir(data_dir)
    return (root / TXN_FILENAME).is_file() and (root / ID_FILENAME).is_file()


def setup_payload(data_dir: Path | str | None = None) -> dict:
    root = resolve_data_dir(data_dir)
    return {
        "dataset_available": False,
        "source": "MISSING",
        "data_dir": str(root),
        "expected_files": [TXN_FILENAME, ID_FILENAME],
        "setup_message": SETUP_MESSAGE,
        "label": "OFFLINE PUBLIC DATASET EVALUATION",
        "note": "The IEEE-CIS experiment is an offline public-dataset evaluation. It does not represent production payment-fraud performance.",
    }


def _is_transaction_coded_numeric(col: str) -> bool:
    """True for IEEE C/D/V numbered fields (C1, D15, V339). Not DeviceType/DeviceInfo/card*."""
    if len(col) < 2 or col[0] not in {"C", "D", "V"}:
        return False
    return col[1:].isdigit()


def column_dtype_map(columns: list[str]) -> dict[str, str]:
    """Explicit, auditable dtype map. Never force identity categoricals to float32."""
    dtypes: dict[str, str] = {}
    categorical = set(IDENTITY_CATEGORICAL_COLUMNS)
    numeric_identity = set(IDENTITY_NUMERIC_COLUMNS)
    for col in columns:
        if col == JOIN_KEY:
            dtypes[col] = "int32"
        elif col == TARGET_COLUMN:
            dtypes[col] = "int8"
        elif col == TIME_COLUMN:
            dtypes[col] = "int32"
        elif col in numeric_identity or col in TRANSACTION_FLOAT32_COLUMNS or _is_transaction_coded_numeric(col):
            dtypes[col] = "float32"
        elif col in categorical or col.startswith("id_"):
            # Remaining id_* (id_12–id_38 and any unexpected id_*) stay strings.
            # NotFound/Found/New/Unknown must not be coerced to numbers.
            dtypes[col] = "object"
    return dtypes


def _efficient_dtypes(columns: list[str]) -> dict[str, str]:
    return column_dtype_map(columns)


def _read_csv(path: Path, max_rows: int | None, usecols: list[str] | None = None) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    available = list(header.columns)
    if usecols:
        cols = [c for c in usecols if c in available]
        extra = [c for c in available if c not in cols]
        # Keep extra columns in audit by loading them unless the file is huge.
        if max_rows is None or max_rows > 50_000:
            cols = available
        else:
            cols = list(dict.fromkeys(cols + extra[:40]))
    else:
        cols = available
    dtypes = _efficient_dtypes(cols)
    return pd.read_csv(
        path,
        usecols=lambda c: c in set(cols),
        dtype={k: v for k, v in dtypes.items() if k in cols},
        nrows=max_rows,
        low_memory=True,
    )


def load_tables(
    data_dir: Path | str | None = None,
    max_rows: int | None = None,
    transaction: pd.DataFrame | None = None,
    identity: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if transaction is not None and identity is not None:
        meta = {
            "source": "IN_MEMORY",
            "dataset_available": True,
            "max_rows": max_rows,
            "transaction_path": None,
            "identity_path": None,
        }
        txn, ident = transaction.copy(), identity.copy()
        if max_rows is not None:
            txn = txn.sort_values(TIME_COLUMN, kind="mergesort").head(max_rows)
        return txn, ident, meta

    root = resolve_data_dir(data_dir)
    txn_path, id_path = root / TXN_FILENAME, root / ID_FILENAME
    if not txn_path.is_file() or not id_path.is_file():
        raise IeeeDatasetError(SETUP_MESSAGE)

    env_max = os.environ.get("IEEE_MAX_ROWS")
    if max_rows is None and env_max:
        max_rows = int(env_max)

    txn = _read_csv(txn_path, max_rows, TRANSACTION_CORE)
    ident = _read_csv(id_path, None if max_rows is None else max_rows * 2, IDENTITY_CORE)
    meta = {
        "source": "IEEE_CIS_CSV",
        "dataset_available": True,
        "data_dir": str(root),
        "transaction_path": str(txn_path),
        "identity_path": str(id_path),
        "max_rows": max_rows,
        "transaction_bytes": txn_path.stat().st_size,
        "identity_bytes": id_path.stat().st_size,
    }
    return txn, ident, meta


def _classify_columns(df: pd.DataFrame, identity_cols: set[str]) -> dict:
    numeric, categorical, time_related, identity = [], [], [], []
    for col in df.columns:
        if col in identity_cols:
            identity.append(col)
        name_l = col.lower()
        if col == TIME_COLUMN or "time" in name_l or col.startswith("D") and col[1:].isdigit():
            time_related.append(col)
        if pd.api.types.is_numeric_dtype(df[col]) and col not in {JOIN_KEY, TARGET_COLUMN}:
            numeric.append(col)
        elif col not in {JOIN_KEY, TARGET_COLUMN}:
            categorical.append(col)
    return {
        "numerical_columns": numeric,
        "categorical_columns": categorical,
        "identity_columns": identity,
        "timestamp_columns": time_related,
    }


def audit_table(df: pd.DataFrame, name: str, identity_cols: set[str] | None = None) -> dict:
    missing = {}
    for col in df.columns:
        pct = float(df[col].isna().mean() * 100.0)
        missing[col] = round(pct, 4)
    dup_rows = int(df.duplicated().sum())
    dup_ids = int(df[JOIN_KEY].duplicated().sum()) if JOIN_KEY in df.columns else None
    y = None
    target = None
    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN].astype("float")
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        other = int((~y.isin([0, 1])).sum())
        target = {
            "positive": pos,
            "negative": neg,
            "other_or_null": other,
            "prevalence": float(pos / len(df)) if len(df) else None,
        }
    kinds = _classify_columns(df, identity_cols or set())
    mem = int(df.memory_usage(deep=True).sum())
    return {
        "name": name,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "target_distribution": target,
        "missing_value_percentages": missing,
        "duplicate_rows": dup_rows,
        "duplicate_transaction_ids": dup_ids,
        "memory_usage_bytes": mem,
        **kinds,
    }


def join_transaction_identity(txn: pd.DataFrame, ident: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if JOIN_KEY not in txn.columns or JOIN_KEY not in ident.columns:
        raise IeeeDatasetError(f"Join requires {JOIN_KEY} on both tables.")
    if TARGET_COLUMN in ident.columns:
        ident = ident.drop(columns=[TARGET_COLUMN])
    n_txn, n_ident = int(len(txn)), int(len(ident))
    txn_dup = int(txn[JOIN_KEY].duplicated().sum())
    ident_dup = int(ident[JOIN_KEY].duplicated().sum())
    ident_key_counts = ident[JOIN_KEY].value_counts()
    one_to_many = int((ident_key_counts > 1).sum())
    if ident_dup:
        ident = ident.drop_duplicates(JOIN_KEY, keep="first")
    txn_ids = set(txn[JOIN_KEY].tolist())
    ident_ids = set(ident[JOIN_KEY].tolist())
    unmatched_identity = int(len(ident_ids - txn_ids))
    matched = int(len(txn_ids & ident_ids))
    ident_cols = [c for c in ident.columns if c != JOIN_KEY]
    joined = txn.merge(ident, on=JOIN_KEY, how="left", validate="m:1" if ident_dup == 0 and txn_dup == 0 else None)
    joined["identity_present"] = joined[JOIN_KEY].isin(ident_ids).astype(np.int8)
    if TARGET_COLUMN not in txn.columns:
        raise IeeeDatasetError("Target isFraud must come from the transaction table.")
    n_after = int(len(joined))
    coverage = float(joined["identity_present"].mean()) if n_after else 0.0
    doc = {
        "join_key": JOIN_KEY,
        "how": "left join identity onto transaction",
        "target_source": "transaction table only",
        "join_not_on_target": True,
        "n_transaction_before": n_txn,
        "n_identity_before": n_ident,
        "n_after_join": n_after,
        "row_count_unchanged": n_after == n_txn,
        "matched_transaction_ids": matched,
        "unmatched_identity_rows": unmatched_identity,
        "duplicate_transaction_keys": txn_dup,
        "duplicate_identity_keys": ident_dup,
        "identity_keys_with_one_to_many": one_to_many,
        "identity_coverage": coverage,
        "identity_columns_added": ident_cols,
        "notes": [
            "Official IEEE-CIS key is TransactionID.",
            "Identity is optional; unmatched transactions keep identity_present=0.",
            "Duplicate identity keys keep the first row after being counted.",
        ],
    }
    if n_after != n_txn and one_to_many == 0:
        raise IeeeDatasetError(f"Join changed row count unexpectedly: {n_txn} -> {n_after}")
    return joined, doc


def write_audit(payload: dict, eval_dir: Path | None = None) -> dict:
    out = Path(eval_dir or EVAL_DIR)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "ieee_data_audit.json"
    md_path = out / "ieee_data_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    md_path.write_text(_audit_markdown(payload))
    return {"json": str(json_path), "md": str(md_path)}


def _audit_markdown(payload: dict) -> str:
    txn = payload.get("transaction") or {}
    ident = payload.get("identity") or {}
    joined = payload.get("joined") or {}
    join = payload.get("join") or {}
    missing = joined.get("missing_value_percentages") or {}
    top_missing = sorted(missing.items(), key=lambda kv: kv[1], reverse=True)[:20]
    lines = [
        "# IEEE-CIS data audit",
        "",
        "The IEEE-CIS experiment is an offline public-dataset evaluation. It does not represent production payment-fraud performance.",
        "",
        f"- Source: `{payload.get('source')}`",
        f"- Dataset available: `{payload.get('dataset_available')}`",
        f"- Label: OFFLINE PUBLIC DATASET EVALUATION",
        "",
        "## Transaction table",
        f"- Rows: {txn.get('n_rows')}",
        f"- Columns: {txn.get('n_columns')}",
        f"- Duplicate rows: {txn.get('duplicate_rows')}",
        f"- Duplicate TransactionIDs: {txn.get('duplicate_transaction_ids')}",
        f"- Memory bytes: {txn.get('memory_usage_bytes')}",
        f"- Target: {txn.get('target_distribution')}",
        "",
        "## Identity table",
        f"- Rows: {ident.get('n_rows')}",
        f"- Columns: {ident.get('n_columns')}",
        f"- Duplicate TransactionIDs: {ident.get('duplicate_transaction_ids')}",
        "",
        "## Join",
        f"- Key: `{join.get('join_key')}` (left join identity onto transaction)",
        f"- Rows before (txn): {join.get('n_transaction_before')}",
        f"- Rows after: {join.get('n_after_join')}",
        f"- Unmatched identity rows: {join.get('unmatched_identity_rows')}",
        f"- Identity coverage: {join.get('identity_coverage')}",
        f"- One-to-many identity keys: {join.get('identity_keys_with_one_to_many')}",
        "",
        "## Joined frame",
        f"- Rows: {joined.get('n_rows')}",
        f"- Columns: {joined.get('n_columns')}",
        f"- Numerical columns: {len(joined.get('numerical_columns') or [])}",
        f"- Categorical columns: {len(joined.get('categorical_columns') or [])}",
        f"- Identity columns: {len(joined.get('identity_columns') or [])}",
        f"- Time-related columns: {joined.get('timestamp_columns')}",
        "",
        "## Highest missingness (joined)",
    ]
    for col, pct in top_missing:
        lines.append(f"- `{col}`: {pct}%")
    if payload.get("setup_message"):
        lines += ["", "## Setup", payload["setup_message"]]
    lines += [
        "",
        "Suspicious or unused columns are listed in the leakage report rather than dropped silently.",
    ]
    return "\n".join(lines) + "\n"
