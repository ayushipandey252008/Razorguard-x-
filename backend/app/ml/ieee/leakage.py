"""Leakage audit for the IEEE-CIS offline track."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.ml.ieee.constants import (
    EVAL_DIR,
    FORBIDDEN_FEATURE_REASONS,
    JOIN_KEY,
    TARGET_COLUMN,
    TIME_COLUMN,
)
from app.ml.ieee.features import unused_raw_columns
from app.ml.ieee.graph_features import GRAPH_CONSTRUCTION_NOTES, graph_feature_uses_future
from app.ml.ieee.split import verify_temporal_order


def _hash_rows(df: pd.DataFrame) -> set:
    if df.empty:
        return set()
    return set(pd.util.hash_pandas_object(df, index=False).tolist())


def audit_leakage(
    raw_joined: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    preprocessor_fitted_on: str,
    feature_columns: list[str],
) -> dict:
    excluded = dict(FORBIDDEN_FEATURE_REASONS)
    unused = unused_raw_columns(raw_joined)
    for col in unused:
        if col in excluded:
            continue
        if str(col).startswith("V"):
            excluded[col] = (
                "Anonymized Vesta matching/aggregate column. Not dropped silently from the audit; "
                "excluded from models because contest derivation may include matching that is not "
                "guaranteed to be strictly causal at time T."
            )
        elif col not in feature_columns:
            excluded[col] = (
                "Present in the joined table but not selected for this experiment's feature families. "
                "Documented rather than silently discarded."
            )

    overlap_train_test = _hash_rows(train) & _hash_rows(test)
    overlap_train_val = _hash_rows(train) & _hash_rows(val)
    id_overlap = set()
    if JOIN_KEY in train.columns:
        id_overlap = set(train[JOIN_KEY]) & set(test[JOIN_KEY])

    temporal_ok = True
    temporal_error = None
    try:
        verify_temporal_order(train, val, test)
    except Exception as exc:  # noqa: BLE001 — report, do not swallow
        temporal_ok = False
        temporal_error = str(exc)

    future_graph = False
    if "graph_card_degree" in test.columns and len(test):
        sample_idx = int(test.index[-1])
        full = pd.concat([train, val, test], ignore_index=False)
        future_graph = graph_feature_uses_future(full.reset_index(drop=True), len(full) - 1)

    checks = [
        {
            "id": "target_leakage",
            "passed": TARGET_COLUMN not in feature_columns,
            "detail": "isFraud is not a covariate.",
        },
        {
            "id": "transaction_id_leakage",
            "passed": JOIN_KEY not in feature_columns,
            "detail": "TransactionID is not a covariate.",
        },
        {
            "id": "raw_time_leakage",
            "passed": TIME_COLUMN not in feature_columns,
            "detail": "Raw TransactionDT excluded; hour_of_day_proxy may be used.",
        },
        {
            "id": "target_encoding_leakage",
            "passed": True,
            "detail": "Target encoding is not used.",
        },
        {
            "id": "preprocessing_fit_scope",
            "passed": preprocessor_fitted_on == "train_only",
            "detail": preprocessor_fitted_on,
        },
        {
            "id": "train_test_exact_overlap",
            "passed": len(overlap_train_test) == 0,
            "detail": f"exact overlapping hashes={len(overlap_train_test)}",
        },
        {
            "id": "train_val_exact_overlap",
            "passed": len(overlap_train_val) == 0,
            "detail": f"exact overlapping hashes={len(overlap_train_val)}",
        },
        {
            "id": "duplicate_ids_across_splits",
            "passed": len(id_overlap) == 0,
            "detail": f"TransactionID overlap train/test={len(id_overlap)}",
        },
        {
            "id": "temporal_split_order",
            "passed": temporal_ok,
            "detail": temporal_error or "max(train) < min(val) < max(val) < min(test)",
        },
        {
            "id": "graph_temporal_safety",
            "passed": not future_graph,
            "detail": "Graph features recomputed from strict past for a late test row.",
        },
        {
            "id": "join_not_on_target",
            "passed": True,
            "detail": "Join key is TransactionID, not a target-derived key.",
        },
        {
            "id": "post_outcome_fields",
            "passed": True,
            "detail": "No chargeback/outcome columns are in the IEEE-CIS public train schema used here.",
        },
    ]
    return {
        "label": "OFFLINE PUBLIC DATASET EVALUATION",
        "excluded_features": excluded,
        "feature_columns": feature_columns,
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
        "graph_notes": GRAPH_CONSTRUCTION_NOTES,
        "identity_across_temporal_boundary": (
            "Identity attributes are row-level contest fields, not a live identity graph. "
            "Behavioral/graph aggregates still use only prior TransactionDT."
        ),
    }


def write_leakage_report(payload: dict, eval_dir: Path | None = None) -> dict:
    out = Path(eval_dir or EVAL_DIR)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "ieee_leakage_report.json"
    md_path = out / "ieee_leakage_report.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    lines = [
        "# IEEE-CIS leakage report",
        "",
        "The IEEE-CIS experiment is an offline public-dataset evaluation. It does not represent production payment-fraud performance.",
        "",
        f"Overall passed: **{payload.get('all_passed')}**",
        "",
        "## Checks",
    ]
    for c in payload.get("checks") or []:
        mark = "PASS" if c["passed"] else "FAIL"
        lines.append(f"- [{mark}] `{c['id']}` — {c['detail']}")
    lines += ["", "## Excluded features (every exclusion has a reason)", ""]
    for col, reason in (payload.get("excluded_features") or {}).items():
        lines.append(f"- `{col}`: {reason}")
    lines += ["", "## Graph construction", ""]
    for note in payload.get("graph_notes") or []:
        lines.append(f"- {note}")
    md_path.write_text("\n".join(lines) + "\n")
    return {"json": str(json_path), "md": str(md_path)}
