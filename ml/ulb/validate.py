from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.ulb.constants import PCA_COLUMNS, REQUIRED_COLUMNS, TARGET_COLUMN
from ml.ulb.errors import DatasetValidationError


def validate_ulb_frame(df: pd.DataFrame, *, source: str = "dataframe") -> dict:
    """Inspect a ULB-schema frame. Raises DatasetValidationError on fatal issues."""
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    n = int(len(df))
    if n == 0:
        errors.append("Dataset is empty")

    duplicate_rows = int(df.duplicated().sum()) if n else 0
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} exact duplicate rows (cleaning may drop them)")

    missing_values = {c: int(df[c].isna().sum()) for c in df.columns} if n else {}
    missing_total = int(sum(missing_values.values()))

    inf_counts = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            inf_counts[c] = int(np.isinf(pd.to_numeric(df[c], errors="coerce")).sum())
    inf_total = int(sum(inf_counts.values()))
    if inf_total:
        warnings.append(f"{inf_total} infinite values (cleaning converts them to NaN)")

    class_counts = {}
    invalid_targets = 0
    if TARGET_COLUMN in df.columns:
        as_num = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        invalid_targets = int((~as_num.isin([0, 1])).sum())
        if invalid_targets:
            errors.append(f"Target Class must be 0 or 1; found {invalid_targets} invalid values")
        class_counts = {
            "legitimate": int((as_num == 0).sum()),
            "fraud": int((as_num == 1).sum()),
        }
        if class_counts.get("fraud", 0) == 0:
            errors.append("No fraud rows (Class=1)")
        if class_counts.get("legitimate", 0) == 0:
            errors.append("No legitimate rows (Class=0)")
    else:
        errors.append("Target column Class is missing")

    non_numeric = []
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.isna().all():
                non_numeric.append(c)
    if non_numeric:
        errors.append(f"Non-numeric columns: {non_numeric}")

    amount_stats = {}
    if "Amount" in df.columns and pd.api.types.is_numeric_dtype(df["Amount"]) and n:
        amt = pd.to_numeric(df["Amount"], errors="coerce")
        amount_stats = {
            "min": float(amt.min()) if amt.notna().any() else None,
            "max": float(amt.max()) if amt.notna().any() else None,
            "mean": float(amt.mean()) if amt.notna().any() else None,
            "median": float(amt.median()) if amt.notna().any() else None,
            "negative_count": int((amt < 0).sum()),
            "zero_count": int((amt == 0).sum()),
        }
        if amount_stats["negative_count"]:
            warnings.append(f"{amount_stats['negative_count']} negative Amount values")

    time_stats = {}
    if "Time" in df.columns and pd.api.types.is_numeric_dtype(df["Time"]) and n:
        t = pd.to_numeric(df["Time"], errors="coerce")
        time_stats = {
            "min": float(t.min()) if t.notna().any() else None,
            "max": float(t.max()) if t.notna().any() else None,
            "span_seconds": float(t.max() - t.min()) if t.notna().any() else None,
            "monotonic_non_decreasing": bool(t.is_monotonic_increasing) if t.notna().any() else False,
        }

    report = {
        "ok": not errors,
        "source": source,
        "n_rows": n,
        "n_columns": int(df.shape[1]),
        "required_columns": REQUIRED_COLUMNS,
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "duplicate_rows": duplicate_rows,
        "missing_values": missing_values,
        "missing_total": missing_total,
        "infinite_values": inf_counts,
        "infinite_total": inf_total,
        "class_counts": class_counts,
        "invalid_target_count": invalid_targets,
        "fraud_prevalence": (
            class_counts["fraud"] / n if n and class_counts.get("fraud") is not None else None
        ),
        "amount": amount_stats,
        "time": time_stats,
        "pca_columns_present": [c for c in PCA_COLUMNS if c in df.columns],
        "errors": errors,
        "warnings": warnings,
    }
    if errors:
        raise DatasetValidationError(
            "ULB dataset validation failed:\n" + "\n".join(f"- {e}" for e in errors)
        )
    return report


def write_validation_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cc = report.get("class_counts") or {}
    amt = report.get("amount") or {}
    tm = report.get("time") or {}
    lines = [
        "# ULB dataset validation",
        "",
        f"Source: `{report.get('source')}`",
        f"Rows: {report.get('n_rows')}  Columns: {report.get('n_columns')}",
        f"Status: {'PASS' if report.get('ok') else 'FAIL'}",
        "",
        "## Class",
        f"- Legitimate (0): {cc.get('legitimate')}",
        f"- Fraud (1): {cc.get('fraud')}",
        f"- Prevalence: {report.get('fraud_prevalence')}",
        "",
        "## Quality",
        f"- Exact duplicate rows: {report.get('duplicate_rows')}",
        f"- Missing values: {report.get('missing_total')}",
        f"- Infinite values: {report.get('infinite_total')}",
        "",
        "## Amount",
        f"- min={amt.get('min')} max={amt.get('max')} mean={amt.get('mean')} median={amt.get('median')}",
        f"- zeros={amt.get('zero_count')} negatives={amt.get('negative_count')}",
        "",
        "## Time",
        f"- min={tm.get('min')} max={tm.get('max')} span_seconds={tm.get('span_seconds')}",
        f"- already sorted: {tm.get('monotonic_non_decreasing')}",
        "",
        "## Warnings",
    ]
    for w in report.get("warnings") or ["none"]:
        lines.append(f"- {w}")
    if report.get("errors"):
        lines += ["", "## Errors"]
        lines += [f"- {e}" for e in report["errors"]]
    path.write_text("\n".join(lines) + "\n")
