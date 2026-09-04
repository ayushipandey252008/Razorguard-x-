from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.ulb.constants import PROCESSED_DIR, RAW_FEATURE_COLUMNS, TARGET_COLUMN
from ml.ulb.errors import DatasetValidationError
from ml.ulb.validate import validate_ulb_frame, write_validation_report


def clean_ulb_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Row-wise cleaning that does **not** learn parameters from the full dataset.

    Scaling / imputation parameters are fitted later on the training split only.
    Exact duplicates are dropped because they are identical on every column
    (including Time and Class) and would otherwise overweight those rows and
    contaminate a random split. Fraud outliers are **not** removed.
    """
    validate_ulb_frame(df, source="pre-clean")
    n_in = int(len(df))
    work = df[RAW_FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    for c in RAW_FEATURE_COLUMNS:
        work[c] = pd.to_numeric(work[c], errors="coerce")
        work[c] = work[c].replace([np.inf, -np.inf], np.nan)
    work[TARGET_COLUMN] = pd.to_numeric(work[TARGET_COLUMN], errors="coerce")
    if work[TARGET_COLUMN].isna().any() or (~work[TARGET_COLUMN].isin([0, 1])).any():
        raise DatasetValidationError("Class contains null or non-binary values after coercion")
    work[TARGET_COLUMN] = work[TARGET_COLUMN].astype(int)

    dup_mask = work.duplicated()
    n_dup = int(dup_mask.sum())
    cleaned = work.loc[~dup_mask].reset_index(drop=True)
    n_missing = int(cleaned[RAW_FEATURE_COLUMNS].isna().sum().sum())

    stats = {
        "rows_in": n_in,
        "exact_duplicates_removed": n_dup,
        "rows_out": int(len(cleaned)),
        "missing_cells_after_inf_to_nan": n_missing,
        "columns_kept": list(cleaned.columns),
        "notes": [
            "Exact duplicates dropped (identical Time, V1–V28, Amount, Class).",
            "Infinite feature values converted to NaN for train-only imputation later.",
            "No outlier clipping: fraud may itself be an outlier.",
            "No full-dataset scaling or imputation.",
            "Class labels preserved as 0/1.",
        ],
    }
    return cleaned, stats


def save_processed(df: pd.DataFrame, stats: dict, dest_dir: Path | None = None) -> Path:
    dest_dir = dest_dir or PROCESSED_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    parquet = dest_dir / "ulb_cleaned.parquet"
    csv = dest_dir / "ulb_cleaned.csv"
    try:
        df.to_parquet(parquet, index=False)
        out = parquet
    except Exception:
        df.to_csv(csv, index=False)
        out = csv
    (dest_dir / "cleaning_stats.json").write_text(json.dumps(stats, indent=2))
    write_validation_report(
        validate_ulb_frame(df, source=str(out)),
        dest_dir / "cleaned_validation.md",
    )
    return out
