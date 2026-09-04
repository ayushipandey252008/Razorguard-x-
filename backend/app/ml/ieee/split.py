"""Chronological IEEE-CIS split. Random stratified split is not official evaluation."""

from __future__ import annotations

import pandas as pd

from app.ml.ieee.constants import TARGET_COLUMN, TEST_FRAC, TIME_COLUMN, TRAIN_FRAC, VAL_FRAC
from app.ml.ieee.errors import IeeeDatasetError


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise IeeeDatasetError("Split fractions must sum to 1.")
    if TIME_COLUMN not in df.columns:
        raise IeeeDatasetError(f"Split requires {TIME_COLUMN}.")
    ordered = df.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    i_train = _snap_cut(ordered, int(train_frac * n))
    i_val = _snap_cut(ordered, int((train_frac + val_frac) * n))
    if not (0 < i_train < i_val < n):
        raise IeeeDatasetError(
            f"Cannot form a strict chronological 3-way split (cuts {i_train}, {i_val}, n={n})."
        )
    train = ordered.iloc[:i_train].reset_index(drop=True)
    val = ordered.iloc[i_train:i_val].reset_index(drop=True)
    test = ordered.iloc[i_val:].reset_index(drop=True)
    verify_temporal_order(train, val, test)
    return train, val, test


def _snap_cut(ordered: pd.DataFrame, i: int) -> int:
    """Move the cut forward so it sits after a timestamp boundary (strict < between splits)."""
    n = len(ordered)
    i = min(max(i, 1), n - 1)
    t = int(ordered.iloc[i - 1][TIME_COLUMN])
    while i < n and int(ordered.iloc[i][TIME_COLUMN]) <= t:
        i += 1
    return i


def verify_temporal_order(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    if train.empty or val.empty or test.empty:
        raise IeeeDatasetError("Each of TRAIN/VALIDATION/TEST must be non-empty.")
    tmax = int(train[TIME_COLUMN].max())
    vmin = int(val[TIME_COLUMN].min())
    vmax = int(val[TIME_COLUMN].max())
    tmin = int(test[TIME_COLUMN].min())
    if not (tmax < vmin):
        raise IeeeDatasetError(f"Temporal leak: max(train_time)={tmax} is not < min(validation_time)={vmin}")
    if not (vmax < tmin):
        raise IeeeDatasetError(f"Temporal leak: max(validation_time)={vmax} is not < min(test_time)={tmin}")


def split_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    def _part(name: str, part: pd.DataFrame) -> dict:
        y = part[TARGET_COLUMN].astype(int)
        return {
            "name": name,
            "n": int(len(part)),
            "fraud": int(y.sum()),
            "legitimate": int((y == 0).sum()),
            "prevalence": float(y.mean()) if len(part) else None,
            "time_min": int(part[TIME_COLUMN].min()),
            "time_max": int(part[TIME_COLUMN].max()),
        }

    verify_temporal_order(train, val, test)
    return {
        "strategy": "chronological",
        "official_evaluation_split": "chronological",
        "random_stratified_used_as_official": False,
        "train": _part("train", train),
        "validation": _part("validation", val),
        "test": _part("test", test),
        "constraints": {
            "max_train_lt_min_validation": True,
            "max_validation_lt_min_test": True,
        },
        "preprocessing_fit_on": "train_only",
    }
