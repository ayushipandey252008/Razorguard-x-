from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.ulb.constants import RANDOM_SEED, TARGET_COLUMN


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split after sorting by Time so test rows occur later than train/val."""
    ordered = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    i_train = int(train_frac * n)
    i_val = int((train_frac + val_frac) * n)
    train = ordered.iloc[:i_train].reset_index(drop=True)
    val = ordered.iloc[i_train:i_val].reset_index(drop=True)
    test = ordered.iloc[i_val:].reset_index(drop=True)
    return train, val, test


def stratified_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Class-stratified random split. Can place later Time rows in train."""
    y = df[TARGET_COLUMN].astype(int)
    train, temp = train_test_split(
        df, test_size=(1.0 - train_frac), random_state=seed, stratify=y
    )
    rel_val = val_frac / (1.0 - train_frac)
    val, test = train_test_split(
        temp, test_size=(1.0 - rel_val), random_state=seed, stratify=temp[TARGET_COLUMN].astype(int)
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def split_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, strategy: str) -> dict:
    def _part(name: str, part: pd.DataFrame) -> dict:
        y = part[TARGET_COLUMN].astype(int)
        return {
            "name": name,
            "n": int(len(part)),
            "fraud": int(y.sum()),
            "legitimate": int((y == 0).sum()),
            "prevalence": float(y.mean()) if len(part) else None,
            "time_min": float(part["Time"].min()) if len(part) else None,
            "time_max": float(part["Time"].max()) if len(part) else None,
        }

    train_max = float(train["Time"].max()) if len(train) else None
    val_min = float(val["Time"].min()) if len(val) else None
    test_min = float(test["Time"].min()) if len(test) else None
    chronological_ok = True
    if train_max is not None and val_min is not None:
        chronological_ok = chronological_ok and val_min >= train_max
    if train_max is not None and test_min is not None:
        chronological_ok = chronological_ok and test_min >= train_max
    return {
        "strategy": strategy,
        "train": _part("train", train),
        "val": _part("val", val),
        "test": _part("test", test),
        "no_future_in_train": chronological_ok,
    }


def overlapping_row_count(a: pd.DataFrame, b: pd.DataFrame) -> int:
    """Count exact row overlaps (all columns) between two frames."""
    if a.empty or b.empty:
        return 0
    key_a = pd.util.hash_pandas_object(a, index=False)
    key_b = pd.util.hash_pandas_object(b, index=False)
    return int(len(set(key_a) & set(key_b)))
