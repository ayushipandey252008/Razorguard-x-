"""Leakage-safe IEEE-CIS preprocessing. Fit on TRAIN only."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.ml.ieee.features import CATEGORICAL_COLUMNS

MISSING_TOKEN = "__MISSING__"


@dataclass
class IeeePreprocessor:
    """Median + missingness indicators for numeric; frequency encoding for categoricals.

    Unseen categories map to 0 frequency. Zeros in the raw data are not treated as missing.
    Identity unavailability is the `identity_present` flag, not a zero-fill of id_* columns.
    """

    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    indicator_columns: list[str] = field(default_factory=list)
    medians: dict[str, float] = field(default_factory=dict)
    frequencies: dict[str, dict[str, float]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    fitted: bool = False
    encoding_strategy: str = (
        "Frequency encoding on train only for high-cardinality categoricals; "
        "unseen -> 0. No target encoding (would require nested temporal OOF)."
    )

    def fit(self, df: pd.DataFrame, feature_columns: list[str]) -> "IeeePreprocessor":
        cols = [c for c in feature_columns if c in df.columns]
        self.categorical_columns = [c for c in cols if c in CATEGORICAL_COLUMNS or df[c].dtype == object]
        self.numeric_columns = [c for c in cols if c not in self.categorical_columns]
        self.indicator_columns = []
        self.medians = {}
        self.frequencies = {}
        n = max(len(df), 1)
        for col in self.numeric_columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.isna().any():
                self.indicator_columns.append(f"{col}_is_missing")
            med = float(s.median()) if s.notna().any() else 0.0
            self.medians[col] = med
        for col in self.categorical_columns:
            raw = df[col].where(df[col].notna(), MISSING_TOKEN).astype(str)
            vc = raw.value_counts()
            self.frequencies[col] = {k: float(v / n) for k, v in vc.items()}
        self.feature_names = (
            list(self.numeric_columns)
            + [f"{c}_freq" for c in self.categorical_columns]
            + list(self.indicator_columns)
        )
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("IeeePreprocessor.transform requires fit() on train only.")
        parts = []
        for col in self.numeric_columns:
            s = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
            filled = s.fillna(self.medians[col]).astype(np.float32)
            parts.append(filled.to_numpy().reshape(-1, 1))
        for col in self.categorical_columns:
            if col in df.columns:
                raw = df[col].where(df[col].notna(), MISSING_TOKEN).astype(str)
            else:
                raw = pd.Series([MISSING_TOKEN] * len(df), index=df.index)
            mapped = raw.map(self.frequencies[col]).fillna(0.0).astype(np.float32)
            parts.append(mapped.to_numpy().reshape(-1, 1))
        for col in self.numeric_columns:
            name = f"{col}_is_missing"
            if name not in self.indicator_columns:
                continue
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                ind = s.isna().astype(np.float32).to_numpy().reshape(-1, 1)
            else:
                ind = np.ones((len(df), 1), dtype=np.float32)
            parts.append(ind)
        if not parts:
            return np.zeros((len(df), 0), dtype=np.float32)
        return np.hstack(parts).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
        return self.fit(df, feature_columns).transform(df)

    def to_meta(self) -> dict:
        return {
            "encoding_strategy": self.encoding_strategy,
            "target_encoding_used": False,
            "one_hot_used": False,
            "numeric_impute": "train_median",
            "zero_fill_all_missing": False,
            "missingness_indicators": self.indicator_columns,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "fitted_on": "train_only",
        }
