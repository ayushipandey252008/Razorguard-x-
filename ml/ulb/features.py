from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ml.ulb.constants import DERIVED_COLUMNS, PCA_COLUMNS, RAW_FEATURE_COLUMNS, SCALE_COLUMNS


class ULBFeatureTransformer:
    """Train-only imputer + scaler. Derived columns are row-wise (no learned params).

    V1–V28 are already PCA components from the dataset publisher; they are
    passed through. Time/Amount/derived numeric columns are median-imputed and
    standardized using **training** statistics only.
    """

    def __init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_names_: list[str] = []
        self.fitted_ = False

    @staticmethod
    def add_derived(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        amount = pd.to_numeric(out["Amount"], errors="coerce").clip(lower=0)
        time = pd.to_numeric(out["Time"], errors="coerce")
        out["log_amount"] = np.log1p(amount.fillna(0))
        out["time_of_day_proxy"] = time % 86400.0
        out["transaction_time_bucket"] = np.floor(time / 3600.0)
        return out

    def fit(self, train: pd.DataFrame) -> "ULBFeatureTransformer":
        raw = train[RAW_FEATURE_COLUMNS]
        self.imputer.fit(raw)
        imputed = pd.DataFrame(self.imputer.transform(raw), columns=RAW_FEATURE_COLUMNS, index=train.index)
        derived = self.add_derived(imputed)
        self.scaler.fit(derived[SCALE_COLUMNS])
        self.feature_names_ = list(PCA_COLUMNS) + list(SCALE_COLUMNS)
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("ULBFeatureTransformer must be fit on training data first")
        raw = df[RAW_FEATURE_COLUMNS]
        imputed = pd.DataFrame(self.imputer.transform(raw), columns=RAW_FEATURE_COLUMNS, index=df.index)
        derived = self.add_derived(imputed)
        scaled = self.scaler.transform(derived[SCALE_COLUMNS])
        pca = derived[PCA_COLUMNS].to_numpy(dtype=float)
        return np.hstack([pca, scaled])

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        return self.fit(train).transform(train)
