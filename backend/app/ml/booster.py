"""Gradient booster loader.

Prefers XGBoost. On macOS, XGBoost needs libomp; we preload it when present.
If XGBoost still cannot load, HistGradientBoostingClassifier is used and the
model version is labeled accordingly. SHAP TreeExplainer works for both.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

_LIBOMP_CANDIDATES = [
    Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
    Path("/usr/local/opt/libomp/lib/libomp.dylib"),
]


def preload_openmp() -> None:
    for path in _LIBOMP_CANDIDATES:
        if path.exists():
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            return


def make_classifier(scale_pos_weight: float, seed: int):
    preload_openmp()
    try:
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=2,
            verbosity=0,
        )
        return clf, "xgboost", f"xgb-iforest-v1"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.08,
            max_iter=120,
            class_weight="balanced",
            random_state=seed,
        )
        return clf, "hist_gbm", "hgb-iforest-v1"
