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


def make_ulb_classifier(scale_pos_weight: float, seed: int):
    """ULB-only classifier. Does not share artifacts with the synthetic product model."""
    preload_openmp()
    try:
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_weight=2,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=2,
            verbosity=0,
        )
        return clf, "xgboost", "ulb-xgb-v1"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.06,
            max_iter=160,
            class_weight="balanced",
            random_state=seed,
        )
        return clf, "hist_gbm", "ulb-hgb-v1"
