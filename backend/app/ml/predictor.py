from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from app.config import get_settings
from app.ml.booster import preload_openmp
from app.ml.features import FEATURE_COLUMNS, features_to_array, row_to_features
from app.utils.logging import Timer, get_logger

log = get_logger("ml.predictor")


class FraudModelService:
    """Loads serialized XGBoost + Isolation Forest artifacts."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = Path(model_dir or get_settings().model_dir)
        self.xgb = None
        self.calibrator = None
        self.iforest = None
        self.version = "untrained"
        self.metrics: dict = {}
        self._explainer = None
        self.ready = False
        self.probability_calibrated = False

    def load_or_train(self) -> None:
        preload_openmp()
        xgb_path = self.model_dir / "xgb_fraud.joblib"
        if not xgb_path.exists():
            log.info("ml_artifacts_missing_training")
            from app.ml.train import train_and_save

            train_and_save(n_samples=4000, model_dir=self.model_dir)
        self.xgb = joblib.load(self.model_dir / "xgb_fraud.joblib")
        cal_path = self.model_dir / "calibrator.joblib"
        if cal_path.exists():
            self.calibrator = joblib.load(cal_path)
            self.probability_calibrated = True
        else:
            self.calibrator = None
            self.probability_calibrated = False
        self.iforest = joblib.load(self.model_dir / "iforest.joblib")
        version_file = self.model_dir / "version.txt"
        self.version = version_file.read_text().strip() if version_file.exists() else "unknown"
        metrics_file = self.model_dir / "metrics.json"
        if metrics_file.exists():
            self.metrics = json.loads(metrics_file.read_text())
        self.ready = True
        log.info("ml_loaded", version=self.version)

    def predict(self, transaction: dict) -> dict:
        if not self.ready:
            self.load_or_train()
        timer = Timer()
        feats = row_to_features(transaction)
        X = features_to_array(feats)
        raw = float(self.xgb.predict_proba(X)[0, 1])
        if self.calibrator is not None:
            if hasattr(self.calibrator, "predict_proba"):
                calibrated = float(self.calibrator.predict_proba(X)[0, 1])
            else:
                calibrated = float(np.clip(self.calibrator.predict([raw])[0], 0.0, 1.0))
        else:
            calibrated = raw
        ml_score = round(calibrated * 100.0, 2)
        shap_top = self._shap(X, feats)
        return {
            "ml_probability_raw": raw,
            "ml_probability": calibrated,
            "probability_calibrated": self.probability_calibrated,
            "ml_score": ml_score,
            "model_version": self.version,
            "shap_top_features": shap_top,
            "features": feats,
            "latency_ms": timer.ms(),
        }

    def isolation_score(self, transaction: dict) -> float:
        if not self.ready:
            self.load_or_train()
        feats = row_to_features(transaction)
        X = features_to_array(feats)
        raw = float(self.iforest.decision_function(X)[0])
        # decision_function: higher = more normal. Map to 0-100 risk.
        # Typical range roughly [-0.5, 0.5]; clip then invert.
        clipped = np.clip(-raw, -0.5, 0.5)
        return float(np.clip((clipped + 0.5) * 100.0, 0, 100))

    def _shap(self, X: np.ndarray, feats: dict) -> list[dict]:
        try:
            import shap

            if self._explainer is None:
                self._explainer = shap.TreeExplainer(self.xgb)
            values = self._explainer.shap_values(X)
            if isinstance(values, list):
                values = values[1]
            arr = np.array(values).reshape(-1)
            ranked = sorted(
                zip(FEATURE_COLUMNS, arr, [feats[c] for c in FEATURE_COLUMNS]),
                key=lambda t: abs(t[1]),
                reverse=True,
            )
            return [
                {"feature": name, "contribution": round(float(contrib), 4), "value": _json_num(val)}
                for name, contrib, val in ranked[:8]
            ]
        except Exception as exc:  # SHAP is best-effort; never fail a scoring request
            log.warning("shap_failed", error=str(exc))
            return []


def _json_num(val):
    if isinstance(val, (np.floating, float)):
        return round(float(val), 4)
    if isinstance(val, (np.integer, int)):
        return int(val)
    return val


model_service = FraudModelService()
