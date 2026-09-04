from __future__ import annotations

import numpy as np

from ml.ulb.constants import RANDOM_SEED


def shap_summary(model, X: np.ndarray, feature_names: list[str], y: np.ndarray | None = None) -> dict:
    """Global mean |SHAP| plus one local fraud and one local legit explanation.

    V1–V28 are anonymized PCA components. Local/global rankings are statistical
    attributions, not human-readable merchant or device meanings.
    """
    try:
        import shap
    except Exception as exc:
        return {"available": False, "reason": f"shap import failed: {exc}"}

    n = min(len(X), 800)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X), size=n, replace=False)
    Xs = X[idx]
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(Xs)
        if isinstance(values, list):
            values = values[1]
        arr = np.asarray(values)
        if arr.ndim == 3:
            arr = arr[:, :, 1]
        mean_abs = np.mean(np.abs(arr), axis=0)
        global_rank = [
            {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
            for i in np.argsort(mean_abs)[::-1]
        ]
        local = []
        if y is not None and len(y) == len(X):
            ys = np.asarray(y)[idx]
            for label, name in ((1, "fraud"), (0, "legitimate")):
                hits = np.where(ys == label)[0]
                if not len(hits):
                    continue
                j = int(hits[0])
                contrib = [
                    {"feature": feature_names[i], "shap": float(arr[j, i]), "value": _num(Xs[j, i])}
                    for i in np.argsort(np.abs(arr[j]))[::-1][:8]
                ]
                local.append({"label": name, "row_in_shap_sample": j, "top_features": contrib})
        return {
            "available": True,
            "n_explained": int(n),
            "global_mean_abs_shap": global_rank,
            "local_examples": local,
            "limitation": (
                "V1–V28 are anonymized PCA components from the dataset publisher. "
                "SHAP ranking does not recover a business feature name."
            ),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _num(v):
    try:
        return round(float(v), 6)
    except Exception:
        return None
