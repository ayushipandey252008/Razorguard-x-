"""Probability calibration fitted on validation scores only.

Never pass chronological test labels into `fit_calibrators`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

EPS = 1e-15


def clip_proba(p) -> np.ndarray:
    arr = np.asarray(p, dtype=float).reshape(-1)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def _log_loss(y_true, y_prob) -> float:
    y = np.asarray(y_true).astype(int)
    p = np.clip(clip_proba(y_prob), EPS, 1.0 - EPS)
    return float(log_loss(y, p, labels=[0, 1]))


def reliability_bins(y_true, y_prob, n_bins: int = 10, strategy: str = "uniform") -> dict:
    """Mean predicted probability vs empirical positive rate per bin.

    On extreme class imbalance most uniform bins are empty or near-zero.
    Quantile bins are reported alongside; neither is a license to refit on test.
    """
    y = np.asarray(y_true).astype(int)
    p = clip_proba(y_prob)
    n = len(y)
    if n == 0:
        return {"strategy": strategy, "n_bins": n_bins, "ece": None, "bins": []}

    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(np.round(edges, 12))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins = []
    ece = 0.0
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == len(edges) - 2:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin": i,
                    "lo": lo,
                    "hi": hi,
                    "count": 0,
                    "n_positive": 0,
                    "mean_predicted": None,
                    "empirical_positive_rate": None,
                    "gap": None,
                }
            )
            continue
        mean_pred = float(p[mask].mean())
        emp = float(y[mask].mean())
        gap = abs(emp - mean_pred)
        ece += (count / n) * gap
        bins.append(
            {
                "bin": i,
                "lo": lo,
                "hi": hi,
                "count": count,
                "n_positive": int(y[mask].sum()),
                "mean_predicted": mean_pred,
                "empirical_positive_rate": emp,
                "gap": gap,
            }
        )
    return {"strategy": strategy, "n_bins": len(edges) - 1, "ece": float(ece), "bins": bins}


def calibration_diagnostics(y_true, y_prob, label: str) -> dict:
    y = np.asarray(y_true).astype(int)
    p = clip_proba(y_prob)
    uniform = reliability_bins(y, p, n_bins=10, strategy="uniform")
    quantile = reliability_bins(y, p, n_bins=10, strategy="quantile")
    unique = int(np.unique(np.round(p, 12)).size)
    return {
        "label": label,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": _log_loss(y, p),
        "ece_uniform_10": uniform["ece"],
        "ece_quantile_10": quantile["ece"],
        "mean_predicted": float(p.mean()),
        "empirical_prevalence": float(y.mean()) if len(y) else None,
        "min_predicted": float(p.min()) if len(p) else None,
        "max_predicted": float(p.max()) if len(p) else None,
        "n_unique_predictions": unique,
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "reliability_uniform": uniform,
        "reliability_quantile": quantile,
        "within_unit_interval": bool(np.all((p >= 0.0) & (p <= 1.0))),
    }


@dataclass
class FittedCalibrators:
    sigmoid: LogisticRegression
    isotonic: IsotonicRegression
    fit_n: int
    fit_n_positive: int
    test_labels_used: bool = False

    def transform(self, raw_prob, method: str) -> np.ndarray:
        raw = clip_proba(raw_prob)
        if method == "raw":
            return raw
        if method == "sigmoid":
            return clip_proba(self.sigmoid.predict_proba(raw.reshape(-1, 1))[:, 1])
        if method == "isotonic":
            return clip_proba(self.isotonic.predict(raw))
        raise ValueError(f"Unknown calibration method: {method}")


def fit_calibrators(raw_val, y_val) -> FittedCalibrators:
    """Fit Platt (sigmoid) and isotonic maps on validation scores.

    Parameters are deliberately validation-only. Do not pass test arrays.
    """
    raw = clip_proba(raw_val)
    y = np.asarray(y_val).astype(int)
    if raw.shape[0] != y.shape[0]:
        raise ValueError("raw_val and y_val length mismatch")
    if y.min() < 0 or y.max() > 1:
        raise ValueError("y_val must be binary 0/1")
    if len(np.unique(y)) < 2:
        raise ValueError("validation labels must contain both classes to fit a calibrator")

    sigmoid = LogisticRegression(solver="lbfgs", max_iter=1000)
    sigmoid.fit(raw.reshape(-1, 1), y)

    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    isotonic.fit(raw, y)

    return FittedCalibrators(
        sigmoid=sigmoid,
        isotonic=isotonic,
        fit_n=int(len(y)),
        fit_n_positive=int(y.sum()),
        test_labels_used=False,
    )


def select_calibration_method(val_diagnostics: dict) -> dict:
    """Choose among raw / sigmoid / isotonic using validation Brier, then ECE.

    Does not look at test diagnostics.
    """
    order = ["raw", "sigmoid", "isotonic"]
    rows = []
    for name in order:
        d = val_diagnostics[name]
        rows.append(
            {
                "method": name,
                "brier": d["brier"],
                "log_loss": d["log_loss"],
                "ece_uniform_10": d["ece_uniform_10"],
                "ece_quantile_10": d["ece_quantile_10"],
                "n_unique_predictions": d["n_unique_predictions"],
            }
        )
    ranked = sorted(
        rows,
        key=lambda r: (
            r["brier"],
            r["ece_uniform_10"] if r["ece_uniform_10"] is not None else 1.0,
            r["log_loss"],
        ),
    )
    selected = ranked[0]["method"]
    notes = [
        "Selection uses validation Brier first, then uniform-bin ECE, then log loss.",
        "Test labels are not used for selection.",
        "On ~0.12% prevalence, Brier is dominated by negatives; ECE bins are sparse.",
        "Isotonic and Platt are both fit on validation, so validation Brier/ECE give isotonic an in-sample advantage. Test Brier is the confirmatory check.",
    ]
    iso = val_diagnostics["isotonic"]
    if selected == "isotonic" and iso["n_unique_predictions"] < 20:
        notes.append(
            f"Isotonic produced {iso['n_unique_predictions']} distinct probabilities. "
            "That stepwise map can improve Brier while compressing ranking (PR-AUC)."
        )
    return {
        "selected_method": selected,
        "ranking": ranked,
        "justification": (
            f"Lowest validation Brier is {selected} "
            f"(Brier={ranked[0]['brier']:.8f}, ECE_uniform={ranked[0]['ece_uniform_10']})."
        ),
        "notes": notes,
    }


def reliability_svg(bins: list[dict], title: str) -> str:
    """Tiny SVG reliability diagram — no matplotlib dependency."""
    w, h, pad = 420, 320, 48
    inner_w, inner_h = w - 2 * pad, h - 2 * pad

    def xy(pred: float, emp: float) -> tuple[float, float]:
        x = pad + pred * inner_w
        y = pad + (1.0 - emp) * inner_h
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#0b1118"/>',
        f'<text x="{w/2}" y="22" fill="#94a3b8" font-size="12" text-anchor="middle">{title}</text>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#334155"/>',
        f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#334155"/>',
    ]
    x0, y0 = xy(0, 0)
    x1, y1 = xy(1, 1)
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="#475569" stroke-dasharray="4 4"/>')
    for b in bins:
        if b.get("mean_predicted") is None:
            continue
        x, y = xy(float(b["mean_predicted"]), float(b["empirical_positive_rate"]))
        r = 3 + min(8, np.sqrt(max(b["count"], 1)) / 40)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="#3ee0c6" fill-opacity="0.85"/>')
    parts.append(
        f'<text x="{w/2}" y="{h-12}" fill="#64748b" font-size="10" text-anchor="middle">'
        "mean predicted probability →</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)
