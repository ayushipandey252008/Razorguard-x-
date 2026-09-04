"""Prototype PSI drift detector. Thresholds are not production standards."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.risk import RiskAssessment
from app.models.transaction import Transaction
from app.utils.logging import get_logger

log = get_logger("ml.drift")

PSI_LOW = 0.10
PSI_HIGH = 0.25
MIN_SAMPLES = 20

DRIFT_FEATURES = (
    "amount",
    "transaction_velocity",
    "hour_of_day",
    "current_device_known",
    "current_location_known",
    "ml_score",
    "final_risk_score",
)


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """Transparent PSI. Equal-width bins from the reference range, with a leftover bucket."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 2 or len(cur) < 2:
        return {
            "psi": None,
            "status": "insufficient",
            "bins": [],
            "reason": "need at least 2 finite values in each window",
        }
    unique = np.unique(ref)
    if len(unique) == 1:
        # Binary / constant: two buckets — equal vs not.
        ref_eq = float(np.mean(ref == unique[0]))
        cur_eq = float(np.mean(cur == unique[0]))
        psi = _psi_pair(ref_eq, cur_eq) + _psi_pair(1 - ref_eq, 1 - cur_eq)
        return {
            "psi": round(float(psi), 4),
            "status": psi_status(psi),
            "bins": [
                {"label": f"eq_{unique[0]}", "reference": round(ref_eq, 4), "current": round(cur_eq, 4)},
                {"label": "other", "reference": round(1 - ref_eq, 4), "current": round(1 - cur_eq, 4)},
            ],
        }
    lo, hi = float(np.min(ref)), float(np.max(ref))
    if lo == hi:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    ref_counts, _ = np.histogram(np.clip(ref, lo, hi), bins=edges)
    cur_counts, _ = np.histogram(np.clip(cur, lo, hi), bins=edges)
    ref_p = (ref_counts + 1e-6) / (ref_counts.sum() + 1e-6 * len(ref_counts))
    cur_p = (cur_counts + 1e-6) / (cur_counts.sum() + 1e-6 * len(cur_counts))
    psi = float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
    bucket = []
    for i in range(len(ref_p)):
        bucket.append(
            {
                "label": f"{edges[i]:.3g}-{edges[i + 1]:.3g}",
                "reference": round(float(ref_p[i]), 4),
                "current": round(float(cur_p[i]), 4),
            }
        )
    return {"psi": round(psi, 4), "status": psi_status(psi), "bins": bucket}


def _psi_pair(e: float, a: float) -> float:
    e = max(float(e), 1e-6)
    a = max(float(a), 1e-6)
    return (a - e) * np.log(a / e)


def psi_status(psi: float | None, low: float = PSI_LOW, high: float = PSI_HIGH) -> str:
    if psi is None:
        return "insufficient"
    if psi < low:
        return "LOW"
    if psi <= high:
        return "MODERATE"
    return "HIGH"


def overall_status(feature_rows: list[dict[str, Any]]) -> str:
    statuses = [r.get("status") for r in feature_rows]
    if statuses and all(s == "insufficient" for s in statuses):
        return "insufficient"
    if any(s == "HIGH" for s in statuses):
        return "drift"
    if any(s == "MODERATE" for s in statuses):
        return "warning"
    if any(s == "insufficient" for s in statuses) and not any(s in {"LOW", "MODERATE", "HIGH"} for s in statuses):
        return "insufficient"
    return "stable"


def recommendation_for(status: str) -> str:
    if status in {"warning", "drift"}:
        return "Review drift and evaluate retraining."
    if status == "insufficient":
        return "Collect more scored transactions before interpreting drift."
    return "No action required."


def _feature_value(txn: Transaction, risk: RiskAssessment | None, name: str) -> float | None:
    ts = _as_utc(txn.timestamp)
    if name == "amount":
        return float(txn.amount)
    if name == "transaction_velocity":
        return float(txn.transaction_velocity)
    if name == "hour_of_day":
        return float(ts.hour) if ts else None
    if name == "current_device_known":
        return float(int(bool(txn.current_device_known)))
    if name == "current_location_known":
        return float(int(bool(txn.current_location_known)))
    if name == "ml_score":
        return float(risk.ml_score) if risk is not None else None
    if name == "final_risk_score":
        return float(risk.final_risk_score) if risk is not None else None
    return None


def compute_drift_from_rows(
    rows: list[tuple[Transaction, RiskAssessment | None]],
    *,
    min_samples: int | None = None,
    psi_low: float | None = None,
    psi_high: float | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    min_samples = min_samples if min_samples is not None else int(getattr(settings, "drift_min_samples", MIN_SAMPLES))
    psi_low = psi_low if psi_low is not None else float(getattr(settings, "drift_psi_low", PSI_LOW))
    psi_high = psi_high if psi_high is not None else float(getattr(settings, "drift_psi_high", PSI_HIGH))

    ordered = sorted(rows, key=lambda pair: _as_utc(pair[0].timestamp) or datetime.min.replace(tzinfo=timezone.utc))
    n = len(ordered)
    if n < min_samples:
        return {
            "status": "insufficient",
            "features": [],
            "reference_window": {"n": 0, "note": "too few scored transactions"},
            "current_window": {"n": n},
            "recommendation": recommendation_for("insufficient"),
            "psi_thresholds": {"low": psi_low, "high": psi_high, "note": "Prototype thresholds, not production standards."},
            "n": n,
        }

    cut = max(1, n // 2)
    reference, current = ordered[:cut], ordered[cut:]
    ref_start = _as_utc(reference[0][0].timestamp)
    ref_end = _as_utc(reference[-1][0].timestamp)
    cur_start = _as_utc(current[0][0].timestamp)
    cur_end = _as_utc(current[-1][0].timestamp)

    features = []
    for name in DRIFT_FEATURES:
        ref_vals = np.array([v for v in (_feature_value(t, r, name) for t, r in reference) if v is not None])
        cur_vals = np.array([v for v in (_feature_value(t, r, name) for t, r in current) if v is not None])
        psi_row = population_stability_index(ref_vals, cur_vals)
        status = psi_status(psi_row.get("psi"), psi_low, psi_high) if psi_row.get("psi") is not None else "insufficient"
        features.append(
            {
                "feature": name,
                "reference_distribution": psi_row.get("bins") or [],
                "current_distribution": psi_row.get("bins") or [],
                "drift_score": psi_row.get("psi"),
                "status": status if psi_row.get("psi") is not None else "insufficient",
            }
        )

    status = overall_status(features)
    return {
        "status": status,
        "features": features,
        "reference_window": {
            "n": len(reference),
            "start": ref_start.isoformat() if ref_start else None,
            "end": ref_end.isoformat() if ref_end else None,
            "rule": "earlier half of scored transactions by timestamp",
        },
        "current_window": {
            "n": len(current),
            "start": cur_start.isoformat() if cur_start else None,
            "end": cur_end.isoformat() if cur_end else None,
            "rule": "later half of scored transactions by timestamp",
        },
        "recommendation": recommendation_for(status),
        "psi_thresholds": {
            "low": psi_low,
            "high": psi_high,
            "interpretation": "<0.10 LOW, 0.10-0.25 MODERATE, >0.25 HIGH",
            "note": "Prototype thresholds, not production standards.",
        },
        "n": n,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


async def compute_drift(db: AsyncSession, **kwargs) -> dict[str, Any]:
    rows = (
        await db.execute(
            select(Transaction, RiskAssessment)
            .join(RiskAssessment, RiskAssessment.transaction_id == Transaction.transaction_id, isouter=True)
        )
    ).all()
    return compute_drift_from_rows([(t, r) for t, r in rows], **kwargs)


def should_emit_alert(
    status: str,
    last_alert_at: datetime | None,
    *,
    cooldown_seconds: int | None = None,
) -> bool:
    if status not in {"warning", "drift"}:
        return False
    settings = get_settings()
    cooldown = cooldown_seconds if cooldown_seconds is not None else int(
        getattr(settings, "drift_alert_cooldown_seconds", 3600)
    )
    if last_alert_at is None:
        return True
    last = _as_utc(last_alert_at)
    now = datetime.now(timezone.utc)
    return (now - last) >= timedelta(seconds=max(0, cooldown))
