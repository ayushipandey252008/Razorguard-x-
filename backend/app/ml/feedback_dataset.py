"""Build an offline labeled dataset from analyst feedback.

Only CONFIRM_FRAUD / CONFIRM_LEGITIMATE are training labels.
NEEDS_REVIEW is not treated as fraud. Scenario-evaluation tags are excluded.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import REPO_ROOT
from app.ml.features import FEATURE_COLUMNS, row_to_features
from app.models.feedback import (
    CONFIRM_FRAUD,
    DEFINED_OUTCOMES,
    NEEDS_REVIEW,
    OUTCOME_FRAUD,
    AnalystFeedback,
)
from app.models.transaction import Transaction
from app.utils.logging import get_logger
from app.utils.redact import redact_secrets

log = get_logger("ml.feedback_dataset")

EVAL_SCENARIO_PREFIX = "eval_"
DEFAULT_EXPORT_DIR = REPO_ROOT / "ml" / "data" / "feedback"


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def is_scenario_eval_tag(tag: str | None) -> bool:
    return bool(tag) and str(tag).startswith(EVAL_SCENARIO_PREFIX)


def label_from_outcome(outcome: str | None) -> int | None:
    if outcome == OUTCOME_FRAUD:
        return 1
    if outcome in DEFINED_OUTCOMES:
        return 0
    return None


def temporal_split(
    records: list[dict[str, Any]],
    *,
    eval_fraction: float = 0.3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by feedback timestamp. Eval is the later slice and must stay untouched."""
    ordered = sorted(records, key=lambda r: r["feedback_created_at"])
    if len(ordered) < 2:
        return ordered, []
    cut = max(1, int(round(len(ordered) * (1.0 - eval_fraction))))
    cut = min(cut, len(ordered) - 1)
    train, eval_rows = ordered[:cut], ordered[cut:]
    return train, eval_rows


def assert_no_future_leakage(train: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    if not train or not eval_rows:
        return
    max_train = max(_as_utc(r["feedback_created_at"]) for r in train)
    min_eval = min(_as_utc(r["feedback_created_at"]) for r in eval_rows)
    if max_train is None or min_eval is None:
        return
    if max_train > min_eval:
        raise ValueError("temporal split leaked: a training label is later than an eval label")


def validate_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    for row in records:
        if row.get("analyst_decision") == NEEDS_REVIEW:
            issues.append("NEEDS_REVIEW leaked into labeled set")
        if row.get("y") not in (0, 1):
            issues.append("undefined label")
        if is_scenario_eval_tag(row.get("scenario_tag")):
            issues.append("scenario evaluation label mixed into feedback dataset")
        if row.get("analyst_decision") == CONFIRM_FRAUD and row.get("y") != 1:
            issues.append("CONFIRM_FRAUD without y=1")
    labeled = [r for r in records if r.get("y") in (0, 1)]
    return {
        "n": len(records),
        "n_labeled": len(labeled),
        "n_positive": sum(1 for r in labeled if r["y"] == 1),
        "n_negative": sum(1 for r in labeled if r["y"] == 0),
        "issues": issues,
        "ok": not issues,
        "feature_columns": FEATURE_COLUMNS,
        "note": (
            "Labels are analyst-confirmed outcomes, not ULB Class. "
            "REVIEW/NEEDS_REVIEW is not fraud. Scenario-eval tags are excluded."
        ),
    }


def records_from_rows(
    pairs: list[tuple[AnalystFeedback, Transaction | None]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fb, txn in pairs:
        y = label_from_outcome(fb.actual_outcome)
        if y is None:
            continue
        tag = txn.scenario_tag if txn is not None else None
        if is_scenario_eval_tag(tag):
            continue
        raw = {
            "amount": txn.amount if txn else 0,
            "account_age_days": txn.account_age_days if txn else 0,
            "failed_attempts": txn.failed_attempts if txn else 0,
            "transaction_velocity": txn.transaction_velocity if txn else 1,
            "previous_transaction_count": txn.previous_transaction_count if txn else 0,
            "previous_average_amount": txn.previous_average_amount if txn else 0,
            "current_device_known": txn.current_device_known if txn else True,
            "current_location_known": txn.current_location_known if txn else True,
            "timestamp": txn.timestamp if txn else fb.created_at,
            "payment_method": txn.payment_method if txn else "UPI",
            "merchant_category": txn.merchant_category if txn else "GROCERY",
        }
        out.append(
            {
                "feedback_id": fb.feedback_id,
                "investigation_id": fb.investigation_id,
                "transaction_id": fb.transaction_id,
                "analyst_decision": fb.analyst_decision,
                "actual_outcome": fb.actual_outcome,
                "y": y,
                "model_prediction_decision": fb.decision_at_prediction_time,
                "model_version": fb.model_version,
                "risk_score": fb.risk_score,
                "ml_probability": fb.ml_probability,
                "feedback_created_at": fb.created_at,
                "scenario_tag": tag,
                "features": row_to_features(raw),
                "raw": raw,
            }
        )
    return out


async def load_feedback_records(db: AsyncSession) -> list[dict[str, Any]]:
    rows = (
        await db.execute(
            select(AnalystFeedback, Transaction)
            .join(Transaction, Transaction.transaction_id == AnalystFeedback.transaction_id, isouter=True)
            .order_by(AnalystFeedback.created_at.asc())
        )
    ).all()
    return records_from_rows([(fb, txn) for fb, txn in rows])


def export_dataset(
    records: list[dict[str, Any]],
    dest: Path | None = None,
    *,
    report: dict[str, Any] | None = None,
) -> Path:
    dest = dest or DEFAULT_EXPORT_DIR
    dest.mkdir(parents=True, exist_ok=True)
    payload = redact_secrets(
        {
            "track": "SYNTHETIC_FEEDBACK",
            "not_ulb": True,
            "not_real_world_fraud": True,
            "n": len(records),
            "records": [
                {
                    "feedback_id": r["feedback_id"],
                    "transaction_id": r["transaction_id"],
                    "analyst_decision": r["analyst_decision"],
                    "actual_outcome": r["actual_outcome"],
                    "y": r["y"],
                    "model_prediction_decision": r.get("model_prediction_decision"),
                    "model_version": r.get("model_version"),
                    "feedback_created_at": r["feedback_created_at"].isoformat()
                    if hasattr(r["feedback_created_at"], "isoformat")
                    else r["feedback_created_at"],
                    "features": r["features"],
                }
                for r in records
            ],
            "validation": report or validate_dataset(records),
        }
    )
    path = dest / "feedback_dataset.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("feedback_dataset_exported", path=str(path), n=len(records))
    return path
