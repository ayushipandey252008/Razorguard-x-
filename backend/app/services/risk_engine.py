from __future__ import annotations

from app.config import get_settings


def combine_scores(
    ml_score: float,
    behavior_score: float,
    rule_score: float,
    graph_score: float,
    triggered_count: int,
) -> dict:
    """Weighted sum of normalized 0–100 component scores.

    THRESHOLD_* are environment-configurable. A cost experiment on synthetic
    validation probabilities lives in ml/evaluation/threshold_calibration.md
    and is not an industry operating point.
    """
    settings = get_settings()
    w = settings.risk_weights
    total_w = sum(w.values()) or 1.0
    final = (
        w["ml"] * ml_score
        + w["behavior"] * behavior_score
        + w["rules"] * rule_score
        + w["graph"] * graph_score
    ) / total_w
    final = round(float(min(100.0, max(0.0, final))), 2)

    if final >= settings.threshold_block:
        decision = "BLOCK"
    elif final >= settings.threshold_review:
        decision = "REVIEW"
    else:
        decision = "APPROVE"

    # Confidence: agreement among components + distance from nearest threshold.
    components = [ml_score, behavior_score, rule_score, graph_score]
    mean = sum(components) / 4
    variance = sum((c - mean) ** 2 for c in components) / 4
    agreement = max(0.0, 1.0 - variance / 2500.0)
    if decision == "APPROVE":
        margin = (settings.threshold_review - final) / settings.threshold_review
    elif decision == "BLOCK":
        margin = (final - settings.threshold_block) / (100 - settings.threshold_block)
    else:
        band = settings.threshold_block - settings.threshold_review
        dist = min(final - settings.threshold_review, settings.threshold_block - final)
        margin = dist / band if band else 0
    confidence = round(float(min(0.95, max(0.35, 0.45 * agreement + 0.45 * max(margin, 0) + 0.1))), 3)
    if triggered_count >= 4:
        confidence = min(0.95, confidence + 0.05)

    return {
        "final_risk_score": final,
        "decision": decision,
        "confidence": confidence,
        "weights": w,
        "thresholds": {
            "review": settings.threshold_review,
            "block": settings.threshold_block,
            "note": "Configurable THRESHOLD_* on 0–100 risk. Not industry-standard.",
        },
    }


def human_explanation(
    final: float,
    decision: str,
    shap: list[dict],
    rules: list,
    anomalies: list[dict],
    graph_evidence: dict,
) -> str:
    parts = [f"Risk score: {final:.0f} → {decision} (prototype thresholds)."]
    factors: list[str] = []
    for r in rules[:5]:
        factors.append(r.explanation if hasattr(r, "explanation") else str(r))
    for a in anomalies[:4]:
        factors.append(a.get("description") or a.get("code"))
    connected = graph_evidence.get("connected_users") or []
    if connected:
        factors.append(f"{len(connected)} connected account(s) via shared device/IP")
    if shap:
        top = shap[0]
        if top.get("contribution", 0) > 0:
            factors.append(f"model feature '{top['feature']}' increased fraud probability")
    if factors:
        parts.append("Contributing factors: " + "; ".join(factors[:6]) + ".")
    else:
        parts.append("No elevated rule, behavior, or graph signals were recorded.")
    parts.append("All factors above are derived from backend scores and tool evidence.")
    return " ".join(parts)
