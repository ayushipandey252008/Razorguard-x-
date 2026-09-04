"""Metrics for SYNTHETIC SCENARIO EVALUATION. Not public-dataset metrics."""

from __future__ import annotations

from typing import Any


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def classification_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    y = [int(r["expected_fraud"]) for r in rows]
    flagged = [1 if r["decision"] in {"BLOCK", "REVIEW"} else 0 for r in rows]
    blocked = [1 if r["decision"] == "BLOCK" else 0 for r in rows]
    tp = sum(1 for a, b in zip(y, flagged) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, flagged) if a == 0 and b == 1)
    tn = sum(1 for a, b in zip(y, flagged) if a == 0 and b == 0)
    fn = sum(1 for a, b in zip(y, flagged) if a == 1 and b == 0)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": len(rows),
        "n_fraud": sum(y),
        "n_legit": len(y) - sum(y),
        "blocked": sum(blocked),
        "reviewed": sum(1 for r in rows if r["decision"] == "REVIEW"),
        "approved": sum(1 for r in rows if r["decision"] == "APPROVE"),
        "flagged": sum(flagged),
    }


def overall_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c = classification_counts(rows)
    precision = _safe_div(c["tp"], c["tp"] + c["fp"])
    recall = _safe_div(c["tp"], c["tp"] + c["fn"])
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return {
        **c,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(_safe_div(c["fp"], c["fp"] + c["tn"]), 4),
        "fraud_catch_rate": round(recall, 4),
        "block_rate": round(_safe_div(c["blocked"], c["n"]), 4),
        "review_rate": round(_safe_div(c["reviewed"], c["n"]), 4),
        "approve_rate": round(_safe_div(c["approved"], c["n"]), 4),
        "label": "SYNTHETIC SCENARIO EVALUATION",
        "note": (
            "Catch rate treats BLOCK+REVIEW as detected vs generator labels. "
            "This is not real-world fraud accuracy and is not ULB evaluation."
        ),
    }


def scenario_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by.setdefault(row["scenario"], []).append(row)
    matrix = []
    for name, group in by.items():
        fraud = sum(int(r["expected_fraud"]) for r in group)
        block = sum(1 for r in group if r["decision"] == "BLOCK")
        review = sum(1 for r in group if r["decision"] == "REVIEW")
        approve = sum(1 for r in group if r["decision"] == "APPROVE")
        catch = _safe_div(sum(1 for r in group if r["expected_fraud"] and r["decision"] != "APPROVE"), fraud)
        matrix.append(
            {
                "scenario": name,
                "n": len(group),
                "fraud": fraud,
                "block": block,
                "review": review,
                "approve": approve,
                "catch": round(catch, 4),
            }
        )
    matrix.sort(key=lambda r: r["scenario"])
    return matrix
