"""Threshold sweeps and three-way APPROVE / REVIEW / BLOCK on calibrated probabilities.

Optimization uses validation predictions only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from ml.ulb.calibration import clip_proba
from ml.ulb.constants import COST_SCENARIOS, DEFAULT_COST_SCENARIO, PROTOTYPE_CAPACITY_CAP


VALID_DECISIONS = ("APPROVE", "REVIEW", "BLOCK")


@dataclass(frozen=True)
class CostConfig:
    false_positive_cost: float
    false_negative_cost: float
    review_cost: float
    scenario_id: str = "custom"
    label: str = "custom"

    @classmethod
    def from_scenario(cls, scenario_id: str) -> "CostConfig":
        row = COST_SCENARIOS[scenario_id]
        return cls(
            false_positive_cost=float(row["false_positive_cost"]),
            false_negative_cost=float(row["false_negative_cost"]),
            review_cost=float(row["review_cost"]),
            scenario_id=str(row["id"]),
            label=str(row["label"]),
        )


def assert_thresholds_ordered(t_review: float, t_block: float) -> None:
    if not (0.0 <= t_review < t_block <= 1.0):
        raise ValueError(
            f"Thresholds must satisfy 0 <= T_REVIEW < T_BLOCK <= 1; got {t_review}, {t_block}"
        )


def three_way_decision(p, t_review: float, t_block: float) -> np.ndarray:
    """Map calibrated probability to APPROVE / REVIEW / BLOCK."""
    assert_thresholds_ordered(t_review, t_block)
    prob = clip_proba(p)
    out = np.full(prob.shape, "REVIEW", dtype=object)
    out[prob < t_review] = "APPROVE"
    out[prob >= t_block] = "BLOCK"
    return out


def binary_threshold_grid() -> np.ndarray:
    coarse = np.round(np.arange(0.01, 0.995, 0.01), 4)
    fine_low = np.round(np.arange(0.001, 0.0505, 0.001), 4)
    fine_high = np.round(np.arange(0.90, 0.9995, 0.001), 4)
    return np.unique(np.concatenate([coarse, fine_low, fine_high, np.array([0.01, 0.99])]))


def three_way_grid() -> np.ndarray:
    """0.01 steps across (0,1). Fine enough for a prototype; dense enough near 0/1 via binary table."""
    return np.round(np.arange(0.01, 1.00, 0.01), 4)


def _confusion_at_cutoff(y_true, y_prob, threshold: float) -> dict:
    y = np.asarray(y_true).astype(int)
    pred = (clip_proba(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": fpr,
        "fnr": fnr,
        "flag_rate": float(pred.mean()),
    }


def sweep_binary_thresholds(y_true, y_prob) -> list[dict]:
    return [_confusion_at_cutoff(y_true, y_prob, t) for t in binary_threshold_grid()]


def expected_cost_three_way(y_true, y_prob, t_review: float, t_block: float, costs: CostConfig) -> dict:
    assert_thresholds_ordered(t_review, t_block)
    y = np.asarray(y_true).astype(int)
    p = clip_proba(y_prob)
    approve = p < t_review
    block = p >= t_block
    review = ~approve & ~block
    fraud = y == 1
    legit = y == 0
    n = len(y)
    total = (
        int((approve & fraud).sum()) * costs.false_negative_cost
        + int((block & legit).sum()) * costs.false_positive_cost
        + int(review.sum()) * costs.review_cost
    )
    caught = int(((review | block) & fraud).sum())
    n_block = int(block.sum())
    return {
        "t_review": float(t_review),
        "t_block": float(t_block),
        "expected_cost_per_txn": float(total / n) if n else 0.0,
        "total_cost": float(total),
        "approve_rate": float(approve.mean()),
        "review_rate": float(review.mean()),
        "block_rate": float(block.mean()),
        "non_approve_rate": float((~approve).mean()),
        "n_approve": int(approve.sum()),
        "n_review": int(review.sum()),
        "n_block": n_block,
        "fn_approved_fraud": int((approve & fraud).sum()),
        "fp_blocked_legit": int((block & legit).sum()),
        "fraud_caught": caught,
        "fraud_catch_rate": float(caught / fraud.sum()) if fraud.sum() else 0.0,
        "block_precision": float((block & fraud).sum() / n_block) if n_block else None,
    }


def optimize_three_way(
    y_true,
    y_prob,
    costs: CostConfig,
    capacity_cap: float | None = None,
) -> dict:
    """Minimize expected cost on validation. Optional cap on review+block rate."""
    y = np.asarray(y_true).astype(int)
    p = clip_proba(y_prob)
    grid = three_way_grid()
    best = None
    evaluated = 0
    for t_review in grid:
        for t_block in grid:
            if t_block <= t_review:
                continue
            row = expected_cost_three_way(y, p, float(t_review), float(t_block), costs)
            evaluated += 1
            if capacity_cap is not None and row["non_approve_rate"] - 1e-12 > capacity_cap:
                continue
            if best is None or row["expected_cost_per_txn"] < best["expected_cost_per_txn"]:
                best = row
    if best is None:
        return {
            "feasible": False,
            "capacity_cap": capacity_cap,
            "evaluated_pairs": evaluated,
            "reason": "no (T_REVIEW, T_BLOCK) pair satisfied the capacity cap",
        }
    best["feasible"] = True
    best["capacity_cap"] = capacity_cap
    best["evaluated_pairs"] = evaluated
    best["cost_scenario"] = {
        "id": costs.scenario_id,
        "label": costs.label,
        "false_positive_cost": costs.false_positive_cost,
        "false_negative_cost": costs.false_negative_cost,
        "review_cost": costs.review_cost,
    }
    return best


def pick_prototype_operating_point(y_true, y_prob, scenario_id: str = DEFAULT_COST_SCENARIO) -> dict:
    """Unconstrained min-cost plus a capacity-capped prototype.

    If the unconstrained optimum reviews almost everything, that is reported rather than hidden.
    The documented prototype uses a 5% non-approve cap when a feasible pair exists.
    """
    costs = CostConfig.from_scenario(scenario_id)
    unconstrained = optimize_three_way(y_true, y_prob, costs, capacity_cap=None)
    capped = optimize_three_way(y_true, y_prob, costs, capacity_cap=PROTOTYPE_CAPACITY_CAP)
    impractical = unconstrained.get("non_approve_rate", 0) >= 0.25
    if capped.get("feasible"):
        selected = dict(capped)
        selected_source = (
            f"min expected cost on validation under non-approve rate <= {PROTOTYPE_CAPACITY_CAP:.0%}"
        )
    else:
        selected = dict(unconstrained)
        selected_source = "unconstrained min expected cost (capacity cap infeasible)"
    notes = []
    if impractical:
        notes.append(
            "Unconstrained optimum flags a large share of traffic because missed-fraud cost "
            f"({costs.false_negative_cost}) dwarfs review cost ({costs.review_cost}). "
            "That is a math result, not an operations policy."
        )
    if capped.get("feasible") and impractical:
        notes.append(
            f"Prototype therefore uses a {PROTOTYPE_CAPACITY_CAP:.0%} review+block capacity cap. "
            "This is a prototype constraint, not an industry-standard threshold."
        )
    if not capped.get("feasible"):
        notes.append("No pair met the 5% capacity cap; unconstrained point is documented instead.")
    return {
        "scenario_id": scenario_id,
        "costs": {
            "false_positive_cost": costs.false_positive_cost,
            "false_negative_cost": costs.false_negative_cost,
            "review_cost": costs.review_cost,
        },
        "unconstrained": unconstrained,
        "capacity_capped": capped,
        "selected": selected,
        "selected_source": selected_source,
        "prototype_label": "PROTOTYPE CALIBRATION",
        "not_industry_standard": True,
        "notes": notes,
    }
