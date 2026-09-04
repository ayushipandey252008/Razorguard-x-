"""Persist IEEE-CIS evaluation results. Does not train models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.ml.ieee.constants import (
    EVAL_DIR,
    LIVE_MODEL_DIR,
    LIVE_MODEL_VERSION,
    ULB_METRICS_PATH,
    ULB_MODEL_DIR,
)

DISCLAIMER = (
    "The IEEE-CIS experiment is an offline public-dataset evaluation. "
    "It does not represent production payment-fraud performance."
)
NOT_AVAILABLE = "not available"
RESULTS_JSON = "ieee_results.json"
RESULTS_MD = "ieee_results.md"

_PROTECTED = [
    LIVE_MODEL_DIR / "version.txt",
    LIVE_MODEL_DIR / "xgb_fraud.joblib",
    LIVE_MODEL_DIR / "calibrator.joblib",
    LIVE_MODEL_DIR / "iforest.joblib",
    ULB_METRICS_PATH,
    EVAL_DIR / "calibration_metrics.json",
    EVAL_DIR / "calibration_report.md",
    EVAL_DIR / "calibration_robustness.json",
    EVAL_DIR / "calibration_robustness_report.md",
    ULB_MODEL_DIR / "model.joblib",
    ULB_MODEL_DIR / "metrics.json",
    ULB_MODEL_DIR / "probability_calibrator.joblib",
]


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integrity_hashes() -> dict:
    live_version = None
    vf = LIVE_MODEL_DIR / "version.txt"
    if vf.is_file():
        live_version = vf.read_text().strip()
    hashes = {str(p): _sha256(p) for p in _PROTECTED}
    return {
        "live_version": live_version or LIVE_MODEL_VERSION,
        "live_matches_expected": live_version == LIVE_MODEL_VERSION if live_version else None,
        "hashes_sha256": hashes,
    }


def _diag_summary(diag: dict | None) -> dict | str:
    if not isinstance(diag, dict):
        return NOT_AVAILABLE
    keys = (
        "label",
        "brier",
        "log_loss",
        "ece_uniform_10",
        "ece_quantile_10",
        "mean_predicted",
        "empirical_prevalence",
        "n_unique_predictions",
        "n_samples",
        "n_positive",
        "pr_auc",
        "roc_auc",
    )
    out = {k: diag[k] if k in diag else NOT_AVAILABLE for k in keys}
    return out


def _file_stat(path_str: str | None) -> dict:
    if not path_str:
        return {"path": None, "exists": False, "bytes": NOT_AVAILABLE, "sha256": NOT_AVAILABLE}
    p = Path(path_str)
    if not p.is_file():
        return {"path": path_str, "exists": False, "bytes": NOT_AVAILABLE, "sha256": NOT_AVAILABLE}
    return {"path": path_str, "exists": True, "bytes": int(p.stat().st_size), "sha256": _sha256(p)}


def build_ieee_results(
    manifest: dict,
    audit: dict | None = None,
    leakage: dict | None = None,
) -> dict:
    """Assemble the operator-facing results document from already-written artifacts."""
    audit = audit or {}
    leakage = leakage or {}
    source = manifest.get("source")
    available = bool(manifest.get("dataset_available"))
    official = source == "IEEE_CIS_CSV" and available and manifest.get("max_rows") is None
    limited = manifest.get("max_rows") is not None
    txn = audit.get("transaction") or {}
    ident = audit.get("identity") or {}
    joined = audit.get("joined") or {}
    join = manifest.get("join") or audit.get("join") or {}
    target = txn.get("target_distribution") or {}
    split = manifest.get("split")
    shap = manifest.get("shap")
    cal = manifest.get("calibration") or {}
    selection = cal.get("selection") or {}
    val_diag = cal.get("validation") or {}
    test_diag = cal.get("test_once") or {}

    if source in {None, "MISSING"} or (not available and source != "SYNTHETIC_FIXTURE_NOT_IEEE_CIS"):
        status = "STOPPED"
        official = False
    elif source == "IEEE_CIS_CSV" and official:
        status = "COMPLETED"
    elif source == "IEEE_CIS_CSV" and limited:
        status = "LIMITED-SAMPLE / NOT OFFICIAL IEEE-CIS RESULT"
        official = False
    else:
        status = "COMPLETED_NOT_OFFICIAL_IEEE_CIS"
        official = False

    missing = joined.get("missing_value_percentages") or txn.get("missing_value_percentages") or {}
    top_missing = sorted(missing.items(), key=lambda kv: kv[1], reverse=True)[:15] if missing else NOT_AVAILABLE

    shap_out: dict | str
    if not isinstance(shap, dict):
        shap_out = NOT_AVAILABLE
    else:
        shap_out = {
            "available": shap.get("available"),
            "reason": shap.get("reason", NOT_AVAILABLE) if not shap.get("available") else None,
            "n_rows_explained": shap.get("n_rows_explained", NOT_AVAILABLE),
            "global_feature_importance": (shap.get("global_feature_importance") or [])[:15],
            "importance_by_family": shap.get("importance_by_family", NOT_AVAILABLE),
            "example_transaction": shap.get("example_transaction", NOT_AVAILABLE),
            "distinction": shap.get("distinction")
            or {
                "model_explanation": "SHAP attributes the model's score to input features.",
                "causal_explanation": "Not provided.",
            },
        }

    leak_checks = leakage.get("checks")
    excluded = leakage.get("excluded_features")
    return {
        "status": status,
        "official_ieee_cis_result": official,
        "limited_sample": bool(limited),
        "max_rows": manifest.get("max_rows"),
        "label": "OFFLINE PUBLIC DATASET EVALUATION",
        "disclaimer": DISCLAIMER,
        "source": source,
        "dataset_available": available,
        "stopped_at": None if status != "STOPPED" else manifest.get("stopped_at") or "VERIFY DATASET",
        "dataset": {
            "name": audit.get("dataset") or "IEEE-CIS Fraud Detection",
            "transaction": _file_stat(audit.get("transaction_path")),
            "identity": _file_stat(audit.get("identity_path")),
            "transaction_rows": txn.get("n_rows", NOT_AVAILABLE),
            "transaction_columns": txn.get("n_columns", NOT_AVAILABLE),
            "identity_rows": ident.get("n_rows", NOT_AVAILABLE),
            "identity_columns": ident.get("n_columns", NOT_AVAILABLE),
            "joined_rows": joined.get("n_rows", NOT_AVAILABLE),
            "joined_columns": joined.get("n_columns", NOT_AVAILABLE),
            "fraud_count": target.get("positive", NOT_AVAILABLE),
            "legitimate_count": target.get("negative", NOT_AVAILABLE),
            "fraud_prevalence": target.get("prevalence", NOT_AVAILABLE),
            "duplicate_transaction_rows": txn.get("duplicate_rows", NOT_AVAILABLE),
            "duplicate_transaction_ids": txn.get("duplicate_transaction_ids", NOT_AVAILABLE),
            "duplicate_identity_ids": ident.get("duplicate_transaction_ids", NOT_AVAILABLE),
            "numerical_columns": len(joined.get("numerical_columns") or txn.get("numerical_columns") or []),
            "categorical_columns": len(joined.get("categorical_columns") or txn.get("categorical_columns") or []),
            "time_range": {
                "field": "TransactionDT",
                "interpretation": "IEEE-CIS contest timedelta in seconds from a withheld reference, not a wall clock.",
                "joined_timestamp_columns": joined.get("timestamp_columns") or txn.get("timestamp_columns") or NOT_AVAILABLE,
                "split": {
                    "train_min": (split or {}).get("train", {}).get("time_min", NOT_AVAILABLE) if split else NOT_AVAILABLE,
                    "train_max": (split or {}).get("train", {}).get("time_max", NOT_AVAILABLE) if split else NOT_AVAILABLE,
                    "validation_min": (split or {}).get("validation", {}).get("time_min", NOT_AVAILABLE) if split else NOT_AVAILABLE,
                    "validation_max": (split or {}).get("validation", {}).get("time_max", NOT_AVAILABLE) if split else NOT_AVAILABLE,
                    "test_min": (split or {}).get("test", {}).get("time_min", NOT_AVAILABLE) if split else NOT_AVAILABLE,
                    "test_max": (split or {}).get("test", {}).get("time_max", NOT_AVAILABLE) if split else NOT_AVAILABLE,
                }
                if split
                else NOT_AVAILABLE,
            },
            "memory_usage_bytes": {
                "transaction": txn.get("memory_usage_bytes", NOT_AVAILABLE),
                "identity": ident.get("memory_usage_bytes", NOT_AVAILABLE),
                "joined": joined.get("memory_usage_bytes", NOT_AVAILABLE),
            },
            "top_missingness_percent": top_missing,
        },
        "join": join or NOT_AVAILABLE,
        "split": split if split else NOT_AVAILABLE,
        "split_configuration": manifest.get("split_configuration", NOT_AVAILABLE),
        "leakage": {
            "all_passed": leakage.get("all_passed", manifest.get("leakage_all_passed", NOT_AVAILABLE)),
            "checks": leak_checks if leak_checks is not None else NOT_AVAILABLE,
            "excluded_feature_count": len(excluded) if isinstance(excluded, dict) else NOT_AVAILABLE,
            "excluded_features": excluded if isinstance(excluded, dict) else NOT_AVAILABLE,
            "report": "ml/evaluation/ieee_leakage_report.json",
        },
        "feature_families": manifest.get("feature_configuration", NOT_AVAILABLE),
        "experiments_test": manifest.get("experiments_test", NOT_AVAILABLE),
        "experiments_validation": manifest.get("experiments_validation", NOT_AVAILABLE),
        "graph_ablation": manifest.get("graph_ablation", NOT_AVAILABLE),
        "calibration": {
            "selected_method": selection.get("selected_method", NOT_AVAILABLE),
            "justification": selection.get("justification", NOT_AVAILABLE),
            "notes": selection.get("notes", NOT_AVAILABLE),
            "ranking": selection.get("ranking", NOT_AVAILABLE),
            "validation": {
                "raw": _diag_summary(val_diag.get("raw")),
                "sigmoid": _diag_summary(val_diag.get("sigmoid")),
                "isotonic": _diag_summary(val_diag.get("isotonic")),
            }
            if val_diag
            else NOT_AVAILABLE,
            "test_once": {
                "raw": _diag_summary(test_diag.get("raw")),
                "calibrated": _diag_summary(test_diag.get("calibrated")),
            }
            if test_diag
            else NOT_AVAILABLE,
            "pr_auc_on_validation_calibrators": NOT_AVAILABLE,
            "note": "Validation calibrator PR-AUC/ROC-AUC were not stored in calibration_diagnostics. Frozen-test PR-AUC/ROC-AUC below use the selected calibrator applied once to TEST.",
        }
        if cal
        else NOT_AVAILABLE,
        "thresholds": manifest.get("thresholds", NOT_AVAILABLE),
        "policy_on_test": manifest.get("policy_on_test", NOT_AVAILABLE),
        "frozen_test_metrics": manifest.get("frozen_test_metrics", NOT_AVAILABLE),
        "shap": shap_out,
        "candidates": manifest.get("candidates") if manifest.get("candidates") is not None else NOT_AVAILABLE,
        "active_live_model": manifest.get("active_live_model") or LIVE_MODEL_VERSION,
        "ieee_status": manifest.get("ieee_status") or "OFFLINE CANDIDATE",
        "auto_activated": bool(manifest.get("auto_activated")),
        "model_probability_separate_from_final_risk_score": True,
        "cross_dataset": manifest.get("cross_dataset", NOT_AVAILABLE),
        "runtime": {
            "total_seconds": manifest.get("runtime_seconds", NOT_AVAILABLE),
            "audit_seconds": NOT_AVAILABLE,
            "join_seconds": NOT_AVAILABLE,
            "preprocessing_seconds": NOT_AVAILABLE,
            "graph_feature_seconds": NOT_AVAILABLE,
            "training_seconds": NOT_AVAILABLE,
            "inference_seconds": NOT_AVAILABLE,
            "calibration_seconds": NOT_AVAILABLE,
            "peak_memory": NOT_AVAILABLE,
            "note": "The pipeline recorded a single wall-clock total. Stage timers were not separately persisted.",
            "python_version": manifest.get("python_version", NOT_AVAILABLE),
            "platform": manifest.get("platform", NOT_AVAILABLE),
            "package_versions": manifest.get("package_versions", NOT_AVAILABLE),
            "trained_at": manifest.get("trained_at", NOT_AVAILABLE),
        },
        "integrity": integrity_hashes(),
        "limitations": [
            DISCLAIMER,
            "IEEE-CIS metrics are not comparable to ULB or live synthetic scores.",
            "model_probability is not the product final_risk_score and is not a production fraud probability.",
            "IEEE candidates stay CANDIDATE and are not auto-activated.",
            "Anonymized id_*/V* names are not business meanings unless contest documentation defines them.",
            "SHAP is a model explanation, not a causal explanation." if isinstance(shap, dict) and shap.get("available") else "SHAP was not available or not produced.",
        ],
    }


def results_markdown(payload: dict) -> str:
    def cell(v) -> str:
        if v is None or v == NOT_AVAILABLE:
            return "not available"
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    ds = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
    split = payload.get("split") if isinstance(payload.get("split"), dict) else {}
    leak = payload.get("leakage") or {}
    rows = payload.get("experiments_test") if isinstance(payload.get("experiments_test"), list) else []
    abl = payload.get("graph_ablation") if isinstance(payload.get("graph_ablation"), dict) else {}
    cal = payload.get("calibration") if isinstance(payload.get("calibration"), dict) else {}
    thr = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}
    frozen = payload.get("frozen_test_metrics") if isinstance(payload.get("frozen_test_metrics"), dict) else {}
    policy = payload.get("policy_on_test") if isinstance(payload.get("policy_on_test"), dict) else {}
    shap = payload.get("shap") if isinstance(payload.get("shap"), dict) else {}
    cross = (payload.get("cross_dataset") or {}).get("table") if isinstance(payload.get("cross_dataset"), dict) else []
    join = payload.get("join") if isinstance(payload.get("join"), dict) else {}
    rt = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    time_info = ds.get("time_range") if isinstance(ds.get("time_range"), dict) else {}
    split_times = time_info.get("split") if isinstance(time_info.get("split"), dict) else {}
    mem = ds.get("memory_usage_bytes") if isinstance(ds.get("memory_usage_bytes"), dict) else {}
    lines = [
        "# IEEE-CIS results",
        "",
        DISCLAIMER,
        "",
        f"**Status:** {payload.get('status')}",
        f"**Official IEEE-CIS result:** {payload.get('official_ieee_cis_result')}",
        f"**Label:** OFFLINE PUBLIC DATASET EVALUATION",
        f"**Source:** `{payload.get('source')}`",
        f"**Limited sample:** {payload.get('limited_sample')} (max_rows={payload.get('max_rows')})",
        f"**ACTIVE MODEL:** `{payload.get('active_live_model')}`",
        f"**IEEE-CIS:** {payload.get('ieee_status')} (auto_activated={payload.get('auto_activated')})",
        "",
        "## Dataset",
        "",
        f"- Transaction rows/columns: {ds.get('transaction_rows')} / {ds.get('transaction_columns')}",
        f"- Identity rows/columns: {ds.get('identity_rows')} / {ds.get('identity_columns')}",
        f"- Fraud / legitimate / prevalence: {ds.get('fraud_count')} / {ds.get('legitimate_count')} / {ds.get('fraud_prevalence')}",
        f"- Duplicate txn IDs: {ds.get('duplicate_transaction_ids')}; duplicate identity IDs: {ds.get('duplicate_identity_ids')}",
        f"- Identity join coverage: {join.get('identity_coverage', 'not available')}",
        f"- Unmatched identity rows: {join.get('unmatched_identity_rows', 'not available')}",
        f"- Numerical / categorical columns (joined): {ds.get('numerical_columns')} / {ds.get('categorical_columns')}",
        f"- TransactionDT range (train min → test max): {split_times.get('train_min', 'not available')} → {split_times.get('test_max', 'not available')}",
        f"- TransactionDT is a contest timedelta in seconds, not a wall clock.",
        f"- Memory bytes (txn / identity / joined): {mem.get('transaction', 'not available')} / {mem.get('identity', 'not available')} / {mem.get('joined', 'not available')}",
        "",
        "## Chronological split (70 / 15 / 15)",
        "",
        "| Split | Rows | Fraud | Prevalence | time_min | time_max |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("train", "validation", "test"):
        part = split.get(name) or {}
        lines.append(
            f"| {name} | {cell(part.get('n'))} | {cell(part.get('fraud'))} | {cell(part.get('prevalence'))} | {cell(part.get('time_min'))} | {cell(part.get('time_max'))} |"
        )
    cons = split.get("constraints") or {}
    lines += [
        "",
        f"- max(train) < min(validation): {cons.get('max_train_lt_min_validation', 'not available')}",
        f"- max(validation) < min(test): {cons.get('max_validation_lt_min_test', 'not available')}",
        "",
        "## Leakage",
        "",
        f"- All checks passed: {leak.get('all_passed')}",
        f"- Excluded features: {leak.get('excluded_feature_count')}",
    ]
    checks = leak.get("checks")
    if isinstance(checks, list):
        for c in checks:
            mark = "PASS" if c.get("passed") else "FAIL"
            lines.append(f"- [{mark}] `{c.get('id')}` — {c.get('detail')}")
    else:
        lines.append("- Checks: not available")
    lines += [
        "",
        "## Experiments (frozen chronological TEST, threshold 0.5, uncalibrated scores)",
        "",
        "| Experiment | Features | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    if rows:
        for r in rows:
            lines.append(
                f"| {r.get('Experiment')} | {r.get('Features')} | {cell(r.get('PR-AUC'))} | {cell(r.get('ROC-AUC'))} | {cell(r.get('Precision'))} | {cell(r.get('Recall'))} | {cell(r.get('F1'))} | {cell(r.get('FPR'))} |"
            )
    else:
        lines.append("| not available | not available | not available | not available | not available | not available | not available | not available |")
    wg = abl.get("with_graph") or {}
    ng = abl.get("without_graph") or {}
    improved = abl.get("improved") or {}
    lines += [
        "",
        "## Graph ablation (combined families, TEST @ 0.5, uncalibrated)",
        "",
        f"- Without graph PR-AUC / recall / precision / F1 / FPR: {cell(ng.get('pr_auc'))} / {cell(ng.get('recall'))} / {cell(ng.get('precision'))} / {cell(ng.get('f1'))} / {cell(ng.get('false_positive_rate'))}",
        f"- With graph PR-AUC / recall / precision / F1 / FPR: {cell(wg.get('pr_auc'))} / {cell(wg.get('recall'))} / {cell(wg.get('precision'))} / {cell(wg.get('f1'))} / {cell(wg.get('false_positive_rate'))}",
        f"- Improved PR-AUC: {improved.get('pr_auc', 'not available')}; recall: {improved.get('recall', 'not available')}; precision: {improved.get('precision', 'not available')}; F1: {improved.get('f1', 'not available')}; FPR: {improved.get('false_positive_rate', 'not available')}",
        "",
        "## Calibration (fit on validation only)",
        "",
        f"- Selected: {cal.get('selected_method', 'not available')}",
        f"- Justification: {cal.get('justification', 'not available')}",
    ]
    ranking = cal.get("ranking")
    if isinstance(ranking, list):
        lines += ["", "| Method | Val Brier | Val log loss | Val ECE | unique preds |", "| --- | --- | --- | --- | --- |"]
        for row in ranking:
            lines.append(
                f"| {row.get('method')} | {cell(row.get('brier'))} | {cell(row.get('log_loss'))} | {cell(row.get('ece_uniform_10'))} | {cell(row.get('n_unique_predictions'))} |"
            )
    test_cal = (cal.get("test_once") or {}) if isinstance(cal.get("test_once"), dict) else {}
    raw_t = test_cal.get("raw") if isinstance(test_cal.get("raw"), dict) else {}
    cal_t = test_cal.get("calibrated") if isinstance(test_cal.get("calibrated"), dict) else {}
    lines += [
        "",
        f"- TEST raw Brier / log loss / ECE: {cell(raw_t.get('brier'))} / {cell(raw_t.get('log_loss'))} / {cell(raw_t.get('ece_uniform_10'))}",
        f"- TEST calibrated Brier / log loss / ECE: {cell(cal_t.get('brier'))} / {cell(cal_t.get('log_loss'))} / {cell(cal_t.get('ece_uniform_10'))}",
        f"- Validation calibrator PR-AUC/ROC-AUC: {cal.get('pr_auc_on_validation_calibrators', 'not available')}",
        "",
        "## Thresholds (validation-selected, applied once to TEST)",
        "",
        f"- APPROVE below: {thr.get('approve_below', 'not available')}",
        f"- REVIEW: {thr.get('review_from', 'not available')} to {thr.get('review_to', 'not available')}",
        f"- BLOCK above: {thr.get('block_above', 'not available')}",
        f"- Source: {thr.get('source', 'not available')}",
        f"- TEST policy counts: {policy.get('decisions', 'not available')}",
        f"- TEST fraud catch (REVIEW or BLOCK): {policy.get('fraud_catch_rate_review_or_block', 'not available')}",
        f"- model_probability is not final_risk_score.",
        "",
        "## Frozen chronological TEST (selected calibrator, threshold 0.5)",
        "",
        f"- PR-AUC: {cell(frozen.get('pr_auc'))}",
        f"- ROC-AUC: {cell(frozen.get('roc_auc'))}",
        f"- Precision / recall / F1: {cell(frozen.get('precision'))} / {cell(frozen.get('recall'))} / {cell(frozen.get('f1'))}",
        f"- FPR / FNR / catch rate: {cell(frozen.get('false_positive_rate'))} / {cell(frozen.get('false_negative_rate'))} / {cell(frozen.get('fraud_catch_rate'))}",
        f"- Confusion: {frozen.get('confusion_matrix', 'not available')}",
        f"- Prevalence: {cell(frozen.get('fraud_prevalence'))}",
        "",
        "## SHAP (model explanation, not causality)",
        "",
        f"- Available: {shap.get('available', 'not available')}",
        f"- Rows explained: {shap.get('n_rows_explained', 'not available')}",
    ]
    for item in (shap.get("global_feature_importance") or [])[:10]:
        lines.append(f"- `{item.get('feature')}`: {cell(item.get('mean_abs_shap'))}")
    if not shap.get("available"):
        lines.append(f"- {shap.get('reason', 'not available')}")
    lines += [
        "",
        "## Candidates",
        "",
        f"- Status: OFFLINE CANDIDATE. Live model unchanged: `{payload.get('active_live_model')}`.",
    ]
    cands = payload.get("candidates")
    if isinstance(cands, list) and cands:
        for c in cands:
            lines.append(
                f"- `{c.get('id')}` families={c.get('feature_families')} features={c.get('feature_count')} path=`{c.get('artifact_path')}`"
            )
    else:
        lines.append("- Candidates: not available")
    lines += ["", "## Cross-dataset (not equivalent)", "", "| Dataset | Rows | Fraud | Features | PR-AUC | ROC-AUC |", "| --- | --- | --- | --- | --- | --- |"]
    if cross:
        for row in cross:
            lines.append(
                f"| {row.get('Dataset')} | {cell(row.get('Rows'))} | {cell(row.get('Fraud'))} | {cell(row.get('Features'))} | {cell(row.get('PR-AUC'))} | {cell(row.get('ROC-AUC'))} |"
            )
    else:
        lines.append("| not available | not available | not available | not available | not available | not available |")
    lines += [
        "",
        "Different feature spaces, time periods, fraud prevalence, collection processes, entity information, and evaluation conditions. Metrics are not interchangeable.",
        "",
        "## Runtime",
        "",
        f"- Total seconds: {rt.get('total_seconds', 'not available')}",
        f"- Stage timers (audit/join/preprocess/graph/train/infer/calibrate): {rt.get('audit_seconds')} / {rt.get('join_seconds')} / {rt.get('preprocessing_seconds')} / {rt.get('graph_feature_seconds')} / {rt.get('training_seconds')} / {rt.get('inference_seconds')} / {rt.get('calibration_seconds')}",
        f"- Peak memory: {rt.get('peak_memory', 'not available')}",
        f"- Platform: {rt.get('platform', 'not available')}",
        "",
        "## Integrity",
        "",
        f"- Live version: `{(payload.get('integrity') or {}).get('live_version')}`",
        "",
        "## Limitations",
        "",
    ]
    for item in payload.get("limitations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def write_ieee_results(payload: dict, eval_dir: Path | None = None) -> dict:
    out = Path(eval_dir or EVAL_DIR)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / RESULTS_JSON
    md_path = out / RESULTS_MD
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    md_path.write_text(results_markdown(payload))
    return {"json": str(json_path), "md": str(md_path), "status": payload.get("status")}


def persist_ieee_results_from_eval_dir(eval_dir: Path | None = None) -> dict:
    """Rewrite ieee_results from an existing successful (or STOPPED) eval directory. No training."""
    root = Path(eval_dir or EVAL_DIR)
    manifest_path = root / "ieee_experiment_manifest.json"
    if not manifest_path.is_file():
        payload = {
            "status": "STOPPED",
            "official_ieee_cis_result": False,
            "label": "OFFLINE PUBLIC DATASET EVALUATION",
            "disclaimer": DISCLAIMER,
            "source": "MISSING",
            "dataset_available": False,
            "stopped_at": "VERIFY DATASET",
            "reason": "ieee_experiment_manifest.json is missing; cannot persist results.",
            "active_live_model": LIVE_MODEL_VERSION,
            "ieee_status": "OFFLINE CANDIDATE",
            "integrity": integrity_hashes(),
        }
        write_ieee_results(payload, root)
        return payload
    manifest = json.loads(manifest_path.read_text())
    audit = {}
    leak = {}
    ap = root / "ieee_data_audit.json"
    lp = root / "ieee_leakage_report.json"
    if ap.is_file():
        audit = json.loads(ap.read_text())
    if lp.is_file():
        leak = json.loads(lp.read_text())
    payload = build_ieee_results(manifest, audit, leak)
    write_ieee_results(payload, root)
    return payload
