"""End-to-end IEEE-CIS offline pipeline. Isolated from live scoring and ULB."""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

import pandas as pd

from app.ml.ieee.adapter import (
    audit_table,
    ieee_files_present,
    join_transaction_identity,
    load_tables,
    resolve_data_dir,
    setup_payload,
    write_audit,
)
from app.ml.ieee.constants import (
    BASELINE_VERSION,
    COMBINED_VERSION,
    DATASET_ID,
    DATASET_NAME,
    EVAL_DIR,
    GRAPH_VERSION,
    IEEE_MODEL_DIR,
    LIVE_MODEL_VERSION,
    RANDOM_SEED,
    TARGET_COLUMN,
    TRACK,
    ULB_METRICS_PATH,
)
from app.ml.ieee.evaluate import (
    calibration_diagnostics,
    classification_metrics,
    fit_calibrators,
    policy_summary,
    select_calibration_method,
    select_three_way_thresholds,
    shap_summary,
)
from app.ml.ieee.features import EXPERIMENTS, add_behavioral_features, add_transaction_timing, columns_for_families
from app.ml.ieee.fixture import make_ieee_fixture
from app.ml.ieee.graph_features import GRAPH_CONSTRUCTION_NOTES, add_graph_features
from app.ml.ieee.leakage import audit_leakage, write_leakage_report
from app.ml.ieee.report import build_ieee_results, write_ieee_results
from app.ml.ieee.split import chronological_split, split_summary
from app.ml.ieee.train import (
    experiment_row,
    family_lookup,
    save_candidate,
    train_experiment,
    version_for_experiment,
)


def _pkg(name: str) -> str | None:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return None


def _cross_dataset(ieee_summary: dict | None, source: str) -> dict:
    ulb = None
    if ULB_METRICS_PATH.exists():
        ulb = json.loads(ULB_METRICS_PATH.read_text())
    ulb_half = (ulb or {}).get("metrics_at_half") or {}
    ulb_dist = (ulb or {}).get("class_distribution") or {}
    ulb_feats = (ulb or {}).get("feature_names") or (ulb or {}).get("feature_columns") or []
    rows = [
        {
            "Dataset": "ULB Credit Card Fraud Detection",
            "Rows": (ulb or {}).get("n_rows_cleaned") or ulb_half.get("n_samples"),
            "Fraud": ulb_dist.get("full_fraud") or (ulb or {}).get("n_fraud_cleaned") or ulb_half.get("n_fraud"),
            "Features": len(ulb_feats),
            "PR-AUC": (ulb or {}).get("pr_auc"),
            "ROC-AUC": (ulb or {}).get("roc_auc"),
            "source": "committed ulb_metrics.json" if ulb else "unavailable",
        }
    ]
    if ieee_summary and source == "IEEE_CIS_CSV":
        rows.append(
            {
                "Dataset": DATASET_NAME,
                "Rows": ieee_summary.get("n_rows"),
                "Fraud": ieee_summary.get("n_fraud"),
                "Features": ieee_summary.get("n_features"),
                "PR-AUC": ieee_summary.get("pr_auc"),
                "ROC-AUC": ieee_summary.get("roc_auc"),
                "source": "measured IEEE-CIS chronological test",
            }
        )
    else:
        rows.append(
            {
                "Dataset": DATASET_NAME,
                "Rows": None,
                "Fraud": None,
                "Features": None,
                "PR-AUC": None,
                "ROC-AUC": None,
                "source": "not reported — IEEE-CIS CSVs missing or fixture-only run",
            }
        )
    return {
        "label": "OFFLINE PUBLIC DATASET EVALUATION",
        "not_equivalent": True,
        "caveats": [
            "ULB and IEEE-CIS are different datasets, time periods, feature spaces, and prevalence regimes.",
            "Metrics are not interchangeable and are not production payment-fraud accuracy.",
            "IEEE-CIS identity/device fields do not exist on ULB PCA columns.",
            "Do not mix IEEE rows into ULB evaluation or synthetic scenario labels into IEEE evaluation.",
        ],
        "table": rows,
    }


def run_ieee_pipeline(
    *,
    data_dir: Path | str | None = None,
    eval_dir: Path | str | None = None,
    model_dir: Path | str | None = None,
    max_rows: int | None = None,
    allow_fixture: bool = False,
    write_reports: bool = True,
    n_estimators: int | None = None,
    seed: int = RANDOM_SEED,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    transaction: pd.DataFrame | None = None,
    identity: pd.DataFrame | None = None,
) -> dict:
    t0 = time.perf_counter()
    eval_path = Path(eval_dir or EVAL_DIR)
    model_path = Path(model_dir or IEEE_MODEL_DIR)
    data_root = resolve_data_dir(data_dir)
    source = "IEEE_CIS_CSV"

    if transaction is None or identity is None:
        if ieee_files_present(data_root):
            txn, ident, load_meta = load_tables(data_root, max_rows=max_rows)
        elif allow_fixture:
            txn, ident = make_ieee_fixture(seed=seed)
            load_meta = {
                "source": "SYNTHETIC_FIXTURE_NOT_IEEE_CIS",
                "dataset_available": False,
                "data_dir": str(data_root),
                "setup_message": setup_payload(data_root)["setup_message"],
            }
            source = "SYNTHETIC_FIXTURE_NOT_IEEE_CIS"
        else:
            payload = setup_payload(data_root)
            payload["stopped_at"] = "VERIFY DATASET"
            payload["active_live_model"] = LIVE_MODEL_VERSION
            payload["ieee_status"] = "OFFLINE CANDIDATE"
            payload["auto_activated"] = False
            if write_reports:
                write_audit(payload, eval_path)
                eval_path.mkdir(parents=True, exist_ok=True)
                (eval_path / "ieee_experiment_manifest.json").write_text(json.dumps(payload, indent=2))
                (eval_path / "ieee_cross_dataset.json").write_text(json.dumps(_cross_dataset(None, "MISSING"), indent=2))
                write_ieee_results(build_ieee_results(payload, payload, {}), eval_path)
            return payload
    else:
        txn, ident, load_meta = load_tables(transaction=transaction, identity=identity, max_rows=max_rows)
        source = load_meta.get("source") or "IN_MEMORY"
        if allow_fixture and source == "IN_MEMORY":
            source = "SYNTHETIC_FIXTURE_NOT_IEEE_CIS"
            load_meta["source"] = source
            load_meta["dataset_available"] = False

    txn_audit = audit_table(txn, "transaction", set())
    ident_audit = audit_table(ident, "identity", set(c for c in ident.columns if c != "TransactionID"))
    joined, join_doc = join_transaction_identity(txn, ident)
    if joined["TransactionID"].duplicated().any():
        joined = joined.drop_duplicates("TransactionID", keep="first")
    del txn, ident
    joined_audit = audit_table(joined, "joined", set(join_doc.get("identity_columns_added") or []))
    joined = add_transaction_timing(joined)
    joined = add_behavioral_features(joined)
    joined = add_graph_features(joined)

    audit_payload = {
        "label": "OFFLINE PUBLIC DATASET EVALUATION",
        "dataset_id": DATASET_ID,
        "dataset": DATASET_NAME,
        "source": source,
        "dataset_available": source == "IEEE_CIS_CSV",
        "disclaimer": "The IEEE-CIS experiment is an offline public-dataset evaluation. It does not represent production payment-fraud performance.",
        "transaction": txn_audit,
        "identity": ident_audit,
        "joined": joined_audit,
        "join": join_doc,
        **{k: load_meta.get(k) for k in ("data_dir", "max_rows", "transaction_path", "identity_path", "setup_message")},
    }
    if write_reports:
        write_audit(audit_payload, eval_path)

    train, val, test = chronological_split(joined, train_frac, val_frac, test_frac)
    splits = split_summary(train, val, test)
    del joined

    feature_all = columns_for_families(EXPERIMENTS["F_combined"], list(train.columns))
    leak = audit_leakage(train, train, val, test, "train_only", feature_all)
    if write_reports:
        write_leakage_report(leak, eval_path)

    results = []
    for name, families in EXPERIMENTS.items():
        results.append(
            train_experiment(name, families, train, val, test, seed=seed, n_estimators=n_estimators)
        )

    comparison_test = [experiment_row(r, "test") for r in results]
    comparison_val = [experiment_row(r, "val") for r in results]

    by_name = {r["experiment"]: r for r in results}
    with_graph = by_name["F_combined"]
    without_graph = by_name["ablation_no_graph"]
    ablation = {
        "without_graph": without_graph["test_metrics"],
        "with_graph": with_graph["test_metrics"],
        "improved": {
            metric: _improved(without_graph["test_metrics"].get(metric), with_graph["test_metrics"].get(metric), higher_better=metric != "false_positive_rate")
            for metric in ("pr_auc", "recall", "precision", "f1", "false_positive_rate")
        },
        "honest_note": "If graph features do not improve a metric, that is reported as false. This does not replace live NetworkX/Neo4j.",
        "construction": GRAPH_CONSTRUCTION_NOTES,
    }

    selected = with_graph
    calibrators = fit_calibrators(selected["p_val"], selected["y_val"])
    val_diag = {
        "raw": calibration_diagnostics(selected["y_val"], selected["p_val"], "raw"),
        "sigmoid": calibration_diagnostics(selected["y_val"], calibrators.transform(selected["p_val"], "sigmoid"), "sigmoid"),
        "isotonic": calibration_diagnostics(selected["y_val"], calibrators.transform(selected["p_val"], "isotonic"), "isotonic"),
    }
    selection = select_calibration_method(val_diag)
    method = selection["selected_method"]
    p_test_cal = calibrators.transform(selected["p_test"], method)
    p_val_cal = calibrators.transform(selected["p_val"], method)
    test_diag = {
        "raw": calibration_diagnostics(selected["y_test"], selected["p_test"], "raw"),
        "calibrated": calibration_diagnostics(selected["y_test"], p_test_cal, method),
    }
    thresholds = select_three_way_thresholds(selected["y_val"], p_val_cal)
    frozen_test = classification_metrics(selected["y_test"], p_test_cal, threshold=0.5)
    policy = policy_summary(selected["y_test"], p_test_cal, thresholds)
    shap = shap_summary(selected["clf"], selected["preprocessor"].transform(train.head(24)), selected["preprocessor"].feature_names, family_lookup())

    split_counts = {
        "train_rows": splits["train"]["n"],
        "train_fraud_rows": splits["train"]["fraud"],
        "validation_rows": splits["validation"]["n"],
        "test_rows": splits["test"]["n"],
    }
    extra = {
        "dataset": DATASET_NAME if source == "IEEE_CIS_CSV" else "SYNTHETIC_FIXTURE_NOT_IEEE_CIS",
        "source": source,
        "dataset_version": load_meta.get("transaction_path"),
        "data_dir": str(data_root),
        **split_counts,
    }

    saved = []
    if write_reports:
        for exp_name, ver in (
            ("A_transaction_only", BASELINE_VERSION),
            ("F_combined", COMBINED_VERSION),
            ("E_transaction_graph", GRAPH_VERSION),
        ):
            saved.append(save_candidate(by_name[exp_name], ver, extra, model_path))

    ieee_summary = {
        "n_rows": txn_audit["n_rows"],
        "n_fraud": (txn_audit.get("target_distribution") or {}).get("positive"),
        "n_features": selected["n_model_features"],
        "pr_auc": frozen_test.get("pr_auc") if source == "IEEE_CIS_CSV" else None,
        "roc_auc": frozen_test.get("roc_auc") if source == "IEEE_CIS_CSV" else None,
    }
    cross = _cross_dataset(ieee_summary, source)

    elapsed = time.perf_counter() - t0
    manifest = {
        "label": "OFFLINE PUBLIC DATASET EVALUATION",
        "track": TRACK,
        "source": source,
        "dataset_available": source == "IEEE_CIS_CSV",
        "disclaimer": "The IEEE-CIS experiment is an offline public-dataset evaluation. It does not represent production payment-fraud performance.",
        "random_seed": seed,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "pandas": _pkg("pandas"),
            "numpy": _pkg("numpy"),
            "scikit-learn": _pkg("scikit-learn"),
            "xgboost": _pkg("xgboost"),
        },
        "split": splits,
        "split_configuration": {"train": train_frac, "validation": val_frac, "test": test_frac, "strategy": "chronological"},
        "max_rows": load_meta.get("max_rows"),
        "n_estimators": n_estimators,
        "feature_configuration": {k: v for k, v in EXPERIMENTS.items()},
        "leakage_all_passed": leak.get("all_passed"),
        "experiments_test": comparison_test,
        "experiments_validation": comparison_val,
        "graph_ablation": ablation,
        "calibration": {
            "selection": selection,
            "validation": val_diag,
            "test_once": test_diag,
        },
        "thresholds": thresholds,
        "frozen_test_metrics": frozen_test if source == "IEEE_CIS_CSV" else {**frozen_test, "not_ieee_cis_public_result": True, "source": source},
        "policy_on_test": policy,
        "shap": shap,
        "candidates": saved,
        "active_live_model": LIVE_MODEL_VERSION,
        "ieee_status": "OFFLINE CANDIDATE",
        "cross_dataset": cross,
        "runtime_seconds": elapsed,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "join": join_doc,
        "audit_paths": {"eval_dir": str(eval_path)},
        "model_probability_separate_from_final_risk_score": True,
        "auto_activated": False,
    }
    if write_reports:
        eval_path.mkdir(parents=True, exist_ok=True)
        (eval_path / "ieee_experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        (eval_path / "ieee_cross_dataset.json").write_text(json.dumps(cross, indent=2, default=str))
        _write_eval_markdown(eval_path, manifest, audit_payload)
        write_ieee_results(build_ieee_results(manifest, audit_payload, leak), eval_path)
    return manifest


def _improved(before, after, higher_better: bool) -> bool | None:
    if before is None or after is None:
        return None
    return (after > before) if higher_better else (after < before)


def _write_eval_markdown(eval_path: Path, manifest: dict, audit: dict) -> None:
    rows = manifest.get("experiments_test") or []
    lines = [
        "# IEEE-CIS evaluation",
        "",
        "The IEEE-CIS experiment is an offline public-dataset evaluation. It does not represent production payment-fraud performance.",
        "",
        f"- Source: `{manifest.get('source')}`",
        f"- Live active model (unchanged): `{manifest.get('active_live_model')}`",
        f"- IEEE status: OFFLINE CANDIDATE",
        f"- Runtime seconds: {manifest.get('runtime_seconds')}",
        "",
        "## Split",
        json.dumps(manifest.get("split"), indent=2),
        "",
        "## Experiment comparison (frozen chronological test, threshold 0.5)",
        "",
        "| Experiment | Features | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['Experiment']} | {r['Features']} | {r['PR-AUC']} | {r['ROC-AUC']} | {r['Precision']} | {r['Recall']} | {r['F1']} | {r['FPR']} |"
        )
    if manifest.get("source") != "IEEE_CIS_CSV":
        lines += ["", "Fixture/in-memory metrics are **not** IEEE-CIS public-dataset results."]
    (eval_path / "ieee_evaluation.md").write_text("\n".join(lines) + "\n")
