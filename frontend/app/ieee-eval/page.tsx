"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

function fmt(v: unknown) {
  if (v == null || v === false) return "—";
  if (typeof v === "number" && Number.isFinite(v)) return v.toFixed(4);
  return String(v);
}

function Metric({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="rounded-lg bg-ink-950 p-3">
      <div className="text-[10px] uppercase text-slate-500">{label}</div>
      <div className="font-mono text-sm">{fmt(value)}</div>
    </div>
  );
}

export default function IeeeEvalPage() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api("/api/v1/ml/ieee-evaluation")
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="text-flare">{error}</p>;
  if (!data) return <p className="text-slate-400">Loading IEEE-CIS evaluation…</p>;

  const audit = data.audit || {};
  const txn = audit.transaction || {};
  const ident = audit.identity || {};
  const join = data.join || audit.join || {};
  const split = data.split || {};
  const leakage = data.leakage || {};
  const experiments = Array.isArray(data.experiments) ? data.experiments : [];
  const cal = data.calibration || {};
  const thresholds = data.thresholds || {};
  const ablation = data.graph_ablation || {};
  const test = data.frozen_test_metrics || {};
  const families = data.feature_families || {};
  const cross = data.cross_dataset?.table || [];
  const publicResult = data.dataset_available === true && data.source === "IEEE_CIS_CSV";

  return (
    <div className="space-y-5 max-w-5xl">
      <header>
        <h1 className="text-2xl font-semibold">IEEE-CIS evaluation</h1>
        <p className="text-[11px] uppercase tracking-wider text-ember mt-1">
          OFFLINE PUBLIC DATASET EVALUATION — not live production performance
        </p>
        <p className="text-sm text-slate-400 mt-2">{data.disclaimer}</p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Model status</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-2 text-sm">
          <div>ACTIVE MODEL</div>
          <div className="font-mono text-right">{data.active_live_model || data.active_model?.version}</div>
          <div>IEEE-CIS</div>
          <div className="text-right">
            <Badge>OFFLINE CANDIDATE</Badge>
          </div>
          <div>Auto-activated</div>
          <div className="font-mono text-right">{String(data.auto_activated ?? false)}</div>
          <div>Source</div>
          <div className="font-mono text-right">{data.source || "MISSING"}</div>
        </CardContent>
      </Card>

      {!data.available || data.dataset_available === false ? (
        <Card>
          <CardHeader>
            <CardTitle>Dataset setup</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-slate-300 space-y-2">
            <p>{data.reason || audit.setup_message || "IEEE-CIS CSVs were not found."}</p>
            <p className="text-xs text-slate-500">
              Expected files: train_transaction.csv and train_identity.csv in IEEE_DATA_DIR. The prototype does not
              download this dataset. Fixture metrics, if shown, are not IEEE-CIS results.
            </p>
          </CardContent>
        </Card>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Dataset audit</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-2 text-sm">
          <div>Transaction rows</div>
          <div className="font-mono text-right">{txn.n_rows ?? "—"}</div>
          <div>Transaction columns</div>
          <div className="font-mono text-right">{txn.n_columns ?? "—"}</div>
          <div>Identity rows</div>
          <div className="font-mono text-right">{ident.n_rows ?? "—"}</div>
          <div>Fraud (txn)</div>
          <div className="font-mono text-right">{txn.target_distribution?.positive ?? "—"}</div>
          <div>Prevalence</div>
          <div className="font-mono text-right">{fmt(txn.target_distribution?.prevalence)}</div>
          <div>Join coverage</div>
          <div className="font-mono text-right">{fmt(join.identity_coverage)}</div>
          <div>Unmatched identity</div>
          <div className="font-mono text-right">{join.unmatched_identity_rows ?? "—"}</div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Temporal split</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <p className="text-xs text-slate-500">Chronological 70/15/15. Preprocessing is fit on train only.</p>
          {["train", "validation", "test"].map((name) => {
            const part = split[name] || {};
            return (
              <div key={name} className="grid grid-cols-4 gap-2 text-xs font-mono">
                <div className="uppercase text-slate-500">{name}</div>
                <div>n={part.n ?? "—"}</div>
                <div>fraud={part.fraud ?? "—"}</div>
                <div>prev={fmt(part.prevalence)}</div>
              </div>
            );
          })}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Leakage checks</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1 text-xs">
          <p>Overall: {String(leakage.all_passed ?? "—")}</p>
          {(leakage.checks || []).map((c: any) => (
            <div key={c.id} className="flex justify-between gap-4">
              <span className="font-mono">{c.id}</span>
              <span className={c.passed ? "text-mint" : "text-flare"}>{c.passed ? "PASS" : "FAIL"}</span>
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Feature families</CardTitle>
        </CardHeader>
        <CardContent className="text-xs font-mono space-y-1">
          {Object.entries(families).map(([k, v]) => (
            <div key={k}>
              {k}: {Array.isArray(v) ? v.join(" + ") : String(v)}
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Experiment comparison</CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          {!publicResult ? (
            <p className="text-xs text-ember mb-2">
              Numbers below are not IEEE-CIS public-dataset results unless source is IEEE_CIS_CSV.
            </p>
          ) : null}
          <table className="w-full text-xs">
            <thead className="text-slate-500">
              <tr>
                {["Experiment", "Features", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "FPR"].map((h) => (
                  <th key={h} className="text-left py-1 pr-2">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {experiments.map((row: any) => (
                <tr key={row.Experiment}>
                  <td className="py-1 pr-2">{row.Experiment}</td>
                  <td className="pr-2">{row.Features}</td>
                  <td>{fmt(row["PR-AUC"])}</td>
                  <td>{fmt(row["ROC-AUC"])}</td>
                  <td>{fmt(row.Precision)}</td>
                  <td>{fmt(row.Recall)}</td>
                  <td>{fmt(row.F1)}</td>
                  <td>{fmt(row.FPR)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Graph ablation</CardTitle>
        </CardHeader>
        <CardContent className="text-xs space-y-2">
          <p>{ablation.honest_note}</p>
          <div className="grid grid-cols-2 gap-2">
            <div>PR-AUC without graph</div>
            <div className="font-mono text-right">{fmt(ablation.without_graph?.pr_auc)}</div>
            <div>PR-AUC with graph</div>
            <div className="font-mono text-right">{fmt(ablation.with_graph?.pr_auc)}</div>
            <div>Graph improved PR-AUC</div>
            <div className="font-mono text-right">{String(ablation.improved?.pr_auc ?? "—")}</div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Calibration</CardTitle>
        </CardHeader>
        <CardContent className="text-xs space-y-2">
          <p>Selected: {cal.selection?.selected_method || "—"}</p>
          <p className="text-slate-500">{cal.selection?.justification}</p>
          <p className="text-slate-500">model_probability is not the live final_risk_score.</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Thresholds (validation only)</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-2 text-xs">
          <div>APPROVE below</div>
          <div className="font-mono text-right">{fmt(thresholds.approve_below)}</div>
          <div>BLOCK above</div>
          <div className="font-mono text-right">{fmt(thresholds.block_above)}</div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Frozen chronological test</CardTitle>
        </CardHeader>
        <CardContent>
          {test.not_ieee_cis_public_result ? (
            <p className="text-xs text-ember mb-2">These figures are not IEEE-CIS public-dataset results.</p>
          ) : null}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            <Metric label="PR-AUC" value={publicResult ? test.pr_auc : null} />
            <Metric label="ROC-AUC" value={publicResult ? test.roc_auc : null} />
            <Metric label="Precision" value={publicResult ? test.precision : null} />
            <Metric label="Recall" value={publicResult ? test.recall : null} />
            <Metric label="F1" value={publicResult ? test.f1 : null} />
            <Metric label="FPR" value={publicResult ? test.false_positive_rate : null} />
            <Metric label="FNR" value={publicResult ? test.false_negative_rate : null} />
            <Metric label="Prevalence" value={publicResult ? test.fraud_prevalence : null} />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Cross-dataset (not equivalent)</CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-slate-500">
              <tr>
                {["Dataset", "Rows", "Fraud", "Features", "PR-AUC", "ROC-AUC"].map((h) => (
                  <th key={h} className="text-left py-1 pr-2">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cross.map((row: any) => (
                <tr key={row.Dataset}>
                  <td className="py-1 pr-2">{row.Dataset}</td>
                  <td>{row.Rows ?? "—"}</td>
                  <td>{row.Fraud ?? "—"}</td>
                  <td>{row.Features ?? "—"}</td>
                  <td>{fmt(row["PR-AUC"])}</td>
                  <td>{fmt(row["ROC-AUC"])}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-[11px] text-slate-500 mt-2">
            Different datasets, time periods, feature spaces, prevalence, and entity information. Metrics are not
            interchangeable.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
