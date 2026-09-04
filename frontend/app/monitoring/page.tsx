"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function ModelMonitoringPage() {
  const [status, setStatus] = useState<any>(null);
  const [drift, setDrift] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api("/api/v1/ml/model-status"), api("/api/v1/ml/drift")])
      .then(([s, d]) => {
        setStatus(s);
        setDrift(d);
      })
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="text-flare">{error}</p>;
  if (!status) return <p className="text-slate-400">Loading model monitoring…</p>;

  const active = status.active_model || {};
  const candidate = status.candidate_model;
  const feedback = status.feedback || {};
  const features = Array.isArray(drift?.features) ? drift.features : [];
  const evalMetrics = active.last_evaluation || {};

  return (
    <div className="space-y-5 max-w-4xl">
      <header>
        <h1 className="text-2xl font-semibold">Model monitoring</h1>
        <p className="text-sm text-slate-400">
          Prototype drift and feedback. Candidates stay offline. This is not production model ops.
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Active model</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-2 text-sm">
          <div>Version</div>
          <div className="font-mono text-right">{active.version || "—"}</div>
          <div>Dataset</div>
          <div className="font-mono text-right">{active.dataset || "—"}</div>
          <div>Training rows</div>
          <div className="font-mono text-right">{active.training_rows ?? "—"}</div>
          <div>Last evaluation PR-AUC</div>
          <div className="font-mono text-right">{fmt(evalMetrics.pr_auc)}</div>
          <div>Status</div>
          <div className="text-right">
            <Badge>{active.status || "ACTIVE"}</Badge>
          </div>
          <div>IEEE-CIS</div>
          <div className="text-right">
            <Badge>{status.ieee_cis?.status || "OFFLINE CANDIDATE"}</Badge>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Drift</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <div className="grid grid-cols-2 gap-2">
            <div>Overall status</div>
            <div className="font-mono text-right">{drift?.status || status.drift_status?.status || "—"}</div>
            <div>Last checked</div>
            <div className="font-mono text-right text-xs">{drift?.checked_at || status.drift_status?.checked_at || "—"}</div>
            <div>Recommendation</div>
            <div className="text-right text-slate-300">{drift?.recommendation || "—"}</div>
          </div>
          <p className="text-[11px] text-slate-500">
            PSI thresholds are prototype heuristics (&lt;0.10 low, 0.10–0.25 moderate, &gt;0.25 high), not production standards.
          </p>
          <table className="w-full text-xs">
            <thead className="text-slate-500">
              <tr>
                <th className="text-left py-1">Feature</th>
                <th className="text-right">PSI</th>
                <th className="text-right">Status</th>
              </tr>
            </thead>
            <tbody>
              {features.length === 0 ? (
                <tr>
                  <td colSpan={3} className="py-2 text-slate-500">
                    Not enough scored transactions for a drift window.
                  </td>
                </tr>
              ) : (
                features.map((row: any) => (
                  <tr key={row.feature} className="border-t border-white/5">
                    <td className="py-1">{row.feature}</td>
                    <td className="text-right font-mono">{row.drift_score ?? "—"}</td>
                    <td className="text-right">{row.status}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Feedback</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-3 gap-3 text-sm">
          <Stat label="Confirmed fraud" value={feedback.CONFIRM_FRAUD} />
          <Stat label="Confirmed legitimate" value={feedback.CONFIRM_LEGITIMATE} />
          <Stat label="Needs review" value={feedback.NEEDS_REVIEW} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Candidate model</CardTitle>
        </CardHeader>
        <CardContent className="text-sm space-y-2">
          {!candidate ? (
            <p className="text-slate-500">No offline candidate. Training never auto-replaces the live model.</p>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              <div>Version</div>
              <div className="font-mono text-right">{candidate.version}</div>
              <div>Dataset</div>
              <div className="font-mono text-right">{candidate.dataset || "SYNTHETIC_FEEDBACK"}</div>
              <div>Status</div>
              <div className="text-right">
                <Badge>{candidate.status}</Badge>
              </div>
              <div>PR-AUC</div>
              <div className="font-mono text-right">{fmt(candidate.metrics?.pr_auc)}</div>
            </div>
          )}
          <p className="text-[11px] text-slate-500">There is no automatic model activation in this prototype.</p>
        </CardContent>
      </Card>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | undefined }) {
  return (
    <div className="rounded-lg bg-ink-950 p-3">
      <div className="text-[10px] uppercase text-slate-500">{label}</div>
      <div className="font-mono text-lg">{value ?? 0}</div>
    </div>
  );
}

function fmt(value: unknown) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return Number(value).toFixed(3);
}
