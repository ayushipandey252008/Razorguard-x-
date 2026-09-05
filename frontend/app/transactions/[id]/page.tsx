"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { InvestigationReportView } from "@/components/investigation-report";
import { formatInr, formatTime } from "@/lib/utils";
import { toast } from "sonner";
import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function TransactionDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  async function load() {
    try {
      setData(await api(`/api/v1/transactions/${id}`));
      setError(null);
    } catch (e: any) {
      setError(e.message);
    }
  }

  useEffect(() => {
    load();
  }, [id]);

  async function runInvestigation() {
    const target = data?.investigation?.id || id;
    setRunning(true);
    try {
      await api(`/api/v1/investigations/${target}/run`, { method: "POST" });
      toast.success("Investigation completed from tool evidence");
      await load();
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setRunning(false);
    }
  }

  if (error) return <p className="text-flare">{error}</p>;
  if (!data) return <Skeleton className="h-96" />;

  const t = data.transaction;
  const r = data.risk;
  const report = data.investigation?.ai_report;

  const scoreBars = r
    ? [
        { name: "ML", v: r.ml_score },
        { name: "Behavior", v: r.behavior_score },
        { name: "Rules", v: r.rule_score },
        { name: "Graph", v: r.graph_score },
        { name: "Final", v: r.final_risk_score },
      ]
    : [];

  return (
    <div className="space-y-6">
      <header className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:flex-wrap">
        <div className="min-w-0">
          <div className="text-[11px] uppercase tracking-[0.25em] text-slate-500">Transaction file</div>
          <h1 className="mt-1 break-all font-mono text-base sm:text-xl">{t.transaction_id}</h1>
          <p className="mt-1 break-words text-sm text-slate-400">
            {t.user_id} → {t.merchant_id} · {formatInr(t.amount)} {t.currency} · {formatTime(t.timestamp)}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-right">
            <div className="font-mono text-3xl font-semibold sm:text-4xl">{r?.final_risk_score ?? "—"}</div>
            <div className="text-[11px] text-slate-500">final risk / 100</div>
          </div>
          <Badge decision={r?.decision} className="px-3 py-1 text-sm">
            {r?.decision}
          </Badge>
        </div>
      </header>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Why this score</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm leading-relaxed text-slate-300">{r?.explanation}</p>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={scoreBars}>
                  <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 100]} />
                  <Tooltip contentStyle={{ background: "#121923", border: "1px solid #243044" }} />
                  <Bar dataKey="v" fill="#3ee0c6" radius={4} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <Metric label="ML risk score" value={`${r?.ml_score ?? "—"} / 100`} />
              <Metric
                label={r?.probability_calibrated ? "Calibrated P(fraud)" : "Uncalibrated model output"}
                value={
                  r?.probability_calibrated
                    ? Number(r.ml_probability).toFixed(3)
                    : "not a probability"
                }
              />
              <Metric label="Model" value={r?.model_version} />
              <Metric label="Confidence" value={r?.confidence} />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Payment facts</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <Row k="Device" v={t.device_id} known={t.current_device_known} />
            <Row k="IP" v={t.ip_address} />
            <Row k="Location" v={t.location} known={t.current_location_known} />
            <Row k="Method" v={t.payment_method} />
            <Row k="Category" v={t.merchant_category} />
            <Row k="Account age" v={`${t.account_age_days}d`} />
            <Row k="Velocity" v={t.transaction_velocity} />
            <Row k="Failed attempts" v={t.failed_attempts} />
            <Row k="Prev count / avg" v={`${t.previous_transaction_count} / ${formatInr(t.previous_average_amount)}`} />
            <Row k="Payment token" v={t.payment_identifier} />
            {t.scenario_tag && <Row k="Scenario tag" v={t.scenario_tag} />}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>User behavior baseline</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            {!data.user_baseline ? (
              <p className="text-slate-500">No payment-user profile is stored for this identifier.</p>
            ) : (
              <>
                <Row k="Typical amount" v={formatInr(data.user_baseline.typical_amount)} />
                <Row
                  k="This amount vs typical"
                  v={data.user_baseline.amount_vs_typical != null ? `${data.user_baseline.amount_vs_typical}×` : "—"}
                />
                <Row k="Typical hour" v={`${data.user_baseline.typical_hour}:00`} />
                <Row k="Home location" v={data.user_baseline.home_location} />
                <Row k="Known devices" v={(data.user_baseline.known_devices || []).join(", ") || "—"} />
                <Row k="Known locations" v={(data.user_baseline.known_locations || []).join(", ") || "—"} />
              </>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Connected entities</CardTitle>
          </CardHeader>
          <CardContent className="text-sm space-y-2">
            <Row k="Connected users" v={(r?.graph_evidence?.connected_users || []).join(", ") || "none"} />
            <Row k="Device users" v={(r?.graph_evidence?.device_users || []).join(", ") || "—"} />
            <Row k="IP users" v={(r?.graph_evidence?.ip_users || []).join(", ") || "—"} />
            <Row k="Cluster" v={r?.graph_evidence?.cluster_id || "none"} />
            <Row k="Suspicious relationships" v={r?.graph_evidence?.suspicious_relationship_count ?? 0} />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Triggered rules</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {!r?.triggered_rules?.length ? (
              <p className="text-sm text-slate-500">No deterministic rules fired.</p>
            ) : (
              r.triggered_rules.map((rule: any) => (
                <div key={rule.rule_id} className="rounded-lg border border-white/10 p-3">
                  <div className="flex flex-col gap-1 text-sm sm:flex-row sm:justify-between">
                    <span className="font-medium">{rule.rule_name}</span>
                    <span className="font-mono text-ember">+{rule.score_contribution}</span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">{rule.explanation}</p>
                  <pre className="mt-2 text-[11px] text-slate-500 overflow-x-auto">
                    {JSON.stringify(rule.evidence, null, 2)}
                  </pre>
                </div>
              ))
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>SHAP feature contributions</CardTitle>
          </CardHeader>
          <CardContent>
            {!r?.shap_top_features?.length ? (
              <p className="text-sm text-slate-500">SHAP values were not available for this prediction.</p>
            ) : (
              <ul className="space-y-2">
                {r.shap_top_features.map((f: any) => (
                  <li key={f.feature} className="flex min-w-0 items-center gap-2 text-sm sm:gap-3">
                    <span className="w-24 shrink-0 truncate font-mono text-xs text-slate-400 sm:w-44" title={f.feature}>
                      {f.feature}
                    </span>
                    <div className="h-2 min-w-0 flex-1 rounded bg-white/5">
                      <div
                        className={`h-2 rounded ${f.contribution > 0 ? "bg-flare" : "bg-mint"}`}
                        style={{ width: `${Math.min(100, Math.abs(f.contribution) * 80)}%` }}
                      />
                    </div>
                    <span className="w-14 shrink-0 text-right font-mono text-xs sm:w-16">{f.contribution}</span>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Behavioral anomalies</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {!r?.anomalies?.length ? (
              <p className="text-sm text-slate-500">No explicit behavioral flags.</p>
            ) : (
              r.anomalies.map((a: any) => (
                <div key={a.code} className="text-sm border border-white/10 rounded-lg p-3">
                  <div className="font-medium">{a.code}</div>
                  <div className="text-slate-400 text-xs">{a.description}</div>
                </div>
              ))
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Graph evidence</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-xs text-slate-400 overflow-x-auto">
              {JSON.stringify(r?.graph_evidence, null, 2)}
            </pre>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <CardTitle>AI investigation</CardTitle>
          <Button size="sm" className="w-full sm:w-auto" onClick={runInvestigation} disabled={running}>
            {running ? "Running tools…" : "Run agent"}
          </Button>
        </CardHeader>
        <CardContent>
          <InvestigationReportView
            report={report}
            provider={data.investigation?.agent_provider}
          />
          {!data.investigation && !report ? (
            <p className="text-xs text-slate-500 mt-3">
              APPROVE decisions do not auto-open a case. Running the agent on this page creates one from the transaction id.
            </p>
          ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Audit trail</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm">
            {(data.audit_trail || []).map((a: any) => (
              <li key={a.id} className="flex flex-col gap-1 border-b border-white/5 pb-2 sm:flex-row sm:gap-4">
                <span className="w-auto text-xs text-slate-500 sm:w-44">{formatTime(a.timestamp)}</span>
                <span className="break-all font-mono text-xs text-mint sm:w-40">{a.action}</span>
                <span className="break-all text-xs text-slate-400">{a.actor}</span>
              </li>
            ))}
            {!data.audit_trail?.length && <p className="text-slate-500 text-sm">No audit events stored.</p>}
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: any }) {
  return (
    <div className="rounded-lg bg-ink-950/50 p-3">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="font-mono text-xs mt-1 break-all">{String(value)}</div>
    </div>
  );
}

function Row({ k, v, known }: { k: string; v: any; known?: boolean }) {
  return (
    <div className="flex flex-col gap-0.5 sm:flex-row sm:justify-between sm:gap-4">
      <span className="shrink-0 text-slate-500">{k}</span>
      <span className="min-w-0 break-all text-left font-mono text-xs sm:text-right">
        {String(v)}
        {known === false && <span className="ml-2 text-ember">new</span>}
        {known === true && <span className="ml-2 text-mint">known</span>}
      </span>
    </div>
  );
}
