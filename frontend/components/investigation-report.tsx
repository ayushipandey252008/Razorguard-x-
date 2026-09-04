"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

function providerLabel(provider?: string | null) {
  if (!provider) return "not run";
  if (provider === "llm") return "OpenAI / llm";
  if (provider.includes("fallback")) return "deterministic_fallback";
  return provider;
}

function statusMark(status?: string) {
  if (status === "success" || status === "ok") return "✓";
  if (status === "unavailable") return "–";
  if (status === "error") return "✕";
  return "·";
}

const NO_RULES_NOTE = "No deterministic rules fired";

/** Empty triggered[] is not enough: that is also the shape when the tool was never collected. */
export function ruleEvidenceState(ruleEvidence?: { triggered?: any[]; note?: string | null } | null) {
  const triggered = Array.isArray(ruleEvidence?.triggered) ? ruleEvidence!.triggered : [];
  if (triggered.length) return "rules" as const;
  const note = String(ruleEvidence?.note || "");
  if (note.includes(NO_RULES_NOTE)) return "none_fired" as const;
  return "not_collected" as const;
}

export function ruleEvidenceCopy(ruleEvidence?: { triggered?: any[]; note?: string | null } | null) {
  const state = ruleEvidenceState(ruleEvidence);
  const triggered = Array.isArray(ruleEvidence?.triggered) ? ruleEvidence!.triggered : [];
  if (state === "rules") {
    return (
      <ul className="space-y-1">
        {triggered.map((r: any) => (
          <li key={r.rule_id || r.rule_name}>
            <span className="font-mono text-ember">{r.rule_id || r.rule_name}</span>{" "}
            <span className="text-slate-400">{r.rule_name}</span>
            {r.severity ? <span className="text-xs text-slate-500"> · {r.severity}</span> : null}
          </li>
        ))}
      </ul>
    );
  }
  if (state === "none_fired") {
    return <p className="text-slate-500">No deterministic rules fired.</p>;
  }
  return <p className="text-slate-500">Rule evidence was not collected.</p>;
}

export function InvestigationReportView({
  report,
  provider,
}: {
  report: any;
  provider?: string | null;
}) {
  if (!report) {
    return (
      <p className="text-sm text-slate-500">
        Not run yet. The agent only uses registered backend tools — it cannot invent evidence.
      </p>
    );
  }

  const rec = report.recommendation || report.recommended_action;
  const risk = report.risk_level;
  const usedProvider = report.provider || provider;
  const findings: string[] = report.key_findings || report.suspicious_signals || [];
  const model = report.model_evidence || report.risk_assessment || {};
  const ruleEvidence = report.rule_evidence || {};
  const graph = report.graph_evidence || report.potential_fraud_ring || {};
  const trace: any[] = report.tool_trace || [];
  const confidence = report.confidence_qualitative;

  return (
    <div className="space-y-5 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] uppercase tracking-[0.25em] text-slate-500">Investigation</span>
        {risk && <Badge>{`Risk: ${risk}`}</Badge>}
        {rec && (
          <Badge decision={rec} className="text-sm px-3 py-1">
            {rec}
          </Badge>
        )}
        <Badge>{providerLabel(usedProvider)}</Badge>
      </div>

      <div>
        <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-1">Summary</h3>
        <p className="text-slate-300 leading-relaxed">{report.summary || "No summary."}</p>
      </div>

      <section>
        <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-2">Key findings</h3>
        {findings.length ? (
          <ul className="space-y-1">
            {findings.map((f, i) => (
              <li key={i} className="text-slate-200">
                <span className="text-mint mr-2">✓</span>
                {f}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-slate-500">No highlighted findings from tool evidence.</p>
        )}
      </section>

      <section className="rounded-lg border border-white/10 p-3 space-y-1">
        <h3 className="text-xs uppercase tracking-wider text-slate-500">Model evidence</h3>
        <p>
          Fraud probability:{" "}
          <span className="font-mono">
            {model.ml_probability == null ? "unavailable" : Number(model.ml_probability).toFixed(4)}
          </span>
        </p>
        <p>
          Model: <span className="font-mono">{model.model_version || "unavailable"}</span>
        </p>
        <p>
          Engine decision: <span className="font-mono">{model.decision || report.risk_engine_decision || "—"}</span>
        </p>
        {model.unavailable && <p className="text-ember text-xs">{model.reason}</p>}
        <p className="text-xs text-slate-500">
          Copied from the risk engine. The investigation agent does not calculate this probability.
        </p>
      </section>

      <section className="rounded-lg border border-white/10 p-3 space-y-2">
        <h3 className="text-xs uppercase tracking-wider text-slate-500">Rule evidence</h3>
        {ruleEvidenceCopy(ruleEvidence)}
      </section>

      <section className="rounded-lg border border-white/10 p-3 space-y-1">
        <h3 className="text-xs uppercase tracking-wider text-slate-500">Graph evidence</h3>
        {graph.cluster_found || graph.identified ? (
          <>
            <p>
              Cluster: <span className="font-mono">{graph.cluster_id}</span>
            </p>
            <p>{graph.cluster_size || graph.user_count || (graph.connected_users || []).length} entities</p>
            <p>{graph.fraud_associated_nodes ?? 0} fraud-associated nodes</p>
            <p>
              Shared devices: {(graph.shared_devices || []).length} · Shared IPs:{" "}
              {(graph.shared_ips || []).length}
            </p>
          </>
        ) : (
          <p className="text-slate-400">
            {graph.reason || graph.message || "No connected suspicious cluster found"}
          </p>
        )}
      </section>

      <section>
        <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-2">Investigation trace</h3>
        {trace.length ? (
          <ol className="space-y-1 font-mono text-xs">
            {trace.map((t, i) => (
              <li key={`${t.tool}-${i}`} className="flex gap-2 items-start">
                <span
                  className={
                    t.status === "error"
                      ? "text-flare"
                      : t.status === "unavailable"
                        ? "text-ember"
                        : "text-mint"
                  }
                >
                  {statusMark(t.status)}
                </span>
                <span className="text-slate-200">{t.tool}</span>
                <span className="text-slate-500">
                  {t.status}
                  {t.duration_ms != null ? ` · ${t.duration_ms}ms` : ""}
                </span>
                {t.result_summary ? <span className="text-slate-500 truncate">{t.result_summary}</span> : null}
              </li>
            ))}
          </ol>
        ) : (
          <p className="text-slate-500">No tool calls recorded.</p>
        )}
      </section>

      <section>
        <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-1">Limitations</h3>
        <p className="text-slate-400 text-xs leading-relaxed">{report.limitations}</p>
        {confidence ? (
          <p className="text-xs text-slate-500 mt-2">
            Confidence: {confidence.kind || "qualitative"}
            {confidence.level ? ` · ${confidence.level}` : ""}
            {typeof report.confidence === "number" ? ` · engine field ${report.confidence}` : ""}.{" "}
            {confidence.note}
          </p>
        ) : null}
      </section>
    </div>
  );
}

export function InvestigationPanelCard({
  report,
  provider,
  title = "Investigation",
}: {
  report: any;
  provider?: string | null;
  title?: string;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <InvestigationReportView report={report} provider={provider} />
      </CardContent>
    </Card>
  );
}
