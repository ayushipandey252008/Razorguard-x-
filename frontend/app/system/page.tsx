"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { PrototypeCalibrationCard } from "@/components/prototype-calibration";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function SystemPage() {
  const [health, setHealth] = useState<any>(null);
  const [analytics, setAnalytics] = useState<any>(null);

  useEffect(() => {
    api("/api/v1/health").then(setHealth).catch(console.error);
    api("/api/v1/analytics").then(setAnalytics).catch(console.error);
  }, []);

  return (
    <div className="space-y-5 max-w-3xl">
      <h1 className="text-xl font-semibold sm:text-2xl">Model bay</h1>
      <Card>
        <CardHeader>
          <CardTitle>Runtime</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="overflow-x-auto whitespace-pre-wrap break-all text-xs">{JSON.stringify(health, null, 2)}</pre>
          {health?.graph_cluster_thresholds && (
            <p className="text-xs text-slate-500 mt-3">
              Graph cluster thresholds are prototype heuristics, not production-grade: min users{" "}
              {health.graph_cluster_thresholds.min_cluster_users}, shared device/IP{" "}
              {health.graph_cluster_thresholds.shared_device_accounts}/
              {health.graph_cluster_thresholds.shared_ip_accounts}. LLM provider:{" "}
              {health.llm?.provider || "deterministic_fallback"}
              {health.llm?.configured ? "" : " (no API key configured)"}.
            </p>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Active product model (SYNTHETIC_DATASET)</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="max-h-[480px] overflow-x-auto whitespace-pre-wrap break-all text-xs">
            {JSON.stringify(analytics?.model, null, 2)}
          </pre>
        </CardContent>
      </Card>
      <EventInfrastructure />
      <OfflineUlbCard />
      <IeeeOfflineCard />
      <CalibrationSection />
      <p className="text-xs text-slate-500">
        Extension points in docs/: feature store, GNNs, online learning.
        Isolation Forest is global; amount/hour/device/location/velocity checks are per-user overlays.
        Neo4j and Kafka are optional transports with in-process fallbacks — not production clusters.
      </p>
    </div>
  );
}

function EventInfrastructure() {
  const [status, setStatus] = useState<any>(null);

  useEffect(() => {
    api("/api/v1/events/status")
      .then(setStatus)
      .catch(() => setStatus({ available: false, reason: "event status unavailable" }));
  }, []);

  const busLabel = status?.active === "kafka" ? "Kafka" : "In-process";
  const kafkaLabel = status?.kafka_connected ? "Connected" : "unavailable";
  const recent = Array.isArray(status?.recent_events) ? status.recent_events.slice(0, 8) : [];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Event Infrastructure</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <p className="text-[11px] uppercase tracking-wider text-ember">
          Durable outbox · optional Kafka transport · synchronous risk decision is unchanged
        </p>
        {!status ? (
          <p className="text-slate-500">Loading event bus status…</p>
        ) : status.available === false ? (
          <p className="text-slate-500">{status.reason}</p>
        ) : (
          <>
            <p className="text-xs font-medium">
              Durable event delivery:{" "}
              {status.outbox?.durable_event_delivery || status.outbox?.enabled ? "enabled" : "off"}
            </p>
            <div className="grid grid-cols-1 gap-2 text-xs sm:grid-cols-2">
              <div>Event Bus</div>
              <div className="break-all font-mono sm:text-right">{busLabel}</div>
              <div>Active bus</div>
              <div className="break-all font-mono sm:text-right">{busLabel}</div>
              <div>Configured</div>
              <div className="break-all font-mono sm:text-right">{status.configured || "inprocess"}</div>
              <div>Kafka</div>
              <div className="break-all font-mono sm:text-right">{kafkaLabel}</div>
              {status.fallback ? (
                <>
                  <div>Fallback</div>
                  <div className="font-mono sm:text-right">true</div>
                </>
              ) : null}
              <div>Durable event delivery</div>
              <div className="font-mono sm:text-right">
                {status.outbox?.durable_event_delivery || status.outbox?.enabled ? "enabled" : "off"}
              </div>
            </div>
            {status.outbox ? (
              <div className="grid grid-cols-1 gap-2 text-xs sm:grid-cols-2">
                <div>Outbox pending</div>
                <div className="font-mono sm:text-right">{status.outbox.pending ?? "—"}</div>
                <div>Outbox processing</div>
                <div className="font-mono sm:text-right">{status.outbox.processing ?? "—"}</div>
                <div>Outbox published</div>
                <div className="font-mono sm:text-right">{status.outbox.published ?? "—"}</div>
                <div>Outbox failed</div>
                <div className="font-mono sm:text-right">{status.outbox.failed ?? "—"}</div>
              </div>
            ) : null}
            <div>
              <div className="text-[10px] uppercase text-slate-500 mb-1">Recent events</div>
              {recent.length === 0 ? (
                <p className="text-xs text-slate-500">No domain events in this process yet.</p>
              ) : (
                <ul className="text-xs font-mono space-y-1">
                  {recent.map((row: any) => (
                    <li key={row.event_id || `${row.event_type}-${row.timestamp}`}>
                      {row.event_type}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function CalibrationSection() {
  const [calibration, setCalibration] = useState<any>(null);
  useEffect(() => {
    api("/api/v1/ml/offline-evaluation")
      .then((d) => setCalibration(d.calibration))
      .catch(() => setCalibration({ available: false, reason: "offline evaluation endpoint unavailable" }));
  }, []);
  return <PrototypeCalibrationCard calibration={calibration} />;
}

function OfflineUlbCard() {
  const [ulb, setUlb] = useState<any>(null);

  useEffect(() => {
    api("/api/v1/ml/offline-evaluation")
      .then((d) => setUlb(d.ulb))
      .catch(() => setUlb({ available: false, reason: "offline evaluation endpoint unavailable" }));
  }, []);

  const metrics = [
    ["PR-AUC", ulb?.pr_auc],
    ["ROC-AUC", ulb?.roc_auc],
    ["Precision", ulb?.precision],
    ["Recall", ulb?.recall],
    ["F1", ulb?.f1],
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Model evaluation — OFFLINE EVALUATION</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <p className="text-[11px] uppercase tracking-wider text-ember">
          Not live synthetic scores · not Razorpay data · not mixed with the product model
        </p>
        {!ulb ? (
          <p className="text-slate-500">Loading ULB metrics…</p>
        ) : !ulb.available ? (
          <p className="text-slate-500">{ulb.reason || "ULB metrics not generated yet."}</p>
        ) : (
          <>
            <div className="grid grid-cols-1 gap-2 text-xs sm:grid-cols-2">
              <div>Dataset</div>
              <div className="break-all font-mono sm:text-right">{ulb.dataset}</div>
              <div>Model</div>
              <div className="break-all font-mono sm:text-right">{ulb.model}</div>
              <div>Model version</div>
              <div className="break-all font-mono sm:text-right">{ulb.model_version}</div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
              {metrics.map(([label, value]) => (
                <div key={label} className="rounded-lg bg-ink-950 p-3">
                  <div className="text-[10px] uppercase text-slate-500">{label}</div>
                  <div className="font-mono text-lg">
                    {value != null && Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "—"}
                  </div>
                </div>
              ))}
            </div>
            <p className="text-xs text-slate-500">
              Figures at probability cutoff 0.5 on the chronological test split. PR-AUC is threshold-free.
              {ulb.operating_point ? ` ${ulb.operating_point}.` : ""} {ulb.note}
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function IeeeOfflineCard() {
  const [ieee, setIeee] = useState<any>(null);

  useEffect(() => {
    api("/api/v1/ml/ieee-evaluation")
      .then(setIeee)
      .catch(() => setIeee({ available: false, reason: "IEEE evaluation endpoint unavailable" }));
  }, []);

  return (
    <Card>
      <CardHeader>
        <CardTitle>IEEE-CIS — OFFLINE PUBLIC DATASET EVALUATION</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <p className="text-[11px] uppercase tracking-wider text-ember">
          Not live scores · not ULB · not production fraud accuracy
        </p>
        <div className="grid grid-cols-1 gap-2 text-xs sm:grid-cols-2">
          <div>ACTIVE MODEL</div>
          <div className="break-all font-mono sm:text-right">{ieee?.active_live_model || "xgb-iforest-v1-calibrated"}</div>
          <div>IEEE-CIS</div>
          <div className="font-mono sm:text-right">OFFLINE CANDIDATE</div>
        </div>
        <p className="text-xs text-slate-500">
          {ieee?.disclaimer ||
            "The IEEE-CIS experiment is an offline public-dataset evaluation. It does not represent production payment-fraud performance."}
        </p>
        {ieee?.reason ? <p className="text-xs text-slate-500">{ieee.reason}</p> : null}
        <a className="text-xs text-mint underline" href="/ieee-eval">
          Open IEEE-CIS evaluation
        </a>
      </CardContent>
    </Card>
  );
}
