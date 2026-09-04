"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { PrototypeCalibrationCard } from "@/components/prototype-calibration";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function AnalyticsPage() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api("/api/v1/analytics")
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="text-flare">{error}</p>;
  if (!data) return <p className="text-slate-400">Loading telemetry…</p>;

  const decisions = Object.entries(data.decisions || {}).map(([k, v]) => ({ name: k, count: v }));
  const metrics = data.model?.metrics || {};

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Telemetry</h1>
        <p className="text-sm text-slate-400">{data.disclaimer}</p>
      </header>
      <div className="grid gap-4 lg:grid-cols-2">
        <ChartCard title="Volume & flagged">
          <LineChart data={data.volume || []}>
            <CartesianGrid stroke="#1a2433" />
            <XAxis dataKey="date" stroke="#64748b" fontSize={11} />
            <YAxis stroke="#64748b" fontSize={11} />
            <Tooltip contentStyle={{ background: "#121923" }} />
            <Line type="monotone" dataKey="count" stroke="#3ee0c6" />
            <Line type="monotone" dataKey="flagged" stroke="#ff4d6d" />
          </LineChart>
        </ChartCard>
        <ChartCard title="Approve / review / block">
          <BarChart data={decisions}>
            <XAxis dataKey="name" stroke="#64748b" />
            <YAxis stroke="#64748b" />
            <Tooltip contentStyle={{ background: "#121923" }} />
            <Bar dataKey="count" fill="#ffb020" />
          </BarChart>
        </ChartCard>
        <ChartCard title="Risk distribution">
          <BarChart data={data.risk_distribution || []}>
            <XAxis dataKey="bucket" stroke="#64748b" />
            <YAxis stroke="#64748b" />
            <Tooltip contentStyle={{ background: "#121923" }} />
            <Bar dataKey="count" fill="#7aa2ff" />
          </BarChart>
        </ChartCard>
        <ChartCard title="Flagged by location">
          <BarChart data={data.by_location || []}>
            <XAxis dataKey="location" stroke="#64748b" fontSize={11} />
            <YAxis stroke="#64748b" />
            <Tooltip contentStyle={{ background: "#121923" }} />
            <Bar dataKey="flagged" fill="#ff4d6d" />
          </BarChart>
        </ChartCard>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Model evaluation (held-out synthetic test)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            {["precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate"].map(
              (k) => (
                <div key={k} className="rounded-lg bg-ink-950 p-3">
                  <div className="text-[10px] uppercase text-slate-500">{k}</div>
                  <div className="font-mono text-lg">{metrics[k] != null ? Number(metrics[k]).toFixed(3) : "—"}</div>
                </div>
              )
            )}
          </div>
          <p className="text-xs text-slate-500 mt-3">Version {data.model?.version}. Accuracy is not the primary metric.</p>
        </CardContent>
      </Card>
      <UlbOfflineCard />
      <CalibrationSection />
      <Card>
        <CardHeader>
          <CardTitle>Merchant category mix</CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-slate-500">
              <tr>
                <th>Category</th>
                <th>Approve</th>
                <th>Review</th>
                <th>Block</th>
              </tr>
            </thead>
            <tbody>
              {(data.by_category || []).map((r: any) => (
                <tr key={r.category} className="border-t border-white/5">
                  <td className="py-2">{r.category}</td>
                  <td>{r.APPROVE}</td>
                  <td>{r.REVIEW}</td>
                  <td>{r.BLOCK}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  );
}

function ChartCard({ title, children }: { title: string; children: any }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function CalibrationSection() {
  const [calibration, setCalibration] = useState<any>(null);
  useEffect(() => {
    api("/api/v1/ml/offline-evaluation")
      .then((d) => setCalibration(d.calibration))
      .catch(() => setCalibration({ available: false }));
  }, []);
  return <PrototypeCalibrationCard calibration={calibration} />;
}

function UlbOfflineCard() {
  const [ulb, setUlb] = useState<any>(null);
  useEffect(() => {
    api("/api/v1/ml/offline-evaluation")
      .then((d) => setUlb(d.ulb))
      .catch(() => setUlb({ available: false }));
  }, []);
  if (!ulb?.available) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>ULB offline evaluation</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-slate-500">
            OFFLINE EVALUATION only. {ulb?.reason || "Run PYTHONPATH=. python ml/training/train_ulb.py to generate metrics."}
          </p>
        </CardContent>
      </Card>
    );
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle>ULB Credit Card Fraud Detection — OFFLINE EVALUATION</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <p className="text-xs text-slate-500">
          Dataset: {ulb.dataset} · Model: {ulb.model} · Version: {ulb.model_version}. Not mixed with synthetic live scores.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {[
            ["PR-AUC", ulb.pr_auc],
            ["ROC-AUC", ulb.roc_auc],
            ["Precision", ulb.precision],
            ["Recall", ulb.recall],
            ["F1", ulb.f1],
          ].map(([label, value]) => (
            <div key={String(label)} className="rounded-lg bg-ink-950 p-3">
              <div className="text-[10px] uppercase text-slate-500">{label}</div>
              <div className="font-mono text-lg">{value != null ? Number(value).toFixed(3) : "—"}</div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
