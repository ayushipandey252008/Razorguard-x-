"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

function fmt(value: unknown, digits = 4) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

export function PrototypeCalibrationCard({ calibration }: { calibration: any }) {
  if (!calibration) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Probability calibration</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-slate-500">Loading prototype calibration…</p>
        </CardContent>
      </Card>
    );
  }

  if (!calibration.available) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Probability calibration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <p className="text-[11px] uppercase tracking-wider text-ember">PROTOTYPE CALIBRATION</p>
          <p className="text-slate-500">{calibration.reason || "Calibration metrics have not been generated."}</p>
        </CardContent>
      </Card>
    );
  }

  const t = calibration.operating_thresholds || {};
  const costs = calibration.cost_assumptions || {};
  const test = calibration.test_once || {};

  return (
    <Card>
      <CardHeader>
        <CardTitle>Probability calibration</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <p className="text-[11px] uppercase tracking-wider text-ember">
          PROTOTYPE CALIBRATION · not industry-standard thresholds · offline ULB only
        </p>
        <p className="text-xs text-slate-500">
          Selected method: <span className="font-mono text-slate-200">{calibration.selected_method}</span>
          {" · "}
          booster <span className="font-mono">{calibration.booster_model_version}</span>
          {" · "}
          calibrated identity <span className="font-mono">{calibration.calibrated_model_version}</span>.
          Calibrated probability is not the product final risk score.
        </p>
        <div>
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-2">Raw vs calibrated (validation)</div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            <Metric label="Raw Brier" value={fmt(calibration.raw?.brier, 6)} />
            <Metric label="Calibrated Brier" value={fmt(calibration.calibrated?.brier, 6)} />
            <Metric label="Raw log loss" value={fmt(calibration.raw?.log_loss, 4)} />
            <Metric label="Calibrated log loss" value={fmt(calibration.calibrated?.log_loss, 4)} />
          </div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-2">Chronological test (once)</div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            <Metric label="Test raw Brier" value={fmt(test.raw_brier, 6)} />
            <Metric label="Test calibrated Brier" value={fmt(test.calibrated_brier, 6)} />
            <Metric label="Test raw PR-AUC" value={fmt(test.raw_pr_auc, 3)} />
            <Metric label="Test calibrated PR-AUC" value={fmt(test.pr_auc, 3)} />
          </div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-2">Operating thresholds</div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
            <Metric label="Approve below" value={fmt(t.approve_below, 2)} />
            <Metric label="Review" value={`${fmt(t.review_from, 2)}–${fmt(t.review_to, 2)}`} />
            <Metric label="Block above" value={fmt(t.block_above, 2)} />
          </div>
          <p className="text-xs text-slate-500 mt-2">
            Cost scenario {calibration.cost_scenario}: FN={costs.false_negative_cost}, FP=
            {costs.false_positive_cost}, review={costs.review_cost}. Not an empirically estimated loss given default.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-ink-950 p-3">
      <div className="text-[10px] uppercase text-slate-500">{label}</div>
      <div className="font-mono text-lg">{value}</div>
    </div>
  );
}
