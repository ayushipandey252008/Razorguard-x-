"use client";

import { useMemo, useState } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";

const ALL = [
  "normal_payment",
  "stolen_account",
  "card_testing",
  "high_velocity",
  "unusual_amount",
  "new_device",
  "shared_device",
  "shared_ip",
  "device_farm",
  "fraud_ring",
];

export default function ScenarioEvaluationPage() {
  const [selected, setSelected] = useState<string[]>(["normal_payment", "stolen_account", "card_testing", "fraud_ring"]);
  const [count, setCount] = useState(8);
  const [seed, setSeed] = useState(42);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<any>(null);

  const overall = result?.overall || {};
  const matrix = result?.scenario_matrix || [];

  async function run() {
    setBusy(true);
    try {
      const data = await api("/api/v1/simulation/evaluate", {
        method: "POST",
        body: JSON.stringify({
          scenarios: selected,
          count_per_scenario: count,
          seed,
          run_investigations: false,
        }),
      });
      setResult(data);
      toast.success(`Scored ${data.n} synthetic scenario payments`);
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setBusy(false);
    }
  }

  const toggle = (name: string) => {
    setSelected((cur) => (cur.includes(name) ? cur.filter((x) => x !== name) : [...cur, name]));
  };

  const catchRate = useMemo(() => overall.fraud_catch_rate, [overall]);

  return (
    <div className="space-y-5 max-w-4xl">
      <header>
        <h1 className="text-xl font-semibold sm:text-2xl">Scenario evaluation</h1>
        <p className="text-[11px] uppercase tracking-wider text-ember">Synthetic scenario evaluation — not public dataset evaluation</p>
        <p className="text-sm text-slate-400 mt-1">
          Generator labels describe synthetic patterns. They are not ULB Class labels and not real-world fraud accuracy.
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Run</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {ALL.map((name) => (
              <button
                key={name}
                type="button"
                onClick={() => toggle(name)}
                className={`min-h-9 rounded-full border px-3 py-2 text-xs ${
                  selected.includes(name) ? "border-mint/40 bg-mint/10 text-mint" : "border-white/10 text-slate-400"
                }`}
              >
                {name}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap gap-3 items-end">
            <label className="text-sm">
              Transactions per scenario
              <input
                type="number"
                min={1}
                max={50}
                className="mt-1 block h-10 w-24 rounded-md bg-ink-900 border border-white/10 px-3"
                value={count}
                onChange={(e) => setCount(Number(e.target.value))}
              />
            </label>
            <label className="text-sm">
              Seed
              <input
                type="number"
                min={0}
                className="mt-1 block h-10 w-24 rounded-md bg-ink-900 border border-white/10 px-3"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
              />
            </label>
            <Button className="w-full sm:w-auto" onClick={run} disabled={busy || selected.length === 0}>
              {busy ? "Scoring…" : "Evaluate"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {result ? (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Overall metrics</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <Stat label="Fraud catch rate" value={catchRate} />
              <Stat label="False positive rate" value={overall.false_positive_rate} />
              <Stat label="Review rate" value={overall.review_rate} />
              <Stat label="Precision" value={overall.precision} />
              <Stat label="Recall" value={overall.recall} />
              <Stat label="F1" value={overall.f1} />
              <Stat label="Block rate" value={overall.block_rate} />
              <Stat label="Approve rate" value={overall.approve_rate} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Scenario matrix</CardTitle>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              <table className="w-full min-w-[32rem] text-xs">
                <thead className="text-slate-500">
                  <tr>
                    <th className="text-left py-1">Scenario</th>
                    <th className="text-right">Fraud</th>
                    <th className="text-right">Block</th>
                    <th className="text-right">Review</th>
                    <th className="text-right">Approve</th>
                    <th className="text-right">Catch</th>
                  </tr>
                </thead>
                <tbody>
                  {matrix.map((row: any) => (
                    <tr key={row.scenario} className="border-t border-white/5">
                      <td className="py-1">{row.scenario}</td>
                      <td className="text-right font-mono">{row.fraud}</td>
                      <td className="text-right font-mono">{row.block}</td>
                      <td className="text-right font-mono">{row.review}</td>
                      <td className="text-right font-mono">{row.approve}</td>
                      <td className="text-right font-mono">{row.catch}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </>
      ) : null}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | undefined }) {
  return (
    <div className="rounded-lg bg-ink-950 p-3">
      <div className="text-[10px] uppercase text-slate-500">{label}</div>
      <div className="font-mono text-lg">
        {value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(3)}
      </div>
    </div>
  );
}
