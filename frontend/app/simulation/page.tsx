"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import Link from "next/link";

const SCENARIOS = [
  ["normal", "Typical user spend"],
  ["stolen_account", "High-value unknown device"],
  ["card_testing", "Tiny amounts, high velocity"],
  ["account_takeover", "New device + destination shift"],
  ["device_farm", "Many users, one device"],
  ["fraud_ring", "Coordinated shared infrastructure"],
  ["velocity_attack", "Burst of payments"],
];

export default function SimulationPage() {
  const [scenario, setScenario] = useState("fraud_ring");
  const [count, setCount] = useState(8);
  const [result, setResult] = useState<any>(null);
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    try {
      const data = await api("/api/v1/simulation/run", {
        method: "POST",
        body: JSON.stringify({ scenario, count }),
      });
      setResult(data);
      toast.success(`Scored ${data.count} synthetic payments through the live pipeline`);
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Simulation range</h1>
        <p className="text-sm text-slate-400">
          Generated payments are scored by the real ML, rules, graph, and risk engine. Results are not faked.
        </p>
      </header>
      <Card>
        <CardContent className="p-5 flex flex-wrap gap-3 items-end">
          <label className="text-sm">
            Scenario
            <select
              className="mt-1 block h-10 rounded-md bg-ink-900 border border-white/10 px-3"
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
            >
              {SCENARIOS.map(([id, label]) => (
                <option key={id} value={id}>
                  {id} — {label}
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm">
            Count
            <input
              type="number"
              min={1}
              max={80}
              className="mt-1 block h-10 w-24 rounded-md bg-ink-900 border border-white/10 px-3"
              value={count}
              onChange={(e) => setCount(Number(e.target.value))}
            />
          </label>
          <Button onClick={run} disabled={busy}>
            {busy ? "Scoring…" : "Fire scenario"}
          </Button>
        </CardContent>
      </Card>
      {result && (
        <div className="space-y-4">
          <div className="grid sm:grid-cols-4 gap-3">
            <Stat label="Generated" value={result.count} />
            <Stat label="Flagged" value={result.detected_flagged} />
            <Stat label="FP vs injected label" value={result.false_positives_vs_injected_label} />
            <Stat label="Clusters" value={result.detected_clusters?.length || 0} />
          </div>
          <p className="text-xs text-slate-500">{result.note}</p>
          <Card>
            <CardHeader>
              <CardTitle>Pipeline output</CardTitle>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-xs text-slate-500">
                  <tr>
                    <th>Txn</th>
                    <th>User</th>
                    <th>Score</th>
                    <th>Decision</th>
                    <th>Injected</th>
                  </tr>
                </thead>
                <tbody>
                  {result.transactions.map((t: any) => (
                    <tr key={t.transaction_id} className="border-t border-white/5">
                      <td className="py-2 font-mono text-xs">
                        <Link className="text-mint" href={`/transactions/${t.transaction_id}`}>
                          {t.transaction_id.slice(0, 16)}
                        </Link>
                      </td>
                      <td className="font-mono text-xs">{t.user_id}</td>
                      <td className="font-mono">{t.final_risk_score}</td>
                      <td>
                        <Badge decision={t.decision}>{t.decision}</Badge>
                      </td>
                      <td>{t.injected_label ? "pattern" : "normal"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: any }) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="text-[11px] uppercase text-slate-500">{label}</div>
        <div className="text-2xl font-mono">{value}</div>
      </CardContent>
    </Card>
  );
}
