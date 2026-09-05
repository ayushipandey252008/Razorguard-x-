"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { TableScroll } from "@/components/layout/table-scroll";
import { formatInr, formatTime } from "@/lib/utils";
import { useLiveEvents } from "@/hooks/use-live-events";
import type { Transaction } from "@/types";

export default function DashboardPage() {
  const [stats, setStats] = useState<any>(null);
  const [txns, setTxns] = useState<Transaction[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const [a, t] = await Promise.all([api("/api/v1/analytics"), api<Transaction[]>("/api/v1/transactions?limit=8")]);
      setStats(a);
      setTxns(t);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const live = useLiveEvents(
    useCallback(() => {
      load();
    }, [load])
  );

  if (loading) {
    return (
      <div className="grid gap-4 md:grid-cols-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <Skeleton key={i} className="h-28" />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="p-6 text-flare">
          Could not reach the API. Start the backend on :8000 then refresh. {error}
        </CardContent>
      </Card>
    );
  }

  const t = stats.totals;
  const tiles = [
    ["Transactions", t.transactions],
    ["Reviewed / closed cases", t.reviewed],
    ["High-risk (review+block)", t.high_risk],
    ["Flag rate", `${((t.fraud_rate || 0) * 100).toFixed(1)}%`],
    ["Blocked", t.blocked],
    ["Potential rings", t.potential_fraud_rings],
    ["Active investigations", t.active_investigations],
    ["Live socket", live ? "connected" : "offline"],
  ];

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold sm:text-2xl">Command floor</h1>
        <p className="text-sm text-slate-400 mt-1">
          Live scoring of synthetic payments. Numbers below are queried from the API, not placeholders.
        </p>
      </header>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {tiles.map(([label, value]) => (
          <Card key={label}>
            <CardHeader>
              <CardTitle className="text-[11px] uppercase tracking-[0.2em] text-slate-500">{label}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="font-mono text-2xl font-semibold text-slate-50 sm:text-3xl">{value}</div>
            </CardContent>
          </Card>
        ))}
      </div>
      <Card>
        <CardHeader className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <CardTitle>Latest scored payments</CardTitle>
          <Link href="/transactions" className="text-xs text-mint">
            Open live wire →
          </Link>
        </CardHeader>
        <CardContent>
          {txns.length === 0 ? (
            <p className="text-sm text-slate-500">No transactions yet. Run a simulation or POST a payment.</p>
          ) : (
            <>
              <ul className="space-y-3 md:hidden">
                {txns.map((row) => (
                  <li key={row.transaction_id} className="rounded-lg border border-white/10 p-3 text-sm">
                    <div className="flex items-start justify-between gap-3">
                      <Link className="min-w-0 break-all font-mono text-xs text-mint" href={`/transactions/${row.transaction_id}`}>
                        {row.transaction_id}
                      </Link>
                      <Badge decision={row.decision}>{row.decision || "—"}</Badge>
                    </div>
                    <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-slate-400">
                      <span className="break-all font-mono">{row.user_id}</span>
                      <span className="text-right">{formatInr(row.amount)}</span>
                      <span className="font-mono">score {row.final_risk_score ?? "—"}</span>
                      <span className="text-right">{formatTime(row.timestamp)}</span>
                    </div>
                  </li>
                ))}
              </ul>
              <TableScroll className="hidden md:block">
                <table className="w-full min-w-[640px] text-sm">
                  <thead className="text-left text-[11px] uppercase tracking-wider text-slate-500">
                    <tr>
                      <th className="pb-2">Txn</th>
                      <th>User</th>
                      <th>Amount</th>
                      <th>Score</th>
                      <th>Decision</th>
                      <th>When</th>
                    </tr>
                  </thead>
                  <tbody>
                    {txns.map((row) => (
                      <tr key={row.transaction_id} className="border-t border-white/5">
                        <td className="py-2 font-mono text-xs">
                          <Link className="text-mint" href={`/transactions/${row.transaction_id}`}>
                            {row.transaction_id.slice(0, 14)}
                          </Link>
                        </td>
                        <td className="font-mono text-xs">{row.user_id}</td>
                        <td>{formatInr(row.amount)}</td>
                        <td className="font-mono">{row.final_risk_score ?? "—"}</td>
                        <td>
                          <Badge decision={row.decision}>{row.decision || "—"}</Badge>
                        </td>
                        <td className="text-xs text-slate-500">{formatTime(row.timestamp)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </TableScroll>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
