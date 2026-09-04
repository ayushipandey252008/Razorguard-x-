"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { formatInr, formatTime } from "@/lib/utils";
import { useLiveEvents } from "@/hooks/use-live-events";
import type { Transaction } from "@/types";

export default function TransactionsPage() {
  const [rows, setRows] = useState<Transaction[]>([]);
  const [flash, setFlash] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setRows(await api("/api/v1/transactions?limit=80"));
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

  const connected = useLiveEvents(
    useCallback(
      (e: any) => {
        setFlash((prev) => [e, ...prev].slice(0, 8));
        load();
      },
      [load]
    )
  );

  return (
    <div className="space-y-5">
      <header className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Live wire</h1>
          <p className="text-sm text-slate-400">WebSocket {connected ? "connected" : "not connected"} · updates without refresh</p>
        </div>
      </header>
      {flash.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {flash.map((e, i) => (
            <Badge key={i} decision={e.decision}>
              {e.type}: {e.transaction_id?.slice(0, 10)} {e.final_risk_score}
            </Badge>
          ))}
        </div>
      )}
      <Card>
        <CardHeader>
          <CardTitle>Scored transactions</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <Skeleton className="h-48" />
          ) : error ? (
            <p className="text-flare text-sm">{error}</p>
          ) : rows.length === 0 ? (
            <p className="text-slate-500 text-sm">Empty. Seed or simulate traffic.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-[11px] uppercase text-slate-500">
                  <tr>
                    <th className="pb-2">ID</th>
                    <th>User</th>
                    <th>Merchant</th>
                    <th>Amount</th>
                    <th>Loc</th>
                    <th>Score</th>
                    <th>Decision</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.transaction_id} className="border-t border-white/5 hover:bg-white/[0.03]">
                      <td className="py-2 font-mono text-xs">
                        <Link className="text-mint" href={`/transactions/${row.transaction_id}`}>
                          {row.transaction_id}
                        </Link>
                      </td>
                      <td className="font-mono text-xs">{row.user_id}</td>
                      <td className="font-mono text-xs">{row.merchant_id}</td>
                      <td>{formatInr(row.amount)}</td>
                      <td>{row.location}</td>
                      <td className="font-mono">{row.final_risk_score ?? "—"}</td>
                      <td>
                        <Badge decision={row.decision}>{row.decision || "—"}</Badge>
                      </td>
                      <td className="text-xs text-slate-500">{formatTime(row.timestamp)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
