"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { TableScroll } from "@/components/layout/table-scroll";
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
      <header>
        <h1 className="text-xl font-semibold sm:text-2xl">Live wire</h1>
        <p className="mt-1 text-sm text-slate-400">
          WebSocket {connected ? "connected" : "not connected"} · updates without refresh
        </p>
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
            <TableScroll>
              <table className="w-full min-w-[760px] text-sm">
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
                      <td className="max-w-[10rem] truncate py-2 font-mono text-xs" title={row.transaction_id}>
                        <Link className="text-mint" href={`/transactions/${row.transaction_id}`}>
                          {row.transaction_id}
                        </Link>
                      </td>
                      <td className="max-w-[8rem] truncate font-mono text-xs" title={row.user_id}>
                        {row.user_id}
                      </td>
                      <td className="max-w-[8rem] truncate font-mono text-xs" title={row.merchant_id}>
                        {row.merchant_id}
                      </td>
                      <td className="whitespace-nowrap">{formatInr(row.amount)}</td>
                      <td className="max-w-[7rem] truncate" title={row.location}>
                        {row.location}
                      </td>
                      <td className="font-mono">{row.final_risk_score ?? "—"}</td>
                      <td>
                        <Badge decision={row.decision}>{row.decision || "—"}</Badge>
                      </td>
                      <td className="whitespace-nowrap text-xs text-slate-500">{formatTime(row.timestamp)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </TableScroll>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
