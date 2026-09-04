"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { formatTime } from "@/lib/utils";
import { useLiveEvents } from "@/hooks/use-live-events";

export default function InvestigationsPage() {
  const [rows, setRows] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      setRows(await api("/api/v1/investigations"));
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

  useLiveEvents(
    useCallback(
      (e: any) => {
        if (e?.type === "transaction_processed" || e?.type === "high_risk_alert") {
          load();
        }
      },
      [load]
    )
  );

  return (
    <div className="space-y-5">
      <h1 className="text-2xl font-semibold">Case files</h1>
      {loading && <Skeleton className="h-40" />}
      {error && <p className="text-flare">{error}</p>}
      {!loading && !rows.length && <p className="text-slate-500">No investigations yet.</p>}
      <div className="grid gap-4 md:grid-cols-2">
        {rows.map((inv) => (
          <Link key={inv.id} href={`/investigations/${inv.id}`}>
            <Card className="hover:border-mint/30 transition-colors">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="font-mono text-xs">{inv.transaction_id}</CardTitle>
                <Badge decision={inv.decision}>{inv.status}</Badge>
              </CardHeader>
              <CardContent className="text-sm space-y-1">
                <div className="flex justify-between">
                  <span className="text-slate-500">Severity</span>
                  <span>{inv.severity}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Risk</span>
                  <span className="font-mono">{inv.final_risk_score ?? "—"}</span>
                </div>
                <div className="text-xs text-slate-500">{formatTime(inv.created_at)}</div>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
