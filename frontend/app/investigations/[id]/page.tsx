"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { InvestigationReportView } from "@/components/investigation-report";
import { toast } from "sonner";
import Link from "next/link";

export default function InvestigationDetail() {
  const { id } = useParams<{ id: string }>();
  const [data, setData] = useState<any>(null);
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);

  async function load() {
    setData(await api(`/api/v1/investigations/${id}`));
  }
  useEffect(() => {
    load().catch((e) => toast.error(e.message));
  }, [id]);

  async function run() {
    setBusy(true);
    try {
      await api(`/api/v1/investigations/${id}/run`, { method: "POST" });
      toast.success("Agent finished");
      await load();
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function feedback(decision: string) {
    if (reason.trim().length < 3) {
      toast.error("Record a reason");
      return;
    }
    setBusy(true);
    try {
      await api("/api/v1/feedback", {
        method: "POST",
        body: JSON.stringify({ investigation_id: id, decision, reason }),
      });
      toast.success(`Feedback ${decision}`);
      await load();
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function decide(decision: string) {
    if (reason.trim().length < 3) {
      toast.error("Record a reason");
      return;
    }
    setBusy(true);
    try {
      await api(`/api/v1/investigations/${id}/decision`, {
        method: "POST",
        body: JSON.stringify({ decision, reason }),
      });
      toast.success(`Recorded ${decision}`);
      await load();
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setBusy(false);
    }
  }

  if (!data) return <p className="text-slate-400">Loading case…</p>;
  const report = data.ai_report;

  return (
    <div className="space-y-5">
      <header className="flex flex-col gap-3 sm:flex-row sm:justify-between">
        <div className="min-w-0">
          <h1 className="text-xl font-semibold sm:text-2xl">Investigation</h1>
          <p className="mt-1 break-all font-mono text-sm text-slate-400">
            {data.id} ·{" "}
            <Link className="text-mint" href={`/transactions/${data.transaction_id}`}>
              {data.transaction_id}
            </Link>
          </p>
        </div>
        <Badge className="self-start">{data.status}</Badge>
      </header>

      <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap">
        <Button size="sm" className="w-full sm:w-auto" onClick={run} disabled={busy}>
          Run AI investigation
        </Button>
        <Button size="sm" className="w-full sm:w-auto" variant="outline" onClick={() => decide("APPROVE")} disabled={busy}>
          Approve
        </Button>
        <Button size="sm" className="w-full sm:w-auto" variant="danger" onClick={() => decide("BLOCK")} disabled={busy}>
          Block
        </Button>
        <Button size="sm" className="w-full sm:w-auto" variant="warn" onClick={() => decide("ESCALATE")} disabled={busy}>
          Escalate
        </Button>
      </div>
      <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap">
        <Button size="sm" className="w-full sm:w-auto" variant="danger" onClick={() => feedback("CONFIRM_FRAUD")} disabled={busy}>
          Confirm fraud
        </Button>
        <Button size="sm" className="w-full sm:w-auto" variant="outline" onClick={() => feedback("CONFIRM_LEGITIMATE")} disabled={busy}>
          Confirm legitimate
        </Button>
        <Button size="sm" className="w-full sm:w-auto" variant="ghost" onClick={() => feedback("NEEDS_REVIEW")} disabled={busy}>
          Needs review
        </Button>
      </div>
      <Input placeholder="Analyst reason (required for human decision)" value={reason} onChange={(e) => setReason(e.target.value)} />

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>AI report</CardTitle>
          </CardHeader>
          <CardContent>
            {!report ? (
              <p className="text-sm text-slate-500">Not run yet. The agent only uses backend tools — it cannot invent evidence.</p>
            ) : (
              <InvestigationReportView report={report} provider={data.agent_provider} />
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Analyst decisions & audit</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            {(data.analyst_decisions || []).map((d: any) => (
              <div key={d.id} className="border border-white/10 rounded-lg p-3">
                <div className="font-medium">
                  {d.decision} by {d.actor_email}
                </div>
                <div className="text-slate-400">{d.reason}</div>
                <div className="text-xs text-slate-500">Previous AI rec: {d.previous_ai_recommendation || "—"}</div>
              </div>
            ))}
            {(data.audit_history || []).slice(0, 12).map((a: any, i: number) => (
              <div key={i} className="text-xs text-slate-500">
                {a.action} · {a.actor}
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
