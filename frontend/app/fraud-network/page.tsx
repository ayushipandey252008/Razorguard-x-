"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  type Node,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const COLORS: Record<string, string> = {
  USER: "#3ee0c6",
  DEVICE: "#7aa2ff",
  IP: "#ffb020",
  MERCHANT: "#ff4d6d",
  LOCATION: "#c084fc",
  PAYMENT: "#94a3b8",
  TRANSACTION: "#38bdf8",
};

export default function FraudNetworkPage() {
  const [snap, setSnap] = useState<{ nodes: any[]; edges: any[]; metrics?: any; graph_backend?: string }>({
    nodes: [],
    edges: [],
  });
  const [metrics, setMetrics] = useState<any>(null);
  const [selected, setSelected] = useState<any>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [clusters, setClusters] = useState<any[]>([]);

  useEffect(() => {
    api("/api/v1/graph/snapshot").then(setSnap).catch(console.error);
    api("/api/v1/graph/metrics").then(setMetrics).catch(console.error);
    api("/api/v1/graph/clusters").then(setClusters).catch(console.error);
  }, []);

  const backend = metrics?.graph_backend || snap.graph_backend || "networkx";
  const connected = metrics?.graph_connected;
  const entityCounts = metrics?.entity_counts || snap.metrics?.entity_counts || {};
  const relCounts = metrics?.relationship_counts || snap.metrics?.relationship_counts || {};

  const neighborIds = useMemo(() => {
    if (!selectedId) return null;
    const ids = new Set<string>([selectedId]);
    for (const e of snap.edges) {
      if (e.source === selectedId) ids.add(e.target);
      if (e.target === selectedId) ids.add(e.source);
    }
    return ids;
  }, [selectedId, snap.edges]);

  const { nodes, edges } = useMemo(() => {
    const grouped: Record<string, any[]> = {};
    for (const n of snap.nodes) {
      (grouped[n.entity_type] ||= []).push(n);
    }
    const types = Object.keys(COLORS);
    const rfNodes: Node[] = [];
    types.forEach((type, col) => {
      (grouped[type] || []).forEach((n, row) => {
        const related = !neighborIds || neighborIds.has(n.id);
        const suspicious = (n.degree || 0) >= 3 && (type === "DEVICE" || type === "IP");
        rfNodes.push({
          id: n.id,
          position: { x: col * 220, y: row * 90 },
          data: { label: `${n.entity_type}\n${n.entity_key}` },
          style: {
            background: n.id === selectedId ? "#1a3d38" : "#121923",
            color: COLORS[type],
            border: `1px solid ${suspicious ? "#ff4d6d" : COLORS[type]}`,
            fontSize: 11,
            fontFamily: "IBM Plex Mono, monospace",
            whiteSpace: "pre",
            width: 160,
            opacity: related ? 1 : 0.22,
          },
        });
      });
    });
    const rfEdges: Edge[] = snap.edges.map((e, i) => {
      const linked = !neighborIds || (neighborIds.has(e.source) && neighborIds.has(e.target));
      const selectedEdge = Boolean(selectedId && (e.source === selectedId || e.target === selectedId));
      return {
        id: `e${i}`,
        source: e.source,
        target: e.target,
        label: (e.rel_types || []).join(","),
        style: {
          stroke: selectedEdge ? "#3ee0c6" : "#334155",
          strokeWidth: selectedEdge ? 2 : 1,
          opacity: linked ? 1 : 0.15,
        },
      };
    });
    return { nodes: rfNodes, edges: rfEdges };
  }, [snap, neighborIds, selectedId]);

  const onNodeClick = useCallback(async (_: any, node: Node) => {
    setSelectedId(node.id);
    try {
      const info = await api(`/api/v1/graph/${encodeURIComponent(node.id)}`);
      setSelected(info);
    } catch (e) {
      setSelected({ error: String(e) });
    }
  }, []);

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-2xl font-semibold">Entity graph</h1>
        <p className="text-sm text-slate-400">
          Relationship map via GraphBackend. Clusters are potential rings, not confirmed fraud.
        </p>
      </header>
      <div className="flex flex-wrap gap-2 items-center">
        <Badge>Graph backend: {backend === "neo4j" ? "Neo4j" : "NetworkX"}</Badge>
        {connected === false && <Badge>not connected{metrics?.reason ? ` · ${metrics.reason}` : ""}</Badge>}
        {metrics?.fallback && <Badge>NetworkX fallback</Badge>}
      </div>
      <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6 text-xs">
        {["USER", "DEVICE", "IP", "MERCHANT", "LOCATION", "TRANSACTION"].map((k) => (
          <div key={k} className="rounded-lg border border-white/10 p-2">
            <div className="text-slate-500">{k}</div>
            <div className="font-mono text-lg">{entityCounts[k] ?? 0}</div>
          </div>
        ))}
      </div>
      <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
        <div className="h-[640px] rounded-xl border border-white/10 overflow-hidden bg-ink-900">
          <ReactFlow nodes={nodes} edges={edges} fitView onNodeClick={onNodeClick}>
            <Background color="#1a2433" />
            <Controls />
            <MiniMap nodeColor="#3ee0c6" maskColor="rgba(7,9,13,0.8)" />
          </ReactFlow>
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Relationships</CardTitle>
            </CardHeader>
            <CardContent className="text-xs space-y-1">
              {!Object.keys(relCounts).length && <p className="text-slate-500">No edges yet.</p>}
              {Object.entries(relCounts).map(([rel, n]) => (
                <div key={rel} className="flex justify-between gap-2 font-mono">
                  <span className="text-slate-400">{rel}</span>
                  <span>{String(n)}</span>
                </div>
              ))}
              <p className="text-[10px] text-slate-500 pt-2">
                {metrics?.node_count ?? snap.nodes.length} nodes · {metrics?.edge_count ?? snap.edges.length}{" "}
                edges
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Selected entity</CardTitle>
            </CardHeader>
            <CardContent>
              {!selected ? (
                <p className="text-sm text-slate-500">Click a node to load neighbors.</p>
              ) : (
                <pre className="text-[11px] overflow-x-auto text-slate-300">{JSON.stringify(selected, null, 2)}</pre>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Potential rings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {!clusters.length && <p className="text-sm text-slate-500">No clusters with ≥3 users.</p>}
              {clusters.map((c) => (
                <div key={c.cluster_id} className="text-xs border border-white/10 rounded-lg p-3">
                  <div className="font-medium">
                    {c.user_count} users · score {c.graph_risk_score}
                  </div>
                  <p className="text-slate-400 mt-1">{c.explanation}</p>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
