"""Grounding checks for investigation reports. Not LLM accuracy."""

from __future__ import annotations

import re
from typing import Any

REQUIRED_TOOLS = (
    "get_transaction",
    "get_model_explanation",
    "get_triggered_rules",
)


def evaluate_investigation_grounding(result: dict[str, Any]) -> dict[str, Any]:
    report = result.get("report") or {}
    trace = result.get("tool_trace") or []
    tools_called = [t.get("tool") for t in trace]
    evidence_map = {}
    for entry in trace:
        name = entry.get("tool")
        if name:
            evidence_map[name] = entry.get("result") or {}

    missing_tools = [t for t in REQUIRED_TOOLS if t not in tools_called]
    unavailable = [
        name
        for name, payload in evidence_map.items()
        if isinstance(payload, dict) and payload.get("unavailable")
    ]

    narrative = " ".join(
        str(report.get(k) or "")
        for k in ("summary", "narrative", "rationale", "explanation", "recommended_action")
        if report.get(k)
    )
    evidence_blob = " ".join(str(v) for v in evidence_map.values())
    referenced = []
    for name in tools_called:
        token = name.replace("_", " ")
        if name in narrative or token in narrative.lower() or name in str(report):
            referenced.append(name)
    # Also count structured evidence fields the report copies.
    for key in ("evidence", "graph_evidence", "rules", "model_evidence"):
        if report.get(key):
            referenced.append(key)

    unsupported: list[str] = []
    for number in re.findall(r"\b\d{4,}\b", narrative):
        if number not in evidence_blob and number not in str(report.get("transaction_id") or ""):
            unsupported.append(number)

    limitations = str(report.get("limitations") or "")
    unavailable_represented = True
    for name in unavailable:
        if name not in limitations and "unavailable" not in limitations.lower() and "not available" not in limitations.lower():
            # Still OK if the report simply omitted that tool.
            unavailable_represented = bool(limitations)

    return {
        "provider": result.get("provider"),
        "tool_calls": len(trace),
        "tools_called": tools_called,
        "tool_trace_complete": not missing_tools,
        "missing_required_tools": missing_tools,
        "evidence_referenced": sorted(set(referenced)),
        "unsupported_numeric_claims": unsupported[:8],
        "unavailable_tools": unavailable,
        "unavailable_evidence_represented": bool(limitations) or not unavailable,
        "has_recommendation": bool(report.get("recommended_action") or report.get("recommendation")),
        "has_limitations": bool(limitations),
        "has_graph_evidence": bool(report.get("graph_evidence") or "find_fraud_cluster" in tools_called or "find_connected_accounts" in tools_called),
        "has_rules": bool(report.get("triggered_rules") or "get_triggered_rules" in tools_called),
        "has_model_evidence": bool(report.get("model_evidence") or "get_model_explanation" in tools_called),
        "note": (
            "Grounding checklist against tool traces. "
            "This is not a claim of LLM accuracy or real-world investigative quality."
        ),
    }
