"""Investigation agent.

Flow: request → investigator → tool registry → structured evidence →
LLM or deterministic synthesis → grounded InvestigationReport.

The LLM never queries the database, never runs SQL, never calculates
fraud probability, and never invents evidence.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.prompts import SYSTEM_PROMPT, investigation_user_prompt, wrap_untrusted
from app.agents.provider import get_provider
from app.agents.registry import registry
from app.agents.schemas import VALID_RECOMMENDATIONS, VALID_RISK_LEVELS
from app.agents.tools import TOOL_SPECS, ToolBox
from app.utils.logging import Timer, get_logger
from app.utils.redact import redact_secrets, redact_text

log = get_logger("agent")

MAX_TOOL_ITERATIONS = 12


async def run_investigation(
    db: AsyncSession,
    transaction_id: str,
    investigation_id: str | None = None,
) -> dict:
    timer = Timer()
    tools = ToolBox(db)
    seed = await tools.get_transaction(transaction_id)
    if seed.get("unavailable"):
        report = _empty_report(transaction_id, investigation_id)
        result = {
            "provider": "deterministic_fallback",
            "model": None,
            "report": report,
            "tool_trace": [],
            "latency_ms": timer.ms(),
            "fallback_reason": seed.get("reason"),
        }
        log.info(
            "investigation_complete",
            investigation_id=investigation_id,
            transaction_id=transaction_id,
            provider="deterministic_fallback",
            model=None,
            latency_ms=result["latency_ms"],
            tool_calls=0,
            tool_failures=0,
            fallback=True,
        )
        return result

    provider = get_provider()
    if not getattr(provider, "supports_tool_calling", False):
        result = await _deterministic(tools, seed, investigation_id=investigation_id)
        result["latency_ms"] = timer.ms()
        _log_complete(result, investigation_id, transaction_id, fallback=False)
        return result

    try:
        result = await _llm_loop(provider, tools, seed, investigation_id=investigation_id)
        result["latency_ms"] = timer.ms()
        _log_complete(result, investigation_id, transaction_id, fallback=False)
        return result
    except Exception as exc:
        safe_error = redact_text(str(exc))
        log.warning(
            "llm_failed_fallback",
            investigation_id=investigation_id,
            transaction_id=transaction_id,
            provider="llm",
            model=getattr(provider, "model", None),
            error=safe_error,
        )
        result = await _deterministic(tools, seed, investigation_id=investigation_id)
        result["provider"] = "deterministic_fallback"
        result["fallback_reason"] = f"LLM provider error; used deterministic investigator."
        result["latency_ms"] = timer.ms()
        result["report"]["provider"] = "deterministic_fallback"
        result["report"]["limitations"] = (
            (result["report"].get("limitations") or "")
            + " LLM provider failed; used deterministic investigator. No API secrets are stored."
        )
        _log_complete(result, investigation_id, transaction_id, fallback=True)
        return result


def _log_complete(result: dict, investigation_id: str | None, transaction_id: str, fallback: bool) -> None:
    trace = result.get("tool_trace") or []
    failures = [t for t in trace if t.get("status") in {"error", "unavailable"}]
    log.info(
        "investigation_complete",
        investigation_id=investigation_id,
        transaction_id=transaction_id,
        provider=result.get("provider"),
        model=result.get("model"),
        latency_ms=result.get("latency_ms"),
        tool_calls=len(trace),
        tool_failures=len(failures),
        fallback=fallback,
    )


async def _invoke_tool(tools: ToolBox, name: str, arguments: dict) -> dict:
    tool_timer = Timer()
    result = await tools.call(name, arguments)
    status = result.get("status") or ("unavailable" if result.get("unavailable") else "success")
    duration = result.get("duration_ms", tool_timer.ms())
    if status == "error":
        log.warning("tool_failure", tool=name, status=status, duration_ms=duration)
    else:
        log.info("tool_call", tool=name, status=status, duration_ms=duration)
    return {
        "tool": name,
        "arguments": redact_secrets(arguments or {}),
        "status": status,
        "duration_ms": duration,
        "result_summary": _summarize_result(name, result),
        "result": redact_secrets(result),
    }


async def _llm_loop(provider, tools: ToolBox, seed: dict, investigation_id: str | None = None) -> dict:
    txn_id = seed["transaction_id"]
    model_snap = await tools.get_model_explanation(txn_id)
    risk_snapshot = {
        "decision": model_snap.get("decision") if not model_snap.get("unavailable") else None,
        "ml_probability": model_snap.get("ml_probability") if not model_snap.get("unavailable") else None,
        "ml_score": model_snap.get("ml_score") if not model_snap.get("unavailable") else None,
        "final_risk_score": model_snap.get("final_risk_score") if not model_snap.get("unavailable") else None,
        "model_version": model_snap.get("model_version") if not model_snap.get("unavailable") else None,
        "note": "Copy these risk-engine values. Do not invent a probability.",
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": investigation_user_prompt(txn_id, seed, risk_snapshot),
        },
    ]
    trace: list[dict] = []
    evidence_map: dict[str, dict] = {}
    for _ in range(MAX_TOOL_ITERATIONS):
        msg = await provider.complete_with_tools(messages, TOOL_SPECS)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            report = _parse_report(msg.get("content") or "")
            evidence_map = await _ensure_grounding_tools(tools, seed, evidence_map, trace)
            grounded = _ground_report(
                report,
                seed,
                evidence_map,
                trace,
                provider_name="llm",
                model_name=getattr(provider, "model", None),
                investigation_id=investigation_id,
            )
            return {
                "provider": "llm",
                "model": getattr(provider, "model", None),
                "report": grounded,
                "tool_trace": trace,
            }
        messages.append(msg)
        for call in tool_calls:
            fn = (call.get("function") or {}).get("name") or ""
            raw_args = (call.get("function") or {}).get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError:
                args = {}
            entry = await _invoke_tool(tools, fn, args if isinstance(args, dict) else {})
            trace.append(entry)
            evidence_map[fn] = entry["result"]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", fn),
                    "name": fn,
                    "content": wrap_untrusted(
                        json.dumps(entry["result"], default=str),
                        fn,
                    ),
                }
            )
    fallback = await _deterministic(tools, seed, investigation_id=investigation_id)
    fallback["provider"] = "deterministic_fallback"
    fallback["fallback_reason"] = "LLM tool-calling loop reached the iteration cap."
    fallback["tool_trace"] = trace + fallback.get("tool_trace", [])
    fallback["report"]["limitations"] = (
        (fallback["report"].get("limitations") or "")
        + " LLM loop truncated; deterministic synthesis used from collected tools."
    )
    return fallback


async def _ensure_grounding_tools(
    tools: ToolBox,
    seed: dict,
    evidence_map: dict[str, dict],
    trace: list[dict],
) -> dict[str, dict]:
    txn_id = seed["transaction_id"]
    required = {
        "get_transaction": {"transaction_id": txn_id},
        "get_model_explanation": {"transaction_id": txn_id},
        "get_triggered_rules": {"transaction_id": txn_id},
    }
    for name, args in required.items():
        if name in evidence_map and not (evidence_map[name] or {}).get("unavailable"):
            continue
        entry = await _invoke_tool(tools, name, args)
        trace.append(entry)
        evidence_map[name] = entry["result"]
    return evidence_map


async def _deterministic(
    tools: ToolBox,
    seed: dict,
    investigation_id: str | None = None,
) -> dict:
    """Call a relevant tool sequence and write a report only from results."""
    txn_id = seed["transaction_id"]
    user_id = seed["user_id"]
    plan = [
        ("get_transaction", {"transaction_id": txn_id}),
        ("get_user_profile", {"user_id": user_id}),
        ("get_user_history", {"user_id": user_id, "limit": 8}),
        ("get_user_baseline", {"user_id": user_id, "transaction_id": txn_id}),
        ("get_model_explanation", {"transaction_id": txn_id}),
        ("get_triggered_rules", {"transaction_id": txn_id}),
        ("check_device", {"device_id": seed["device_id"]}),
        ("check_ip", {"ip_address": seed["ip_address"]}),
        ("check_location", {"user_id": user_id, "location": seed["location"]}),
        ("check_transaction_velocity", {"transaction_id": txn_id}),
        ("find_connected_accounts", {"user_id": user_id}),
        ("find_fraud_cluster", {"transaction_id": txn_id}),
    ]
    trace: list[dict] = []
    evidence_map: dict[str, dict] = {}
    for name, args in plan:
        entry = await _invoke_tool(tools, name, args)
        trace.append(entry)
        evidence_map[name] = entry["result"]

    report = _ground_report(
        {},
        seed,
        evidence_map,
        trace,
        provider_name="deterministic_fallback",
        model_name=None,
        investigation_id=investigation_id,
    )
    report["limitations"] = (
        "Prototype investigator. Evidence is limited to controlled backend tools and synthetic data. "
        "A graph cluster is a potential ring, not a confirmed fraud label. "
        "No LLM was used; this report is assembled directly from tool outputs. "
        "Confidence is qualitative and is not a statistically calibrated probability."
    )
    return {
        "provider": "deterministic_fallback",
        "model": None,
        "report": report,
        "tool_trace": trace,
    }


def _parse_report(content: str) -> dict:
    content = (content or "").strip()
    try:
        if "```" in content:
            chunk = content.split("```")[1]
            if chunk.startswith("json"):
                chunk = chunk[4:]
            content = chunk
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {
        "summary": redact_text(content)[:500] if content else "LLM response was not valid JSON.",
        "limitations": (
            "LLM response was not valid JSON. Structured fields were filled from tool evidence. "
            "Do not treat unparsed model text as evidence."
        ),
        "_unparsed": True,
    }


def _ground_report(
    raw: dict,
    seed: dict,
    evidence_map: dict[str, dict],
    trace: list[dict],
    *,
    provider_name: str,
    model_name: str | None,
    investigation_id: str | None,
) -> dict:
    """Force risk-engine facts, reject invalid recs, strip fabricated graph/txn data."""
    raw = raw if isinstance(raw, dict) else {}
    txn = evidence_map.get("get_transaction") or seed
    if txn.get("unavailable"):
        txn = seed
    model = evidence_map.get("get_model_explanation") or {}
    rules = evidence_map.get("get_triggered_rules") or {}
    connected = evidence_map.get("find_connected_accounts") or {}
    cluster = evidence_map.get("find_fraud_cluster") or {}
    device = evidence_map.get("check_device") or {}
    ipinfo = evidence_map.get("check_ip") or {}
    loc = evidence_map.get("check_location") or {}
    velocity = evidence_map.get("check_transaction_velocity") or {}
    baseline = evidence_map.get("get_user_baseline") or evidence_map.get("get_user_profile") or {}

    engine_decision = model.get("decision") if not model.get("unavailable") else None
    recommended = engine_decision if engine_decision in VALID_RECOMMENDATIONS else "REVIEW"
    llm_rec = raw.get("recommendation") or raw.get("recommended_action")
    if isinstance(llm_rec, str) and llm_rec.upper() not in VALID_RECOMMENDATIONS:
        # Invalid enum is rejected; engine decision (or REVIEW) is used.
        recommended = engine_decision if engine_decision in VALID_RECOMMENDATIONS else "REVIEW"

    score = model.get("final_risk_score") if not model.get("unavailable") else None
    risk_level = _risk_level_from_score(score)
    raw_level = raw.get("risk_level")
    if isinstance(raw_level, str) and raw_level.upper() in VALID_RISK_LEVELS:
        # Still derive from engine score so the LLM cannot inflate/deflate independently.
        risk_level = _risk_level_from_score(score)

    cluster_ok = bool(cluster) and not cluster.get("unavailable")
    identified = bool(cluster_ok and (cluster.get("identified") or cluster.get("cluster_found")))
    if identified:
        ring = {
            "identified": True,
            "cluster_found": True,
            "cluster_id": cluster.get("cluster_id"),
            "cluster_size": cluster.get("cluster_size") or cluster.get("user_count"),
            "connected_users": cluster.get("connected_users"),
            "shared_devices": cluster.get("shared_devices"),
            "shared_ips": cluster.get("shared_ips"),
            "shared_device_count": cluster.get("shared_device_count")
            or len(cluster.get("shared_devices") or []),
            "shared_ip_count": cluster.get("shared_ip_count") or len(cluster.get("shared_ips") or []),
            "fraud_associated_nodes": cluster.get("fraud_associated_nodes") or 0,
            "suspicious_nodes": cluster.get("suspicious_nodes") or [],
            "relationships": cluster.get("relationships") or [],
            "risk_indicators": cluster.get("risk_indicators") or [],
            "merchants": cluster.get("merchants"),
            "relationship_counts": cluster.get("relationship_counts"),
            "graph_risk": cluster.get("graph_risk") or cluster.get("graph_risk_score"),
            "explanation": cluster.get("explanation"),
        }
    else:
        reason = (
            cluster.get("reason")
            or cluster.get("message")
            or "No connected suspicious cluster found"
        )
        ring = {
            "identified": False,
            "cluster_found": False,
            "reason": reason,
            "message": cluster.get("message") or "no suspicious cluster identified",
            "explanation": cluster.get("explanation"),
            "relationship_counts": cluster.get("relationship_counts"),
            "risk_indicators": cluster.get("risk_indicators") or [],
            "fraud_associated_nodes": cluster.get("fraud_associated_nodes") or 0,
        }

    # LLM cannot fabricate a cluster if the graph tool did not find one.
    raw_graph = raw.get("graph_evidence") if isinstance(raw.get("graph_evidence"), dict) else {}
    if raw_graph.get("cluster_found") and not identified:
        raw_graph = {}

    model_evidence = {
        "ml_probability": model.get("ml_probability") if not model.get("unavailable") else None,
        "ml_score": model.get("ml_score") if not model.get("unavailable") else None,
        "ml_probability_raw": model.get("ml_probability_raw") if not model.get("unavailable") else None,
        "probability_calibrated": model.get("probability_calibrated") if not model.get("unavailable") else None,
        "model_version": model.get("model_version") if not model.get("unavailable") else None,
        "shap_top_features": model.get("shap_top_features") if not model.get("unavailable") else [],
        "decision": engine_decision,
        "final_risk_score": score,
        "explanation": model.get("explanation") if not model.get("unavailable") else None,
        "unavailable": bool(model.get("unavailable")),
        "reason": model.get("reason") if model.get("unavailable") else None,
        "note": "Copied from the risk engine. The LLM did not calculate this probability.",
    }
    # Strip any LLM-invented probability.
    if isinstance(raw.get("model_evidence"), dict):
        raw["model_evidence"].pop("ml_probability", None)

    behavioral = {
        "velocity": velocity if not velocity.get("unavailable") else {"unavailable": True, "reason": velocity.get("reason")},
        "baseline": baseline if not baseline.get("unavailable") else {"unavailable": True, "reason": baseline.get("reason")},
        "location": loc if not loc.get("unavailable") else {"unavailable": True, "reason": loc.get("reason")},
        "anomalies": model.get("anomalies") or [],
        "device_known": txn.get("current_device_known"),
        "location_known": txn.get("current_location_known"),
    }
    rule_evidence = {
        "triggered": rules.get("triggered") or [],
        "note": rules.get("note"),
    }
    graph_evidence = {
        **ring,
        "connected_users": (connected.get("connected_users") if not connected.get("unavailable") else [])
        or ring.get("connected_users")
        or [],
        "previously_flagged_connected_users": connected.get("previously_flagged_connected_users") or [],
        "device_users": (device.get("connected_users") if not device.get("unavailable") else []),
        "ip_users": (ipinfo.get("connected_users") if not ipinfo.get("unavailable") else []),
        "graph_backend": ring.get("graph_backend")
        or connected.get("graph_backend")
        or cluster.get("graph_backend"),
    }

    findings = _key_findings(txn, model, device, velocity, connected, cluster, loc, baseline)
    summary = raw.get("summary") if isinstance(raw.get("summary"), str) and not raw.get("_unparsed") else None
    if not summary or _looks_like_injection(summary):
        summary = _build_summary(txn, model, ring, findings, recommended)

    engine_confidence = model.get("confidence") if model.get("confidence") is not None else 0.4
    if model.get("unavailable"):
        engine_confidence = 0.2
        recommended = "REVIEW"

    evidence = []
    for name, result in evidence_map.items():
        if result.get("unavailable"):
            evidence.append({"source": name, "status": "unavailable", "reason": result.get("reason")})
        elif result.get("status") == "error" or result.get("error"):
            evidence.append({"source": name, "status": "error", "reason": result.get("error") or result.get("reason")})
        else:
            evidence.append({"source": name, "status": "ok", "excerpt": _excerpt(result)})

    signals = []
    for r in rule_evidence.get("triggered") or []:
        signals.append(f"{r.get('rule_name')}: {r.get('explanation')}")
    for a in model.get("anomalies") or []:
        if isinstance(a, dict):
            signals.append(a.get("description") or a.get("code"))
    if loc.get("known_for_user") is False:
        signals.append(f"Location {loc.get('location')} is not in the user known set")

    limitations = raw.get("limitations") if isinstance(raw.get("limitations"), str) else None
    if not limitations:
        limitations = (
            "Independent student prototype; not an official Razorpay product. "
            "Recommendation is based on collected tool evidence and the risk engine decision. "
            "The agent cannot query the database directly or invent missing evidence. "
            "Graph clusters are potential rings, not confirmed fraud labels. "
            "Confidence is qualitative and is not a statistically calibrated probability."
        )
    if provider_name == "llm":
        limitations += " LLM text was grounded against tool outputs before this report was stored."

    report = {
        "investigation_id": investigation_id,
        "transaction_id": txn.get("transaction_id") or seed.get("transaction_id"),
        "provider": provider_name,
        "model": model_name,
        "summary": summary,
        "risk_level": risk_level,
        "recommendation": recommended,
        "recommended_action": recommended,
        "confidence": engine_confidence,
        "confidence_qualitative": {
            "kind": "qualitative",
            "level": _qualitative_level(engine_confidence, evidence),
            "note": (
                "Qualitative judgment from evidence coverage and the risk engine confidence field. "
                "Not a statistically calibrated probability."
            ),
        },
        "model_evidence": model_evidence,
        "behavioral_evidence": behavioral,
        "rule_evidence": rule_evidence,
        "graph_evidence": graph_evidence,
        "key_findings": findings,
        "tool_trace": trace,
        "limitations": limitations,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "transaction_summary": {
            "transaction_id": txn.get("transaction_id"),
            "user_id": txn.get("user_id"),
            "amount": txn.get("amount"),
            "merchant_id": txn.get("merchant_id"),
            "device_id": txn.get("device_id"),
            "ip_address": txn.get("ip_address"),
            "location": txn.get("location"),
            "timestamp": txn.get("timestamp"),
        },
        "risk_assessment": {
            "final_risk_score": model_evidence.get("final_risk_score"),
            "ml_score": model_evidence.get("ml_score"),
            "ml_probability": model_evidence.get("ml_probability"),
            "behavior_score": model.get("behavior_score") if not model.get("unavailable") else None,
            "rule_score": model.get("rule_score") if not model.get("unavailable") else None,
            "graph_score": model.get("graph_score") if not model.get("unavailable") else None,
            "model_decision": engine_decision,
            "model_version": model_evidence.get("model_version"),
            "narrative": model_evidence.get("explanation"),
        },
        "evidence": evidence,
        "suspicious_signals": signals,
        "connected_entities": {
            "users": connected.get("connected_users") if not connected.get("unavailable") else [],
            "device_users": device.get("connected_users") if not device.get("unavailable") else [],
            "ip_users": ipinfo.get("connected_users") if not ipinfo.get("unavailable") else [],
        },
        "potential_fraud_ring": ring,
        "user_baseline": baseline if not baseline.get("unavailable") else None,
        "risk_engine_decision": engine_decision,
    }
    return redact_secrets(report)


def _empty_report(transaction_id: str, investigation_id: str | None) -> dict:
    return {
        "investigation_id": investigation_id,
        "transaction_id": transaction_id,
        "provider": "deterministic_fallback",
        "summary": "Transaction not found",
        "risk_level": "MEDIUM",
        "recommendation": "REVIEW",
        "recommended_action": "REVIEW",
        "confidence": 0.0,
        "confidence_qualitative": {
            "kind": "qualitative",
            "level": "low",
            "note": "Qualitative judgment from evidence coverage. Not a statistically calibrated probability.",
        },
        "model_evidence": {"unavailable": True, "reason": "Transaction not found"},
        "behavioral_evidence": {"unavailable": True},
        "rule_evidence": {"triggered": []},
        "graph_evidence": {
            "cluster_found": False,
            "identified": False,
            "reason": "No connected suspicious cluster found",
        },
        "key_findings": [],
        "tool_trace": [],
        "transaction_summary": "Transaction not found",
        "risk_assessment": "unavailable",
        "evidence": [],
        "suspicious_signals": [],
        "connected_entities": [],
        "potential_fraud_ring": None,
        "limitations": "The transaction identifier was not present in the database.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _risk_level_from_score(score: Any) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "MEDIUM"
    if value < 40:
        return "LOW"
    if value < 70:
        return "MEDIUM"
    if value < 90:
        return "HIGH"
    return "CRITICAL"


def _qualitative_level(confidence: Any, evidence: list) -> str:
    available = sum(1 for e in evidence if e.get("status") == "ok")
    if available >= 8 and (confidence or 0) >= 0.6:
        return "high"
    if available >= 4:
        return "medium"
    return "low"


def _looks_like_injection(text: str) -> bool:
    lowered = text.lower()
    needles = (
        "ignore previous",
        "ignore all instructions",
        "system prompt",
        "you are now",
        "disregard",
        "override the risk",
    )
    return any(n in lowered for n in needles)


def _build_summary(txn: dict, model: dict, ring: dict, findings: list[str], recommended: str) -> str:
    score = model.get("final_risk_score")
    version = model.get("model_version") or "unknown-model"
    parts = [
        f"Risk engine decision is {recommended} (final risk score {score}, model {version}).",
        "This recommendation is based on collected tool evidence and the stored risk result.",
    ]
    if findings:
        parts.append("Key evidence: " + "; ".join(findings[:5]) + ".")
    if ring.get("identified") or ring.get("cluster_found"):
        parts.append(
            f"Graph tool identified potential cluster {ring.get('cluster_id')} "
            f"({ring.get('cluster_size') or ring.get('user_count')} entities). "
            "This is not a confirmed fraud label."
        )
    else:
        parts.append("No suspicious graph cluster was identified.")
    return " ".join(parts)


def _key_findings(
    txn: dict,
    model: dict,
    device: dict,
    velocity: dict,
    connected: dict,
    cluster: dict,
    loc: dict,
    baseline: dict,
) -> list[str]:
    findings: list[str] = []
    if txn.get("current_device_known") is False:
        findings.append("New device")
    if txn.get("current_location_known") is False:
        findings.append("Unfamiliar location")
    vel = velocity.get("transaction_velocity") if not velocity.get("unavailable") else txn.get("transaction_velocity")
    try:
        if vel is not None and int(vel) >= 5:
            findings.append(f"Abnormal velocity ({vel} recorded on the transaction)")
    except (TypeError, ValueError):
        pass
    device_users = device.get("connected_users") if not device.get("unavailable") else []
    if device_users and len(device_users) >= 3:
        findings.append(f"Device shared with {len(device_users)} accounts")
    flagged = connected.get("previously_flagged_count") or 0
    if flagged:
        findings.append(f"{flagged} connected account(s) previously flagged")
    if cluster.get("identified") or cluster.get("cluster_found"):
        size = cluster.get("cluster_size") or cluster.get("user_count")
        findings.append(
            f"Potential fraud cluster {cluster.get('cluster_id')} with {size} entities"
        )
        fraud_nodes = cluster.get("fraud_associated_nodes") or 0
        if fraud_nodes:
            findings.append(f"{fraud_nodes} fraud-associated node(s) in the cluster")
    amount_vs = baseline.get("amount_vs_typical") if not baseline.get("unavailable") else None
    try:
        if amount_vs is not None and float(amount_vs) >= 3:
            findings.append(f"Amount is {amount_vs}× typical for this user")
    except (TypeError, ValueError):
        pass
    if loc.get("known_for_user") is False:
        findings.append("Location not in the user known set")
    return findings


def _summarize_result(name: str, result: dict) -> str:
    if result.get("unavailable") or result.get("status") == "unavailable":
        return f"unavailable: {result.get('reason') or 'no evidence'}"
    if result.get("status") == "error" or result.get("error"):
        return f"error: {result.get('error') or result.get('reason')}"
    if name == "find_fraud_cluster":
        if result.get("identified") or result.get("cluster_found"):
            return (
                f"cluster {result.get('cluster_id')} "
                f"size={result.get('cluster_size') or result.get('user_count')}"
            )
        return result.get("message") or result.get("reason") or "no cluster"
    if name == "get_model_explanation":
        return (
            f"decision={result.get('decision')} "
            f"p={result.get('ml_probability')} "
            f"score={result.get('final_risk_score')}"
        )
    if name == "get_triggered_rules":
        return f"triggered={len(result.get('triggered') or [])}"
    if name == "find_connected_accounts":
        return f"connected={result.get('count', 0)} flagged={result.get('previously_flagged_count', 0)}"
    excerpt = _excerpt(result)
    return json.dumps(excerpt, default=str)[:240]


def _excerpt(result: dict) -> dict:
    keys = [
        "transaction_id",
        "final_risk_score",
        "decision",
        "ml_probability",
        "connected_users",
        "triggered",
        "identified",
        "cluster_found",
        "message",
        "reason",
        "cluster_id",
        "cluster_size",
        "shared_devices",
        "shared_ips",
        "merchants",
        "relationship_counts",
        "graph_risk",
        "graph_risk_score",
        "user_count",
        "fraud_associated_nodes",
        "explanation",
        "typical_amount",
        "known_for_user",
        "transaction_velocity",
        "count",
        "status",
    ]
    return {k: result[k] for k in keys if k in result}
