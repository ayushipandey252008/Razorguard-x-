from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agents.provider import llm_is_configured
from app.config import get_settings
from app.events.factory import event_bus_status
from app.graph.factory import graph_status
from app.graph.rings import prototype_graph_thresholds
from app.ml.predictor import model_service
from app.security.auth import decode_token
from app.services.events import TXN_CHANNEL, event_bus

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    settings = get_settings()
    bus = event_bus_status()
    return {
        "status": "ok",
        "app": settings.app_name,
        "model_ready": model_service.ready,
        "model_version": model_service.version,
        "event_backend": event_bus.backend,
        "event_bus": {
            "configured": bus.get("configured"),
            "active": bus.get("active"),
            "fallback": bus.get("fallback"),
            "kafka_connected": bus.get("kafka_connected"),
            "reason": bus.get("reason"),
        },
        "outbox_enabled": settings.outbox_enabled,
        "durable_event_delivery": settings.outbox_enabled,
        "prototype": True,
        "llm": {
            "configured": llm_is_configured(),
            "provider": "llm" if llm_is_configured() else "deterministic_fallback",
            "model": settings.llm_model if llm_is_configured() else None,
        },
        "graph_cluster_thresholds": prototype_graph_thresholds(),
        **graph_status(),
    }


@router.websocket("/ws/transactions")
async def txn_socket(ws: WebSocket, token: str | None = None):
    await ws.accept()
    settings = get_settings()
    if not token:
        if settings.is_production:
            await ws.send_json({"type": "error", "detail": "token required"})
            await ws.close(code=4401)
            return
    else:
        try:
            decode_token(token)
        except ValueError:
            await ws.send_json({"type": "error", "detail": "invalid token"})
            await ws.close()
            return
    queue = event_bus.subscribe(TXN_CHANNEL)
    try:
        await ws.send_json({"type": "hello", "channel": TXN_CHANNEL, "backend": event_bus.backend})
        while True:
            payload = await queue.get()
            await ws.send_json(payload)
    except WebSocketDisconnect:
        event_bus.unsubscribe(TXN_CHANNEL, queue)
    except Exception:
        event_bus.unsubscribe(TXN_CHANNEL, queue)
        try:
            await ws.close()
        except Exception:
            pass
