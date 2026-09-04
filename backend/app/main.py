from __future__ import annotations

import uuid

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1 import api_router
from app.config import assert_secure_settings, get_settings
from app.database import init_db
from app.ml.predictor import model_service
from app.rate_limit import limiter
from app.services.events import event_bus
from app.utils.logging import Timer, configure_logging, get_logger

import structlog

configure_logging()
log = get_logger("main")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    assert_secure_settings(settings)
    await init_db()
    await event_bus.connect()
    from app.events.factory import close_event_bus, connect_event_bus

    await connect_event_bus()
    model_service.load_or_train()
    from app.services.bootstrap import ensure_seeded

    await ensure_seeded()
    from app.events.factory import event_bus_status
    from app.graph.factory import graph_status

    log.info(
        "startup_complete",
        model=model_service.version,
        environment=settings.environment,
        **graph_status(),
        **{f"event_{k}": v for k, v in event_bus_status().items() if k in {"configured", "active", "fallback", "kafka_connected"}},
        outbox_enabled=settings.outbox_enabled,
    )
    from app.events.outbox_worker import start_outbox_worker, stop_outbox_worker

    if settings.outbox_enabled and settings.outbox_background_worker:
        await start_outbox_worker()
    yield
    from app.graph.factory import close_graph_store

    close_graph_store()
    await stop_outbox_worker()
    await close_event_bus()


app = FastAPI(
    title="RazorGuard X API",
    description=(
        "Agentic real-time payment fraud & fraud-ring intelligence prototype. "
        "Synthetic data only. Not affiliated with Razorpay. Not a production fraud system."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Correlation-ID"],
)
app.include_router(api_router)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    if isinstance(exc, (HTTPException, StarletteHTTPException, RateLimitExceeded)):
        raise exc
    log.exception("unhandled_error", path=request.url.path, method=request.method)
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


@app.middleware("http")
async def request_context(request: Request, call_next):
    timer = Timer()
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    correlation_id = request.headers.get("X-Correlation-ID") or request_id
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id, correlation_id=correlation_id)
    try:
        response = await call_next(request)
        latency_ms = timer.ms()
        log.info(
            "request",
            path=request.url.path,
            method=request.method,
            status=response.status_code,
            latency_ms=latency_ms,
            correlation_id=correlation_id,
        )
        response.headers["X-Process-Time-ms"] = str(latency_ms)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Correlation-ID"] = correlation_id
        return response
    except Exception:
        log.exception("request_failed", path=request.url.path, method=request.method)
        raise
