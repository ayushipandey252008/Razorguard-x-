"""Standalone outbox publisher. Does not start FastAPI or Kafka consumers.

    PYTHONPATH=. python -m app.workers.outbox
"""

from __future__ import annotations

import asyncio
import signal

from app.config import get_settings
from app.database import init_db
from app.events.factory import close_event_bus, connect_event_bus
from app.events.outbox_worker import run_outbox_loop
from app.utils.logging import configure_logging, get_logger

configure_logging()
log = get_logger("workers.outbox")


async def amain() -> None:
    settings = get_settings()
    if not settings.outbox_enabled:
        raise SystemExit("OUTBOX_ENABLED=false")
    await init_db()
    await connect_event_bus()
    log.info("outbox_process_started")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    try:
        await run_outbox_loop(stop)
    finally:
        await close_event_bus()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
