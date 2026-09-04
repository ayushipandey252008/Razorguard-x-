"""Standalone Kafka/in-process event consumer. Does not start FastAPI.

Run:

    PYTHONPATH=. python -m app.workers.event_consumer

Consumes transactions, risk-results, investigations, alerts, and feedback
through the existing idempotent handlers. Also drains the transactional
outbox so a dedicated process can publish without the API holding Kafka
inside a DB transaction.
"""

from __future__ import annotations

import asyncio
import signal

from app.config import get_settings
from app.database import init_db
from app.events.factory import close_event_bus, connect_event_bus, event_bus_status
from app.events.outbox_worker import run_outbox_loop
from app.utils.logging import configure_logging, get_logger

configure_logging()
log = get_logger("workers.event_consumer")


async def amain() -> None:
    settings = get_settings()
    await init_db()
    await connect_event_bus(start_consumers=True)
    status = event_bus_status()
    log.info(
        "event_consumer_started",
        configured=status.get("configured"),
        active=status.get("active"),
        kafka_connected=status.get("kafka_connected"),
        outbox_enabled=settings.outbox_enabled,
        note="at-least-once delivery; handlers are idempotent",
    )
    stop = asyncio.Event()

    def _stop() -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    tasks = []
    if settings.outbox_enabled:
        tasks.append(asyncio.create_task(run_outbox_loop(stop), name="outbox-worker"))

    try:
        await stop.wait()
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await close_event_bus()
        log.info("event_consumer_stopped")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
