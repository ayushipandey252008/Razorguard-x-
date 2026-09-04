from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import SessionLocal
from app.graph.factory import graph_status, graph_store
from app.graph.service import ingest_transaction
from app.ml.predictor import model_service
from app.models.app_user import AppUser
from app.models.merchant import Merchant
from app.models.model_version import ModelVersion
from app.models.payment_user import PaymentUser
from app.models.transaction import Transaction
from app.schemas.common import TransactionCreate
from app.security.auth import hash_password
from app.services.pipeline import process_transaction
from app.services.synthetic import MERCHANTS, SEED_USERS, WATCHLISTED, scenario_transactions
from app.utils.ids import new_id
from app.utils.logging import get_logger

log = get_logger("bootstrap")


async def ensure_seeded() -> None:
    async with SessionLocal() as db:
        await record_active_model(db)
        count = (await db.execute(select(func.count()).select_from(AppUser))).scalar() or 0
        if count:
            # NetworkX is in-memory: rebuild from SQL on every boot.
            # Neo4j persists: MERGE existing transactions only when empty or GRAPH_REBUILD_ON_START.
            status = graph_status()
            backend_name = getattr(graph_store, "name", "networkx")
            rebuild = True
            if backend_name == "neo4j" and not get_settings().graph_rebuild_on_start:
                node_count = graph_store.node_count() if hasattr(graph_store, "node_count") else 0
                if node_count > 0:
                    rebuild = False
                    log.info(
                        "skip_graph_rebuild",
                        graph_backend="neo4j",
                        node_count=node_count,
                        reason="persistent graph already populated",
                    )
            if rebuild:
                if backend_name == "networkx":
                    graph_store.clear()
                txns = (await db.execute(select(Transaction))).scalars().all()
                for t in txns:
                    ingest_transaction(
                        {
                            "transaction_id": t.transaction_id,
                            "user_id": t.user_id,
                            "device_id": t.device_id,
                            "ip_address": t.ip_address,
                            "merchant_id": t.merchant_id,
                            "location": t.location,
                            "payment_identifier": t.payment_identifier,
                            "account_age_days": t.account_age_days,
                            "merchant_category": t.merchant_category,
                        }
                    )
                log.info(
                    "graph_rebuilt_from_sql",
                    graph_backend=backend_name,
                    transactions=len(txns),
                    graph_status=status.get("graph_backend"),
                )
            return
        await seed(db)


async def record_active_model(db: AsyncSession) -> None:
    if not model_service.ready:
        return
    existing = (
        await db.execute(select(ModelVersion).where(ModelVersion.version == model_service.version))
    ).scalar_one_or_none()
    if existing:
        existing.is_active = True
        existing.status = "ACTIVE"
        existing.model_id = existing.model_id or model_service.version
        existing.dataset = existing.dataset or "SYNTHETIC_DATASET"
        await db.commit()
        return
    others = (await db.execute(select(ModelVersion).where(ModelVersion.is_active.is_(True)))).scalars().all()
    for row in others:
        row.is_active = False
    metrics = model_service.metrics or {}
    db.add(
        ModelVersion(
            id=new_id(),
            version=model_service.version,
            model_id=model_service.version,
            model_type="supervised+iforest",
            metrics=metrics,
            artifact_path=str(model_service.model_dir),
            is_active=True,
            dataset="SYNTHETIC_DATASET",
            feature_set=metrics.get("feature_columns") or [],
            training_rows=int(metrics.get("n_samples") or 0),
            positive_rows=0,
            evaluation_rows=int(metrics.get("n_samples") or 0),
            status="ACTIVE",
        )
    )
    await db.commit()


async def seed(db: AsyncSession) -> None:
    settings = get_settings()
    users = [
        ("admin@razorguard.local", settings.seed_admin_password, "ADMIN", "Asha Admin"),
        ("analyst@razorguard.local", settings.seed_admin_password, "RISK_ANALYST", "Rohan Analyst"),
        ("investigator@razorguard.local", settings.seed_admin_password, "INVESTIGATOR", "Ira Investigator"),
        ("viewer@razorguard.local", settings.seed_admin_password, "VIEWER", "Vik Viewer"),
    ]
    for email, password, role, name in users:
        db.add(
            AppUser(
                id=new_id(),
                email=email,
                hashed_password=hash_password(password),
                role=role,
                display_name=name,
                is_active=True,
            )
        )
    for u in SEED_USERS:
        db.add(PaymentUser(**u))
    for mid, name, cat, loc in MERCHANTS:
        db.add(
            Merchant(
                merchant_id=mid,
                name=name,
                category=cat,
                location=loc,
                is_watchlisted=mid in WATCHLISTED,
            )
        )
    await db.commit()

    # Seed a mix of live pipeline transactions so the dashboard is not empty.
    import random

    rng = random.Random(7)
    mix = (
        scenario_transactions("normal", 8, rng)
        + scenario_transactions("stolen_account", 1, rng)
        + scenario_transactions("fraud_ring", 6, rng)
        + scenario_transactions("card_testing", 3, rng)
    )
    for row in mix:
        payload = TransactionCreate(
            user_id=row["user_id"],
            merchant_id=row["merchant_id"],
            amount=row["amount"],
            currency=row["currency"],
            timestamp=row["timestamp"],
            device_id=row["device_id"],
            ip_address=row["ip_address"],
            location=row["location"],
            payment_method=row["payment_method"],
            merchant_category=row["merchant_category"],
            account_age_days=row["account_age_days"],
            failed_attempts=row["failed_attempts"],
            transaction_velocity=row["transaction_velocity"],
            previous_transaction_count=row["previous_transaction_count"],
            previous_average_amount=row["previous_average_amount"],
            current_device_known=row["current_device_known"],
            current_location_known=row["current_location_known"],
            payment_identifier=row["payment_identifier"],
            scenario_tag=row["scenario_tag"],
        )
        await process_transaction(db, payload, actor="seed")
    log.info("seed_complete")
