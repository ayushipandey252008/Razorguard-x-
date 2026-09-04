from fastapi import APIRouter

from app.api.v1 import analytics, auth, events, feedback, graph, health, investigations, ml, risk, simulation, transactions

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(transactions.router)
api_router.include_router(risk.router)
api_router.include_router(investigations.router)
api_router.include_router(graph.router)
api_router.include_router(events.router)
api_router.include_router(analytics.router)
api_router.include_router(simulation.router)
api_router.include_router(ml.router)
api_router.include_router(feedback.router)
