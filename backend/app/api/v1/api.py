"""
MediGenius — api/v1/api.py
Router aggregator: collects all v1 endpoint routers into one.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import chat, health, review, session

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(session.router)
api_router.include_router(review.router)
