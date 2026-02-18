"""API v1 router composition"""
from fastapi import APIRouter
from .endpoints import chat, session, health

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(chat.router, tags=["Chat"])
api_router.include_router(session.router, tags=["Session"])
