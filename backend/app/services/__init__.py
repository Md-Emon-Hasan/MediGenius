"""
MediGenius — services/__init__.py
Exports service singletons.
"""

from app.services.chat_service import ChatService, chat_service
from app.services.database_service import DatabaseService, db_service

__all__ = ["DatabaseService", "db_service", "ChatService", "chat_service"]
