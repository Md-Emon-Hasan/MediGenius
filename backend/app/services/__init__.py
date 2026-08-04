"""
MediGenius — services/__init__.py
Exports service singletons.
"""

from app.services.chat_service import ChatService
from app.services.database_service import DatabaseService, db_service

# The ChatService singleton is deliberately not re-exported here: binding it as
# `chat_service` shadows the app.services.chat_service submodule on this package, and
# mock.patch("app.services.chat_service.<name>") then resolves to the instance instead
# of the module (breaks on Python 3.10, silently works on 3.11+).
# Import it from its own module: from app.services.chat_service import chat_service
__all__ = ["DatabaseService", "db_service", "ChatService"]
