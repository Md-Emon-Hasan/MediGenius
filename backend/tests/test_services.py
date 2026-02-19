"""Tests for services — Deep Modular Architecture"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.session import get_engine, get_session_factory  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.database_service import DatabaseService  # noqa: E402


class TestChatService:
    """Tests for ChatService"""

    def test_chat_service_initialization(self):
        service = ChatService()
        assert service.workflow_app is None
        assert service.conversation_states == {}

    def test_initialize_workflow(self):
        service = ChatService()
        import sys
        chat_module = sys.modules['app.services.chat_service']
        with patch.object(chat_module, 'create_workflow') as mock_create:
            mock_create.return_value = MagicMock()
            service.initialize_workflow()
            assert service.workflow_app is not None
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_success(self):
        service = ChatService()
        service.workflow_app = MagicMock()
        service.workflow_app.ainvoke = AsyncMock(return_value={
            "generation": "Test response",
            "source": "Test Source"
        })
        from app.services import db_service
        with patch.object(db_service, 'save_message'):
            result = await service.process_message("test-session", "Hello")
            assert result["success"] is True
            assert result["response"] == "Test response"
            assert result["source"] == "Test Source"

    @pytest.mark.asyncio
    async def test_process_message_no_workflow(self):
        service = ChatService()
        with pytest.raises(ValueError, match="Workflow not initialized"):
            await service.process_message("test-session", "Hello")

    @pytest.mark.asyncio
    async def test_process_message_fallback_sync(self):
        service = ChatService()
        service.workflow_app = MagicMock()
        service.workflow_app.ainvoke = AsyncMock(side_effect=AttributeError)
        service.workflow_app.invoke = MagicMock(return_value={
            "generation": "Sync response",
            "source": "Sync Source"
        })
        from app.services import db_service
        with patch.object(db_service, 'save_message'):
            result = await service.process_message("test-session", "Hello")
            assert result["success"] is True
            assert result["response"] == "Sync response"

    def test_clear_conversation(self):
        service = ChatService()
        service.conversation_states["test-session"] = {"question": "old"}
        service.clear_conversation("test-session")
        assert service.conversation_states["test-session"]["question"] == ""

    def test_clear_conversation_nonexistent(self):
        service = ChatService()
        service.clear_conversation("nonexistent")  # Should not raise


class TestDatabaseService:
    """Tests for DatabaseService"""

    def test_database_service_initialization(self):
        test_db = "test_init.db"
        if os.path.exists(test_db):
            os.remove(test_db)

        test_engine = get_engine(test_db)
        test_session = get_session_factory(test_engine)
        service = DatabaseService(session_local=test_session, engine_instance=test_engine)
        service.init_db()
        assert os.path.exists(test_db)

        test_engine.dispose()
        os.remove(test_db)

    def test_save_and_retrieve_message(self):
        test_db = "test_save.db"
        if os.path.exists(test_db):
            os.remove(test_db)

        test_engine = get_engine(test_db)
        test_session = get_session_factory(test_engine)
        service = DatabaseService(session_local=test_session, engine_instance=test_engine)
        service.init_db()
        service.save_message("sess1", "user", "Hello", None)
        service.save_message("sess1", "assistant", "Hi there", "AI")

        history = service.get_chat_history("sess1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["content"] == "Hi there"

        test_engine.dispose()
        os.remove(test_db)

    def test_get_all_sessions(self):
        test_db = "test_sessions.db"
        if os.path.exists(test_db):
            os.remove(test_db)

        test_engine = get_engine(test_db)
        test_session = get_session_factory(test_engine)
        service = DatabaseService(session_local=test_session, engine_instance=test_engine)
        service.init_db()
        service.save_message("sess1", "user", "Message 1")
        service.save_message("sess2", "user", "Message 2")

        sessions = service.get_all_sessions()
        assert len(sessions) >= 2
        session_ids = [s["session_id"] for s in sessions]
        assert "sess1" in session_ids
        assert "sess2" in session_ids

        test_engine.dispose()
        os.remove(test_db)

    def test_delete_session(self):
        test_db = "test_delete.db"
        if os.path.exists(test_db):
            os.remove(test_db)

        test_engine = get_engine(test_db)
        test_session = get_session_factory(test_engine)
        service = DatabaseService(session_local=test_session, engine_instance=test_engine)
        service.init_db()
        service.save_message("sess_del", "user", "Delete me")
        assert len(service.get_chat_history("sess_del")) == 1

        service.delete_session("sess_del")
        assert len(service.get_chat_history("sess_del")) == 0

        test_engine.dispose()
        os.remove(test_db)
