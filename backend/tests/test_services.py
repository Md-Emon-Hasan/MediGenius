"""Tests for service layer"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add app to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))

from services.chat_service import ChatService  # noqa: E402
from services.database_service import DatabaseService  # noqa: E402


class TestChatService:
    """Tests for ChatService"""

    def test_chat_service_initialization(self):
        """Test chat service initialization"""
        service = ChatService()
        assert service.workflow_app is None
        assert service.conversation_states == {}

    def test_initialize_workflow(self):
        """Test workflow initialization"""
        service = ChatService()
        with patch('services.chat_service.create_workflow') as mock_create:
            mock_create.return_value = MagicMock()
            service.initialize_workflow()
            assert service.workflow_app is not None
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_success(self):
        """Test successful message processing"""
        service = ChatService()
        service.workflow_app = MagicMock()
        service.workflow_app.ainvoke = AsyncMock(return_value={
            "generation": "Test response",
            "source": "Test Source"
        })

        with patch('services.chat_service.db_service.save_message'):
            result = await service.process_message("test-session", "Hello")
            assert result["success"] is True
            assert result["response"] == "Test response"
            assert result["source"] == "Test Source"

    @pytest.mark.asyncio
    async def test_process_message_no_workflow(self):
        """Test message processing without initialized workflow"""
        service = ChatService()
        with pytest.raises(ValueError, match="Workflow not initialized"):
            await service.process_message("test-session", "Hello")

    @pytest.mark.asyncio
    async def test_process_message_fallback_sync(self):
        """Test fallback to sync invoke"""
        service = ChatService()
        service.workflow_app = MagicMock()
        service.workflow_app.ainvoke = AsyncMock(side_effect=AttributeError)
        service.workflow_app.invoke = MagicMock(return_value={
            "generation": "Sync response",
            "source": "Sync Source"
        })

        with patch('services.chat_service.db_service.save_message'):
            result = await service.process_message("test-session", "Hello")
            assert result["success"] is True
            assert result["response"] == "Sync response"

    def test_clear_conversation(self):
        """Test conversation clearing"""
        service = ChatService()
        service.conversation_states["test-session"] = {"question": "old"}
        service.clear_conversation("test-session")
        assert service.conversation_states["test-session"]["question"] == ""

    def test_clear_conversation_nonexistent(self):
        """Test clearing non-existent conversation"""
        service = ChatService()
        service.clear_conversation("nonexistent")  # Should not raise


class TestDatabaseService:
    """Tests for DatabaseService"""

    def test_database_service_initialization(self):
        """Test database service initialization"""
        test_db = "test_init.db"
        if os.path.exists(test_db):
            os.remove(test_db)

        service = DatabaseService(db_path=test_db)
        assert service.db_path == test_db
        assert os.path.exists(test_db)

        service.engine.dispose()
        os.remove(test_db)

    def test_save_and_retrieve_message(self):
        """Test saving and retrieving messages"""
        test_db = "test_save.db"
        if os.path.exists(test_db):
            os.remove(test_db)

        service = DatabaseService(db_path=test_db)
        service.save_message("sess1", "user", "Hello", None)
        service.save_message("sess1", "assistant", "Hi there", "AI")

        history = service.get_chat_history("sess1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["content"] == "Hi there"

        service.engine.dispose()
        os.remove(test_db)

    def test_get_all_sessions(self):
        """Test retrieving all sessions"""
        test_db = "test_sessions.db"
        if os.path.exists(test_db):
            os.remove(test_db)

        service = DatabaseService(db_path=test_db)
        service.save_message("sess1", "user", "Message 1")
        service.save_message("sess2", "user", "Message 2")

        sessions = service.get_all_sessions()
        assert len(sessions) >= 2
        session_ids = [s["session_id"] for s in sessions]
        assert "sess1" in session_ids
        assert "sess2" in session_ids

        service.engine.dispose()
        os.remove(test_db)

    def test_delete_session(self):
        """Test session deletion"""
        test_db = "test_delete.db"
        if os.path.exists(test_db):
            os.remove(test_db)

        service = DatabaseService(db_path=test_db)
        service.save_message("sess_del", "user", "Delete me")
        assert len(service.get_chat_history("sess_del")) == 1

        service.delete_session("sess_del")
        assert len(service.get_chat_history("sess_del")) == 0

        service.engine.dispose()
        os.remove(test_db)
