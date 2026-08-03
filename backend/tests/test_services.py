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

    @pytest.fixture(autouse=True)
    def _mock_memory_store_by_default(self):
        """add_exchange runs directly in process_message, outside any mocked graph — keep it off
        the real Chroma store here; tests that care about it patch/assert on it explicitly."""
        with patch("app.services.chat_service.memory_store.add_exchange"):
            yield

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
    async def test_process_message_crisis_bypasses_workflow(self):
        service = ChatService()
        service.workflow_app = MagicMock()
        service.workflow_app.ainvoke = AsyncMock(return_value={"generation": "should not be used"})
        from app.services import db_service
        with patch.object(db_service, 'save_message'):
            result = await service.process_message("test-session", "I want to kill myself")
            assert result["safety"]["blocked"] is True
            assert result["safety"]["category"] == "crisis"
            assert result["disclaimer"] is None
            service.workflow_app.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_message_normal_has_disclaimer(self):
        service = ChatService()
        service.workflow_app = MagicMock()
        service.workflow_app.ainvoke = AsyncMock(return_value={
            "generation": "Test response",
            "source": "Test Source"
        })
        from app.services import db_service
        with patch.object(db_service, 'save_message'):
            result = await service.process_message("test-session", "What is diabetes?")
            assert result["safety"]["blocked"] is False
            assert result["disclaimer"]

    @pytest.mark.asyncio
    async def test_process_message_refused_topic_bypasses_workflow(self):
        service = ChatService()
        service.workflow_app = MagicMock()
        service.workflow_app.ainvoke = AsyncMock(return_value={"generation": "should not be used"})
        from app.services import db_service
        with patch.object(db_service, 'save_message'):
            result = await service.process_message("test-session", "what is the dose of paracetamol for my baby")
            assert result["safety"]["refused_topic"] == "pediatric_dosing"
            service.workflow_app.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_message_strips_ungrounded_figure(self):
        from langchain_core.documents import Document
        service = ChatService()
        service.workflow_app = MagicMock()
        service.workflow_app.ainvoke = AsyncMock(return_value={
            "generation": "Take 999mg every 2 hours.",
            "source": "Test Source",
            "documents": [Document(page_content="General information about pain relief.")],
        })
        from app.services import db_service
        with patch.object(db_service, 'save_message'):
            result = await service.process_message("test-session", "what helps with a headache")
            assert "999mg" not in result["response"]
            assert len(result["safety"]["figures_removed"]) > 0

    @pytest.mark.asyncio
    async def test_process_message_rehydrates_history_on_cold_start(self):
        service = ChatService()
        service.workflow_app = MagicMock()
        service.workflow_app.ainvoke = AsyncMock(return_value={"generation": "Test response", "source": "Test Source"})
        from app.services import db_service
        with patch.object(db_service, 'save_message'), \
             patch.object(db_service, 'get_chat_history', return_value=[
                 {"role": "user", "content": "old question", "source": None},
                 {"role": "assistant", "content": "old answer", "source": "Test Source"},
             ]) as mock_history:
            await service.process_message("new-session", "Hello again")
            mock_history.assert_called_once_with("new-session")
            passed_state = service.workflow_app.ainvoke.call_args[0][0]
            assert passed_state["conversation_history"][0]["content"] == "old question"

    @pytest.mark.asyncio
    async def test_process_message_stores_successful_exchange_in_memory(self):
        service = ChatService()
        service.workflow_app = MagicMock()
        service.workflow_app.ainvoke = AsyncMock(return_value={"generation": "Test response", "source": "Test Source"})
        from app.services import db_service
        with patch.object(db_service, 'save_message'), \
             patch("app.services.chat_service.memory_store.add_exchange") as mock_add:
            await service.process_message("test-session", "What is diabetes?")
            mock_add.assert_called_once_with("test-session", "What is diabetes?", "Test response")

    @pytest.mark.asyncio
    async def test_process_message_degraded_answer_skips_memory_storage(self):
        service = ChatService()
        service.workflow_app = MagicMock()
        service.workflow_app.ainvoke = AsyncMock(return_value={"generation": "", "source": "System Message"})
        from app.services import db_service
        with patch.object(db_service, 'save_message'), \
             patch("app.services.chat_service.memory_store.add_exchange") as mock_add:
            await service.process_message("test-session", "What is diabetes?")
            mock_add.assert_not_called()

    @pytest.mark.asyncio
    async def test_crisis_message_never_reaches_recall_or_memory(self):
        # the safety gate returns before the graph (and MemoryAgent's recall) ever runs
        service = ChatService()
        service.workflow_app = MagicMock()
        service.workflow_app.ainvoke = AsyncMock(return_value={"generation": "should not run"})
        from app.services import db_service
        with patch.object(db_service, 'save_message'), \
             patch("app.tools.memory_store.recall") as mock_recall, \
             patch("app.services.chat_service.memory_store.add_exchange") as mock_add:
            result = await service.process_message("test-session", "I want to kill myself")
            assert result["safety"]["blocked"] is True
            mock_recall.assert_not_called()
            mock_add.assert_not_called()
            service.workflow_app.ainvoke.assert_not_called()

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

        with patch("app.services.database_service.memory_store.delete_session_memory") as mock_mem_del:
            service.delete_session("sess_del")
            mock_mem_del.assert_called_once_with("sess_del")
        assert len(service.get_chat_history("sess_del")) == 0

        test_engine.dispose()
        os.remove(test_db)
