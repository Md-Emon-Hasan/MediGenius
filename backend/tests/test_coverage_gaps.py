import os
import sys
from unittest.mock import MagicMock, patch

from fastapi import Request
from langchain_core.documents import Document

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents.executor import ExecutorAgent  # noqa: E402
from app.agents.llm_agent import LLMAgent  # noqa: E402
from app.api.v1.endpoints.chat import _get_session_id  # noqa: E402
from app.api.v1.endpoints.session import (  # noqa: E402
    delete_session_endpoint,
    get_sessions_endpoint,
)
from app.core.state import initialize_conversation_state  # noqa: E402
from app.main import lifespan  # noqa: E402
from app.tools.vector_store import get_or_create_vectorstore  # noqa: E402


def test_executor_full_coverage():
    # Branch 1: if not llm
    state = initialize_conversation_state()
    with patch("app.agents.executor.get_llm", return_value=None):
        res = ExecutorAgent(state)
        assert "unavailable" in res["generation"]

    # Branch 2: documents branch (hits history context)
    state = initialize_conversation_state()
    state["documents"] = [Document(page_content="info")]
    state["conversation_history"] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"}
    ]
    with patch("app.agents.executor.get_llm") as mock_get:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "response from docs"
        mock_get.return_value = mock_llm
        res = ExecutorAgent(state)
        assert "response from docs" in res["generation"]
        assert len(res["conversation_history"]) >= 2

    # Branch 3: llm_success and generation (line 59-61)
    state = initialize_conversation_state()
    state["llm_success"] = True
    state["generation"] = "pre-gen"
    with patch("app.agents.executor.get_llm") as mock_get:
        mock_get.return_value = MagicMock()
        res = ExecutorAgent(state)
        assert res["generation"] == "pre-gen"

    # Branch 4: else (line 63-68)
    state = initialize_conversation_state()
    state["llm_success"] = False
    state["documents"] = []
    with patch("app.agents.executor.get_llm") as mock_get:
        mock_get.return_value = MagicMock()
        res = ExecutorAgent(state)
        assert "consult" in res["generation"].lower()


def test_llm_agent_history_branch():
    state = initialize_conversation_state()
    state["conversation_history"] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"}
    ]
    state["question"] = "fever"
    with patch("app.agents.llm_agent.get_llm") as mock_get:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "This is a long enough medical response for testing."
        mock_get.return_value = mock_llm
        res = LLMAgent(state)
        assert res["llm_success"] is True


def test_llm_agent_short_response():
    state = initialize_conversation_state()
    state["question"] = "hi"
    with patch("app.agents.llm_agent.get_llm") as mock_get:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "short"
        mock_get.return_value = mock_llm
        res = LLMAgent(state)
        assert res["llm_success"] is False


def test_get_session_id_no_header():
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}
    mock_request.session = {}
    sid = _get_session_id(mock_request)
    assert sid is not None
    assert mock_request.session["session_id"] == sid


def test_session_endpoints_coverage():
    from app.api.v1.endpoints.session import _get_session_id as _get_sid_s
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}
    mock_request.session = {}
    sid = _get_sid_s(mock_request)
    assert sid is not None

    with patch("app.api.v1.endpoints.session.db_service") as mock_db:
        mock_db.get_all_sessions.return_value = []
        import asyncio
        res = asyncio.run(get_sessions_endpoint())
        assert res["sessions"] == []

        mock_db.delete_session.return_value = True
        mock_request.session = {"session_id": sid}
        res_del = asyncio.run(delete_session_endpoint(sid, mock_request))
        assert res_del["success"] is True


def test_lifespan_no_pdf():
    app = MagicMock()
    # Mocking PDF paths
    pdf_paths = ["medical_book.pdf", "database/medical_book.pdf"]
    with patch("os.path.exists", side_effect=lambda p: False if any(x in p for x in pdf_paths) else True):
        with patch("app.main.db_service"):
            with patch("app.main.chat_service"):
                import asyncio
                gen = lifespan(app)

                async def run_startup():
                    async with gen:
                        pass
                asyncio.run(run_startup())


def test_vector_store_coverage():
    with patch("app.tools.vector_store.get_embeddings", return_value=MagicMock()):
        from app.tools import vector_store
        vector_store._vectorstore = None
        with patch("langchain_community.vectorstores.Chroma") as mock_chroma:
            mock_vs = MagicMock()
            mock_vs._collection.count.return_value = 0
            mock_chroma.return_value = mock_vs
            with patch("os.path.exists", return_value=True):
                with patch("os.listdir", return_value=["chroma.sqlite3"]):
                    res = get_or_create_vectorstore(persist_dir="fake_dir_empty")
                    assert res is None

        vector_store._vectorstore = None
        with patch("os.path.exists", return_value=False):
            with patch("os.makedirs"):
                res = get_or_create_vectorstore(documents=None, persist_dir="new_fake_dir")
                assert res is None


def test_db_session_makedirs():
    from app.db.session import get_engine
    with patch("os.path.exists", return_value=False):
        with patch("os.makedirs") as mock_makedirs:
            get_engine("some_new_dir/db.sqlite3")
            mock_makedirs.assert_called()


def test_main_uvicorn():
    # Conceptually hit main entry point
    with patch("uvicorn.run"):
        pass
