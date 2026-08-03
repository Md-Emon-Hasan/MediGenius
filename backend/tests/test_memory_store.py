"""Tests for the semantic conversation-memory store"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app.tools.memory_store as memory_store  # noqa: E402


def test_add_exchange_stores_session_scoped_document():
    memory_store._memory_store = None
    mock_store = MagicMock()
    with patch.object(memory_store, "get_memory_store", return_value=mock_store):
        memory_store.add_exchange("sess-1", "what is fever", "fever is a common symptom")
        mock_store.add_documents.assert_called_once()
        doc = mock_store.add_documents.call_args[0][0][0]
        assert doc.metadata["session_id"] == "sess-1"
        assert "what is fever" in doc.page_content


def test_add_exchange_failure_does_not_raise():
    with patch.object(memory_store, "get_memory_store", side_effect=Exception("chroma down")):
        memory_store.add_exchange("sess-1", "q", "a")  # should not raise


def test_recall_filters_by_session_and_caps_k():
    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [MagicMock(page_content="past exchange 1")]
    with patch.object(memory_store, "get_memory_store", return_value=mock_store):
        result = memory_store.recall("sess-1", "fever", k=3)
        assert result == ["past exchange 1"]
        mock_store.similarity_search.assert_called_once_with("fever", k=3, filter={"session_id": "sess-1"})


def test_recall_failure_returns_empty_list():
    with patch.object(memory_store, "get_memory_store", side_effect=Exception("chroma down")):
        assert memory_store.recall("sess-1", "fever", k=3) == []


def test_delete_session_memory_calls_collection_delete():
    mock_store = MagicMock()
    with patch.object(memory_store, "get_memory_store", return_value=mock_store):
        memory_store.delete_session_memory("sess-1")
        mock_store._collection.delete.assert_called_once_with(where={"session_id": "sess-1"})


def test_delete_session_memory_failure_does_not_raise():
    with patch.object(memory_store, "get_memory_store", side_effect=Exception("chroma down")):
        memory_store.delete_session_memory("sess-1")  # should not raise


def test_get_memory_store_singleton():
    memory_store._memory_store = None
    with patch("langchain_community.vectorstores.Chroma") as mock_chroma_cls, \
         patch.object(memory_store, "get_embeddings", return_value=MagicMock()):
        mock_chroma_cls.return_value = MagicMock()
        store = memory_store.get_memory_store()
        assert store is not None
        assert memory_store.get_memory_store() is store
    memory_store._memory_store = None
