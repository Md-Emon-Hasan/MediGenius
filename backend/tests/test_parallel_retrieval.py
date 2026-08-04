"""Tests for the concurrent RAG/Wikipedia/Tavily retrieval fan-out"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents.parallel_retrieval_agent import (  # noqa: E402
    ParallelRetrievalAgent,
    _rag_sync,
    _tavily_sync,
    _wiki_sync,
)
from app.core.state import initialize_conversation_state  # noqa: E402


def _state(question="fever"):
    state = initialize_conversation_state()
    state["question"] = question
    return state


@pytest.mark.asyncio
async def test_rag_hit_wins_over_other_sources():
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document(page_content="RAG content " * 10)]
    mock_wiki = MagicMock()
    mock_wiki.run.return_value = "Wikipedia content " * 10
    mock_tavily = MagicMock()
    mock_tavily.invoke.return_value = [{"content": "Tavily content " * 10, "url": "http://x.com"}]

    with patch("app.agents.parallel_retrieval_agent.get_retriever", return_value=mock_retriever), \
         patch("app.agents.parallel_retrieval_agent.get_wikipedia_wrapper", return_value=mock_wiki), \
         patch("app.agents.parallel_retrieval_agent.get_tavily_search", return_value=mock_tavily):
        res = await ParallelRetrievalAgent(_state())
        assert res["source"] == "Medical Literature Database"
        assert res["rag_success"] is True


@pytest.mark.asyncio
async def test_tavily_wins_when_rag_empty():
    mock_wiki = MagicMock()
    mock_wiki.run.return_value = "Wikipedia content " * 10
    mock_tavily = MagicMock()
    mock_tavily.invoke.return_value = [{"content": "Tavily content " * 10, "url": "http://x.com"}]

    with patch("app.agents.parallel_retrieval_agent.get_retriever", return_value=None), \
         patch("app.agents.parallel_retrieval_agent.get_wikipedia_wrapper", return_value=mock_wiki), \
         patch("app.agents.parallel_retrieval_agent.get_tavily_search", return_value=mock_tavily):
        res = await ParallelRetrievalAgent(_state())
        assert res["source"] == "Current Medical Research & News"


@pytest.mark.asyncio
async def test_all_sources_fail():
    with patch("app.agents.parallel_retrieval_agent.get_retriever", return_value=None), \
         patch("app.agents.parallel_retrieval_agent.get_wikipedia_wrapper", return_value=None), \
         patch("app.agents.parallel_retrieval_agent.get_tavily_search", return_value=None):
        res = await ParallelRetrievalAgent(_state())
        assert res["rag_success"] is False
        assert res["documents"] == []


@pytest.mark.asyncio
async def test_branch_timeout_does_not_crash_others():
    import time

    mock_retriever = MagicMock()
    mock_retriever.invoke = MagicMock(side_effect=lambda q: time.sleep(0.3))
    mock_tavily = MagicMock()
    mock_tavily.invoke.return_value = [{"content": "Tavily content " * 10, "url": "http://x.com"}]

    with patch("app.agents.parallel_retrieval_agent.BRANCH_TIMEOUT", 0.05), \
         patch("app.agents.parallel_retrieval_agent.get_retriever", return_value=mock_retriever), \
         patch("app.agents.parallel_retrieval_agent.get_wikipedia_wrapper", return_value=None), \
         patch("app.agents.parallel_retrieval_agent.get_tavily_search", return_value=mock_tavily):
        res = await ParallelRetrievalAgent(_state())
        assert res["source"] == "Current Medical Research & News"


@pytest.mark.asyncio
async def test_branch_exception_does_not_crash_others():
    mock_retriever = MagicMock()
    mock_retriever.invoke.side_effect = Exception("vector store down")
    mock_wiki = MagicMock()
    mock_wiki.run.return_value = "Wikipedia content " * 10

    with patch("app.agents.parallel_retrieval_agent.get_retriever", return_value=mock_retriever), \
         patch("app.agents.parallel_retrieval_agent.get_wikipedia_wrapper", return_value=mock_wiki), \
         patch("app.agents.parallel_retrieval_agent.get_tavily_search", return_value=None):
        res = await ParallelRetrievalAgent(_state())
        assert res["source"] == "Wikipedia Medical Information"


def test_rag_sync_uses_cached_retrieval():
    mock_retriever = MagicMock()
    cached_docs = [Document(page_content="cached " * 20)]
    with patch("app.agents.parallel_retrieval_agent.get_retriever", return_value=mock_retriever), \
         patch("app.agents.parallel_retrieval_agent.cache.get_retrieval", return_value=cached_docs), \
         patch("app.agents.parallel_retrieval_agent.cache.set_retrieval") as mock_set:
        docs, source = _rag_sync("fever")
        assert source == "Medical Literature Database"
        mock_retriever.invoke.assert_not_called()
        mock_set.assert_not_called()


def test_wiki_sync_uses_cached_result():
    mock_wiki = MagicMock()
    with patch("app.agents.parallel_retrieval_agent.get_wikipedia_wrapper", return_value=mock_wiki), \
         patch("app.agents.parallel_retrieval_agent.cache.get_retrieval", return_value="cached " * 30), \
         patch("app.agents.parallel_retrieval_agent.cache.set_retrieval") as mock_set:
        docs, source = _wiki_sync("fever")
        assert source == "Wikipedia Medical Information"
        mock_wiki.run.assert_not_called()
        mock_set.assert_not_called()


def test_wiki_sync_short_content_returns_none():
    mock_wiki = MagicMock()
    mock_wiki.run.return_value = "short"
    with patch("app.agents.parallel_retrieval_agent.get_wikipedia_wrapper", return_value=mock_wiki), \
         patch("app.agents.parallel_retrieval_agent.cache.get_retrieval", return_value=None), \
         patch("app.agents.parallel_retrieval_agent.cache.set_retrieval"):
        docs, source = _wiki_sync("fever")
        assert docs is None and source == "wiki"


def test_tavily_sync_exception_returns_none():
    mock_tavily = MagicMock()
    mock_tavily.invoke.side_effect = Exception("network down")
    with patch("app.agents.parallel_retrieval_agent.get_tavily_search", return_value=mock_tavily):
        docs, source = _tavily_sync("fever")
        assert docs is None and source == "tavily"


def test_tavily_sync_no_valid_results_returns_none():
    mock_tavily = MagicMock()
    mock_tavily.invoke.return_value = [{"content": "", "url": "http://x.com"}]
    with patch("app.agents.parallel_retrieval_agent.get_tavily_search", return_value=mock_tavily):
        docs, source = _tavily_sync("fever")
        assert docs is None and source == "tavily"
