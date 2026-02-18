"""Tests for tools — Deep Modular Architecture"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app.tools.llm_client as llm_module  # noqa: E402
import app.tools.tavily_search as tavily_module  # noqa: E402
import app.tools.wikipedia_search as wiki_module  # noqa: E402
from app.tools.llm_client import get_llm  # noqa: E402
from app.tools.pdf_loader import process_pdf  # noqa: E402
from app.tools.tavily_search import get_tavily_search  # noqa: E402
from app.tools.wikipedia_search import get_wikipedia_wrapper  # noqa: E402


def test_get_llm_no_key():
    llm_module._llm_instance = None
    with patch('app.tools.llm_client.GROQ_API_KEY', None):
        result = get_llm()
        assert result is None


def test_get_llm_with_key():
    llm_module._llm_instance = None
    with patch('app.tools.llm_client.GROQ_API_KEY', 'fake-key'):
        # Patch at the source since ChatGroq is lazily imported inside the function
        with patch('langchain_groq.ChatGroq') as mock_groq:
            mock_groq.return_value = MagicMock()
            result = get_llm()
            assert result is not None
    llm_module._llm_instance = None  # reset


def test_get_wikipedia():
    wiki_module._wiki_wrapper = None
    # Patch at the source since WikipediaAPIWrapper is lazily imported inside the function
    with patch('langchain_community.utilities.wikipedia.WikipediaAPIWrapper') as mock_wiki:
        mock_wiki.return_value = MagicMock()
        wrapper = get_wikipedia_wrapper()
        assert wrapper is not None
        # Singleton check
        assert get_wikipedia_wrapper() == wrapper
    wiki_module._wiki_wrapper = None  # reset


def test_get_tavily_no_key():
    tavily_module._tavily_search = None
    with patch('app.tools.tavily_search.TAVILY_API_KEY', None):
        result = get_tavily_search()
        assert result is None


def test_get_tavily_with_key():
    tavily_module._tavily_search = None
    with patch('app.tools.tavily_search.TAVILY_API_KEY', 'fake-key'):
        # Patch at the source since TavilySearchResults is lazily imported inside the function
        with patch('langchain_community.tools.tavily_search.tool.TavilySearchResults') as mock_tav:
            mock_tav.return_value = MagicMock()
            result = get_tavily_search()
            assert result is not None
    tavily_module._tavily_search = None  # reset


def test_pdf_loader():
    # Patch at the source since PyPDFLoader is lazily imported inside the function
    with patch('langchain_community.document_loaders.PyPDFLoader') as mock_loader_cls:
        mock_loader = MagicMock()
        mock_loader.load.return_value = []
        mock_loader_cls.return_value = mock_loader

        with patch('app.tools.pdf_loader.split_documents') as mock_split:
            mock_split.return_value = ["chunk1"]
            res = process_pdf("path.pdf")
            assert res == ["chunk1"]
