"""Tests for tools — Deep Modular Architecture"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app.tools.duckduckgo_search as ddg_module  # noqa: E402
import app.tools.llm_client as llm_module  # noqa: E402
import app.tools.tavily_search as tavily_module  # noqa: E402
import app.tools.vector_store as vs_module  # noqa: E402
import app.tools.wikipedia_search as wiki_module  # noqa: E402
from app.tools.duckduckgo_search import get_duckduckgo_search  # noqa: E402
from app.tools.llm_client import get_llm  # noqa: E402
from app.tools.pdf_loader import process_pdf, split_documents  # noqa: E402
from app.tools.tavily_search import get_tavily_search  # noqa: E402
from app.tools.vector_store import (  # noqa: E402
    get_embeddings,
    get_or_create_vectorstore,
    get_retriever,
)
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
        with patch('langchain_community.tools.tavily_search.TavilySearchResults') as mock_tav:
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


def test_get_duckduckgo_no_import():
    ddg_module._ddg_search = None
    # Patch the actual source to trigger ImportError in the local import
    with patch('langchain_community.tools.DuckDuckGoSearchRun', side_effect=ImportError):
        # We need to be careful with __import__ patching
        with patch('app.tools.duckduckgo_search.logger') as mock_log:
            res = get_duckduckgo_search()
            assert res is None
            mock_log.warning.assert_called()


def test_get_duckduckgo_success():
    ddg_module._ddg_search = None
    with patch('langchain_community.tools.DuckDuckGoSearchRun') as mock_ddg:
        mock_ddg.return_value = MagicMock()
        res = get_duckduckgo_search()
        assert res is not None
    ddg_module._ddg_search = None


def test_vector_store_embeddings():
    vs_module._embeddings = None
    with patch('langchain_huggingface.embeddings.HuggingFaceEmbeddings') as mock_emb:
        mock_emb.return_value = MagicMock()
        res = get_embeddings()
        assert res is not None
    vs_module._embeddings = None


def test_vector_store_get_or_create():
    vs_module._vectorstore = None
    vs_module._embeddings = MagicMock()

    with patch('langchain_community.vectorstores.Chroma') as mock_chroma_cls:
        mock_vs = MagicMock()
        mock_vs._collection.count.return_value = 5
        mock_chroma_cls.return_value = mock_vs

        # Test loading existing
        with patch('os.path.exists', return_value=True):
            with patch('os.listdir', return_value=['chroma.sqlite3']):
                res = get_or_create_vectorstore(persist_dir="fake")
                assert res is not None

        vs_module._vectorstore = None
        # Test creation from docs
        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs'):
                res = get_or_create_vectorstore(documents=[MagicMock()], persist_dir="new")
                assert res is not None

    vs_module._vectorstore = None


def test_get_retriever():
    vs_module._vectorstore = MagicMock()
    vs_module._vectorstore.as_retriever.return_value = MagicMock()
    res = get_retriever()
    assert res is not None

    vs_module._vectorstore = None
    with patch('app.tools.vector_store.get_or_create_vectorstore', return_value=None):
        assert get_retriever() is None


def test_split_documents():
    mock_doc = MagicMock()
    with patch('langchain_text_splitters.RecursiveCharacterTextSplitter') as mock_splitter_cls:
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = [mock_doc]
        mock_splitter_cls.from_tiktoken_encoder.return_value = mock_splitter

        res = split_documents([mock_doc])
        assert len(res) == 1
