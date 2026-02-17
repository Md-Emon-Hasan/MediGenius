import os
import sys
from unittest.mock import MagicMock, patch

from tools.llm_client import get_llm
from tools.pdf_loader import process_pdf
from tools.search_tools import get_tavily_search, get_wikipedia_wrapper

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))


def test_get_llm():
    with patch('os.getenv') as mock_env:
        # Case 1: No API Key
        mock_env.return_value = None
        # Must reset singleton if any
        import tools.llm_client
        tools.llm_client._llm_instance = None
        assert get_llm() is None

        # Case 2: API Key present
        mock_env.return_value = "key"
        with patch('tools.llm_client.ChatGroq') as mock_groq:
            llm = get_llm()
            assert llm is not None
            mock_groq.assert_called_once()


def test_get_wikipedia():
    import tools.search_tools
    tools.search_tools._wiki_wrapper = None

    with patch('tools.search_tools.WikipediaAPIWrapper') as mock_wiki:
        wrapper = get_wikipedia_wrapper()
        assert wrapper is not None
        mock_wiki.assert_called_once()
        # Singleton check
        assert get_wikipedia_wrapper() == wrapper
        assert mock_wiki.call_count == 1


def test_get_tavily():
    import tools.search_tools
    tools.search_tools._tavily_search = None

    with patch('os.getenv') as mock_env:
        mock_env.return_value = None
        assert get_tavily_search() is None

        mock_env.return_value = "key"
        with patch('tools.search_tools.TavilySearchResults') as mock_tav:
            tools.search_tools._tavily_search = None
            tav = get_tavily_search()
            assert tav is not None
            mock_tav.assert_called_once()


def test_pdf_loader():
    with patch('tools.pdf_loader.PyPDFLoader') as mock_loader_cls:
        mock_loader = MagicMock()
        mock_loader.load.return_value = []
        mock_loader_cls.return_value = mock_loader

        with patch('tools.pdf_loader.split_documents') as mock_split:
            mock_split.return_value = ["chunk1"]
            res = process_pdf("path.pdf")
            assert res == ["chunk1"]
