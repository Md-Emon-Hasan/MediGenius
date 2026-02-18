"""
MediGenius — tools/duckduckgo_search.py
DuckDuckGo search tool — placeholder for future use.

This module provides a free, no-API-key alternative to Tavily for web search.
Activate by installing: pip install duckduckgo-search langchain-community
"""

from app.core.logging_config import logger

_ddg_search = None


def get_duckduckgo_search(max_results: int = 3):
    """
    Return a cached DuckDuckGoSearchRun instance.

    Note: Currently a placeholder. Uncomment the implementation below
    and install `duckduckgo-search` to activate.
    """
    global _ddg_search
    if _ddg_search is None:
        try:
            from langchain_community.tools import DuckDuckGoSearchRun

            _ddg_search = DuckDuckGoSearchRun()
            logger.info("DuckDuckGo search tool initialized")
        except ImportError:
            logger.warning(
                "DuckDuckGo search not available. "
                "Install with: pip install duckduckgo-search"
            )
            return None
    return _ddg_search
