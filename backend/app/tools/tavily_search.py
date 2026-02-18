"""
MediGenius — tools/tavily_search.py
Tavily web search tool singleton.
"""

from app.core.config import TAVILY_API_KEY
from app.core.logging_config import logger

_tavily_search = None


def get_tavily_search():
    """Return a cached TavilySearchResults instance, or None if API key is missing."""
    global _tavily_search
    if _tavily_search is None:
        if not TAVILY_API_KEY:
            logger.warning("TAVILY_API_KEY not found in environment variables")
            return None
        from langchain_community.tools.tavily_search import TavilySearchResults

        _tavily_search = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)
        logger.info("Tavily search tool initialized")
    return _tavily_search
