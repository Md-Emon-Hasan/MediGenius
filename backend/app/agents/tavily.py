"""
MediGenius — agents/tavily.py
TavilyAgent: searches the web via Tavily for current medical information.
"""

from langchain_core.documents import Document

from app.core.logging_config import logger
from app.core.state import AgentState
from app.tools.tavily_search import get_tavily_search


def TavilyAgent(state: AgentState) -> AgentState:
    """Search the web via Tavily for current medical research and news."""
    tavily = get_tavily_search()
    if not tavily:
        state["documents"] = []
        state["tavily_success"] = False
        state["tavily_attempted"] = True
        return state

    search_query = f"{state['question']} medical health treatment symptoms"
    try:
        results = tavily.invoke(search_query)
    except Exception as e:
        logger.error("Tavily: Search failed: %s", str(e))
        results = []

    valid_results = [
        r
        for r in (results or [])
        if isinstance(r, dict) and r.get("content") and len(r["content"].strip()) > 50
    ]

    if valid_results:
        state["documents"] = [
            Document(
                page_content=r["content"],
                metadata={"url": r.get("url", ""), "title": r.get("title", "")},
            )
            for r in valid_results
        ]
        state["tavily_success"] = True
        state["source"] = "Current Medical Research & News"
        logger.info("Tavily: Found %d results", len(valid_results))
    else:
        state["documents"] = []
        state["tavily_success"] = False
        logger.info("Tavily: No valid results")

    state["tavily_attempted"] = True
    return state
