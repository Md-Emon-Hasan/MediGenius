"""
MediGenius — agents/wikipedia.py
WikipediaAgent: searches Wikipedia for medical information.
"""

from langchain_core.documents import Document

from app.core.logging_config import logger
from app.core.state import AgentState
from app.tools.wikipedia_search import get_wikipedia_wrapper


def WikipediaAgent(state: AgentState) -> AgentState:
    """Search Wikipedia for medical information as a fallback source."""
    wiki = get_wikipedia_wrapper()
    if not wiki:
        state["documents"] = []
        state["wiki_success"] = False
        state["wiki_attempted"] = True
        return state

    search_query = f"{state['question']} medical symptoms treatment"
    content = wiki.run(search_query)

    # Retry with raw question if first search is too short
    if not content or len(content.strip()) < 100:
        content = wiki.run(state["question"])

    if content and len(content.strip()) > 100:
        state["documents"] = [Document(page_content=content)]
        state["wiki_success"] = True
        state["source"] = "Wikipedia Medical Information"
        logger.info("Wikipedia: Found relevant content")
    else:
        state["documents"] = []
        state["wiki_success"] = False
        logger.info("Wikipedia: No relevant content found")

    state["wiki_attempted"] = True
    return state
