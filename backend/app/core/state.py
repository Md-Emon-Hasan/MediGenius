"""
MediGenius — core/state.py
AgentState TypedDict and state helper functions.
"""

from typing import Dict, List, Optional, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    """Shared state passed between all LangGraph agent nodes."""

    question: str
    documents: List[Document]
    generation: str
    source: str
    search_query: Optional[str]
    conversation_history: List[Dict]
    llm_attempted: bool
    llm_success: bool
    rag_attempted: bool
    rag_success: bool
    wiki_attempted: bool
    wiki_success: bool
    tavily_attempted: bool
    tavily_success: bool
    current_tool: Optional[str]
    retry_count: int


def initialize_conversation_state() -> AgentState:
    """Return a fresh AgentState with all fields at their defaults."""
    return {
        "question": "",
        "documents": [],
        "generation": "",
        "source": "",
        "search_query": None,
        "conversation_history": [],
        "llm_attempted": False,
        "llm_success": False,
        "rag_attempted": False,
        "rag_success": False,
        "wiki_attempted": False,
        "wiki_success": False,
        "tavily_attempted": False,
        "tavily_success": False,
        "current_tool": None,
        "retry_count": 0,
    }


def reset_query_state(state: AgentState) -> AgentState:
    """Reset per-query flags while preserving conversation history."""
    state.update(
        {
            "question": "",
            "documents": [],
            "generation": "",
            "source": "",
            "search_query": None,
            "llm_attempted": False,
            "llm_success": False,
            "rag_attempted": False,
            "rag_success": False,
            "wiki_attempted": False,
            "wiki_success": False,
            "tavily_attempted": False,
            "tavily_success": False,
            "current_tool": None,
            "retry_count": 0,
        }
    )
    return state
