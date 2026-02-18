"""Tests for LangGraph workflow routing — Deep Modular Architecture"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.langgraph_workflow import (  # noqa: E402
    _route_after_llm,
    _route_after_planner,
    _route_after_rag,
    _route_after_wiki,
)
from app.core.state import initialize_conversation_state  # noqa: E402


def test_routing_logic():
    state = initialize_conversation_state()

    # Planner routing
    state["current_tool"] = "retriever"
    assert _route_after_planner(state) == "retriever"
    state["current_tool"] = "other"
    assert _route_after_planner(state) == "llm_agent"

    # LLM routing
    state["llm_success"] = True
    assert _route_after_llm(state) == "executor"
    state["llm_success"] = False
    assert _route_after_llm(state) == "retriever"

    # RAG routing
    state["rag_success"] = True
    assert _route_after_rag(state) == "executor"
    state["rag_success"] = False
    assert _route_after_rag(state) == "llm_agent"

    # Wiki routing
    state["wiki_success"] = True
    assert _route_after_wiki(state) == "executor"
    state["wiki_success"] = False
    assert _route_after_wiki(state) == "tavily"
