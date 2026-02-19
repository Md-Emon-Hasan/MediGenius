"""Tests for LangGraph routing functions"""
from app.core.langgraph_workflow import (
    _route_after_llm,
    _route_after_llm_fallback,
    _route_after_planner,
    _route_after_rag,
    _route_after_tavily,
    _route_after_wiki,
    create_workflow,
)


def test_route_after_planner():
    assert _route_after_planner({"current_tool": "retriever"}) == "retriever"
    assert _route_after_planner({"current_tool": "llm_agent"}) == "llm_agent"


def test_route_after_llm():
    assert _route_after_llm({"llm_success": True}) == "executor"
    assert _route_after_llm({"llm_success": False}) == "retriever"


def test_route_after_rag():
    assert _route_after_rag({"rag_success": True}) == "executor"
    assert _route_after_rag({"rag_success": False}) == "llm_agent"


def test_route_after_llm_fallback():
    assert _route_after_llm_fallback({"llm_success": True}) == "executor"
    assert _route_after_llm_fallback({"llm_success": False}) == "wikipedia"


def test_route_after_wiki():
    assert _route_after_wiki({"wiki_success": True}) == "executor"
    assert _route_after_wiki({"wiki_success": False}) == "tavily"


def test_route_after_tavily():
    assert _route_after_tavily({}) == "executor"


def test_create_workflow():
    workflow = create_workflow()
    assert workflow is not None
