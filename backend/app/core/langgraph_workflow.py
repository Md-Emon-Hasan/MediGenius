"""
MediGenius — core/langgraph_workflow.py
LangGraph StateGraph definition, routing functions, and workflow factory.
"""

from langgraph.graph import END, StateGraph

from app.agents.executor import ExecutorAgent
from app.agents.explanation import ExplanationAgent
from app.agents.llm_agent import LLMAgent
from app.agents.memory import MemoryAgent
from app.agents.planner import PlannerAgent
from app.agents.retriever import RetrieverAgent
from app.agents.tavily import TavilyAgent
from app.agents.wikipedia import WikipediaAgent
from app.core.state import AgentState


# ── Routing Functions ──────────────────────────────────────────────────────────
def _route_after_planner(state: AgentState) -> str:
    return "retriever" if state["current_tool"] == "retriever" else "llm_agent"


def _route_after_llm(state: AgentState) -> str:
    return "executor" if state.get("llm_success") else "retriever"


def _route_after_rag(state: AgentState) -> str:
    return "executor" if state.get("rag_success") else "llm_agent"


def _route_after_llm_fallback(state: AgentState) -> str:
    return "executor" if state.get("llm_success") else "wikipedia"


def _route_after_wiki(state: AgentState) -> str:
    return "executor" if state.get("wiki_success") else "tavily"


def _route_after_tavily(state: AgentState) -> str:
    return "executor"


# ── Workflow Factory ───────────────────────────────────────────────────────────
def create_workflow():
    """Build and compile the LangGraph agentic workflow."""
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("memory", MemoryAgent)
    workflow.add_node("planner", PlannerAgent)
    workflow.add_node("llm_agent", LLMAgent)
    workflow.add_node("retriever", RetrieverAgent)
    workflow.add_node("wikipedia", WikipediaAgent)
    workflow.add_node("tavily", TavilyAgent)
    workflow.add_node("executor", ExecutorAgent)
    workflow.add_node("explanation", ExplanationAgent)

    # Entry point
    workflow.set_entry_point("memory")

    # Edges
    workflow.add_edge("memory", "planner")
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"retriever": "retriever", "llm_agent": "llm_agent"},
    )
    workflow.add_conditional_edges(
        "llm_agent",
        _route_after_llm,
        {"executor": "executor", "retriever": "retriever"},
    )
    workflow.add_conditional_edges(
        "retriever",
        _route_after_rag,
        {"executor": "executor", "llm_agent": "llm_agent"},
    )
    workflow.add_conditional_edges(
        "wikipedia",
        _route_after_wiki,
        {"executor": "executor", "tavily": "tavily"},
    )
    workflow.add_conditional_edges(
        "tavily", _route_after_tavily, {"executor": "executor"}
    )
    workflow.add_edge("executor", END)

    return workflow.compile()
