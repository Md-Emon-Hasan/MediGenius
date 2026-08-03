"""
MediGenius — core/langgraph_workflow.py
LangGraph StateGraph definition, routing functions, and workflow factory.
"""

from langgraph.graph import END, StateGraph

from app.agents.diagnosis_verification_sub_agent import DiagnosisVerificationSubAgent
from app.agents.executor import ExecutorAgent
from app.agents.explanation import ExplanationAgent
from app.agents.llm_agent import LLMAgent
from app.agents.medical_supervisor_agent import MedicalSupervisorAgent
from app.agents.memory import MemoryAgent
from app.agents.parallel_retrieval_agent import ParallelRetrievalAgent
from app.agents.planner import PlannerAgent
from app.agents.symptom_analysis_sub_agent import SymptomAnalysisSubAgent
from app.core.state import AgentState


# ── Routing Functions ──────────────────────────────────────────────────────────
def _route_after_supervisor(state: AgentState) -> str:
    return "symptom_analysis" if state.get("needs_symptom_analysis") else "planner"


def _route_after_planner(state: AgentState) -> str:
    return "parallel_retrieval" if state["current_tool"] == "retriever" else "llm_agent"


def _route_after_parallel_retrieval(state: AgentState) -> str:
    return "executor" if state.get("rag_success") else "llm_agent"


# superseded by _route_after_parallel_retrieval — kept because they're still unit-tested
# and because retriever/wikipedia/tavily nodes stay importable for standalone use
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
    workflow.add_node("supervisor", MedicalSupervisorAgent)
    workflow.add_node("symptom_analysis", SymptomAnalysisSubAgent)
    workflow.add_node("planner", PlannerAgent)
    workflow.add_node("llm_agent", LLMAgent)
    workflow.add_node("parallel_retrieval", ParallelRetrievalAgent)
    workflow.add_node("executor", ExecutorAgent)
    workflow.add_node("diagnosis_verification", DiagnosisVerificationSubAgent)
    workflow.add_node("explanation", ExplanationAgent)

    # Entry point
    workflow.set_entry_point("memory")

    # Edges
    workflow.add_edge("memory", "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {"symptom_analysis": "symptom_analysis", "planner": "planner"},
    )
    workflow.add_edge("symptom_analysis", "planner")
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"parallel_retrieval": "parallel_retrieval", "llm_agent": "llm_agent"},
    )
    workflow.add_conditional_edges(
        "parallel_retrieval",
        _route_after_parallel_retrieval,
        {"executor": "executor", "llm_agent": "llm_agent"},
    )
    workflow.add_edge("llm_agent", "executor")
    workflow.add_edge("executor", "diagnosis_verification")
    workflow.add_edge("diagnosis_verification", END)

    return workflow.compile()
