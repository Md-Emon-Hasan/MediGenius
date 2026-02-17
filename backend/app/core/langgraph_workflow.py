from agents.executor_agent import ExecutorAgent
from agents.explanation_agent import ExplanationAgent
from agents.llm_agent import LLMAgent
from agents.memory_agent import MemoryAgent
from agents.planner_agent import PlannerAgent
from agents.retriever_agent import RetrieverAgent
from agents.tavily_agent import TavilyAgent
from agents.wikipedia_agent import WikipediaAgent
from core.state import AgentState
from langgraph.graph import END, StateGraph


def route_after_planner(state: AgentState):
    if state["current_tool"] == "retriever":
        return "retriever"
    else:
        return "llm_agent"


def route_after_llm(state: AgentState):
    if state.get("llm_success", False):
        return "executor"
    else:
        return "retriever"


def route_after_rag(state: AgentState):
    if state.get("rag_success", False):
        return "executor"
    else:
        return "llm_agent"  # Try LLM if RAG fails


def route_after_llm_fallback(state: AgentState):
    if state.get("llm_success", False):
        return "executor"
    else:
        return "wikipedia"


def route_after_wiki(state: AgentState):
    if state.get("wiki_success", False):
        return "executor"
    else:
        return "tavily"


def route_after_tavily(state: AgentState):
    return "executor"


def create_workflow():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("memory", MemoryAgent)
    workflow.add_node("planner", PlannerAgent)
    workflow.add_node("llm_agent", LLMAgent)
    workflow.add_node("retriever", RetrieverAgent)
    workflow.add_node("wikipedia", WikipediaAgent)
    workflow.add_node("tavily", TavilyAgent)
    workflow.add_node("executor", ExecutorAgent)
    workflow.add_node("explanation", ExplanationAgent)

    # Set entry point
    workflow.set_entry_point("memory")

    # Add edges
    workflow.add_edge("memory", "planner")

    # Conditional edges with improved fallback logic
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "retriever": "retriever",
            "llm_agent": "llm_agent"
        }
    )

    # If initial LLM attempt
    workflow.add_conditional_edges(
        "llm_agent",
        route_after_llm,
        {
            "executor": "executor",
            "retriever": "retriever"
        }
    )

    # If retriever fails, try LLM
    workflow.add_conditional_edges(
        "retriever",
        route_after_rag,
        {
            "executor": "executor",
            "llm_agent": "llm_agent"
        }
    )

    # After Wiki
    workflow.add_conditional_edges(
        "wikipedia",
        route_after_wiki,
        {
            "executor": "executor",
            "tavily": "tavily"
        }
    )

    # After Tavily
    workflow.add_conditional_edges(
        "tavily",
        route_after_tavily,
        {
            "executor": "executor"
        }
    )

    workflow.add_edge("executor", END)

    return workflow.compile()
