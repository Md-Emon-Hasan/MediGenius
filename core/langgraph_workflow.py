# core/langgraph_workflow.py
from langgraph.graph import StateGraph
from langgraph.graph import END
from agents import MemoryAgent
from agents import PlannerAgent
from agents import LLMAgent
from agents import RetrieverAgent
from agents import WikipediaAgent
from agents import DuckDuckGoAgent
from agents import ExecutorAgent
from agents import ExplanationAgent

from core.state import AgentState

def setup_workflow():
    workflow = StateGraph(AgentState)
    
    # Add all agent nodes
    workflow.add_node("memory", MemoryAgent.process)
    workflow.add_node("planner", PlannerAgent.process)
    workflow.add_node("llm_agent", LLMAgent.process)
    workflow.add_node("retriever", RetrieverAgent.process)
    workflow.add_node("wikipedia", WikipediaAgent.process)
    workflow.add_node("duckduckgo", DuckDuckGoAgent.process)
    workflow.add_node("executor", ExecutorAgent.process)
    workflow.add_node("explanation", ExplanationAgent.process)
    
    # Set entry point
    workflow.set_entry_point("memory")
    
    # Define edges and conditional routing
    workflow.add_edge("memory", "planner")
    workflow.add_edge("planner", "llm_agent")
    
    def route_after_llm(state: AgentState):
        if state.get("llm_success", False):
            return "executor"
        return "retriever"
    
    workflow.add_conditional_edges(
        "llm_agent",
        route_after_llm,
        {"executor": "executor", "retriever": "retriever"}
    )
    
    def route_after_rag(state: AgentState):
        if state.get("rag_success", False):
            return "executor"
        return "wikipedia"
    
    workflow.add_conditional_edges(
        "retriever",
        route_after_rag,
        {"executor": "executor", "wikipedia": "wikipedia"}
    )
    
    def route_after_wiki(state: AgentState):
        if state.get("wiki_success", False):
            return "executor"
        return "duckduckgo"
    
    workflow.add_conditional_edges(
        "wikipedia",
        route_after_wiki,
        {"executor": "executor", "duckduckgo": "duckduckgo"}
    )
    
    def route_after_ddg(state: AgentState):
        return "executor"
    
    workflow.add_conditional_edges(
        "duckduckgo",
        route_after_ddg,
        {"executor": "executor"}
    )
    
    workflow.add_edge("executor", "explanation")
    workflow.add_edge("explanation", END)
    
    return workflow.compile()