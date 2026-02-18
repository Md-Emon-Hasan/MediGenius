"""
MediGenius — agents/memory.py
MemoryAgent: trims conversation history to the last 20 turns.
"""

from app.core.state import AgentState


def MemoryAgent(state: AgentState) -> AgentState:
    """Trim conversation history to the last 20 turns to avoid context overflow."""
    history = state.get("conversation_history", [])
    if len(history) > 20:
        history = history[-20:]
    state["conversation_history"] = history
    return state
