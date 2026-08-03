"""
MediGenius — agents/memory.py
MemoryAgent: trims conversation history to the last 20 turns and recalls relevant
past exchanges from this session's semantic memory.
"""

from app.core.config import MAX_RECALLED_MEMORIES
from app.core.state import AgentState
from app.tools import memory_store


def format_recalled_memories(state: AgentState) -> str:
    """Prompt-ready block of recalled memories — empty string if there are none."""
    memories = state.get("recalled_memories")
    if not memories:
        return ""
    joined = "\n---\n".join(memories)
    return f"\nRelevant past exchanges from this conversation:\n{joined}\n"


def MemoryAgent(state: AgentState) -> AgentState:
    """Trim conversation history to the last 20 turns and recall relevant past exchanges."""
    history = state.get("conversation_history", [])
    if len(history) > 20:
        history = history[-20:]
    state["conversation_history"] = history

    session_id = state.get("session_id")
    question = state.get("question")
    if session_id and question:
        state["recalled_memories"] = memory_store.recall(session_id, question, k=MAX_RECALLED_MEMORIES)
    else:
        state["recalled_memories"] = []

    return state
