# agents/memory_agent.py
from core.state import AgentState

class MemoryAgent:
    @staticmethod
    def process(state: AgentState) -> AgentState:
        history = state.get("conversation_history", [])
        if len(history) > 20:
            history = history[-20:]  # keep last 20 messages
        state["conversation_history"] = history
        return state