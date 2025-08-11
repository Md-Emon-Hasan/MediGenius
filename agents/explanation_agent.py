# agents/explanation_agent.py
from core.state import AgentState

class ExplanationAgent:
    @staticmethod
    def process(state: AgentState) -> AgentState:
        explanation = "This response is generated using a combination of medical literature and AI reasoning."
        state["conversation_history"].append(f"AI Explanation: {explanation}")
        return state