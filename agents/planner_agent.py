# agents/planner_agent.py
from core.state import AgentState

class PlannerAgent:
    @staticmethod
    def process(state: AgentState) -> AgentState:
        question = state["question"].lower()
        medical_keywords = ["pain", "fever", "treatment", "symptom", "diagnosis", 
                          "cancer", "disease", "virus", "bacteria", "infection"]

        if any(word in question for word in medical_keywords):
            state["current_tool"] = "llm"
        else:
            state["current_tool"] = "llm"
        state["retry_count"] = 0
        return state