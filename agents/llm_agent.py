# agents/llm_agent.py
from tools.llm_client import get_llm
from core.state import AgentState

class LLMAgent:
    llm = get_llm()
    
    @classmethod
    def process(cls, state: AgentState) -> AgentState:
        try:
            ctx = "\n".join(state.get("conversation_history", [])[-10:])
            prompt = f"""You are a compassionate and knowledgeable medical AI assistant and doctor helping a patient. Your conversational skill should be a professional consultant with a human touch.

Patient's History:
{ctx}

Patient's Question:
{state['question']}

Respond like an experienced doctor in 2–3 sentences. Be clear, professional, and confident. Do not mention sources or uncertainty."""

            response = cls.llm.invoke(prompt)
            answer = response.content.strip()

            if answer:
                state["generation"] = answer
                state["llm_success"] = True
            else:
                state["llm_success"] = False
        except Exception:
            state["llm_success"] = False

        state["llm_attempted"] = True
        return state