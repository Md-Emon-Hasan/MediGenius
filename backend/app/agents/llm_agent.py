"""
MediGenius — agents/llm_agent.py
LLMAgent: generates a direct response from the LLM without RAG.
"""

from app.core.logging_config import logger
from app.core.state import AgentState
from app.tools import model_gateway


def LLMAgent(state: AgentState) -> AgentState:
    """Generate a response directly from the LLM (no retrieval)."""
    if not model_gateway.is_available():
        state["llm_success"] = False
        state["llm_attempted"] = True
        state["generation"] = "Medical AI service is temporarily unavailable."
        return state

    # Build conversation context
    history_context = ""
    for item in state.get("conversation_history", [])[-5:]:
        if item.get("role") == "user":
            history_context += f"Patient: {item.get('content', '')}\n"
        elif item.get("role") == "assistant":
            history_context += f"Doctor: {item.get('content', '')}\n"

    prompt = (
        "You are a compassionate and knowledgeable medical AI assistant helping a patient.\n\n"
        f"Conversation History:\n{history_context}\n"
        f"Current Patient Question:\n{state['question']}\n\n"
        "Provide a helpful medical response in 2-3 sentences. Be clear, professional, and caring."
    )

    result = model_gateway.generate(prompt, tier="synthesis")
    answer = result["content"]

    if answer and len(answer) > 10:
        state["generation"] = answer
        state["llm_success"] = True
        state["source"] = "AI Medical Knowledge"
        state["model_used"] = result["model_used"]
        state["model_fallback"] = result["fallback"]
        logger.info("LLM: Generated response successfully")
    else:
        state["llm_success"] = False
        logger.warning("LLM: Response too short or empty")

    state["llm_attempted"] = True
    return state
