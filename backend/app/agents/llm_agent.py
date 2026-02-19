"""
MediGenius — agents/llm_agent.py
LLMAgent: generates a direct response from the LLM without RAG.
"""

from app.core.logging_config import logger
from app.core.state import AgentState
from app.tools.llm_client import get_llm


def LLMAgent(state: AgentState) -> AgentState:
    """Generate a response directly from the LLM (no retrieval)."""
    llm = get_llm()
    if not llm:
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

    response = llm.invoke(prompt)
    answer = (
        response.content.strip()
        if hasattr(response, "content")
        else str(response).strip()
    )

    if answer and len(answer) > 10:
        state["generation"] = answer
        state["llm_success"] = True
        state["source"] = "AI Medical Knowledge"
        logger.info("LLM: Generated response successfully")
    else:
        state["llm_success"] = False
        logger.warning("LLM: Response too short or empty")

    state["llm_attempted"] = True
    return state
