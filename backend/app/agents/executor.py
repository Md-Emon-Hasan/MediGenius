"""
MediGenius — agents/executor.py
ExecutorAgent: synthesizes the final response using the LLM and gathered context.
"""

from app.agents.symptom_analysis_sub_agent import format_symptom_context
from app.core import safety_router
from app.core.logging_config import logger
from app.core.state import AgentState
from app.tools import model_gateway

FALLBACK_ANSWER = (
    "I understand your concern about your symptoms. For accurate medical advice, "
    "please consult with a healthcare professional who can properly evaluate your condition."
)


def ExecutorAgent(state: AgentState) -> AgentState:
    """Synthesize the final patient response from retrieved documents or LLM knowledge."""
    question = state["question"]
    source_info = state.get("source", "Unknown")

    # Build recent conversation context
    history_context = ""
    for item in state.get("conversation_history", [])[-3:]:
        if item.get("role") == "user":
            history_context += f"Patient: {item.get('content', '')}\n"
        elif item.get("role") == "assistant":
            history_context += f"Doctor: {item.get('content', '')}\n"

    if not model_gateway.is_available():
        answer = (
            "Medical AI service temporarily unavailable. "
            "Please consult a healthcare professional."
        )
        source_info = "System Message"

    elif state.get("documents") and len(state["documents"]) > 0:
        content = "\n\n".join(
            [safety_router.sanitize_external_text(doc.page_content[:1000]) for doc in state["documents"][:3]]
        )
        state["evidence_text"] = content
        prompt = (
            "You are an experienced medical doctor providing helpful consultation.\n\n"
            f"Previous Conversation:\n{history_context}\n"
            f"Patient's Current Question:\n{question}\n"
            f"{format_symptom_context(state)}\n"
            f"Medical Information:\n{content}\n\n"
            "Provide a clear, caring response in 2-4 sentences. Be professional and reassuring."
        )
        result = model_gateway.generate(prompt, tier="synthesis")
        if result["content"]:
            answer = result["content"]
            state["model_used"] = result["model_used"]
            state["model_fallback"] = result["fallback"]
            logger.info("Executor: Generated response from documents")
        else:
            answer = FALLBACK_ANSWER
            source_info = "System Message"

    elif state.get("llm_success") and state.get("generation"):
        answer = state["generation"]
        logger.info("Executor: Using pre-generated LLM response")

    else:
        answer = FALLBACK_ANSWER
        source_info = "System Message"

    state["generation"] = answer
    state["source"] = source_info
    state["conversation_history"].append({"role": "user", "content": question})
    state["conversation_history"].append(
        {"role": "assistant", "content": answer, "source": source_info}
    )
    return state
