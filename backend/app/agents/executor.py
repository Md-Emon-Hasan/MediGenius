"""
MediGenius — agents/executor.py
ExecutorAgent: synthesizes the final response using the LLM and gathered context.
"""

from app.core.logging_config import logger
from app.core.state import AgentState
from app.tools.llm_client import get_llm


def ExecutorAgent(state: AgentState) -> AgentState:
    """Synthesize the final patient response from retrieved documents or LLM knowledge."""
    llm = get_llm()
    question = state["question"]
    source_info = state.get("source", "Unknown")

    # Build recent conversation context
    history_context = ""
    for item in state.get("conversation_history", [])[-3:]:
        if item.get("role") == "user":
            history_context += f"Patient: {item.get('content', '')}\n"
        elif item.get("role") == "assistant":
            history_context += f"Doctor: {item.get('content', '')}\n"

    if not llm:
        answer = (
            "Medical AI service temporarily unavailable. "
            "Please consult a healthcare professional."
        )
        source_info = "System Message"

    elif state.get("documents") and len(state["documents"]) > 0:
        content = "\n\n".join(
            [doc.page_content[:1000] for doc in state["documents"][:3]]
        )
        prompt = (
            "You are an experienced medical doctor providing helpful consultation.\n\n"
            f"Previous Conversation:\n{history_context}\n"
            f"Patient's Current Question:\n{question}\n\n"
            f"Medical Information:\n{content}\n\n"
            "Provide a clear, caring response in 2-4 sentences. Be professional and reassuring."
        )
        try:
            response = llm.invoke(prompt)
            answer = (
                response.content.strip()
                if hasattr(response, "content")
                else str(response).strip()
            )
            logger.info("Executor: Generated response from documents")
        except Exception as e:
            logger.error("Executor: LLM generation failed: %s", str(e))
            answer = (
                "I understand your concern about your symptoms. For accurate medical advice, "
                "please consult with a healthcare professional who can properly evaluate your condition."
            )
            source_info = "System Message"

    elif state.get("llm_success") and state.get("generation"):
        answer = state["generation"]
        logger.info("Executor: Using pre-generated LLM response")

    else:
        answer = (
            "I understand your concern about your symptoms. For accurate medical advice, "
            "please consult with a healthcare professional who can properly evaluate your condition."
        )
        source_info = "System Message"

    state["generation"] = answer
    state["source"] = source_info
    state["conversation_history"].append({"role": "user", "content": question})
    state["conversation_history"].append(
        {"role": "assistant", "content": answer, "source": source_info}
    )
    return state
