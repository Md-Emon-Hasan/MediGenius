from core.state import AgentState
from tools.llm_client import get_llm


def ExecutorAgent(state: AgentState) -> AgentState:
    llm = get_llm()

    if not llm:
        # Fallback if LLM not available
        state["generation"] = "Medical AI service temporarily unavailable. Please consult a healthcare professional."
        state["source"] = "System Message"
        return state

    question = state["question"]
    source_info = state.get("source", "Unknown")

    # Get conversation context
    history_context = ""
    for item in state.get("conversation_history", [])[-3:]:
        if item.get('role') == 'user':
            history_context += f"Patient: {item.get('content', '')}\n"
        elif item.get('role') == 'assistant':
            history_context += f"Doctor: {item.get('content', '')}\n"

    # If we have documents from retrieval
    if state.get("documents") and len(state["documents"]) > 0:
        content = "\n\n".join([doc.page_content[:1000] for doc in state["documents"][:3]])

        prompt = f"""You are an experienced medical doctor providing helpful consultation.

Previous Conversation:
{history_context}

Patient's Current Question:
{question}

Medical Information:
{content}

Provide a clear, caring response in 2-4 sentences. Be professional and reassuring."""

        response = llm.invoke(prompt)
        answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()

        state["generation"] = answer
        state["source"] = source_info

        # Add to conversation history
        state["conversation_history"].append({
            'role': 'user',
            'content': question
        })
        state["conversation_history"].append({
            'role': 'assistant',
            'content': answer,
            'source': source_info
        })
        return state

    # If LLM was successful earlier
    if state.get("llm_success", False) and state.get("generation"):
        answer = state["generation"]

        state["conversation_history"].append({
            'role': 'user',
            'content': question
        })
        state["conversation_history"].append({
            'role': 'assistant',
            'content': answer,
            'source': source_info
        })
        return state

    # Fallback response
    fallback_response = (
        "I understand your concern about your symptoms. For accurate medical advice, "
        "please consult with a healthcare professional who can properly evaluate your condition."
    )
    state["generation"] = fallback_response
    state["source"] = "System Message"

    state["conversation_history"].append({
        'role': 'user',
        'content': question
    })
    state["conversation_history"].append({
        'role': 'assistant',
        'content': fallback_response,
        'source': 'System Message'
    })

    return state
