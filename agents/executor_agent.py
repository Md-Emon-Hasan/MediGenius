# agents/executor_agent.py
from tools.llm_client import get_llm
from core.state import AgentState

class ExecutorAgent:
    llm = get_llm()
    
    @classmethod
    def process(cls, state: AgentState) -> AgentState:
        context = state.get("conversation_history", [])
        question = state["question"]

        # Use docs if available
        if state.get("documents") and len(state["documents"]) > 0:
            content = "\n".join([doc.page_content for doc in state["documents"]])
            prompt = f"""You are a kind, highly experienced professional medical doctor speaking directly with a patient. Be clear, supportive and concise like human response.

Conversation Context:
{"".join(context[-6:])}

Patient's Question:
{question}

Relevant Medical Information:
{content}

Guidelines:
- Answer in 2-3 sentences.
- Do not mention sources.
- Speak like a caring human doctor."""

            response = cls.llm.invoke(prompt)
            answer = response.content.strip()
            state["generation"] = answer
            state["source"] = "retrieved_docs"
            state["conversation_history"].append(f"Doctor: {answer}")
            return state

        # If no docs but LLM succeeded earlier, use that generation
        if state.get("llm_success", False) and state.get("generation"):
            state["conversation_history"].append(f"Doctor: {state['generation']}")
            state["source"] = "llm_knowledge"
            return state

        # Otherwise fallback response
        state["generation"] = "I couldn't find enough information to answer your question right now. Please consult a licensed medical professional."
        state["source"] = "none"
        state["conversation_history"].append(state["generation"])
        return state