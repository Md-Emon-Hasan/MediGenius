# agents/retriever_agent.py
from tools.vector_store import get_retriever
from core.state import AgentState

class RetrieverAgent:
    retriever = get_retriever()
    
    @classmethod
    def process(cls, state: AgentState) -> AgentState:
        query = state["question"]
        context = "\n".join(state.get("conversation_history", [])[-6:])
        combined_query = f"Context: {context}\nQuestion: {query}"

        try:
            docs = cls.retriever.invoke(combined_query)
            if docs and len(docs) > 0:
                state["documents"] = docs
                state["rag_success"] = True
                state["conversation_history"].append("AI: Retrieved documents from medical PDF database.")
            else:
                state["documents"] = []
                state["rag_success"] = False
        except Exception:
            state["documents"] = []
            state["rag_success"] = False

        state["rag_attempted"] = True
        return state