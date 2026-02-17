from core.state import AgentState
from tools.vector_store import get_retriever


def RetrieverAgent(state: AgentState) -> AgentState:
    query = state["question"]

    # Get retriever
    retriever = get_retriever()

    if not retriever:
        print("RAG: No retriever available - vector database not initialized")
        state["documents"] = []
        state["rag_success"] = False
        state["rag_attempted"] = True
        return state

    # Create context from conversation history
    context_parts = []
    for item in state.get("conversation_history", [])[-3:]:
        if item.get('role') == 'user':
            context_parts.append(f"Context: {item.get('content', '')}")

    context = " | ".join(context_parts)
    combined_query = f"{query} {context}" if context else query

    # Retrieve documents
    docs = retriever.invoke(combined_query)

    if docs and len(docs) > 0:
        valid_docs = [doc for doc in docs if len(doc.page_content.strip()) > 50]
        if valid_docs:
            state["documents"] = valid_docs
            state["rag_success"] = True
            state["source"] = "Medical Literature Database"
            print(f"RAG: Found {len(valid_docs)} relevant documents")
        else:
            state["documents"] = []
            state["rag_success"] = False
            print("RAG: No valid documents found")
    else:
        state["documents"] = []
        state["rag_success"] = False
        print("RAG: No documents retrieved")

    state["rag_attempted"] = True
    return state
