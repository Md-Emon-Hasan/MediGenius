from langgraph.graph import StateGraph, END
from typing import Dict, Any
import logging
from modules.state_management import AgentState

logger = logging.getLogger("medical_chatbot")

class MedicalWorkflow:
    def __init__(self, llm_service, retrieval_system):
        self.llm = llm_service
        self.retriever = retrieval_system
        self.workflow = StateGraph(AgentState)
        self._build_workflow()

    def _build_workflow(self):
        # Add nodes
        self.workflow.add_node("llm_query", self.query_llm)
        self.workflow.add_node("rag_query", self._retrieve_docs)
        self.workflow.add_node("wiki_query", self._retrieve_wiki)
        self.workflow.add_node("ddg_query", self._retrieve_duckduckgo)
        self.workflow.add_node("generate", self.generate_response)

        # Set entry point
        self.workflow.set_entry_point("llm_query")

        # Add conditional edges
        self.workflow.add_conditional_edges(
            "llm_query",
            self._route_after_llm,
            {"generate": "generate", "rag_query": "rag_query"}
        )

        self.workflow.add_conditional_edges(
            "rag_query",
            self._route_after_rag,
            {"generate": "generate", "wiki_query": "wiki_query"}
        )

        self.workflow.add_conditional_edges(
            "wiki_query",
            self._route_after_wiki,
            {"generate": "generate", "ddg_query": "ddg_query"}
        )

        self.workflow.add_edge("ddg_query", "generate")
        self.workflow.add_edge("generate", END)

    def query_llm(self, state: AgentState):
        ctx = "\n".join(state.get("conversation_history", []))
        prompt = f"""You are a kind, highly experienced professional medical doctor speaking directly with a patient. Imagine you are sitting in front of them — caring, respectful, and thoughtful in your words.

Patient's History:
{ctx}

Patient's Question:
{state['question']}

Respond like an experienced doctor in 1–2 sentences. Be clear, professional, and confident. Do not mention sources, uncertainty, or say 'I don't know'. Your conversational skill should be a professional consultant with a human touch."""

        logger.info("Attempting LLM response")
        res = self.llm.generate_response(prompt)
        
        convo = state["conversation_history"] + [
            f"User: {state['question']}",
            f"Doctor: {res.strip()}"
        ]
        
        return {
            "generation": res.strip(),
            "source": "llm_knowledge",
            "llm_attempted": True,
            "conversation_history": convo
        }

    def _retrieve_docs(self, state: AgentState):
        """Second attempt with RAG"""
        context = "\n".join(state.get("conversation_history", []))
        query = f"{state['question']}"

        if context:
            query = f"Context: {context}\nQuestion: {query}"

        logger.info("Attempting document retrieval")
        docs = self.retriever.retrieve_documents(query)
        return {
            "documents": docs,
            "rag_attempted": True,
            "search_query": query,
            "source": "documents",
            "conversation_history": state["conversation_history"] + [
                "AI: Searching medical documents..."
            ]
        }

    def _retrieve_wiki(self, state: AgentState):
        logger.info("Attempting Wikipedia retrieval")
        doc = self.retriever.retrieve_wikipedia(state["question"])
        return {
            "documents": [doc],
            "source": "wikipedia",
            "conversation_history": state["conversation_history"] + [
                "AI: Searching Wikipedia..."
            ],
            "wiki_attempted": True
        }

    def _retrieve_duckduckgo(self, state: AgentState):
        """Final fallback with DuckDuckGo"""
        logger.info("Attempting web search")
        doc = self.retriever.retrieve_web(state["question"])
        return {
            "documents": [doc],
            "source": "duckduckgo",
            "conversation_history": state["conversation_history"] + [
                "AI: Searching DuckDuckGo..."
            ],
            "ddg_attempted": True
        }

    def _route_after_llm(self, state):
        if state.get("generation"):
            return "generate"
        return "rag_query"

    def _route_after_rag(self, state):
        if state.get("documents"):
            return "generate"
        return "wiki_query"

    def _route_after_wiki(self, state):
        if state.get("documents"):
            return "generate"
        return "ddg_query"

    def generate_response(self, state: AgentState):
        """Generate final response based on available information"""
        context = state.get("conversation_history", [])

        if state.get("source") == "llm_knowledge":
            return state  # Already has LLM response

        if state.get("documents"):
            content = "\n".join([doc.page_content for doc in state["documents"]])
            source = state["source"]  # Preserve original source

            prompt = f"""You are a kind, highly experienced professional medical doctor speaking directly with a patient. Imagine you are sitting in front of them — caring, respectful, and thoughtful in your words.

Patient's Conversation Context:
{"".join(context[-3:])}

Patient's Question:
{state['question']}

Relevant Medical Information:
{content}

Guidelines:
1. Respond like a doctor who is calm, confident, and helpful.
2. Give clear, supportive advice in 1–2 sentences.
3. Do not mention any source, search, or say things like 'according to...'
4. Never mention 'Wikipedia', 'DuckDuckGo' or 'retrieved information'—speak like a doctor directly responding.
5. Encourage professional consultation if needed, but don't defer the question.
"""
            logger.info("Generating response from retrieved information")
            response = self.llm.generate_response(prompt)

            return {
                "generation": response.strip(),
                "source": source,
                "conversation_history": context + [f"Doctor: {response.strip()}"]
            }

        return {
            "generation": "I couldn't find enough information to answer your question right now. Please consult a licensed medical professional.",
            "source": "none",
            "conversation_history": context + ["Doctor: I couldn't find enough information to answer your question."]
        }

    def compile(self):
        logger.info("Compiling workflow")
        return self.workflow.compile()