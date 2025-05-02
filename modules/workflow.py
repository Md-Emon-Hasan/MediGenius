from langgraph.graph import StateGraph, END
from typing import Dict, Any
from enum import Enum
import logging
from modules.state_management import AgentState

logger = logging.getLogger("medical_chatbot")

class Source(Enum):
    LLM = "llm_knowledge"
    RAG = "documents"
    WIKI = "wikipedia"
    DDG = "duckduckgo"
    NONE = "none"

class PromptBuilder:
    @staticmethod
    def doctor_prompt(context, question, content="") -> str:
        return f"""
        You are a kind, experienced doctor...

        Context:
        {context}

        Question:
        {question}

        Medical Info:
        {content}
        """

class MedicalWorkflow:
    def __init__(self, llm_service, retrieval_system):
        self.llm = llm_service
        self.retriever = retrieval_system
        self.workflow = StateGraph(AgentState)
        self._build_workflow()

    def _build_workflow(self):
        self.workflow.add_node("llm_query", self.query_llm)
        self.workflow.add_node("rag_query", self._retrieve_docs)
        self.workflow.add_node("wiki_query", self._retrieve_wiki)
        self.workflow.add_node("ddg_query", self._retrieve_duckduckgo)
        self.workflow.add_node("generate", self.generate_response)

        self.workflow.set_entry_point("llm_query")

        self.workflow.add_conditional_edges(
            "llm_query", self._route_after_llm, {"generate": "generate", "rag_query": "rag_query"}
        )
        self.workflow.add_conditional_edges(
            "rag_query", self._route_after_rag, {"generate": "generate", "wiki_query": "wiki_query"}
        )
        self.workflow.add_conditional_edges(
            "wiki_query", self._route_after_wiki, {"generate": "generate", "ddg_query": "ddg_query"}
        )
        self.workflow.add_edge("ddg_query", "generate")
        self.workflow.add_edge("generate", END)

    def query_llm(self, state: AgentState):
        try:
            ctx = "\n".join(state.get("conversation_history", []))
            prompt = PromptBuilder.doctor_prompt(ctx, state['question'])
            
            logger.info("Attempting LLM response")
            res = self.llm.generate_response(prompt)
            
            convo = state["conversation_history"] + [
                f"User: {state['question']}",
                f"Doctor: {res.strip()}"
            ]
            
            return {
                "generation": res.strip(),
                "source": Source.LLM.value,
                "llm_attempted": True,
                "conversation_history": convo
            }
        except Exception as e:
            logger.exception("Error generating LLM response")
            return {
                "generation": "Sorry, I encountered an error generating the response.",
                "source": Source.LLM.value,
                "llm_attempted": True,
                "conversation_history": state["conversation_history"]
            }

    def _retrieve_docs(self, state: AgentState):
        try:
            context = "\n".join(state.get("conversation_history", []))
            query = f"{state['question']}"

            if context:
                query = f"Context: {context}\nQuestion: {query}"

            logger.info(f"Attempting RAG retrieval for query: {query}")
            docs = self.retriever.retrieve_documents(query)
            return {
                "documents": docs,
                "rag_attempted": True,
                "search_query": query,
                "source": Source.RAG.value,
                "conversation_history": state["conversation_history"] + ["AI: Searching medical documents..."]
            }
        except Exception as e:
            logger.exception("Error retrieving documents with RAG")
            return {
                "documents": [],
                "rag_attempted": True,
                "conversation_history": state["conversation_history"] + ["AI: Document retrieval failed."]
            }

    def _retrieve_wiki(self, state: AgentState):
        try:
            logger.info("Attempting Wikipedia retrieval")
            doc = self.retriever.retrieve_wikipedia(state["question"])
            return {
                "documents": [doc],
                "source": Source.WIKI.value,
                "conversation_history": state["conversation_history"] + ["AI: Searching Wikipedia..."],
                "wiki_attempted": True
            }
        except Exception as e:
            logger.exception("Error retrieving from Wikipedia")
            return {
                "documents": [],
                "source": Source.WIKI.value,
                "conversation_history": state["conversation_history"] + ["AI: Wikipedia retrieval failed."]
            }

    def _retrieve_duckduckgo(self, state: AgentState):
        try:
            logger.info("Attempting DuckDuckGo search")
            doc = self.retriever.retrieve_web(state["question"])
            return {
                "documents": [doc],
                "source": Source.DDG.value,
                "conversation_history": state["conversation_history"] + ["AI: Searching DuckDuckGo..."],
                "ddg_attempted": True
            }
        except Exception as e:
            logger.exception("Error retrieving from DuckDuckGo")
            return {
                "documents": [],
                "source": Source.DDG.value,
                "conversation_history": state["conversation_history"] + ["AI: DuckDuckGo retrieval failed."]
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
        try:
            context = state.get("conversation_history", [])
            
            if state.get("source") == Source.LLM.value:
                return state

            if state.get("documents"):
                content = "\n".join([doc.page_content for doc in state["documents"]])
                prompt = PromptBuilder.doctor_prompt("".join(context[-3:]), state['question'], content)
                logger.info("Generating response from retrieved information")
                response = self.llm.generate_response(prompt)

                return {
                    "generation": response.strip(),
                    "source": state["source"],
                    "conversation_history": context + [f"Doctor: {response.strip()}"]
                }

            return {
                "generation": "I couldn't find enough information to answer your question right now. Please consult a licensed medical professional.",
                "source": Source.NONE.value,
                "conversation_history": context + ["Doctor: I couldn't find enough information to answer your question."]
            }
        except Exception as e:
            logger.exception("Error generating final response")
            return {
                "generation": "An error occurred while processing your request.",
                "source": Source.NONE.value,
                "conversation_history": state["conversation_history"] + ["Doctor: An error occurred."]
            }

    def compile(self):
        logger.info("Compiling workflow")
        return self.workflow.compile()
