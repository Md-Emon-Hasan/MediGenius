from typing import TypedDict, List, Optional
from langchain.schema import Document
import logging

logger = logging.getLogger("medical_chatbot")

class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    source: str
    search_query: Optional[str]
    conversation_history: List[str]
    llm_attempted: bool
    rag_attempted: bool
    wiki_attempted: bool
    ddg_attempted: bool

class StateManager:
    def __init__(self):
        self.state = self._initialize_state()

    def _initialize_state(self) -> AgentState:
        return {
            "question": "",
            "documents": [],
            "generation": "",
            "source": "",
            "search_query": None,
            "conversation_history": [],
            "llm_attempted": False,
            "rag_attempted": False,
            "wiki_attempted": False,
            "ddg_attempted": False
        }

    def update_state(self, new_state: dict) -> AgentState:
        logger.debug(f"Updating state with: {new_state.keys()}")
        self.state.update(new_state)
        return self.state

    def reset_conversation(self):
        logger.info("Resetting conversation state")
        self.state["conversation_history"] = []

    def get_state(self) -> AgentState:
        return self.state