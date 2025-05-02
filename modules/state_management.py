from typing import TypedDict, List, Optional
from langchain.schema import Document
import logging

# Set up logging for debugging state changes
logger = logging.getLogger("medical_chatbot")

class AgentState(TypedDict):
    """
    Represents the full state of the AI agent during a conversation cycle.
    Tracks question, memory, fallback attempts, and sources.
    """
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
    """
    Manages the conversation and fallback state across the agent's lifecycle.
    Ensures proper tracking of what strategies have been attempted and logs transitions.
    """

    def __init__(self):
        """
        Initializes the conversation state when the agent starts or resets.
        """
        self.state = self._initialize_state()

    def _initialize_state(self) -> AgentState:
        """
        Sets default initial values for a new conversation state.

        Returns:
            AgentState: Initial state dictionary with all fields reset.
        """
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
        """
        Updates the internal agent state with new information.

        Args:
            new_state (dict): Dictionary containing state keys and updated values.

        Returns:
            AgentState: Updated agent state.
        """
        logger.debug(f"Updating state with keys: {list(new_state.keys())}")
        self.state.update(new_state)
        return self.state

    def reset_conversation(self):
        """
        Clears only the conversation history (not the full state) for continued interactions.
        Useful when the user starts a new question but fallback tracking should persist.
        """
        logger.info("Resetting conversation history")
        self.state["conversation_history"] = []

    def get_state(self) -> AgentState:
        """
        Returns the current internal state of the agent.

        Returns:
            AgentState: Complete state dictionary.
        """
        return self.state
