"""
MediGenius — services/chat_service.py
ChatService: orchestrates the LangGraph agentic workflow for each chat message.
"""

from datetime import datetime
from typing import Any, Dict

from app.core.langgraph_workflow import create_workflow
from app.core.logging_config import logger
from app.core.state import initialize_conversation_state, reset_query_state
from app.services.database_service import db_service


class ChatService:
    """Orchestrates the agentic workflow for each chat message."""

    def __init__(self):
        self.workflow_app = None
        self.conversation_states: Dict[str, Dict] = {}
        logger.info("ChatService initialized")

    def initialize_workflow(self) -> None:
        """Compile and cache the LangGraph workflow (called once at startup)."""
        if not self.workflow_app:
            logger.info("Initializing LangGraph workflow...")
            self.workflow_app = create_workflow()
            logger.info("LangGraph workflow initialized successfully")

    async def process_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """Run the agentic pipeline for a single user message."""
        logger.info("Processing message for session %s...", session_id[:8])

        if not self.workflow_app:
            raise ValueError("Workflow not initialized")

        # Persist user message
        db_service.save_message(session_id, "user", message)

        # Initialize or retrieve conversation state
        if session_id not in self.conversation_states:
            self.conversation_states[session_id] = initialize_conversation_state()

        state = self.conversation_states[session_id]
        state = reset_query_state(state)
        state["question"] = message

        # Run workflow (async preferred, sync fallback)
        try:
            result = await self.workflow_app.ainvoke(state)
        except AttributeError:
            logger.warning("Falling back to sync invoke")
            result = self.workflow_app.invoke(state)

        self.conversation_states[session_id].update(result)

        response_text = result.get("generation", "Unable to generate response.")
        source = result.get("source", "Unknown")

        # Persist assistant response
        db_service.save_message(session_id, "assistant", response_text, source)

        return {
            "response": response_text,
            "source": source,
            "timestamp": datetime.now().strftime("%I:%M %p"),
            "success": bool(result.get("generation")),
        }

    def clear_conversation(self, session_id: str) -> None:
        """Reset the in-memory conversation state for a session."""
        if session_id in self.conversation_states:
            self.conversation_states[session_id] = initialize_conversation_state()
            logger.info("Conversation cleared for session %s", session_id[:8])


# Module-level singleton
chat_service = ChatService()
