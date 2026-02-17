"""Chat service layer for MediGenius"""
from datetime import datetime
from typing import Any, Dict

from core.langgraph_workflow import create_workflow
from core.logging_config import logger
from core.state import initialize_conversation_state, reset_query_state
from services.database_service import db_service


class ChatService:
    """Service class for chat operations"""

    def __init__(self):
        """Initialize chat service"""
        self.workflow_app = None
        self.conversation_states: Dict[str, Dict] = {}
        logger.info("ChatService initialized")

    def initialize_workflow(self):
        """Initialize the LangGraph workflow"""
        if not self.workflow_app:
            logger.info("Initializing LangGraph workflow...")
            self.workflow_app = create_workflow()
            logger.info("LangGraph workflow initialized successfully")

    async def process_message(
        self,
        session_id: str,
        message: str
    ) -> Dict[str, Any]:
        """Process a chat message"""
        logger.info(f"Processing message for session {session_id[:8]}...")

        if not self.workflow_app:
            logger.error("Workflow not initialized")
            raise ValueError("Workflow not initialized")

        # Save user message
        db_service.save_message(session_id, 'user', message)
        logger.debug(f"User message saved for session {session_id[:8]}")

        # Initialize conversation state if needed
        if session_id not in self.conversation_states:
            self.conversation_states[session_id] = initialize_conversation_state()
            logger.debug(f"New conversation state created for session {session_id[:8]}")

        conversation_state = self.conversation_states[session_id]
        conversation_state = reset_query_state(conversation_state)
        conversation_state["question"] = message

        # Process query
        try:
            result = await self.workflow_app.ainvoke(conversation_state)
            logger.info(f"Message processed successfully for session {session_id[:8]}")
        except AttributeError:
            # Fallback to sync invoke
            logger.warning("Falling back to sync invoke")
            result = self.workflow_app.invoke(conversation_state)

        self.conversation_states[session_id].update(result)

        # Prepare response
        timestamp = datetime.now().strftime("%I:%M %p")
        response_text = result.get('generation', 'Unable to generate response.')
        source = result.get('source', 'Unknown')

        # Save assistant response
        db_service.save_message(session_id, 'assistant', response_text, source)
        logger.debug(f"Assistant response saved for session {session_id[:8]}")

        return {
            "response": response_text,
            "source": source,
            "timestamp": timestamp,
            "success": bool(result.get('generation'))
        }

    def clear_conversation(self, session_id: str):
        """Clear conversation state for a session"""
        if session_id in self.conversation_states:
            self.conversation_states[session_id] = initialize_conversation_state()
            logger.info(f"Conversation cleared for session {session_id[:8]}")


# Global chat service instance
chat_service = ChatService()
