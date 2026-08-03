"""
MediGenius — services/chat_service.py
ChatService: orchestrates the LangGraph agentic workflow for each chat message.
"""

import time
from datetime import datetime
from typing import Any, Dict

from app.core import cache, dosage_grounding, safety_router
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

    def _record_audit(self, session_id: str, **fields) -> None:
        """Audit logging must never fail the request it's describing."""
        try:
            db_service.save_audit_log(session_id, **fields)
        except Exception:
            logger.error("Audit log write failed", exc_info=True)

    async def process_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """Run the agentic pipeline for a single user message."""
        logger.info("Processing message for session %s...", session_id[:8])

        if not self.workflow_app:
            raise ValueError("Workflow not initialized")

        started = time.monotonic()
        db_service.save_message(session_id, "user", message)

        safety = safety_router.evaluate(message)
        if safety["blocked"]:
            db_service.save_message(session_id, "assistant", safety["response"], "Safety Router")
            self._record_audit(
                session_id, safety_category=safety["category"], source="Safety Router",
                latency_ms=(time.monotonic() - started) * 1000,
            )
            return {
                "response": safety["response"],
                "source": "Safety Router",
                "timestamp": datetime.now().strftime("%I:%M %p"),
                "success": True,
                "disclaimer": None,
                "safety": {"blocked": True, "category": safety["category"], "refused_topic": None, "figures_removed": []},
                "verification": None,
            }

        refused_topic = dosage_grounding.check_refusal(message)
        if refused_topic:
            refusal_text = dosage_grounding.refusal_response(refused_topic)
            db_service.save_message(session_id, "assistant", refusal_text, "Safety Router")
            self._record_audit(
                session_id, refused_topic=refused_topic, source="Safety Router",
                latency_ms=(time.monotonic() - started) * 1000,
            )
            return {
                "response": refusal_text,
                "source": "Safety Router",
                "timestamp": datetime.now().strftime("%I:%M %p"),
                "success": True,
                "disclaimer": safety["disclaimer"],
                "safety": {"blocked": False, "category": None, "refused_topic": refused_topic, "figures_removed": []},
                "verification": None,
            }

        cached = cache.get_answer(message)
        if cached is not None:
            response_text, source, figures_removed, model_used, model_fallback, verification = cached
            db_service.save_message(session_id, "assistant", response_text, source)
            self._record_audit(
                session_id, source=source, figures_removed_count=len(figures_removed), model_used=model_used,
                cache_hit=True, latency_ms=(time.monotonic() - started) * 1000,
            )
            return {
                "response": response_text,
                "source": source,
                "timestamp": datetime.now().strftime("%I:%M %p"),
                "success": True,
                "disclaimer": safety["disclaimer"],
                "safety": {
                    "blocked": False, "category": None, "refused_topic": None, "figures_removed": figures_removed,
                    "model_used": model_used, "model_fallback": model_fallback,
                },
                "verification": verification,
            }

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

        source_texts = [doc.page_content for doc in result.get("documents", []) if getattr(doc, "page_content", None)]
        response_text, figures_removed = dosage_grounding.ground_answer(response_text, source_texts)
        if figures_removed:
            logger.warning("dosage_grounding: stripped %d ungrounded figure(s)", len(figures_removed))

        model_used = result.get("model_used")
        model_fallback = bool(result.get("model_fallback"))
        verification = result.get("verification")

        cache.set_answer(message, (response_text, source, figures_removed, model_used, model_fallback, verification))

        # "System Message"/"Safety Router" here means the real answer failed or was held back post-verification
        degraded = source in ("System Message", "Safety Router") or not result.get("generation")
        if model_fallback:
            logger.warning("model_gateway: answer served by fallback model %s", model_used)

        # Persist assistant response
        db_service.save_message(session_id, "assistant", response_text, source)
        self._record_audit(
            session_id, source=source, figures_removed_count=len(figures_removed), model_used=model_used,
            degraded=degraded, latency_ms=(time.monotonic() - started) * 1000,
        )

        return {
            "response": response_text,
            "source": source,
            "timestamp": datetime.now().strftime("%I:%M %p"),
            "success": bool(result.get("generation")),
            "disclaimer": safety["disclaimer"],
            "safety": {
                "blocked": False, "category": None, "refused_topic": None, "figures_removed": figures_removed,
                "model_used": model_used, "model_fallback": model_fallback,
            },
            "verification": verification,
        }

    def clear_conversation(self, session_id: str) -> None:
        """Reset the in-memory conversation state for a session."""
        if session_id in self.conversation_states:
            self.conversation_states[session_id] = initialize_conversation_state()
            logger.info("Conversation cleared for session %s", session_id[:8])


# Module-level singleton
chat_service = ChatService()
