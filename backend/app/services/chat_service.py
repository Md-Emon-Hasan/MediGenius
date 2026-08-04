"""
MediGenius — services/chat_service.py
ChatService: orchestrates the LangGraph agentic workflow for each chat message.
"""

import time
from datetime import datetime
from typing import Any, Dict

from app.agents.drug_interaction_sub_agent import DrugInteractionSubAgent
from app.core import cache, dosage_grounding, safety_router
from app.core.langgraph_workflow import create_workflow
from app.core.logging_config import logger
from app.core.state import initialize_conversation_state, reset_query_state
from app.services.database_service import db_service
from app.tools import memory_store

# node name -> human-readable label, shown to the user while that node is executing
STAGE_LABELS = {
    "safety_router": "Checking safety filters",
    "memory": "Recalling conversation context",
    "supervisor": "Understanding your question",
    "symptom_analysis": "Analyzing reported symptoms",
    "planner": "Deciding how to answer",
    "parallel_retrieval": "Searching medical knowledge base and web",
    "llm_agent": "Consulting medical knowledge",
    "executor": "Generating response",
    "diagnosis_verification": "Verifying answer accuracy",
}


def _stage_event(key: str) -> Dict[str, str]:
    return {"type": "stage", "stage": key, "label": STAGE_LABELS.get(key, key)}


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

    def _rehydrate_history(self, session_id: str) -> list:
        """Reload persisted messages so a session survives a process restart."""
        try:
            history = db_service.get_chat_history(session_id)
        except Exception:
            logger.error("Failed to load persisted history for session %s", session_id[:8], exc_info=True)
            return []
        return [{"role": m["role"], "content": m["content"], "source": m.get("source")} for m in history[-20:]]

    def _try_quick_response(self, session_id: str, message: str, started: float):
        """Handle safety block / refusal / cache hit without touching the graph.

        Returns (payload, safety) if answered here, or (None, safety) if the full
        pipeline still needs to run.
        """
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
                "safety": {
                    "blocked": True,
                    "category": safety["category"],
                    "refused_topic": None,
                    "figures_removed": [],
                },
                "verification": None,
                "symptom_summary": None,
            }, safety

        refused_topic = dosage_grounding.check_refusal(message)
        if refused_topic:
            if refused_topic == "drug_interaction":
                # RxNav's interaction API is gone (NIH discontinued it 2024-01-02) — normalize names, still refer
                refusal_text = DrugInteractionSubAgent({"question": message})["generation"]
            else:
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
                "symptom_summary": None,
            }, safety

        cached = cache.get_answer(message)
        if cached is not None:
            response_text, source, figures_removed, model_used, model_fallback, verification = cached
            db_service.save_message(session_id, "assistant", response_text, source)
            self._record_audit(
                session_id, source=source, figures_removed_count=len(figures_removed), model_used=model_used,
                model_fallback=model_fallback, verification_risk=(verification or {}).get("risk"),
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
                "symptom_summary": None,
            }, safety

        return None, safety

    def _init_query_state(self, session_id: str, message: str) -> Dict:
        """Initialize or retrieve conversation state — rehydrate from SQLite on a cold start."""
        if session_id not in self.conversation_states:
            state = initialize_conversation_state()
            state["conversation_history"] = self._rehydrate_history(session_id)
            self.conversation_states[session_id] = state

        state = self.conversation_states[session_id]
        state = reset_query_state(state)
        state["question"] = message
        state["session_id"] = session_id
        return state

    def _finalize_graph_result(
        self, session_id: str, message: str, started: float, result: Dict, safety: Dict
    ) -> Dict[str, Any]:
        """Post-process a completed graph run into the response payload (grounding, caching, persistence, audit)."""
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

        if not degraded:
            memory_store.add_exchange(session_id, message, response_text)

        # Persist assistant response
        db_service.save_message(session_id, "assistant", response_text, source)
        self._record_audit(
            session_id, source=source, figures_removed_count=len(figures_removed), model_used=model_used,
            model_fallback=model_fallback, verification_risk=(verification or {}).get("risk"),
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
            "symptom_summary": result.get("symptom_summary"),
        }

    async def process_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """Run the agentic pipeline for a single user message."""
        logger.info("Processing message for session %s...", session_id[:8])

        if not self.workflow_app:
            raise ValueError("Workflow not initialized")

        started = time.monotonic()
        db_service.save_message(session_id, "user", message)

        payload, safety = self._try_quick_response(session_id, message, started)
        if payload is not None:
            return payload

        state = self._init_query_state(session_id, message)

        # Run workflow (async preferred, sync fallback)
        try:
            result = await self.workflow_app.ainvoke(state)
        except AttributeError:
            logger.warning("Falling back to sync invoke")
            result = self.workflow_app.invoke(state)

        return self._finalize_graph_result(session_id, message, started, result, safety)

    async def process_message_stream(self, session_id: str, message: str):
        """Same pipeline as process_message, but yields a stage event per executed node.

        Yields {"type": "stage", "stage": ..., "label": ...} as each node completes, then
        exactly one {"type": "final", "payload": {...}} with the same payload process_message
        would have returned.
        """
        logger.info("Processing message for session %s (streaming)...", session_id[:8])

        if not self.workflow_app:
            raise ValueError("Workflow not initialized")

        started = time.monotonic()
        db_service.save_message(session_id, "user", message)

        yield _stage_event("safety_router")
        payload, safety = self._try_quick_response(session_id, message, started)
        if payload is not None:
            yield {"type": "final", "payload": payload}
            return

        state = self._init_query_state(session_id, message)

        merged = dict(state)
        try:
            async for update in self.workflow_app.astream(state, stream_mode="updates"):
                for node_name, delta in update.items():
                    merged.update(delta)
                    yield _stage_event(node_name)
        except (AttributeError, TypeError):
            logger.warning("Falling back to sync invoke (no astream support)")
            merged.update(self.workflow_app.invoke(state))
            yield _stage_event("executor")

        payload = self._finalize_graph_result(session_id, message, started, merged, safety)
        yield {"type": "final", "payload": payload}

    def clear_conversation(self, session_id: str) -> None:
        """Reset the in-memory conversation state for a session."""
        if session_id in self.conversation_states:
            self.conversation_states[session_id] = initialize_conversation_state()
            logger.info("Conversation cleared for session %s", session_id[:8])


# Module-level singleton
chat_service = ChatService()
