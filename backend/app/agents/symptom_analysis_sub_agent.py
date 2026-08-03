"""
MediGenius — agents/symptom_analysis_sub_agent.py
SymptomAnalysisSubAgent: structures reported symptoms and asks clarifying questions.
Produces a summary and a referral recommendation — never a diagnosis or a
confidence-ranked differential.
"""

import json

from app.core.logging_config import logger
from app.core.state import AgentState
from app.tools import model_gateway

NOT_A_DIAGNOSIS_NOTE = "This is a summary of what you've described, not a diagnosis."


def _build_prompt(question: str, history_context: str) -> str:
    return (
        "A patient described symptoms below. Do not diagnose or rank possible conditions. "
        "Just structure what they said and ask up to 2 short clarifying questions a nurse "
        "doing intake would ask (duration, severity, what makes it better or worse).\n\n"
        f"Recent conversation:\n{history_context}\n"
        f"Patient's message:\n{question}\n\n"
        "Reply with strict JSON only, no prose, in this exact shape:\n"
        '{"reported_symptoms": ["..."], "clarifying_questions": ["..."], "referral": "..."}\n'
        'referral should be a short recommendation, e.g. "see a doctor if this persists beyond 3 days".'
    )


def _parse(raw: str):
    try:
        data = json.loads(raw)
        return {
            "reported_symptoms": list(data.get("reported_symptoms", []) or []),
            "clarifying_questions": list(data.get("clarifying_questions", []) or [])[:2],
            "referral": data.get("referral") or "Please see a clinician if symptoms persist or worsen.",
            "note": NOT_A_DIAGNOSIS_NOTE,
        }
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None


def format_symptom_context(state: AgentState) -> str:
    """Prompt-ready block for downstream agents — empty string if no summary exists."""
    summary = state.get("symptom_summary")
    if not summary:
        return ""
    symptoms = ", ".join(summary["reported_symptoms"]) or "none extracted"
    return f"\nStructured symptom summary (not a diagnosis): {symptoms}\n"


def SymptomAnalysisSubAgent(state: AgentState) -> AgentState:
    """Structure reported symptoms into a summary, clarifying questions, and a referral note."""
    if not model_gateway.is_available():
        state["symptom_summary"] = None
        return state

    history_context = " ".join(
        item.get("content", "") for item in state.get("conversation_history", [])[-3:]
        if item.get("role") == "user"
    )
    result = model_gateway.generate(
        _build_prompt(state["question"], history_context), tier="reasoning", max_tokens=800, reasoning_effort="low",
    )
    summary = _parse(result["content"]) if result["content"] else None
    if summary is None:
        logger.warning("symptom_analysis: could not produce a structured summary")
    state["symptom_summary"] = summary
    return state
