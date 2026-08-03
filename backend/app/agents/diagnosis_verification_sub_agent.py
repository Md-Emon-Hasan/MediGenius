"""
MediGenius — agents/diagnosis_verification_sub_agent.py
DiagnosisVerificationSubAgent: one structured LLM call, gated by cheap deterministic
pre-checks, to catch clinical claims the retrieved evidence doesn't actually support.
"""

import json

from app.core import dosage_grounding
from app.core.logging_config import logger
from app.core.state import AgentState
from app.tools import model_gateway

# starting points, not validated against outcome data yet — tune from logged verdicts, see Phase 5 report
RISK_HOLD_THRESHOLD = "high"

HELD_BACK_ANSWER = (
    "I don't have enough verified information to answer this safely. Please check with a "
    "clinician or pharmacist directly rather than relying on this answer."
)

SKIPPED_VERDICT = {"grounded": False, "citations_valid": False, "unsupported_claims": [], "risk": "high", "needs_revision": False}


def _build_verdict_prompt(answer: str, evidence_text: str, figures_flagged: bool) -> str:
    hint = " Note: this answer contains at least one number not found in the evidence." if figures_flagged else ""
    return (
        "You are checking a medical answer against its source evidence, not writing one.\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Answer to check:\n{answer}\n\n"
        f"Reply with strict JSON only, no prose, in this exact shape:{hint}\n"
        '{"grounded": true|false, "citations_valid": true|false, '
        '"unsupported_claims": ["..."], "risk": "low"|"medium"|"high", "needs_revision": true|false}\n'
        "Mark a claim unsupported if the evidence doesn't state it. risk=high if an unsupported claim could cause harm."
    )


def _build_revision_prompt(answer: str, evidence_text: str, unsupported: list) -> str:
    return (
        "Rewrite this medical answer using only what the evidence below actually supports. "
        "Remove or soften any claim not backed by the evidence. Keep it to 2-4 sentences.\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Original answer:\n{answer}\n\n"
        f"Claims to remove or soften: {'; '.join(unsupported) if unsupported else 'none listed'}"
    )


def _parse_verdict(raw: str) -> dict:
    try:
        data = json.loads(raw)
        risk = data.get("risk")
        return {
            "grounded": bool(data.get("grounded", False)),
            "citations_valid": bool(data.get("citations_valid", False)),
            "unsupported_claims": list(data.get("unsupported_claims", []) or []),
            "risk": risk if risk in ("low", "medium", "high") else "high",
            "needs_revision": bool(data.get("needs_revision", False)),
        }
    except (json.JSONDecodeError, TypeError, AttributeError, ValueError):
        return dict(SKIPPED_VERDICT)


def DiagnosisVerificationSubAgent(state: AgentState) -> AgentState:
    """Verify the executor's answer against its evidence before it reaches the patient."""
    answer = state.get("generation", "")
    source = state.get("source", "")

    # static safety-router / system-fallback text was never generated from evidence — nothing to verify
    if not answer or source in ("Safety Router", "System Message"):
        state["verification"] = None
        return state

    evidence = state.get("documents") or []
    if not evidence:
        state["verification"] = dict(SKIPPED_VERDICT)
        logger.info("verification: no evidence available, skipped the LLM call")
        return state

    if not model_gateway.is_available():
        verdict = dict(SKIPPED_VERDICT)
        verdict["risk"] = "medium"
        state["verification"] = verdict
        return state

    evidence_text = "\n\n".join(d.page_content[:800] for d in evidence[:3])
    _, figures_removed = dosage_grounding.ground_answer(answer, [d.page_content for d in evidence])

    result = model_gateway.generate(
        _build_verdict_prompt(answer, evidence_text, bool(figures_removed)), tier="reasoning", max_tokens=400
    )
    verdict = _parse_verdict(result["content"] or "")

    if verdict["needs_revision"] and verdict["risk"] != RISK_HOLD_THRESHOLD:
        revision = model_gateway.generate(
            _build_revision_prompt(answer, evidence_text, verdict["unsupported_claims"]), tier="reasoning", max_tokens=300
        )
        if revision["content"]:
            answer = revision["content"]
            logger.info("verification: answer revised once, no re-verification after")

    if verdict["risk"] == RISK_HOLD_THRESHOLD:
        answer = HELD_BACK_ANSWER
        state["source"] = "Safety Router"
        logger.warning("verification: high-risk answer held back")

    state["generation"] = answer
    state["verification"] = verdict
    return state
