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

SKIPPED_VERDICT = {
    "grounded": False,
    "citations_valid": False,
    "unsupported_claims": [],
    "risk": "high",
    "needs_revision": False,
}


def _build_verdict_prompt(answer: str, evidence_text: str, figures_flagged: bool) -> str:
    hint = " Note: this answer contains at least one number not found in the evidence." if figures_flagged else ""
    return (
        "You are checking a medical answer against its source evidence, not writing one.\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Answer to check:\n{answer}\n\n"
        f"Reply with strict JSON only, no prose, in this exact shape:{hint}\n"
        '{"grounded": true|false, "citations_valid": true|false, '
        '"unsupported_claims": ["..."], "risk": "low"|"medium"|"high", "needs_revision": true|false}\n\n'
        "A general restatement of the evidence in different words, or reasonable paraphrasing, is grounded — "
        "do not flag it just because the wording differs. Only list a claim as unsupported if it states something "
        "the evidence does not say at all (a specific dosage, a diagnosis, a safety claim, a fact not in evidence). "
        "risk=low for ordinary informational content consistent with the evidence. risk=medium for a minor, low-harm "
        "unsupported detail. risk=high only for a specific unsupported claim that could cause direct physical harm "
        "if wrong (dosing, a false safety reassurance, a diagnostic claim). Default to low when in doubt."
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
    evidence = state.get("documents") or []

    # only a document-grounded answer makes a retrieval claim worth checking — static safety/system
    # text and direct-LLM-knowledge answers (source "AI Medical Knowledge") never claimed RAG grounding
    if not answer or source in ("Safety Router", "System Message") or not evidence:
        state["verification"] = None
        return state

    if not model_gateway.is_available():
        verdict = dict(SKIPPED_VERDICT)
        verdict["risk"] = "medium"
        state["verification"] = verdict
        return state

    # reuse the exact text the executor was given — reconstructing a different slice here would
    # flag claims as "unsupported" purely because the verifier saw a different truncation than the writer did
    evidence_text = state.get("evidence_text") or "\n\n".join(d.page_content[:1000] for d in evidence[:3])
    _, figures_removed = dosage_grounding.ground_answer(answer, [d.page_content for d in evidence])

    result = model_gateway.generate(
        _build_verdict_prompt(answer, evidence_text, bool(figures_removed)),
        tier="reasoning", max_tokens=900, reasoning_effort="low",
    )
    verdict = _parse_verdict(result["content"] or "")

    if verdict["needs_revision"] and verdict["risk"] != RISK_HOLD_THRESHOLD:
        revision = model_gateway.generate(
            _build_revision_prompt(answer, evidence_text, verdict["unsupported_claims"]),
            tier="reasoning", max_tokens=700, reasoning_effort="low",
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
