"""
MediGenius — agents/drug_interaction_sub_agent.py
DrugInteractionSubAgent: normalizes mentioned substance names against RxNav (NIH RxNorm,
free, no key) and always refers to a pharmacist. NIH discontinued the RxNav drug-drug
interaction API on 2024-01-02 — it is not coming back — so no code path here ever states
whether an interaction exists. Pure lookup and refusal, no LLM involved.
"""

import re

import httpx

from app.core import dosage_grounding
from app.core.logging_config import logger
from app.core.state import AgentState

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
RXNAV_TIMEOUT = 3.0

_STOPWORDS = {
    "can", "i", "take", "with", "and", "or", "the", "a", "an", "is", "it", "safe", "to", "does",
    "have", "interact", "interaction", "interactions", "between", "my", "for", "of", "drug", "drugs",
    "medication", "medicine", "combine", "mix", "together", "this", "that", "are", "will",
}


def _candidate_terms(question: str) -> list:
    words = re.findall(r"[a-zA-Z]{4,}", (question or "").lower())
    return [w for w in dict.fromkeys(words) if w not in _STOPWORDS]


def _resolve_drug_name(term: str):
    try:
        resp = httpx.get(f"{RXNAV_BASE}/rxcui.json", params={"name": term}, timeout=RXNAV_TIMEOUT)
        resp.raise_for_status()
        if resp.json().get("idGroup", {}).get("rxnormId"):
            return term
    except Exception as e:
        logger.warning("drug_interaction: RxNav lookup failed for a candidate term: %s", str(e))
    return None


def DrugInteractionSubAgent(state: AgentState) -> AgentState:
    """Confirm which mentioned terms are real drugs, then always refer — never assess the interaction itself."""
    question = state.get("question", "")
    recognized = [d for d in (_resolve_drug_name(t) for t in _candidate_terms(question)[:6]) if d]

    if recognized:
        names = " and ".join(recognized[:2])
        answer = (
            f"I recognize {names} as medications, but I can't safely tell you whether they interact — "
            "that needs a pharmacist or your doctor's direct input. Please check with one before "
            "combining them."
        )
    else:
        answer = dosage_grounding.refusal_response("drug_interaction")

    state["generation"] = answer
    state["source"] = "Safety Router"
    return state
