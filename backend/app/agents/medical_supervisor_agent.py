"""
MediGenius — agents/medical_supervisor_agent.py
MedicalSupervisorAgent: decides which optional sub-agents a query needs before the
main retrieval/answer chain runs. Deterministic — a definitional question triggers none.
"""

import re

from app.core.state import AgentState

FIRST_PERSON_PATTERNS = [
    r"\bi (have|feel|am feeling|am experiencing|have been experiencing)\b",
    r"\bi'?ve (been|had)\b",
    r"\bi'?m (having|feeling)\b",
    r"\bmy (head|stomach|chest|back|throat|skin|leg|arm|joint|body)\b",
]
SYMPTOM_KEYWORDS = [
    "fever", "pain", "headache", "nausea", "vomiting", "diarrhea", "cough",
    "rash", "itch", "shortness of breath", "chest pain", "abdominal pain",
    "back pain", "joint pain", "muscle pain", "fatigue", "weakness",
    "dizziness", "numbness", "tingling", "swelling", "bleeding", "bruising",
]

_FIRST_PERSON_RE = re.compile("|".join(FIRST_PERSON_PATTERNS), re.IGNORECASE)


def _looks_like_reported_symptoms(question: str) -> bool:
    q = (question or "").lower()
    has_symptom_word = any(kw in q for kw in SYMPTOM_KEYWORDS)
    return has_symptom_word and bool(_FIRST_PERSON_RE.search(q))


def MedicalSupervisorAgent(state: AgentState) -> AgentState:
    """Flag whether this query describes reported symptoms (vs. a definitional question)."""
    state["needs_symptom_analysis"] = _looks_like_reported_symptoms(state.get("question", ""))
    return state
