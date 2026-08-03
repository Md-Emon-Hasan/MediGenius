"""
MediGenius — core/dosage_grounding.py
Deterministic, regex-only checks: strip ungrounded dosage figures from a generated
answer, and hard-refuse topics where a plausible-sounding wrong answer is most harmful.
No LLM involved — string matching against the retrieved sources only.
"""

import re
from typing import List, Optional, Tuple

REFERRAL_PLACEHOLDER = "[exact dosing removed — please confirm with a clinician or pharmacist]"

# each alternative is non-overlapping by construction (single compiled regex, left-to-right scan)
_DOSAGE_PATTERNS = [
    r"\d+(\.\d+)?\s?(mg/kg|mg/ml|mcg/kg|%)\b",
    r"\d+(\.\d+)?\s?(mg|mcg|g|ml|iu|units?)\b",
    r"(once|twice|three times|\d+\s*times)\s*(a|per)?\s*(day|daily|week)\b",
    r"every\s+\d+\s*(hours?|hrs?|days?)",
    r"for\s+\d+\s*(days?|weeks?|months?)\b",
    r"\d+\s*(years?|yrs?|months?)\s*(old|of age)?\b",
    r"\d+\s*(kg|lbs?|pounds?)\b",
]
_DOSAGE_RE = re.compile("|".join(f"(?:{p})" for p in _DOSAGE_PATTERNS), re.IGNORECASE)

# config-driven refusal rules — these three are where a wrong-but-plausible answer causes the most harm
PEDIATRIC_PATTERNS = [
    r"\b(infant|toddler|newborn|baby)\b.*\b(dose|dosage|mg|medicine|medication)\b",
    r"\b(dose|dosage|mg|medicine|medication)\b.*\b(infant|toddler|newborn|baby)\b",
    r"\bchild(ren)?\b.*\b(dose|dosage|mg|medicine|medication)\b",
    r"\b(dose|dosage|mg|medicine|medication)\b.*\bchild(ren)?\b",
    r"for a \d+[\s-]?(year|month)[\s-]?old",
]
PREGNANCY_PATTERNS = [
    r"\bpregnan\w*\b.*\b(safe|dose|dosage|medication|take|drug)\b",
    r"\b(safe|dose|dosage|medication|take|drug)\b.*\bpregnan\w*\b",
    r"\bbreastfeed\w*\b.*\b(safe|dose|medication|take)\b",
    r"\b(safe|dose|medication|take)\b.*\bbreastfeed\w*\b",
    r"\bnursing mother\b",
]
DRUG_INTERACTION_PATTERNS = [
    r"drug interaction", r"interact with", r"interaction between",
    r"safe to take .*(with|and) ", r"combine .* with", r"mix .* (and|with) ",
]

REFUSED_TOPICS = {
    "pediatric_dosing": re.compile("|".join(PEDIATRIC_PATTERNS), re.IGNORECASE),
    "pregnancy_or_breastfeeding_dosing": re.compile("|".join(PREGNANCY_PATTERNS), re.IGNORECASE),
    "drug_interaction": re.compile("|".join(DRUG_INTERACTION_PATTERNS), re.IGNORECASE),
}

_TOPIC_LABEL = {
    "pediatric_dosing": "medication dosing for children",
    "pregnancy_or_breastfeeding_dosing": "medication safety in pregnancy or breastfeeding",
    "drug_interaction": "drug interactions",
}


def check_refusal(question: str) -> Optional[str]:
    """Return the matched refused-topic key, or None if the question doesn't hit one."""
    for topic, pattern in REFUSED_TOPICS.items():
        if pattern.search(question or ""):
            return topic
    return None


def refusal_response(topic: str) -> str:
    label = _TOPIC_LABEL[topic]
    return (
        f"I can't give a reliable answer on {label} — this needs a pharmacist or your doctor's "
        "direct input, since a wrong answer here could cause real harm. Please check with one "
        "before taking or combining any medication."
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def ground_answer(answer: str, sources: List[str]) -> Tuple[str, List[str]]:
    """Strip any dosage/frequency/duration/age figure not found verbatim in the sources."""
    if not answer:
        return answer, []

    source_blob = _normalize(" ".join(s for s in (sources or []) if s))
    matches = list(_DOSAGE_RE.finditer(answer))
    removed = []
    result = answer
    for m in reversed(matches):
        figure = m.group(0)
        if _normalize(figure) not in source_blob:
            removed.append(figure)
            result = result[: m.start()] + REFERRAL_PLACEHOLDER + result[m.end() :]
    removed.reverse()
    return result, removed
