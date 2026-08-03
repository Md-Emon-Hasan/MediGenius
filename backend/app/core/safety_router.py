"""
MediGenius — core/safety_router.py
Deterministic pre-pipeline safety gate: crisis detection, emergency detection,
disclaimer, and input sanitisation. Runs before the LangGraph pipeline.
No LLM ever decides these outcomes — keyword and pattern matching only.
"""

import re
from typing import Dict, Optional

from app.core.logging_config import logger

# contact details verified against official sources August 2026 — see Phase 1 report, do not edit without a source
HELPLINES = {
    "BD": {"crisis_name": "Kaan Pete Roi", "crisis_contact": "09612-119911 (alt +8801779-554391), 3 PM-3 AM daily", "emergency": "999"},
    "US": {"crisis_name": "988 Suicide & Crisis Lifeline", "crisis_contact": "call or text 988, 24/7", "emergency": "911"},
    "UK": {"crisis_name": "Samaritans", "crisis_contact": "116 123, free, 24/7", "emergency": "999 or 112"},
}
INTERNATIONAL_DIRECTORY = "findahelpline.com"


def _build_crisis_response() -> str:
    bd = HELPLINES["BD"]
    return (
        "I'm really glad you reached out, and I'm concerned about what you shared. "
        "You deserve support from a real person right now, not a chatbot.\n\n"
        f"If you're in Bangladesh: call {bd['crisis_name']} at {bd['crisis_contact']}.\n"
        f"Anywhere else: {INTERNATIONAL_DIRECTORY} lists a free, confidential helpline for your country.\n"
        f"If you are in immediate danger, please call your local emergency number right now ({bd['emergency']} in Bangladesh).\n\n"
        "Please talk to someone — a helpline, a trusted person, or emergency services. "
        "You don't have to go through this alone."
    )


def _build_emergency_response() -> str:
    bd, us, uk = HELPLINES["BD"], HELPLINES["US"], HELPLINES["UK"]
    return (
        "This sounds like it could be a medical emergency. Please stop reading this and call emergency "
        f"services now ({bd['emergency']} in Bangladesh, {us['emergency']} in the US, {uk['emergency']} in "
        "the UK, or your local equivalent), or go to the nearest emergency room.\n\n"
        "I can't safely evaluate or advise on this here — a same-day, in-person medical response is what you need."
    )


CRISIS_RESPONSE = _build_crisis_response()
EMERGENCY_RESPONSE = _build_emergency_response()

FAIL_CLOSED_RESPONSE = (
    "I'm not able to safely process this message right now. If this is a medical emergency, call your local "
    "emergency number immediately (999 in Bangladesh, 911 in the US). If you're struggling emotionally, "
    f"{INTERNATIONAL_DIRECTORY} can connect you with a helpline in your country."
)

DISCLAIMER = (
    "This is general information, not a medical diagnosis. It doesn't replace an in-person evaluation by a "
    "licensed clinician — please consult one for anything beyond general guidance."
)

# over-inclusive on purpose: a missed positive here is worse than an unnecessary helpline shown
CRISIS_PATTERNS = [
    r"kill myself", r"end my life", r"end it all", r"want to die", r"wish i (was|were) dead",
    r"don'?t want to (live|be alive)", r"no reason to live", r"better off dead",
    r"take my (own )?life", r"suicidal", r"suicide", r"self.?harm", r"cutting myself",
    r"hurt myself on purpose", r"can'?t go on", r"can'?t take (it|this) anymore",
    r"not worth living", r"planning to (kill myself|end my life)", r"goodbye forever",
    r"want to overdose", r"planning to overdose", r"overdose on purpose",
    r"swallow all (my|the) pills", r"take all (my|the) pills", r"enough pills to die",
    # Bengali script
    r"আত্মহত্যা", r"মরে যেতে চাই", r"নিজেকে শেষ করে দিতে চাই", r"বেঁচে থাকতে চাই না",
    # transliterated Bengali (Banglish)
    r"attohotta", r"aattohotya", r"more jete chai", r"morjete chai", r"bachte chai na",
    r"suicide korte chai", r"nijeke shesh kore",
]

EMERGENCY_PATTERNS = [
    r"chest pain", r"crushing (pain|feeling) in (my |the )?chest",
    r"can'?t breathe", r"cannot breathe", r"difficulty breathing", r"gasping for air",
    r"face (is )?drooping", r"one side of (my |the )?face (is|feels)", r"slurred speech",
    r"sudden numbness", r"can'?t (lift|move) (my )?arm", r"sudden confusion",
    r"worst headache of my life", r"won'?t stop bleeding", r"severe bleeding",
    r"bleeding heavily", r"throat (is )?closing", r"anaphylaxis",
    r"passed out and (won'?t|not) wak", r"unconscious", r"unresponsive", r"not breathing",
    # Bengali script
    r"বুকে ব্যথা", r"শ্বাস নিতে পারছি না", r"অজ্ঞান হয়ে গেছে",
    # transliterated Bengali
    r"buke betha", r"buke bytha", r"shash nite parchi na", r"ojnan hoye geche",
]

_CRISIS_RE = re.compile("|".join(CRISIS_PATTERNS), re.IGNORECASE)
_EMERGENCY_RE = re.compile("|".join(EMERGENCY_PATTERNS), re.IGNORECASE)

# phrases that could hijack the prompt if they arrive inside a PDF chunk or a web search result
_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|above|prior) instructions?",
    r"disregard (all |any )?(previous|prior) (prompts?|instructions?)",
    r"you are now",
    r"^\s*system\s*:", r"^\s*assistant\s*:",
    r"new instructions?\s*:",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)


def sanitize_external_text(text: str) -> str:
    """Strip instruction-like lines from PDF/web content before it reaches a prompt."""
    if not text:
        return text
    return _INJECTION_RE.sub("[removed]", text)


def evaluate(message: str) -> Dict[str, Optional[str]]:
    """Run the deterministic safety gate on a raw user message before the pipeline runs."""
    try:
        if _CRISIS_RE.search(message or ""):
            logger.warning("safety_router: crisis rule fired")
            return {"blocked": True, "category": "crisis", "response": CRISIS_RESPONSE, "disclaimer": None}

        if _EMERGENCY_RE.search(message or ""):
            logger.warning("safety_router: emergency rule fired")
            return {"blocked": True, "category": "emergency", "response": EMERGENCY_RESPONSE, "disclaimer": None}

        return {"blocked": False, "category": None, "response": None, "disclaimer": DISCLAIMER}
    except Exception:
        logger.error("safety_router: check failed, failing closed")
        return {"blocked": True, "category": "error", "response": FAIL_CLOSED_RESPONSE, "disclaimer": None}
