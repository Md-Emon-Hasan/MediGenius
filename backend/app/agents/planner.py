"""
MediGenius — agents/planner.py
PlannerAgent: decides whether to use RAG retriever or direct LLM.
"""

from app.core.state import AgentState

# ── Medical Keywords ───────────────────────────────────────────────────────────
MEDICAL_KEYWORDS = [
    # Symptoms
    "fever", "pain", "headache", "nausea", "vomiting", "diarrhea", "cough",
    "acne", "pimple", "skin", "rash", "itch", "cold", "flu",
    "shortness of breath", "chest pain", "abdominal pain", "back pain",
    "joint pain", "muscle pain", "fatigue", "weakness", "dizziness",
    "confusion", "memory loss", "seizure", "numbness", "tingling", "swelling",
    "bleeding", "bruising", "weight loss", "weight gain",
    "appetite loss", "sleep problems", "insomnia",
    # Conditions
    "cancer", "diabetes", "hypertension", "heart disease", "stroke", "asthma",
    "copd", "pneumonia", "bronchitis", "covid", "coronavirus",
    "infection", "virus", "bacteria", "fungal", "arthritis", "osteoporosis",
    "thyroid", "kidney disease", "liver disease", "hepatitis", "depression",
    "anxiety", "bipolar", "schizophrenia", "alzheimer", "parkinson", "epilepsy",
    # Medical terms
    "treatment", "therapy", "medication", "medicine", "prescription", "dosage",
    "side effects", "diagnosis", "prognosis", "surgery", "operation",
    "procedure", "test", "lab results", "blood test", "x-ray", "mri",
    "ct scan", "ultrasound", "biopsy", "screening", "prevention", "vaccine",
    "immunization", "rehabilitation", "recovery", "chronic", "acute",
    "syndrome", "disorder", "symptom", "cure", "remedy", "doctor", "hospital",
    # Body parts
    "heart", "lung", "kidney", "liver", "brain", "stomach", "intestine",
    "blood", "bone", "muscle", "nerve", "eye", "ear", "throat",
    "neck", "spine", "joint", "head", "chest", "abdomen", "leg", "arm",
]


def PlannerAgent(state: AgentState) -> AgentState:
    """Decide whether to use RAG retriever or direct LLM based on question content."""
    question = state["question"].lower()
    contains_medical = any(kw in question for kw in MEDICAL_KEYWORDS)
    state["current_tool"] = "retriever" if contains_medical else "llm_agent"
    state["retry_count"] = 0
    return state
