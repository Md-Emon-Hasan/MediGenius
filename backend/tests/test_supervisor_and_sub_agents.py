"""Tests for the supervisor and the symptom/drug-interaction sub-agents"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents.drug_interaction_sub_agent import DrugInteractionSubAgent  # noqa: E402
from app.agents.medical_supervisor_agent import MedicalSupervisorAgent  # noqa: E402
from app.agents.symptom_analysis_sub_agent import (  # noqa: E402
    SymptomAnalysisSubAgent,
    format_symptom_context,
)
from app.core.state import initialize_conversation_state  # noqa: E402


# --- Supervisor ---
def test_supervisor_skips_definitional_question():
    state = initialize_conversation_state()
    state["question"] = "what is diabetes"
    res = MedicalSupervisorAgent(state)
    assert res["needs_symptom_analysis"] is False


def test_supervisor_flags_reported_symptoms():
    state = initialize_conversation_state()
    state["question"] = "I have a headache and fever since yesterday"
    res = MedicalSupervisorAgent(state)
    assert res["needs_symptom_analysis"] is True


def test_supervisor_symptom_word_without_first_person_not_flagged():
    state = initialize_conversation_state()
    state["question"] = "what causes fever in children"
    res = MedicalSupervisorAgent(state)
    assert res["needs_symptom_analysis"] is False


# --- Symptom analysis sub-agent ---
def test_symptom_analysis_no_gateway():
    state = initialize_conversation_state()
    state["question"] = "I have a headache"
    with patch("app.agents.symptom_analysis_sub_agent.model_gateway.is_available", return_value=False):
        res = SymptomAnalysisSubAgent(state)
        assert res["symptom_summary"] is None


def test_symptom_analysis_success():
    state = initialize_conversation_state()
    state["question"] = "I have a headache and mild fever since this morning"
    with patch("app.agents.symptom_analysis_sub_agent.model_gateway.is_available", return_value=True), \
         patch("app.agents.symptom_analysis_sub_agent.model_gateway.generate") as mock_gen:
        mock_gen.return_value = {
            "content": ('{"reported_symptoms": ["headache", "mild fever"], '
                        '"clarifying_questions": ["How long has this lasted?"], '
                        '"referral": "See a doctor if it persists beyond 3 days."}'),
            "model_used": "groq/openai/gpt-oss-120b", "fallback": False, "degraded": False,
        }
        res = SymptomAnalysisSubAgent(state)
        assert res["symptom_summary"]["reported_symptoms"] == ["headache", "mild fever"]
        assert "not a diagnosis" in res["symptom_summary"]["note"].lower()


def test_symptom_analysis_malformed_json():
    state = initialize_conversation_state()
    state["question"] = "I have a headache"
    with patch("app.agents.symptom_analysis_sub_agent.model_gateway.is_available", return_value=True), \
         patch("app.agents.symptom_analysis_sub_agent.model_gateway.generate") as mock_gen:
        mock_gen.return_value = {"content": "not json", "model_used": "x", "fallback": False, "degraded": False}
        res = SymptomAnalysisSubAgent(state)
        assert res["symptom_summary"] is None


def test_format_symptom_context_empty_when_none():
    state = initialize_conversation_state()
    assert format_symptom_context(state) == ""


def test_format_symptom_context_formats_summary():
    state = initialize_conversation_state()
    state["symptom_summary"] = {"reported_symptoms": ["headache", "fever"], "note": "x"}
    ctx = format_symptom_context(state)
    assert "headache" in ctx and "fever" in ctx


# --- Drug interaction sub-agent ---
def test_drug_interaction_recognizes_named_drugs():
    def fake_get(url, params=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"idGroup": {"rxnormId": ["12345"]}}
        return resp

    state = {"question": "can i take ibuprofen with warfarin"}
    with patch("app.agents.drug_interaction_sub_agent.httpx.get", side_effect=fake_get):
        res = DrugInteractionSubAgent(state)
        assert "ibuprofen" in res["generation"] or "warfarin" in res["generation"]
        assert "pharmacist" in res["generation"].lower() or "doctor" in res["generation"].lower()
        assert res["source"] == "Safety Router"


def test_drug_interaction_no_recognized_drugs_falls_back_to_generic_refusal():
    def fake_get(url, params=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"idGroup": {}}
        return resp

    state = {"question": "does this thing interact with that other thing"}
    with patch("app.agents.drug_interaction_sub_agent.httpx.get", side_effect=fake_get):
        res = DrugInteractionSubAgent(state)
        assert "pharmacist" in res["generation"].lower()


def test_drug_interaction_rxnav_failure_falls_back_gracefully():
    state = {"question": "can i take aspirin with ibuprofen"}
    with patch("app.agents.drug_interaction_sub_agent.httpx.get", side_effect=Exception("network down")):
        res = DrugInteractionSubAgent(state)
        assert "pharmacist" in res["generation"].lower()


def test_drug_interaction_never_states_an_interaction_exists():
    def fake_get(url, params=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"idGroup": {"rxnormId": ["1"]}}
        return resp

    state = {"question": "can i take ibuprofen with paracetamol"}
    with patch("app.agents.drug_interaction_sub_agent.httpx.get", side_effect=fake_get):
        res = DrugInteractionSubAgent(state)
        lowered = res["generation"].lower()
        assert "yes" not in lowered.split(".")[0]
        assert "safe to combine" not in lowered
