"""Tests for the diagnosis verification sub-agent"""
import os
import sys
from unittest.mock import patch

from langchain_core.documents import Document

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents.diagnosis_verification_sub_agent import (  # noqa: E402
    HELD_BACK_ANSWER,
    DiagnosisVerificationSubAgent,
)
from app.core.state import initialize_conversation_state  # noqa: E402


def _state_with_answer(answer, documents=None, source="Medical Literature Database"):
    state = initialize_conversation_state()
    state["generation"] = answer
    state["source"] = source
    state["documents"] = documents or []
    return state


def test_static_safety_response_skips_verification():
    state = _state_with_answer("call a helpline", source="Safety Router")
    res = DiagnosisVerificationSubAgent(state)
    assert res["verification"] is None


def test_empty_answer_skips_verification():
    state = _state_with_answer("")
    res = DiagnosisVerificationSubAgent(state)
    assert res["verification"] is None


def test_no_evidence_skips_llm_and_flags_high_risk():
    state = _state_with_answer("Take this medication as directed.", documents=[])
    with patch("app.agents.diagnosis_verification_sub_agent.model_gateway.generate") as mock_gen:
        res = DiagnosisVerificationSubAgent(state)
        mock_gen.assert_not_called()
        assert res["verification"]["risk"] == "high"
        assert res["verification"]["grounded"] is False


def test_gateway_unavailable_skips_llm():
    state = _state_with_answer("Answer", documents=[Document(page_content="evidence")])
    with patch("app.agents.diagnosis_verification_sub_agent.model_gateway.is_available", return_value=False), \
         patch("app.agents.diagnosis_verification_sub_agent.model_gateway.generate") as mock_gen:
        res = DiagnosisVerificationSubAgent(state)
        mock_gen.assert_not_called()
        assert res["verification"]["risk"] == "medium"


def test_low_risk_answer_passed_through_unchanged():
    evidence = [Document(page_content="Fever is a common symptom of viral infections.")]
    state = _state_with_answer("Fever is common with viral infections.", documents=evidence)
    with patch("app.agents.diagnosis_verification_sub_agent.model_gateway.is_available", return_value=True), \
         patch("app.agents.diagnosis_verification_sub_agent.model_gateway.generate") as mock_gen:
        mock_gen.return_value = {
            "content": '{"grounded": true, "citations_valid": true, "unsupported_claims": [], "risk": "low", "needs_revision": false}',
            "model_used": "groq/openai/gpt-oss-120b", "fallback": False, "degraded": False,
        }
        res = DiagnosisVerificationSubAgent(state)
        assert res["generation"] == "Fever is common with viral infections."
        assert res["verification"]["risk"] == "low"
        mock_gen.assert_called_once()


def test_high_risk_answer_is_held_back():
    state = _state_with_answer(
        "This confirms you have a rare autoimmune disease.",
        documents=[Document(page_content="Fever is a common symptom of viral infections.")],
    )
    with patch("app.agents.diagnosis_verification_sub_agent.model_gateway.is_available", return_value=True), \
         patch("app.agents.diagnosis_verification_sub_agent.model_gateway.generate") as mock_gen:
        mock_gen.return_value = {
            "content": ('{"grounded": false, "citations_valid": false, '
                        '"unsupported_claims": ["diagnosis of autoimmune disease"], "risk": "high", "needs_revision": true}'),
            "model_used": "groq/openai/gpt-oss-120b", "fallback": False, "degraded": False,
        }
        res = DiagnosisVerificationSubAgent(state)
        assert res["generation"] == HELD_BACK_ANSWER
        assert res["source"] == "Safety Router"
        assert res["verification"]["risk"] == "high"
        mock_gen.assert_called_once()  # high risk skips the revision call entirely


def test_medium_risk_needs_revision_triggers_one_revision_call():
    state = _state_with_answer(
        "Take it twice a day for best results.",
        documents=[Document(page_content="This medication is generally taken with food.")],
    )
    with patch("app.agents.diagnosis_verification_sub_agent.model_gateway.is_available", return_value=True), \
         patch("app.agents.diagnosis_verification_sub_agent.model_gateway.generate") as mock_gen:
        mock_gen.side_effect = [
            {
                "content": ('{"grounded": false, "citations_valid": false, '
                            '"unsupported_claims": ["twice a day"], "risk": "medium", "needs_revision": true}'),
                "model_used": "groq/openai/gpt-oss-120b", "fallback": False, "degraded": False,
            },
            {
                "content": "Take it with food as generally recommended.",
                "model_used": "groq/openai/gpt-oss-120b", "fallback": False, "degraded": False,
            },
        ]
        res = DiagnosisVerificationSubAgent(state)
        assert res["generation"] == "Take it with food as generally recommended."
        assert res["verification"]["risk"] == "medium"
        assert mock_gen.call_count == 2  # one verify + one revision, no re-verification


def test_malformed_json_fails_closed_to_high_risk():
    state = _state_with_answer("Some answer.", documents=[Document(page_content="evidence")])
    with patch("app.agents.diagnosis_verification_sub_agent.model_gateway.is_available", return_value=True), \
         patch("app.agents.diagnosis_verification_sub_agent.model_gateway.generate") as mock_gen:
        mock_gen.return_value = {"content": "not json at all", "model_used": "x", "fallback": False, "degraded": False}
        res = DiagnosisVerificationSubAgent(state)
        assert res["verification"]["risk"] == "high"
        assert res["generation"] == HELD_BACK_ANSWER
