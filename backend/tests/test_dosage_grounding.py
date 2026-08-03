"""Tests for deterministic dosage grounding and refused-topic rules"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core import dosage_grounding  # noqa: E402


def test_grounded_figure_kept():
    answer = "Take 500mg twice a day."
    sources = ["The recommended adult dose is 500mg twice a day."]
    result, removed = dosage_grounding.ground_answer(answer, sources)
    assert removed == []
    assert "500mg" in result


def test_ungrounded_figure_stripped():
    answer = "Take 750mg every 4 hours."
    sources = ["This medication is used for pain relief."]
    result, removed = dosage_grounding.ground_answer(answer, sources)
    assert len(removed) > 0
    assert "750mg" not in result
    assert dosage_grounding.REFERRAL_PLACEHOLDER in result


def test_no_figures_no_op():
    answer = "This medication is generally used for fever."
    result, removed = dosage_grounding.ground_answer(answer, ["some source"])
    assert result == answer
    assert removed == []


def test_empty_answer():
    result, removed = dosage_grounding.ground_answer("", ["source"])
    assert result == ""
    assert removed == []


def test_pediatric_refusal():
    assert dosage_grounding.check_refusal("what is the right dose of paracetamol for my toddler") == "pediatric_dosing"


def test_pediatric_refusal_age_phrasing_without_child_keyword():
    # caught by an end-to-end benchmark run: "for my" (not just "for a") must also match
    assert dosage_grounding.check_refusal("what dose of paracetamol for my 2 year old") == "pediatric_dosing"
    assert dosage_grounding.check_refusal("how much for a 2-year-old") == "pediatric_dosing"
    assert dosage_grounding.check_refusal("dosing for an 18 month old") == "pediatric_dosing"


def test_pregnancy_refusal():
    assert dosage_grounding.check_refusal("is ibuprofen safe during pregnancy") == "pregnancy_or_breastfeeding_dosing"


def test_drug_interaction_refusal():
    assert dosage_grounding.check_refusal("does ibuprofen have a drug interaction with warfarin") == "drug_interaction"


def test_normal_question_not_refused():
    assert dosage_grounding.check_refusal("what is diabetes") is None


def test_refusal_response_mentions_pharmacist_or_doctor():
    text = dosage_grounding.refusal_response("pediatric_dosing")
    assert "pharmacist" in text.lower() or "doctor" in text.lower()
