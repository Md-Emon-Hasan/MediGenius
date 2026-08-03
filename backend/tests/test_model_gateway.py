"""Tests for the LiteLLM-based model gateway: routing, retry, and fallback"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import litellm  # noqa: E402

from app.tools import model_gateway  # noqa: E402


def _fake_response(text):
    return {"choices": [{"message": {"content": text}}]}


def test_is_available_true_with_key():
    with patch('app.tools.model_gateway.GROQ_API_KEY', 'fake-key'):
        assert model_gateway.is_available() is True


def test_is_available_false_without_key():
    with patch('app.tools.model_gateway.GROQ_API_KEY', None):
        assert model_gateway.is_available() is False


def test_generate_no_key_returns_degraded():
    with patch('app.tools.model_gateway.GROQ_API_KEY', None):
        result = model_gateway.generate("hello")
        assert result["degraded"] is True
        assert result["content"] is None


def test_generate_success_on_primary_model():
    with patch('app.tools.model_gateway.GROQ_API_KEY', 'fake-key'), \
         patch('litellm.completion', return_value=_fake_response("a real medical answer here")) as mock_completion:
        result = model_gateway.generate("hello", tier="synthesis")
        assert result["content"] == "a real medical answer here"
        assert result["model_used"] == model_gateway.SYNTHESIS_MODEL
        assert result["fallback"] is False
        assert result["degraded"] is False
        mock_completion.assert_called_once()


def test_generate_retries_same_model_on_transient_error():
    with patch('app.tools.model_gateway.GROQ_API_KEY', 'fake-key'), \
         patch('litellm.completion') as mock_completion:
        mock_completion.side_effect = [Exception("temporary 503"), _fake_response("recovered answer")]
        result = model_gateway.generate("hello", tier="synthesis")
        assert result["content"] == "recovered answer"
        assert result["fallback"] is False
        assert mock_completion.call_count == 2


def test_generate_drops_tier_on_rate_limit_without_retry():
    with patch('app.tools.model_gateway.GROQ_API_KEY', 'fake-key'), \
         patch('litellm.completion') as mock_completion:
        mock_completion.side_effect = [
            litellm.RateLimitError("rate limited", llm_provider="groq", model="x"),
            _fake_response("answer from fallback tier"),
        ]
        result = model_gateway.generate("hello", tier="synthesis")
        assert result["content"] == "answer from fallback tier"
        assert result["model_used"] == model_gateway.CLASSIFICATION_MODEL
        assert result["fallback"] is True
        assert mock_completion.call_count == 2


def test_generate_all_tiers_fail_returns_degraded():
    with patch('app.tools.model_gateway.GROQ_API_KEY', 'fake-key'), \
         patch('litellm.completion', side_effect=Exception("down")):
        result = model_gateway.generate("hello", tier="synthesis")
        assert result["degraded"] is True
        assert result["content"] is None


def test_generate_short_response_treated_as_failure():
    with patch('app.tools.model_gateway.GROQ_API_KEY', 'fake-key'), \
         patch('litellm.completion', return_value=_fake_response("hi")):
        result = model_gateway.generate("hello", tier="synthesis")
        assert result["degraded"] is True


def test_generate_classification_tier_has_no_fallback_below_it():
    with patch('app.tools.model_gateway.GROQ_API_KEY', 'fake-key'), \
         patch('litellm.completion', side_effect=Exception("down")) as mock_completion:
        model_gateway.generate("hello", tier="classification")
        # primary retried once, no lower tier to drop to
        assert mock_completion.call_count == 2


def test_generate_forwards_reasoning_effort_when_given():
    # gpt-oss models spend max_tokens on hidden reasoning before any visible output — omitting this
    # for "cheap" structured-output calls was the root cause of a real truncation bug, see Phase 6 report
    with patch('app.tools.model_gateway.GROQ_API_KEY', 'fake-key'), \
         patch('litellm.completion', return_value=_fake_response("a real medical answer here")) as mock_completion:
        model_gateway.generate("hello", tier="reasoning", reasoning_effort="low")
        assert mock_completion.call_args.kwargs.get("reasoning_effort") == "low"


def test_generate_omits_reasoning_effort_when_not_given():
    with patch('app.tools.model_gateway.GROQ_API_KEY', 'fake-key'), \
         patch('litellm.completion', return_value=_fake_response("a real medical answer here")) as mock_completion:
        model_gateway.generate("hello", tier="synthesis")
        assert "reasoning_effort" not in mock_completion.call_args.kwargs
