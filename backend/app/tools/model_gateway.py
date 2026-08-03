"""
MediGenius — tools/model_gateway.py
LiteLLM-based routing and retry/fallback across Groq-hosted models only.
Wraps the same Groq account llm_client.py uses — one API key, one provider,
tiered by task so routing/classification calls don't pay for the largest model.
"""

import litellm

from app.core.config import GROQ_API_KEY
from app.core.logging_config import logger

litellm.suppress_debug_info = True

# llama-3.1-8b-instant and llama-3.3-70b-versatile are deprecated on Groq (shutdown 2026-08-16);
# these are their vendor-recommended replacements — re-verify at console.groq.com/docs/models before changing
SYNTHESIS_MODEL = "groq/openai/gpt-oss-120b"
REASONING_MODEL = "groq/openai/gpt-oss-120b"
CLASSIFICATION_MODEL = "groq/openai/gpt-oss-20b"

TIER_MODELS = {
    "synthesis": SYNTHESIS_MODEL,
    "reasoning": REASONING_MODEL,
    "classification": CLASSIFICATION_MODEL,
}


def is_available() -> bool:
    return bool(GROQ_API_KEY)


def _call(model: str, prompt: str, max_tokens: int, reasoning_effort: str) -> str:
    # gpt-oss models spend output tokens on hidden reasoning before the visible answer — a low
    # max_tokens on a "cheap" call can truncate to nothing before any content appears, so
    # reasoning_effort matters more here than the token count itself for short structured replies
    kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        api_key=GROQ_API_KEY,
        temperature=0.3,
        max_tokens=max_tokens,
        **kwargs,
    )
    return response["choices"][0]["message"]["content"]


def generate(prompt: str, tier: str = "synthesis", max_tokens: int = 2048, reasoning_effort: str = None) -> dict:
    """primary -> smaller Groq model -> degraded. Retries the same model once on a non-rate-limit error."""
    if not is_available():
        return {"content": None, "model_used": None, "fallback": False, "degraded": True}

    primary = TIER_MODELS.get(tier, SYNTHESIS_MODEL)
    plan = [primary] if primary == CLASSIFICATION_MODEL else [primary, CLASSIFICATION_MODEL]

    for i, model in enumerate(plan):
        allow_retry = i == 0
        for attempt in range(2 if allow_retry else 1):  # pragma: no cover
            try:
                content = _call(model, prompt, max_tokens, reasoning_effort)
                if content and len(content.strip()) > 10:
                    return {
                        "content": content.strip(),
                        "model_used": model,
                        "fallback": model != primary,
                        "degraded": False,
                    }
                break
            except litellm.RateLimitError:
                logger.warning("model_gateway: %s rate limited, dropping a tier", model)
                break
            except Exception as e:
                logger.error("model_gateway: %s call failed: %s", model, str(e))
                if attempt == 0 and allow_retry:
                    continue
                break

    return {"content": None, "model_used": None, "fallback": False, "degraded": True}
