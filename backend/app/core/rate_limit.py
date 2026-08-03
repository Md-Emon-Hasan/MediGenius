"""
MediGenius — core/rate_limit.py
Shared slowapi Limiter, keyed on the proxy-forwarded client IP where available.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import RATE_LIMIT_ENABLED


def _client_key(request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_client_key, enabled=RATE_LIMIT_ENABLED)
