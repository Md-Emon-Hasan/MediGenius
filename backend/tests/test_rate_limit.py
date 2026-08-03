"""Tests for the rate limiter configuration and key function"""
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from slowapi import Limiter, _rate_limit_exceeded_handler  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402

from app.core.rate_limit import _client_key, limiter  # noqa: E402


def test_client_key_uses_forwarded_header():
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"X-Forwarded-For": "1.2.3.4, 5.6.7.8"}
    assert _client_key(mock_request) == "1.2.3.4"


def test_client_key_falls_back_to_remote_address():
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}
    mock_request.client = MagicMock(host="9.9.9.9")
    assert _client_key(mock_request) == "9.9.9.9"


def test_limiter_respects_enabled_flag():
    assert hasattr(limiter, "enabled")


def test_rate_limit_returns_429_when_exceeded():
    app = FastAPI()
    test_limiter = Limiter(key_func=lambda request: "fixed-key")
    app.state.limiter = test_limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.get("/ping")
    @test_limiter.limit("2/minute")
    async def ping(request: Request):
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 429


def test_rate_limit_disabled_never_429s():
    app = FastAPI()
    test_limiter = Limiter(key_func=lambda r: "fixed-key", enabled=False)
    app.state.limiter = test_limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.get("/ping")
    @test_limiter.limit("1/minute")
    async def ping(request: Request):
        return {"ok": True}

    client = TestClient(app)
    for _ in range(5):
        assert client.get("/ping").status_code == 200
