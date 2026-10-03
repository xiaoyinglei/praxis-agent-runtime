"""Authenticated loopback providers must pass the same startup checks as local models."""

import asyncio
from types import SimpleNamespace

import httpx

from agent_runtime.local_runtime import LocalProviderProbe


def test_local_health_uses_configured_token_only_for_same_origin(monkeypatch):
    import agent_runtime.local_runtime as runtime

    monkeypatch.setenv("TEST_GATEWAY_TOKEN", "fake-proxy")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "test-model"}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        runtime.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)
    )
    for health_url in (None, "http://127.0.0.1:9999/models"):
        spec = SimpleNamespace(
            id="test-model",
            location="local",
            base_url="http://127.0.0.1:18443/v1",
            api_key_env="TEST_GATEWAY_TOKEN",
            runtime=SimpleNamespace(health_url=health_url),
        )
        asyncio.run(LocalProviderProbe().ensure_ready(spec))
    assert seen[0].headers.get("authorization") == "Bearer fake-proxy"
    assert "authorization" not in seen[1].headers


def test_default_port_is_same_origin(monkeypatch):
    import agent_runtime.local_runtime as runtime

    monkeypatch.setenv("TEST_GATEWAY_TOKEN", "fake-proxy")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "test-model"}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        runtime.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)
    )
    spec = SimpleNamespace(
        id="test-model",
        location="local",
        base_url="http://127.0.0.1:80/v1",
        api_key_env="TEST_GATEWAY_TOKEN",
        runtime=SimpleNamespace(health_url="http://127.0.0.1/models"),
    )
    asyncio.run(LocalProviderProbe().ensure_ready(spec))
    assert seen[0].headers.get("authorization") == "Bearer fake-proxy"
