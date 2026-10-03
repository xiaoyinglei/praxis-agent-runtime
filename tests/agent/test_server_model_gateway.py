"""Security and budget behavior of the independently supervised model service."""

import asyncio
import concurrent.futures
import json

import httpx
import pytest
from openai import OpenAI
from starlette.testclient import TestClient

from agent_runtime import server_model_gateway as gateway

KEYS = {"deepseek": "fake-deepseek-master-secret", "groq": "fake-groq-master-secret"}
TOKEN = "fake-agent-proxy-token-secret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def request_body(model="deepseek-flash", **extra):
    return {"model": model, "messages": [{"role": "user", "content": "hello"}], **extra}


def app(tmp_path, handler, keys=None, token=TOKEN, **limits):
    return gateway.create_app(
        KEYS if keys is None else keys,
        token,
        tmp_path / "quota.sqlite3",
        policy=gateway.ServicePolicy(**limits),
        transport=httpx.MockTransport(handler),
    )


def completion(request):
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "id": "fake",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        },
    )


def test_routes_each_model_to_fixed_host_and_credential(tmp_path):
    seen = []

    def upstream(req):
        seen.append((str(req.url), req.headers["authorization"], json.loads(req.content)))
        return completion(req)

    with TestClient(app(tmp_path, upstream)) as client:
        for model in ("deepseek-flash", "openai/gpt-oss-120b"):
            assert client.post("/v1/chat/completions", headers=AUTH, json=request_body(model)).status_code == 200
    assert [(u, k) for u, k, _ in seen] == [
        ("https://api.deepseek.com/v1/chat/completions", f"Bearer {KEYS['deepseek']}"),
        ("https://api.groq.com/openai/v1/chat/completions", f"Bearer {KEYS['groq']}"),
    ]
    assert all(p["max_tokens"] == 4096 for _, _, p in seen)


def test_listing_only_enabled_models_is_authenticated_and_local(tmp_path):
    def never(req):
        pytest.fail("Model listing must not make a provider request")

    with TestClient(app(tmp_path, never, keys={"groq": KEYS["groq"]})) as client:
        assert client.get("/v1/models").status_code == 401
        response = client.get("/v1/models", headers=AUTH)
        assert [v["id"] for v in response.json()["data"]] == ["openai/gpt-oss-120b"]
        assert client.post("/v1/chat/completions", headers=AUTH, json=request_body()).status_code == 400


@pytest.mark.parametrize(
    "extra",
    [
        {"model": "unknown"},
        {"base_url": "http://169.254.169.254"},
        {"api_key": "x"},
        {"max_tokens": True},
        {"max_tokens": 4097},
        {"n": 2},
        {"stream": "true"},
        {"messages": []},
        {"messages": [{"role": "user", "content": [{"image_url": "http://private"}]}]},
    ],
)
def test_rejects_untrusted_or_unbounded_inputs(tmp_path, extra):
    with TestClient(app(tmp_path, lambda req: pytest.fail("Must reject before upstream"))) as client:
        assert client.post("/v1/chat/completions", headers=AUTH, json=request_body(**extra)).status_code == 400


def test_auth_routes_and_body_limit(tmp_path):
    with TestClient(app(tmp_path, completion, max_body_bytes=128)) as client:
        assert client.post("/v1/chat/completions", json=request_body()).status_code == 401
        assert client.get("/v1/models?host=evil", headers=AUTH).status_code == 400
        assert client.get("/v1/models", headers={**AUTH, "Origin": "https://evil"}).status_code == 403
        assert client.get("/v1/models/", headers=AUTH, follow_redirects=False).status_code == 404
        assert client.post("/v1/chat/completions", headers=AUTH, content=b"x" * 129).status_code == 413
        assert client.post("/v1/chat/completions", headers=AUTH, content=b"{").status_code == 400


@pytest.mark.parametrize("status", [302, 401, 429, 500])
def test_upstream_errors_headers_and_redirects_never_disclose_secrets(tmp_path, status):
    def upstream(req):
        return httpx.Response(status, text=KEYS["deepseek"], headers={"Location": "https://evil/", "X-Key": TOKEN})

    with TestClient(app(tmp_path, upstream)) as client:
        response = client.post("/v1/chat/completions", headers=AUTH, json=request_body())
        assert response.status_code == 502
        assert all(secret not in response.text for secret in (*KEYS.values(), TOKEN))
        assert "x-key" not in response.headers and "location" not in response.headers


@pytest.mark.parametrize("mode", ["literal", "escaped", "sse"])
def test_rejects_provider_key_echo_even_from_other_provider(tmp_path, mode):
    key = KEYS["groq"]
    if mode == "literal":
        output = json.dumps({"value": key})
    elif mode == "escaped":
        output = '{"value":"' + "".join(f"\\u{ord(c):04x}" for c in key) + '"}'
    else:
        output = "".join(
            "data: " + json.dumps({"choices": [{"delta": {"content": p}}]}) + "\n\n" for p in (key[:10], key[10:])
        )
    with TestClient(app(tmp_path, lambda req: httpx.Response(200, text=output))) as client:
        response = client.post("/v1/chat/completions", headers=AUTH, json=request_body(stream=mode == "sse"))
        assert response.status_code == 502
        assert key not in response.text


def test_response_size_and_compression_are_bounded(tmp_path):
    for response in (
        httpx.Response(200, content=b"x" * 65),
        httpx.Response(200, stream=httpx.ByteStream(b"invalid"), headers={"Content-Encoding": "gzip"}),
    ):
        with TestClient(app(tmp_path, lambda req, response=response: response, max_response_bytes=64)) as client:
            assert client.post("/v1/chat/completions", headers=AUTH, json=request_body()).status_code == 502


def test_restart_and_proxy_rotation_do_not_reset_budget(tmp_path):
    for token in (TOKEN, "rotated-agent-token-secret"):
        with TestClient(app(tmp_path, completion, token=token, daily_calls=1)) as client:
            response = client.post(
                "/v1/chat/completions", headers={"Authorization": f"Bearer {token}"}, json=request_body()
            )
            assert response.status_code == (200 if token == TOKEN else 429)


def test_failed_upstream_consumes_output_reservation(tmp_path):
    with TestClient(app(tmp_path, lambda req: httpx.Response(500), daily_output_tokens=4096)) as client:
        assert client.post("/v1/chat/completions", headers=AUTH, json=request_body()).status_code == 502
        assert client.post("/v1/chat/completions", headers=AUTH, json=request_body(max_tokens=1)).status_code == 429


def test_sqlite_budget_is_atomic_across_connections_and_days(tmp_path):
    policy = gateway.ServicePolicy(daily_calls=3, daily_output_tokens=100)
    ledgers = [gateway.DailyBudget(tmp_path / "quota.sqlite3", policy) for _ in range(10)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(lambda ledger: ledger.reserve(10, day=20000), ledgers))
    assert sum(results) == 3
    assert not ledgers[0].reserve(1, day=19999)
    assert ledgers[0].reserve(1, day=20001)
    assert not ledgers[1].reserve(1, day=20000)


def test_timeout_releases_concurrency_slot(tmp_path):
    async def upstream(req):
        await asyncio.sleep(0.2)
        return completion(req)

    with TestClient(app(tmp_path, upstream, timeout_seconds=0.03, max_concurrent=1)) as client:
        for _ in range(2):
            assert client.post("/v1/chat/completions", headers=AUTH, json=request_body()).status_code == 502


def test_openai_sdk_accepts_json_and_sse_tool_calls(tmp_path):
    def upstream(req):
        body = json.loads(req.content)
        if not body.get("stream"):
            return completion(req)
        delta = {
            "role": "assistant",
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"x.py"}'},
                }
            ],
        }
        output = "data: " + json.dumps(
            {
                "id": "fake",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        )
        return httpx.Response(200, text=output + "\n\ndata: [DONE]\n\n")

    with TestClient(app(tmp_path, upstream)) as client:
        with OpenAI(api_key=TOKEN, base_url="http://testserver/v1", http_client=client, max_retries=0) as sdk:
            for model in ("deepseek-flash", "openai/gpt-oss-120b"):
                assert (
                    sdk.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}])
                    .choices[0]
                    .message.content
                    == "ok"
                )
                chunks = list(
                    sdk.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}], stream=True)
                )
                assert chunks[0].choices[0].delta.tool_calls[0].function.name == "read_file"


def test_credentials_are_read_from_private_regular_files_only(tmp_path, monkeypatch):
    for name, value in {"deepseek-key": KEYS["deepseek"], "groq-key": KEYS["groq"], "agent-token": TOKEN}.items():
        p = tmp_path / name
        p.write_text(value + "\n")
        p.chmod(0o600)
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "do-not-use-environment-secret")
    assert gateway.load_credentials(["deepseek", "groq"]) == (KEYS, TOKEN)
    (tmp_path / "groq-key").chmod(0o644)
    with pytest.raises(ValueError, match="credential"):
        gateway.load_credentials(["deepseek", "groq"])
    (tmp_path / "groq-key").unlink()
    (tmp_path / "groq-key").symlink_to(tmp_path / "deepseek-key")
    with pytest.raises(ValueError, match="credential"):
        gateway.load_credentials(["deepseek", "groq"])


def test_missing_credentials_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", KEYS["deepseek"])
    with pytest.raises(ValueError, match="credential"):
        gateway.load_credentials(["deepseek"])


@pytest.mark.parametrize(
    "limits", [{"daily_calls": 0}, {"max_tokens": True}, {"timeout_seconds": float("nan")}, {"daily_output_tokens": -1}]
)
def test_invalid_policy_fails_at_startup(limits):
    with pytest.raises(ValueError):
        gateway.ServicePolicy(**limits)


def test_default_sdk_retries_are_each_charged(tmp_path):
    seen = []

    def upstream(req):
        seen.append(req)
        return httpx.Response(500)

    with TestClient(app(tmp_path, upstream, daily_calls=2)) as client:
        with OpenAI(api_key=TOKEN, base_url="http://testserver/v1", http_client=client) as sdk:
            from openai import RateLimitError

            with pytest.raises(RateLimitError):
                sdk.chat.completions.create(model="deepseek-flash", messages=[{"role": "user", "content": "hi"}])
    assert len(seen) == 2


def test_budget_rejects_symlink_or_nonregular_state(tmp_path):
    target = tmp_path / "other-state"
    target.write_text("do not overwrite")
    state = tmp_path / "quota.sqlite3"
    state.symlink_to(target)
    with pytest.raises(ValueError):
        gateway.DailyBudget(state, gateway.ServicePolicy())
    assert target.read_text() == "do not overwrite"
    state.unlink()
    state.mkdir()
    with pytest.raises(ValueError):
        gateway.DailyBudget(state, gateway.ServicePolicy())
