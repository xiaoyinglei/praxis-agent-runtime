from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time

import httpx
import pytest
from pydantic import ValidationError

from agent_runtime.channels.providers import (
    Envelope,
    ProviderClient,
    ProviderError,
    Source,
    parse_command,
    parse_event,
)


def github_source(**updates: object) -> Source:
    updates.setdefault("outbound_principal", "99")
    return Source(source_id="gh", kind="github", repository="owner/repo", webhook_secret_env="GH_WEBHOOK", **updates)


def github_event(text: str = "hello", *, sender_type: str = "User") -> dict[str, object]:
    return {
        "action": "created",
        "repository": {"full_name": "owner/repo"},
        "sender": {"id": 42, "type": sender_type},
        "issue": {"number": 3},
        "comment": {"id": 101, "body": text, "user": {"id": 42}},
    }


def gh_headers(raw: bytes, *, event: str = "issue_comment", delivery: str = "event-1") -> dict[str, str]:
    return {
        "X-Hub-Signature-256": "sha256=" + hmac.new(b"secret", raw, "sha256").hexdigest(),
        "X-GitHub-Delivery": delivery,
        "X-GitHub-Event": event,
    }


def feishu_source(**updates: object) -> Source:
    return Source(
        source_id="fs",
        kind="feishu",
        app_id="app",
        chats=("chat-1",),
        encrypt_key_env="FS_KEY",
        verification_token_env="FS_VERIFY",
        **updates,
    )


def fs_event(text: str = "hi") -> dict[str, object]:
    return {
        "schema": "2.0",
        "header": {"app_id": "app", "token": "verify", "event_id": "e1", "event_type": "im.message.receive_v1"},
        "event": {
            "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_42"}},
            "message": {
                "message_id": "m1",
                "chat_id": "chat-1",
                "root_id": "root-1",
                "message_type": "text",
                "content": json.dumps({"text": text}),
            },
        },
    }


def fs_headers(raw: bytes, now: int = 1700000000) -> dict[str, str]:
    return {
        "X-Lark-Request-Timestamp": str(now),
        "X-Lark-Request-Nonce": "nonce",
        "X-Lark-Signature": hashlib.sha256(str(now).encode() + b"noncekey" + raw).hexdigest(),
    }


@pytest.fixture(autouse=True)
def secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "GH_WEBHOOK": "secret",
        "GH_TOKEN": "token",
        "FS_KEY": "key",
        "FS_VERIFY": "verify",
        "FS_APP_SECRET": "app-secret",
    }.items():
        monkeypatch.setenv(name, value)


def test_github_signature_canonical_identity_and_header_case() -> None:
    payload = github_event("/respond req-1 allow_once\nexact action approved")
    raw = json.dumps(payload).encode()
    result = parse_event(github_source(), gh_headers(raw), raw)
    assert isinstance(result, Envelope)
    assert (result.principal, result.operation, result.request_id, result.action) == (
        "42",
        "respond",
        "req-1",
        "allow_once",
    )
    assert result.conversation_id == result.destination == "github:owner/repo:3"
    compact = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    again = parse_event(github_source(), gh_headers(compact), compact)
    assert isinstance(again, Envelope) and result.digest == again.digest
    assert result.native_id == "issue_comment:101:created"


def test_github_signature_rejects_tamper_and_wrong_repo() -> None:
    raw = json.dumps(github_event()).encode()
    with pytest.raises(ProviderError, match="unauthenticated"):
        parse_event(github_source(), gh_headers(raw), raw + b" ")
    payload = github_event()
    payload["repository"] = {"full_name": "other/repo"}
    raw = json.dumps(payload).encode()
    with pytest.raises(ProviderError, match="forbidden"):
        parse_event(github_source(), gh_headers(raw), raw)


@pytest.mark.parametrize("body", ["> /cancel", "```\n/cancel\n```", "please /cancel", "/cancel\nextra", " /cancel"])
def test_quoted_or_nonexact_control_does_not_cancel(body: str) -> None:
    assert parse_command(body)[0] == "input"


def test_commands_and_bot_authority() -> None:
    assert parse_command("/cancel") == ("cancel", "", None, None)
    assert parse_command("/status") == ("status", "", None, None)
    raw = json.dumps(github_event("/cancel", sender_type="Bot")).encode()
    event = parse_event(github_source(), gh_headers(raw), raw)
    assert isinstance(event, Envelope) and event.operation == "evidence" and event.principal_kind == "bot"


def test_github_check_evidence_is_exact_sha_and_check_identity() -> None:
    payload = {
        "action": "completed",
        "repository": {"full_name": "owner/repo"},
        "sender": {"id": 9, "type": "Bot"},
        "check_run": {
            "id": 77,
            "head_sha": "a" * 40,
            "name": "test",
            "status": "completed",
            "conclusion": "failure",
            "pull_requests": [{"number": 3}],
        },
    }
    raw = json.dumps(payload).encode()
    event = parse_event(github_source(), gh_headers(raw, event="check_run"), raw)
    assert isinstance(event, Envelope) and event.operation == "evidence"
    assert event.evidence["head_sha"] == "a" * 40 and event.evidence["check_id"] == "77"
    payload["check_run"]["pull_requests"].append({"number": 4})  # type: ignore[index,union-attr]
    raw = json.dumps(payload).encode()
    assert parse_event(github_source(), gh_headers(raw, event="check_run"), raw) is None


def test_feishu_exact_root_and_timestamp_and_attachment() -> None:
    payload = fs_event("/status")
    raw = json.dumps(payload).encode()
    event = parse_event(feishu_source(), fs_headers(raw), raw, now=1700000000)
    assert isinstance(event, Envelope) and event.conversation_id == "feishu:chat-1:root-1"
    assert event.operation == "status" and event.principal == "ou_42"
    with pytest.raises(ProviderError, match="unauthenticated"):
        parse_event(feishu_source(), fs_headers(raw), raw, now=1700000301)
    payload["event"]["message"].update(message_type="file", content=json.dumps({"file_key": "file-123"}))  # type: ignore[index,union-attr]
    raw = json.dumps(payload).encode()
    event = parse_event(feishu_source(), fs_headers(raw), raw, now=1700000000)
    assert isinstance(event, Envelope) and event.attachments == ("feishu:file:file-123",)


def test_feishu_encrypted_callback_pkcs7() -> None:
    import base64

    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    plaintext = json.dumps(fs_event()).encode()
    padder = padding.PKCS7(128).padder()
    padded = padder.update(plaintext) + padder.finalize()
    iv = b"0" * 16
    encryptor = Cipher(algorithms.AES(hashlib.sha256(b"key").digest()), modes.CBC(iv)).encryptor()
    raw = json.dumps(
        {"encrypt": base64.b64encode(iv + encryptor.update(padded) + encryptor.finalize()).decode()}
    ).encode()
    assert isinstance(parse_event(feishu_source(), fs_headers(raw), raw, now=1700000000), Envelope)


def test_size_duplicate_json_key_and_config_constraints() -> None:
    source = github_source(max_body_bytes=100)
    raw = json.dumps(github_event()).encode()
    with pytest.raises(ProviderError, match="body_too_large"):
        parse_event(source, gh_headers(raw), raw)
    raw = b'{"action":"created","action":"deleted"}'
    with pytest.raises(ProviderError, match="invalid"):
        parse_event(github_source(), gh_headers(raw), raw)
    with pytest.raises(ValidationError):
        github_source(outbound_token_env="actual token")
    with pytest.raises(ValidationError):
        Source(source_id="x", kind="github", repository="owner/../evil", webhook_secret_env="SECRET")


async def authorize() -> None:
    return None


def test_outbound_ack_and_post_token_authorization() -> None:
    requests: list[httpx.Request] = []
    authorized = False

    async def check() -> None:
        nonlocal authorized
        authorized = True

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "tenant_access_token" in str(request.url):
            assert not authorized
            return httpx.Response(200, json={"code": 0, "tenant_access_token": "t"})
        assert authorized
        return httpx.Response(200, json={"code": 0, "data": {"message_id": "sent-1"}})

    async def run() -> None:
        client = ProviderClient(transport=httpx.MockTransport(handler))
        ack = await client.send(
            feishu_source(app_secret_env="FS_APP_SECRET"),
            "feishu:chat-1:root-1",
            "summary",
            "delivery1",
            authorize=check,
        )
        assert ack.status == "acknowledged" and ack.remote_id == "sent-1"

    asyncio.run(run())
    assert len(requests) == 2 and requests[-1].url.host == "open.feishu.cn"


@pytest.mark.parametrize("status,expected", [(429, "unsent"), (401, "blocked"), (503, "uncertain")])
def test_http_failures_do_not_blind_retry(status: int, expected: str) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, headers={"Retry-After": "3"}, json={"message": "error"})

    async def run() -> None:
        ack = await ProviderClient(transport=httpx.MockTransport(handler)).send(
            github_source(outbound_token_env="GH_TOKEN"), "github:owner/repo:3", "summary", "d1", authorize=authorize
        )
        assert ack.status == expected
        if status == 429:
            assert ack.retry_after == 3

    asyncio.run(run())
    assert calls == 1


def test_transport_unknown_connect_unsent_and_deadline() -> None:
    async def run(error: Exception, expected: str) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise error

        ack = await ProviderClient(transport=httpx.MockTransport(handler)).send(
            github_source(outbound_token_env="GH_TOKEN"), "github:owner/repo:3", "summary", "d1", authorize=authorize
        )
        assert ack.status == expected

    asyncio.run(run(httpx.ConnectError("before request"), "unsent"))
    asyncio.run(run(httpx.ReadTimeout("ACK lost"), "uncertain"))

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for chunk in [b'{"id":', b"1}"]:
                await asyncio.sleep(0.04)
                yield chunk

    async def slow(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=SlowStream())

    async def deadline() -> None:
        before = time.monotonic()
        ack = await ProviderClient(transport=httpx.MockTransport(slow)).send(
            github_source(outbound_token_env="GH_TOKEN", timeout_seconds=0.05),
            "github:owner/repo:3",
            "summary",
            "d1",
            authorize=authorize,
        )
        assert ack.status == "uncertain" and time.monotonic() - before < 0.2

    asyncio.run(deadline())


def test_reconcile_not_found_stays_unknown_and_never_posts() -> None:
    methods = []

    async def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, json=[])

    async def run() -> None:
        result = await ProviderClient(transport=httpx.MockTransport(handler)).reconcile(
            github_source(outbound_token_env="GH_TOKEN", outbound_principal="7"),
            "github:owner/repo:3",
            "d1",
            authorize=authorize,
        )
        assert result is None

    asyncio.run(run())
    assert methods == ["GET"]


def test_token_deadline_is_unsent_and_revocation_prevents_message() -> None:
    sent = []

    async def slow_token(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        await asyncio.sleep(0.1)
        return httpx.Response(200, json={"code": 0, "tenant_access_token": "token"})

    async def run() -> None:
        ack = await ProviderClient(transport=httpx.MockTransport(slow_token)).send(
            feishu_source(app_secret_env="FS_APP_SECRET", timeout_seconds=0.02),
            "feishu:chat-1:root-1",
            "summary",
            "d1",
            authorize=authorize,
        )
        assert ack.status == "unsent" and ack.error == "deadline_exceeded"

    asyncio.run(run())
    assert len(sent) == 1 and "tenant_access_token" in sent[0]
    sent.clear()

    async def token(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        return httpx.Response(200, json={"code": 0, "tenant_access_token": "token"})

    async def revoked() -> None:
        raise PermissionError("revoked")

    async def reject() -> None:
        ack = await ProviderClient(transport=httpx.MockTransport(token)).send(
            feishu_source(app_secret_env="FS_APP_SECRET"), "feishu:chat-1:root-1", "summary", "d1", authorize=revoked
        )
        assert ack.status == "blocked" and ack.error == "authorization_revoked"

    asyncio.run(reject())
    assert len(sent) == 1 and "tenant_access_token" in sent[0]


def test_correlation_requires_expected_remote_author() -> None:
    async def run(author: int, expected: bool) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "GET"
            return httpx.Response(
                200, json=[{"id": 101, "body": "summary\n<!-- praxis-delivery:d1 -->", "user": {"id": author}}]
            )

        ack = await ProviderClient(transport=httpx.MockTransport(handler)).reconcile(
            github_source(outbound_token_env="GH_TOKEN", outbound_principal="7"),
            "github:owner/repo:3",
            "d1",
            authorize=authorize,
        )
        assert (ack is not None) == expected

    asyncio.run(run(8, False))
    asyncio.run(run(7, True))


def test_feishu_unsigned_challenge_cannot_become_unsigned_event() -> None:
    raw = json.dumps({"type": "url_verification", "token": "verify", "challenge": "challenge-1"}).encode()
    assert parse_event(feishu_source(), {}, raw) == {"challenge": "challenge-1"}
    raw = json.dumps(fs_event("/cancel")).encode()
    with pytest.raises(ProviderError, match="unauthenticated"):
        parse_event(feishu_source(), {}, raw)


def test_github_review_is_evidence_even_when_human_requests_cancel() -> None:
    payload = {
        "action": "submitted",
        "repository": {"full_name": "owner/repo"},
        "sender": {"id": 42, "type": "User"},
        "pull_request": {"number": 3, "head": {"sha": "a" * 40}},
        "review": {"id": 99, "body": "/cancel", "commit_id": "a" * 40, "state": "changes_requested"},
    }
    raw = json.dumps(payload).encode()
    event = parse_event(github_source(), gh_headers(raw, event="pull_request_review"), raw)
    assert isinstance(event, Envelope) and event.operation == "evidence" and event.action is None
    assert event.evidence["review_id"] == "99"


def test_malformed_authenticated_payload_is_sanitized() -> None:
    payload = github_event()
    payload["repository"] = {"full_name": 123}
    raw = json.dumps(payload).encode()
    with pytest.raises(ProviderError, match="invalid"):
        parse_event(github_source(), gh_headers(raw), raw)


def test_bot_control_text_is_preserved_as_evidence() -> None:
    raw = json.dumps(github_event("/cancel", sender_type="Bot")).encode()
    event = parse_event(github_source(), gh_headers(raw), raw)
    assert isinstance(event, Envelope) and event.operation == "evidence" and event.text == "/cancel"


def test_response_size_and_malformed_success_preserve_unknown_effect() -> None:
    async def run(response: httpx.Response, expected: str) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return response

        ack = await ProviderClient(transport=httpx.MockTransport(handler)).send(
            github_source(outbound_token_env="GH_TOKEN", max_response_bytes=50),
            "github:owner/repo:3",
            "summary",
            "d1",
            authorize=authorize,
        )
        assert ack.status == expected

    asyncio.run(run(httpx.Response(201, content=b"x" * 100), "uncertain"))
    asyncio.run(run(httpx.Response(201, json={"no_id": "missing"}), "uncertain"))


def test_fixed_destination_and_missing_credentials_do_not_make_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(201, json={"id": 101})

    async def run(destination: str, configured: bool) -> None:
        source = github_source(outbound_token_env="GH_TOKEN" if configured else None)
        ack = await ProviderClient(transport=httpx.MockTransport(handler)).send(
            source, destination, "summary", "d1", authorize=authorize
        )
        assert ack.status == "blocked"

    asyncio.run(run("https://attacker.invalid/", True))
    asyncio.run(run("github:owner/repo:3/../../other", True))
    monkeypatch.delenv("GH_TOKEN")
    asyncio.run(run("github:owner/repo:3", True))
    assert calls == []


@pytest.mark.anyio
async def test_ci_poll_pins_head_app_and_complete_check_set(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "token")
    source = github_source().model_copy(update={"outbound_token_env": "GH_TOKEN"})
    calls = []

    async def remote(request):
        calls.append(str(request.url))
        if "/pulls/" in request.url.path:
            return httpx.Response(200, json={"head": {"sha": "head2"}})
        return httpx.Response(
            200,
            json={
                "total_count": 1,
                "check_runs": [
                    {
                        "id": 9,
                        "app": {"id": 5},
                        "name": "unit",
                        "head_sha": "head2",
                        "status": "completed",
                        "conclusion": "success",
                    }
                ],
            },
        )

    async def authorized():
        pass

    snapshot = await ProviderClient(transport=httpx.MockTransport(remote)).poll_ci(source, 3, authorize=authorized)
    assert snapshot["head_sha"] == "head2"
    assert snapshot["checks"] == [{"id": "5:unit", "state": "pass", "run": 9}]
    assert len(calls) == 2


@pytest.mark.anyio
async def test_feishu_provider_uuid_is_stable_and_bounded_for_full_outbox_id():
    from uuid import UUID

    keys = []

    async def remote(request):
        if request.url.path.endswith("/tenant_access_token/internal"):
            return httpx.Response(200, json={"code": 0, "tenant_access_token": "token"})
        key = json.loads(request.content)["uuid"]
        UUID(key)
        assert len(key) <= 50
        keys.append(key)
        return httpx.Response(200, json={"code": 0, "data": {"message_id": "message1"}})

    async def authorize():
        pass

    client = ProviderClient(transport=httpx.MockTransport(remote))
    source = feishu_source(app_secret_env="FS_APP_SECRET")
    for _ in range(2):
        assert (
            await client.send(source, "feishu:chat-1:root-1", "status", "a" * 64, authorize=authorize)
        ).status == "acknowledged"
    assert keys[0] == keys[1]
