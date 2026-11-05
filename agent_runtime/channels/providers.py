"""Authenticated native envelopes and bounded provider I/O, without runtime authority.

GitHub HMAC authenticates the body, not delivery headers or a timestamp. Persist
both source/event-id conflicts AND source/canonical-digest replay aliases. Feishu
also authenticates a bounded timestamp. Neither parser grants workspace access.
Protocol references: docs.github.com/en/webhooks/using-webhooks/validating-webhook-deliveries
and open.feishu.cn/document/server-docs/event-subscription-guide/event-subscription-configure-/
encrypt-key-encryption-configuration-case (AES format follows larksuite/node-sdk).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import math
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Literal, Self
from uuid import NAMESPACE_URL, uuid5

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,256}\Z")
_ENV = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_REPOSITORY = re.compile(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+\Z")
_OPERATION = Literal["input", "status", "respond", "cancel", "evidence"]


class Source(BaseModel):
    """Local provider configuration. Secrets are references, never serialized values."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    source_id: str
    kind: Literal["github", "feishu"]
    repository: str | None = None
    chats: tuple[str, ...] = ()
    app_id: str | None = None
    bot_open_id: str | None = None
    webhook_secret_env: str | None = None
    outbound_token_env: str | None = None
    outbound_principal: str | None = None
    app_secret_env: str | None = None
    encrypt_key_env: str | None = None
    verification_token_env: str | None = None
    enabled: bool = True
    max_body_bytes: int = Field(default=1024 * 1024, ge=1, le=8 * 1024 * 1024)
    max_response_bytes: int = Field(default=1024 * 1024, ge=1, le=8 * 1024 * 1024)
    timeout_seconds: float = Field(default=10, gt=0, le=60, allow_inf_nan=False)
    max_clock_skew_seconds: int = Field(default=300, ge=1, le=3600)

    @field_validator("source_id", "app_id", "bot_open_id", "outbound_principal")
    @classmethod
    def identifiers(cls, value: str | None) -> str | None:
        if value is not None and not _IDENTIFIER.fullmatch(value):
            raise ValueError("invalid provider identifier")
        return value

    @field_validator(
        "webhook_secret_env", "outbound_token_env", "app_secret_env", "encrypt_key_env", "verification_token_env"
    )
    @classmethod
    def references(cls, value: str | None) -> str | None:
        if value is not None and not _ENV.fullmatch(value):
            raise ValueError("credentials require environment references")
        return value

    @model_validator(mode="after")
    def configuration(self) -> Self:
        if self.kind == "github":
            if (
                not self.repository
                or not _REPOSITORY.fullmatch(self.repository)
                or ".." in self.repository
                or not self.webhook_secret_env
            ):
                raise ValueError("GitHub repository and webhook reference required")
            if any((self.app_id, self.chats, self.app_secret_env, self.encrypt_key_env, self.verification_token_env)):
                raise ValueError("Feishu fields on GitHub source")
            if self.outbound_principal is not None and not self.outbound_principal.isdecimal():
                raise ValueError("GitHub outbound principal requires a numeric ID")
        else:
            if not self.app_id or not self.encrypt_key_env or not self.verification_token_env or not self.chats:
                raise ValueError("Feishu app, chats and authentication references required")
            if any(not _IDENTIFIER.fullmatch(chat) for chat in self.chats):
                raise ValueError("invalid Feishu chat identifier")
            if self.repository or self.webhook_secret_env or self.outbound_token_env:
                raise ValueError("GitHub fields on Feishu source")
        return self


class Envelope(BaseModel):
    """Authenticated content only; canonical runtime resolves all authority and binding."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    source_id: str
    event_id: str
    digest: str
    native_id: str = ""
    principal: str
    principal_kind: Literal["user", "bot"] = "user"
    conversation_id: str
    destination: str
    operation: _OPERATION = "input"
    text: str = ""
    request_id: str | None = None
    action: str | None = None
    attachments: tuple[str, ...] = ()
    evidence: dict[str, Any] = Field(default_factory=dict)


class DeliveryAck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    status: Literal["acknowledged", "unsent", "uncertain", "blocked"]
    remote_id: str | None = None
    retry_after: float | None = None
    error: str | None = None


class ProviderError(ValueError):
    """Sanitized diagnostic; never includes request bodies or credentials."""

    def __init__(
        self, reason: str, *, uncertain: bool = False, retryable: bool = False, retry_after: float | None = None
    ) -> None:
        super().__init__(reason)
        self.reason, self.uncertain, self.retryable = reason, uncertain, retryable
        self.retry_after = retry_after


def _secret(reference: str | None) -> str:
    value = os.environ.get(reference or "")
    if not value:
        raise ProviderError("credential_unconfigured")
    return value


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProviderError("invalid")
    return value


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProviderError("invalid")
        result[key] = value
    return result


def _load(raw: bytes) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError, RecursionError):
        raise ProviderError("invalid") from None


def _text(value: object, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not empty) or len(value) > 100000:
        raise ProviderError("invalid")
    return value


def _identifier(value: object) -> str:
    value = _text(value)
    if not _IDENTIFIER.fullmatch(value):
        raise ProviderError("invalid")
    return value


def _number(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ProviderError("invalid")
    return value


def _digest(payload: Mapping[str, Any], kind: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {"event": kind, "payload": payload},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def parse_command(text: str) -> tuple[_OPERATION, str, str | None, str | None]:
    """Only exact, unquoted command bodies grant a *requested* operation, never authority.

    `/respond REQUEST ACTION` may have continuation text. Ordinary text, code,
    quoted commands and malformed command bodies remain plain coding input.
    """
    if text in {"/status", "/cancel"}:
        return ("status" if text == "/status" else "cancel"), "", None, None
    match = re.fullmatch(r"/respond ([A-Za-z0-9_-]{1,256}) ([A-Za-z0-9_-]{1,64})(?:\n([\s\S]*))?", text)
    if match:
        return "respond", match[3] or "", match[1], match[2]
    if text.startswith("/input "):
        return "input", text[7:], None, None
    return "input", text, None, None


def _github(source: Source, headers: Mapping[str, str], raw: bytes) -> Envelope | None:
    expected = "sha256=" + hmac.new(_secret(source.webhook_secret_env).encode(), raw, "sha256").hexdigest()
    if not hmac.compare_digest(expected, headers.get("x-hub-signature-256", "")):
        raise ProviderError("unauthenticated")
    event_id = _identifier(headers.get("x-github-delivery"))
    payload = _object(_load(raw))
    repository = _text(_object(payload.get("repository")).get("full_name"))
    if repository.casefold() != (source.repository or "").casefold():
        raise ProviderError("forbidden")
    sender = _object(payload.get("sender"))
    if sender.get("type") not in {"User", "Bot"}:
        return None
    principal = str(_number(sender.get("id")))
    event, action = headers.get("x-github-event", ""), payload.get("action")
    evidence: dict[str, Any] = {"provider": "github", "repository": source.repository, "event": event, "action": action}
    content: dict[str, Any]
    evidence_only = sender["type"] == "Bot"
    if event == "issue_comment" and action == "created":
        issue, content = _object(payload.get("issue")), _object(payload.get("comment"))
        number = _number(issue.get("number"))
        text = _text(content.get("body"), empty=True)
        if _object(content.get("user")).get("id") != sender["id"]:
            raise ProviderError("forbidden")
        native_id = f"{event}:{_number(content.get('id'))}:{action}"
    elif event in {"issues", "pull_request"} and action == "opened":
        content = _object(payload.get("issue" if event == "issues" else "pull_request"))
        number = _number(content.get("number"))
        text = _text(content.get("title", ""), empty=True) + "\n" + _text(content.get("body") or "", empty=True)
        native_id = f"{event}:{_number(content.get('id'))}:{action}"
    elif event in {"pull_request_review", "pull_request_review_comment"} and action in {
        "submitted",
        "created",
        "dismissed",
    }:
        pull = _object(payload.get("pull_request"))
        number = _number(pull.get("number"))
        content = _object(payload.get("review" if event == "pull_request_review" else "comment"))
        text = _text(content.get("body") or "", empty=True)
        evidence_only = True
        native_id = f"{event}:{_number(content.get('id'))}:{action}"
        evidence.update(
            head_sha=_object(pull.get("head")).get("sha"),
            review_id=str(content["id"]),
            reviewer_id=str(_object(content.get("user", {})).get("id", principal)),
            submitted_at=content.get("submitted_at"),
            commit_id=content.get("commit_id"),
            path=content.get("path"),
            line=content.get("line"),
            state=content.get("state"),
        )
    elif event in {"check_run", "check_suite", "workflow_run"}:
        content = _object(payload.get(event))
        pulls = content.get("pull_requests")
        if not isinstance(pulls, list) or len(pulls) != 1:
            return None
        number = _number(_object(pulls[0]).get("number"))
        native_id = f"{event}:{_number(content.get('id'))}:{_text(action)}"
        evidence_only, text = True, ""
        evidence.update(
            head_sha=_text(content.get("head_sha")),
            check_id=str(content["id"]),
            check_identity=f"{content.get('app', {}).get('id')}:{content.get('name')}",
            name=content.get("name"),
            status=content.get("status"),
            conclusion=content.get("conclusion"),
        )
    elif event == "pull_request" and action in {"synchronize", "closed", "reopened"}:
        content = _object(payload.get("pull_request"))
        number = _number(content.get("number"))
        native_id = f"{event}:{_number(content.get('id'))}:{action}:{_text(_object(content.get('head')).get('sha'))}"
        evidence_only, text = True, ""
        evidence.update(head_sha=content["head"]["sha"], before_sha=payload.get("before"), merged=content.get("merged"))
    else:
        return None
    evidence["pull_request"] = number
    destination = f"github:{source.repository}:{number}"
    operation, text, request_id, requested_action = (
        ("evidence", text, None, None) if evidence_only else parse_command(text)
    )
    return Envelope(
        source_id=source.source_id,
        event_id=event_id,
        digest=_digest(payload, event),
        native_id=native_id,
        principal=principal,
        principal_kind="bot" if sender["type"] == "Bot" else "user",
        conversation_id=destination,
        destination=destination,
        operation="evidence" if evidence_only else operation,
        text=text,
        request_id=None if evidence_only else request_id,
        action=None if evidence_only else requested_action,
        evidence=evidence if evidence_only else {},
    )


def _decrypt(source: Source, payload: dict[str, Any]) -> dict[str, Any]:
    if "encrypt" not in payload:
        return payload
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    try:
        encoded = base64.b64decode(_text(payload["encrypt"]), validate=True)
        if len(encoded) < 32 or len(encoded) % 16:
            raise ValueError
        cipher = Cipher(
            algorithms.AES(hashlib.sha256(_secret(source.encrypt_key_env).encode()).digest()), modes.CBC(encoded[:16])
        ).decryptor()
        padded = cipher.update(encoded[16:]) + cipher.finalize()
        unpadder = padding.PKCS7(128).unpadder()
        return _object(_load(unpadder.update(padded) + unpadder.finalize()))
    except (ValueError, ProviderError):
        raise ProviderError("unauthenticated") from None


def _feishu(source: Source, headers: Mapping[str, str], raw: bytes, now: float) -> Envelope | dict[str, str] | None:
    timestamp, nonce = headers.get("x-lark-request-timestamp", ""), headers.get("x-lark-request-nonce", "")
    # Setup challenges are side-effect free. The provider may send them unsigned;
    # permit ONLY token-authenticated url_verification, never event execution.
    signed = bool(headers.get("x-lark-signature"))
    if signed:
        try:
            if not nonce or len(nonce) > 256 or abs(now - int(timestamp)) > source.max_clock_skew_seconds:
                raise ValueError
        except ValueError:
            raise ProviderError("unauthenticated") from None
        expected = hashlib.sha256(
            timestamp.encode() + nonce.encode() + _secret(source.encrypt_key_env).encode() + raw
        ).hexdigest()
        if not hmac.compare_digest(expected, headers["x-lark-signature"]):
            raise ProviderError("unauthenticated")
    payload = _decrypt(source, _object(_load(raw)))
    header = _object(payload.get("header", {}))
    challenge = payload.get("type") == "url_verification"
    token = payload.get("token") if challenge else header.get("token")
    if not isinstance(token, str) or not hmac.compare_digest(_secret(source.verification_token_env), token):
        raise ProviderError("unauthenticated")
    if challenge:
        return {"challenge": _text(payload.get("challenge"))}
    if not signed or header.get("app_id") != source.app_id:
        raise ProviderError("unauthenticated")
    if header.get("event_type") != "im.message.receive_v1":
        return None
    event = _object(payload.get("event"))
    sender, message = _object(event.get("sender")), _object(event.get("message"))
    if sender.get("sender_type") != "user":
        return None
    principal = _identifier(_object(sender.get("sender_id")).get("open_id"))
    if principal == source.bot_open_id:
        return None
    chat, message_id = _identifier(message.get("chat_id")), _identifier(message.get("message_id"))
    if chat not in source.chats:
        raise ProviderError("forbidden")
    content = _object(_load(_text(message.get("content")).encode()))
    attachments: tuple[str, ...] = ()
    if message.get("message_type") == "text":
        text = _text(content.get("text"), empty=True)
    elif message.get("message_type") in {"file", "image"}:
        kind = message["message_type"]
        attachments = (f"feishu:{kind}:{_identifier(content.get(kind + '_key'))}",)
        text = ""
    else:
        return None
    root = _identifier(message["root_id"]) if message.get("root_id") else message_id
    destination = f"feishu:{chat}:{root}"
    operation, text, request_id, action = parse_command(text)
    return Envelope(
        source_id=source.source_id,
        event_id=_identifier(header.get("event_id")),
        digest=_digest(payload, "im.message.receive_v1"),
        native_id=message_id,
        principal=principal,
        conversation_id=destination,
        destination=destination,
        operation=operation,
        text=text,
        request_id=request_id,
        action=action,
        attachments=attachments,
    )


def parse_event(
    source: Source, headers: Mapping[str, str], raw: bytes, *, now: float | None = None
) -> Envelope | dict[str, str] | None:
    """Bounded authentication/parsing; host must apply persistent replay and authorization checks."""
    if not source.enabled:
        raise ProviderError("source_disabled")
    if len(raw) > source.max_body_bytes:
        raise ProviderError("body_too_large")
    timestamp = time.time() if now is None else now
    if not math.isfinite(timestamp):
        raise ProviderError("invalid")
    normalized = {key.lower(): value for key, value in headers.items()}
    return _github(source, normalized, raw) if source.kind == "github" else _feishu(source, normalized, raw, timestamp)


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
        if not math.isfinite(seconds):
            return None
        return max(0.0, seconds)
    except ValueError:
        try:
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


class ProviderClient:
    """One attempt per call. The canonical outbox owns retry and uncertainty.

    No arbitrary origins, redirects, environmental proxies, transport retries or
    absence-based replay. A total deadline includes credentials, authorization,
    token requests and streamed response reads. In-flight writes cannot be recalled.
    """

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.transport = transport

    @staticmethod
    def _destination(source: Source, destination: str) -> tuple[str, str]:
        if not source.enabled:
            raise ProviderError("source_disabled")
        if source.kind == "github":
            prefix = f"github:{source.repository}:"
            number = destination.removeprefix(prefix)
            if not destination.startswith(prefix) or not re.fullmatch(r"[1-9][0-9]{0,18}", number):
                raise ProviderError("invalid_destination")
            return f"https://api.github.com/repos/{source.repository}/issues/{number}/comments", ""
        parts = destination.split(":")
        if len(parts) != 3 or parts[0] != "feishu" or parts[1] not in source.chats:
            raise ProviderError("invalid_destination")
        root = _identifier(parts[2])
        return f"https://open.feishu.cn/open-apis/im/v1/messages/{root}/reply", root

    @staticmethod
    def _marker(correlation: str) -> str:
        return f"<!-- praxis-delivery:{_identifier(correlation)} -->"

    @staticmethod
    def _headers(source: Source) -> dict[str, str]:
        return {
            "Authorization": "Bearer " + _secret(source.outbound_token_env),
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _request(
        self,
        source: Source,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: dict[str, Any] | None = None,
        write: bool = False,
    ) -> Any:
        async with httpx.AsyncClient(
            transport=self.transport,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(source.timeout_seconds),
        ) as client:
            try:
                async with client.stream(method, url, headers=headers, json=data) as response:
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > source.max_response_bytes:
                            raise ProviderError("response_too_large", uncertain=write)
                        chunks.append(chunk)
                    retry_after = _retry_after(response.headers.get("Retry-After"))
                    if response.status_code == 429:
                        raise ProviderError("rate_limited", retryable=True, retry_after=retry_after)
                    if response.status_code >= 500 or response.status_code == 408:
                        raise ProviderError("remote_server_error", uncertain=write, retryable=not write)
                    if not 200 <= response.status_code < 300:
                        raise ProviderError("remote_rejected")
                    try:
                        return _load(b"".join(chunks))
                    except ProviderError:
                        raise ProviderError("invalid_remote_response", uncertain=write) from None
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
                raise ProviderError("connection_unavailable", retryable=True) from None
            except httpx.HTTPError:
                raise ProviderError("transport_outcome_unknown", uncertain=write, retryable=not write) from None

    async def send(
        self, source: Source, destination: str, text: str, correlation: str, *, authorize: Callable[[], Awaitable[None]]
    ) -> DeliveryAck:
        write_started = False
        try:
            async with asyncio.timeout(source.timeout_seconds):
                url, _ = self._destination(source, destination)
                marker = self._marker(correlation)
                if len(text) > 24000:
                    raise ProviderError("text_too_large")
                if source.kind == "github":
                    if not source.outbound_principal:
                        raise ProviderError("outbound_identity_required")
                    headers = self._headers(source)
                    data: dict[str, Any] = {"body": text + "\n\n" + marker}
                else:
                    token = await self._request(
                        source,
                        "POST",
                        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
                        data={"app_id": source.app_id, "app_secret": _secret(source.app_secret_env)},
                    )
                    if (
                        not isinstance(token, dict)
                        or token.get("code") != 0
                        or not isinstance(token.get("tenant_access_token"), str)
                        or not token["tenant_access_token"]
                    ):
                        raise ProviderError("credential_rejected")
                    headers = {"Authorization": "Bearer " + token["tenant_access_token"]}
                    data = {
                        "msg_type": "text",
                        "content": json.dumps({"text": text}, ensure_ascii=False),
                        "uuid": str(uuid5(NAMESPACE_URL, f"praxis:{source.source_id}:{correlation}")),
                        "reply_in_thread": True,
                    }
                await authorize()  # last authorization check, after token acquisition
                write_started = True
                response = await self._request(source, "POST", url, headers=headers, data=data, write=True)
                if not isinstance(response, dict):
                    raise ProviderError("invalid_remote_response", uncertain=True)
                if source.kind == "github":
                    remote_id = str(_number(response.get("id")))
                else:
                    if response.get("code") != 0:
                        raise ProviderError("remote_rejected")
                    remote_id = _identifier(_object(response.get("data")).get("message_id"))
                return DeliveryAck(status="acknowledged", remote_id=remote_id)
        except TimeoutError:
            return DeliveryAck(status="uncertain" if write_started else "unsent", error="deadline_exceeded")
        except PermissionError:
            return DeliveryAck(status="blocked", error="authorization_revoked")
        except ProviderError as error:
            # A malformed successful write response may conceal an accepted effect.
            uncertain = error.uncertain or (write_started and error.reason == "invalid")
            return DeliveryAck(
                status="uncertain" if uncertain else "unsent" if error.retryable else "blocked",
                error=error.reason,
                retry_after=error.retry_after,
            )

    async def reconcile(
        self, source: Source, destination: str, correlation: str, *, authorize: Callable[[], Awaitable[None]]
    ) -> DeliveryAck | None:
        """Only positive, author-bound remote correlation proves acceptance. None stays unknown."""
        try:
            async with asyncio.timeout(source.timeout_seconds):
                url, _ = self._destination(source, destination)
                marker = self._marker(correlation)
                await authorize()
                if source.kind != "github" or not source.outbound_principal:
                    return None  # no native durable Feishu UUID lookup; never infer absence
                headers = self._headers(source)
                for page in range(1, 11):
                    await authorize()
                    comments = await self._request(source, "GET", f"{url}?per_page=100&page={page}", headers=headers)
                    if not isinstance(comments, list):
                        raise ProviderError("invalid_remote_response")
                    for comment in comments:
                        if (
                            isinstance(comment, dict)
                            and marker in str(comment.get("body", ""))
                            and isinstance(comment.get("user"), dict)
                            and str(comment["user"].get("id")) == source.outbound_principal
                        ):
                            return DeliveryAck(status="acknowledged", remote_id=str(_number(comment.get("id"))))
                    if len(comments) < 100:
                        return None
        except (TimeoutError, PermissionError, ProviderError):
            return None
        return None

    async def poll_ci(
        self, source: Source, pull_request: int, *, authorize: Callable[[], Awaitable[None]]
    ) -> dict[str, Any]:
        """One bounded snapshot of the current PR head and app-bound check runs."""
        if source.kind != "github" or type(pull_request) is not int or pull_request <= 0:
            raise ProviderError("invalid_ci_target")
        async with asyncio.timeout(source.timeout_seconds):
            headers = self._headers(source)
            base = f"https://api.github.com/repos/{source.repository}"
            await authorize()
            pull = _object(await self._request(source, "GET", f"{base}/pulls/{pull_request}", headers=headers))
            head = _text(_object(pull.get("head")).get("sha"))
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", head):
                raise ProviderError("invalid_remote_head")
            results: dict[str, dict[str, Any]] = {}
            count = 0
            for page in range(1, 11):
                await authorize()
                data = _object(
                    await self._request(
                        source, "GET", f"{base}/commits/{head}/check-runs?per_page=100&page={page}", headers=headers
                    )
                )
                runs = data.get("check_runs")
                if not isinstance(runs, list) or not isinstance(data.get("total_count"), int):
                    raise ProviderError("invalid_remote_checks")
                count += len(runs)
                for value in runs:
                    run = _object(value)
                    if run.get("head_sha") != head:
                        raise ProviderError("check_head_mismatch")
                    app_id = _number(_object(run.get("app")).get("id"))
                    name = _text(run.get("name"))
                    key = f"{app_id}:{name}"
                    run_id = _number(run.get("id"))
                    state = (
                        "pending"
                        if run.get("status") != "completed"
                        else "pass"
                        if run.get("conclusion") == "success"
                        else "fail"
                        if run.get("conclusion") in {"failure", "timed_out", "action_required", "startup_failure"}
                        else "cancel"
                        if run.get("conclusion") == "cancelled"
                        else "skipping"
                    )
                    if key not in results or results[key]["run"] < run_id:
                        results[key] = {"id": key, "state": state, "run": run_id}
                if count >= data["total_count"]:
                    return {"head_sha": head, "checks": list(results.values())}
                if len(runs) < 100:
                    raise ProviderError("incomplete_checks_snapshot")
            raise ProviderError("checks_snapshot_limit")
