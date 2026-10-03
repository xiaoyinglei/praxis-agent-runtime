"""Fixed-route cloud model relay for a separately supervised, non-sudo OS user.

Provider keys are read only from systemd credentials. The Agent has a revocable
proxy token, not provider keys. Same-host root compromise is outside this boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import math
import os
import sqlite3
import stat
import struct
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

# Administrative allowlist, never derived from a request or workspace YAML.
_ROUTES = {
    "deepseek-flash": ("deepseek", "https://api.deepseek.com/v1/chat/completions"),
    "openai/gpt-oss-120b": ("groq", "https://api.groq.com/openai/v1/chat/completions"),
}
_PROVIDERS = frozenset(provider for provider, _ in _ROUTES.values())
_FIELDS = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "stream",
        "stream_options",
        "temperature",
        "top_p",
        "stop",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "seed",
        "response_format",
        "frequency_penalty",
        "presence_penalty",
        "thinking",
        "reasoning_effort",
        "n",
    }
)


@dataclass(frozen=True)
class ServicePolicy:
    daily_calls: int = 200
    daily_output_tokens: int = 200_000
    max_tokens: int = 4096
    max_body_bytes: int = 262_144
    max_response_bytes: int = 8_388_608
    max_concurrent: int = 2
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name == "timeout_seconds":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError("invalid timeout_seconds")
            elif type(value) is not int or value <= 0:
                raise ValueError(f"invalid {name}")


class DailyBudget:
    """Durable pessimistic reservations. Failed calls stay charged; no refunds.

    A single UTC-day high-water mark prevents clock rollback reopening old windows.
    Global across proxy-token rotations and processes sharing the state file.
    """

    def __init__(self, path: Path, policy: ServicePolicy) -> None:
        self.path = path
        self.policy = policy
        # StateDirectory must already exist and belong only to the service user.
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            with os.fdopen(fd, "rb") as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise ValueError("insecure budget state")
                os.fchmod(handle.fileno(), 0o600)
        except OSError:
            raise ValueError("insecure budget state") from None
        with sqlite3.connect(path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY CHECK (id=1), "
                "day INTEGER NOT NULL, calls INTEGER NOT NULL, output_tokens INTEGER NOT NULL)"
            )
        path.chmod(0o600)

    def reserve(self, tokens: int, *, day: int | None = None) -> bool:
        current_day = int(time.time() // 86400) if day is None else day
        with sqlite3.connect(self.path, timeout=5) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT day,calls,output_tokens FROM budget WHERE id=1").fetchone()
            if row is not None and current_day < row[0]:
                return False
            calls, output_tokens = (0, 0) if row is None or current_day > row[0] else (row[1], row[2])
            if calls >= self.policy.daily_calls or output_tokens + tokens > self.policy.daily_output_tokens:
                return False
            db.execute(
                "INSERT OR REPLACE INTO budget VALUES (1,?,?,?)", (current_day, calls + 1, output_tokens + tokens)
            )
        return True


def _valid_secret(value: str) -> bool:
    return 16 <= len(value) <= 4096 and value.isascii() and value.isprintable() and not any(c.isspace() for c in value)


def private_systemd_acl(acl: bytes, uid: int) -> bool:
    """systemd >=254 uses root-owned 0440 files with a service-user read ACL.

    The group mode bits are an ACL mask, not a grant to the owning group.
    Accept only the exact root/service-only ACL; no extra users or groups.
    """
    if len(acl) != 44 or struct.unpack_from("<I", acl)[0] != 2:
        return False
    entries = set(struct.iter_unpack("<HHI", acl[4:]))
    return entries == {
        (1, 4, 0xFFFFFFFF),  # owner (root): read
        (2, 4, uid),  # named service user: read
        (4, 0, 0xFFFFFFFF),  # owning group: no access
        (16, 4, 0xFFFFFFFF),  # ACL mask: read
        (32, 0, 0xFFFFFFFF),  # everyone else: no access
    }


def load_credentials(providers: Sequence[str]) -> tuple[dict[str, str], str]:
    """No .env/environment key fallback. Never include file contents in errors."""
    directory = os.environ.get("CREDENTIALS_DIRECTORY")
    if not directory or not providers or set(providers) - _PROVIDERS:
        raise ValueError("invalid credential configuration")

    def read(name: str) -> str:
        try:
            fd = os.open(Path(directory) / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("insecure credential file")
                if info.st_mode & 0o077:
                    if (
                        stat.S_IMODE(info.st_mode) != 0o440
                        or info.st_uid != 0
                        or not private_systemd_acl(os.getxattr(handle.fileno(), "system.posix_acl_access"), os.getuid())
                    ):
                        raise ValueError("insecure credential ACL")
                value = handle.read(4098).decode("ascii").strip()
                if not _valid_secret(value):
                    raise ValueError("invalid credential value")
                return value
        except (OSError, UnicodeError, ValueError):
            raise ValueError("missing or insecure credential") from None

    return {provider: read(f"{provider}-key") for provider in providers}, read("agent-token")


def create_app(
    keys: Mapping[str, str],
    proxy_token: str,
    state_path: Path,
    *,
    policy: ServicePolicy | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Starlette:
    policy = policy or ServicePolicy()
    keys = dict(keys)
    if not keys or set(keys) - _PROVIDERS or not all(_valid_secret(v) for v in (*keys.values(), proxy_token)):
        raise ValueError("invalid credential configuration")
    if proxy_token in keys.values():
        raise ValueError("proxy credential must differ from provider keys")
    routes = {model: route for model, route in _ROUTES.items() if route[0] in keys}
    budget = DailyBudget(state_path, policy)
    client = httpx.AsyncClient(
        transport=transport,
        trust_env=False,
        follow_redirects=False,
        timeout=policy.timeout_seconds,
        limits=httpx.Limits(max_connections=policy.max_concurrent),
    )
    active = 0

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        yield
        await client.aclose()

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse({"error": {"message": message}}, status_code=status, headers={"Cache-Control": "no-store"})

    async def relay(request: Request) -> Response:
        nonlocal active
        if request.headers.get("origin") is not None:
            return error(403, "browser requests disabled")
        auth = request.headers.getlist("authorization")
        if len(auth) != 1 or not hmac.compare_digest(auth[0].encode(), f"Bearer {proxy_token}".encode()):
            return error(401, "invalid proxy credential")
        if request.url.query:
            return error(400, "query parameters disabled")
        if request.method == "GET":
            return JSONResponse(
                {
                    "object": "list",
                    "data": [
                        {"id": model, "object": "model", "created": 0, "owned_by": "praxis"} for model in sorted(routes)
                    ],
                },
                headers={"Cache-Control": "no-store"},
            )
        if active >= policy.max_concurrent:
            return error(429, "concurrency limit reached")
        active += 1
        try:
            async with asyncio.timeout(policy.timeout_seconds):
                body = bytearray()
                async with asyncio.timeout(min(10, policy.timeout_seconds)):
                    async for chunk in request.stream():
                        if len(body) + len(chunk) > policy.max_body_bytes:
                            return error(413, "request too large")
                        body.extend(chunk)
                try:
                    payload = json.loads(body)
                except (ValueError, UnicodeError, RecursionError):
                    return error(400, "invalid JSON")
                if not isinstance(payload, dict):
                    return error(400, "expected object")
                model = payload.get("model")
                if set(payload) - _FIELDS or not isinstance(model, str) or model not in routes:
                    return error(400, "unsupported model or parameter")
                tokens = payload.get("max_tokens", policy.max_tokens)
                if type(tokens) is not int or not 0 < tokens <= policy.max_tokens or payload.get("n", 1) != 1:
                    return error(400, "invalid output limit")
                if type(payload.get("stream", False)) is not bool:
                    return error(400, "invalid stream flag")
                messages = payload.get("messages")
                if (
                    not isinstance(messages, list)
                    or not messages
                    or any(
                        not isinstance(m, dict) or not isinstance(m.get("content", ""), (str, type(None)))
                        for m in messages
                    )
                ):
                    return error(400, "only text and tool conversations supported")
                payload["max_tokens"] = tokens
                # SQLite work does not block the event loop or body-admission timeout.
                if not await asyncio.to_thread(budget.reserve, tokens):
                    return error(429, "daily usage limit reached")
                provider, url = routes[model]
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {keys[provider]}", "Accept-Encoding": "identity"},
                ) as upstream:
                    if (
                        upstream.status_code != 200
                        or upstream.headers.get("content-encoding", "identity") != "identity"
                    ):
                        return error(502, "provider request failed")
                    output = bytearray()
                    async for chunk in upstream.aiter_bytes():
                        if len(output) + len(chunk) > policy.max_response_bytes:
                            return error(502, "provider response exceeded limit")
                        output.extend(chunk)
                if contains_credential(bytes(output), (*keys.values(), proxy_token)):
                    return error(502, "provider response rejected")
                media = "text/event-stream" if payload.get("stream") else "application/json"
                return Response(bytes(output), media_type=media, headers={"Cache-Control": "no-store"})
        except (httpx.HTTPError, TimeoutError, ValueError, RecursionError, sqlite3.Error):
            return error(502, "model service request failed")
        finally:
            active -= 1

    app = Starlette(
        routes=[Route("/v1/models", relay, methods=["GET"]), Route("/v1/chat/completions", relay, methods=["POST"])],
        lifespan=lifespan,
    )
    app.router.redirect_slashes = False
    return app


def contains_credential(output: bytes, credentials: tuple[str, ...]) -> bool:
    """Defense in depth for literal/JSON-escaped/SSE-fragmented echoes, not arbitrary encodings."""
    text = output.decode("utf-8", errors="replace")
    strings: list[str] = []
    fields: dict[tuple[str, ...], list[str]] = {}

    def collect(value: object, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, str):
            strings.append(value)
            fields.setdefault(path, []).append(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                collect(item, (*path, str(key)))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                collect(item, (*path, str(index)))

    try:
        collect(json.loads(text))
    except ValueError:
        for line in text.splitlines():
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                try:
                    collect(json.loads(line[5:]))
                except ValueError:
                    pass
    decoded = [text, "".join(strings), *("".join(parts) for parts in fields.values())]
    return any(key in value for key in credentials for value in decoded)


def main() -> None:
    import resource

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", action="append", choices=sorted(_PROVIDERS))
    parser.add_argument("--state-dir", type=Path, default=Path("/var/lib/praxis-model"))
    parser.add_argument("--daily-calls", type=int, default=200)
    parser.add_argument("--daily-output-tokens", type=int, default=200_000)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    if os.geteuid() == 0:
        parser.error("run as the dedicated non-root model-service user")
    try:
        keys, token = load_credentials(args.provider or sorted(_PROVIDERS))
        policy = ServicePolicy(daily_calls=args.daily_calls, daily_output_tokens=args.daily_output_tokens)
        app = create_app(keys, token, args.state_dir / "quota.sqlite3", policy=policy)
        # Fixed loopback-only listener; no caller-selectable bind address/workers.
        uvicorn.run(app, host="127.0.0.1", port=18444, workers=1, access_log=False, log_level="critical")
    except (OSError, ValueError, sqlite3.Error):
        parser.exit(1, "Model service failed to start; check private credentials, state directory and loopback port.\n")


if __name__ == "__main__":
    main()
