from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from agent_runtime.core.llm_config import AgentModelsConfig, ModelProvider, ModelSpec
from agent_runtime.core.llm_registry import ModelRegistry
from agent_runtime.model_definition import ModelExecutionDefinition
from agent_runtime.model_probe import ModelProbe, ModelProbeError, ProbeLevel


@dataclass
class _ProviderState:
    mode: str = "ok"
    advertised_model: str = "probe-model"
    requests: list[dict[str, Any]] = field(default_factory=list)
    model_list_requests: int = 0
    stream_started: threading.Event = field(default_factory=threading.Event)
    release_stream: threading.Event = field(default_factory=threading.Event)


class _ProviderServer(ThreadingHTTPServer):
    state: _ProviderState


class _Handler(BaseHTTPRequestHandler):
    server: _ProviderServer

    def log_message(self, _format: str, *args: object) -> None:
        del args

    def do_GET(self) -> None:
        if self.path != "/v1/models":
            self.send_error(404)
            return
        self.server.state.model_list_requests += 1
        if not self._authorized():
            self._json_response(401, {"error": {"message": "bad key"}})
            return
        if self.server.state.mode == "timeout":
            time.sleep(0.25)
        self._json_response(
            200,
            {
                "object": "list",
                "data": [
                    {
                        "id": self.server.state.advertised_model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "probe",
                    }
                ],
            },
        )

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        if not self._authorized():
            self._json_response(401, {"error": {"message": "bad key"}})
            return
        size = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(size))
        self.server.state.requests.append(body)
        if body.get("stream"):
            self._stream_response(body)
            return
        content = "{\"ok\":true}" if "JSON schema" in str(body.get("messages")) else "probe"
        self._json_response(
            200,
            {
                "id": "chatcmpl-probe",
                "object": "chat.completion",
                "created": 0,
                "model": "probe-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    def _stream_response(self, body: dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.server.state.stream_started.set()
        if self.server.state.mode == "blocked_stream":
            self.server.state.release_stream.wait(timeout=5)
            return
        if self.server.state.mode == "malformed_stream":
            self._sse("{not-json")
            return
        if body.get("tools"):
            if self.server.state.mode == "missing_tool":
                self._chunk(delta={"content": "no tool"})
                self._chunk(delta={}, finish_reason="stop")
            else:
                self._chunk(
                    delta={
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-probe",
                                "type": "function",
                                "function": {
                                    "name": "probe_capability",
                                    "arguments": (
                                        "{\"ok\":false}"
                                        if self.server.state.mode == "invalid_tool"
                                        else "{\"ok\":true}"
                                    ),
                                },
                            }
                        ]
                    }
                )
                self._chunk(delta={}, finish_reason="tool_calls")
        else:
            self._chunk(delta={"role": "assistant", "content": "pro"})
            self._chunk(delta={"content": "be"})
            self._chunk(delta={}, finish_reason="stop")
        self._sse("[DONE]")

    def _chunk(
        self,
        *,
        delta: dict[str, Any],
        finish_reason: str | None = None,
    ) -> None:
        self._sse(
            json.dumps(
                {
                    "id": "chatcmpl-probe",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "probe-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
            )
        )

    def _sse(self, data: str) -> None:
        try:
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _authorized(self) -> bool:
        return self.headers.get("authorization") == "Bearer probe-secret"

    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode()
        try:
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.fixture
def provider_server() -> tuple[_ProviderState, str]:
    state = _ProviderState()
    server = _ProviderServer(("127.0.0.1", 0), _Handler)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield state, f"http://{host}:{port}/v1"
    finally:
        state.release_stream.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _probe(
    base_url: str,
    *,
    timeout_seconds: float = 1.0,
    max_output_tokens: int | None = None,
) -> tuple[ModelProbe, ModelExecutionDefinition]:
    config = AgentModelsConfig(
        models={
            "probe": ModelSpec(
                provider=ModelProvider.OPENAI_COMPATIBLE,
                provider_name="probe-provider",
                model="probe-model",
                tokenizer_model="probe-model",
                context_window_tokens=4_096,
                max_context_window_tokens=65_536,
                max_output_tokens=max_output_tokens,
                timeout_seconds=timeout_seconds,
                base_url=base_url,
                api_key_env="PROBE_API_KEY",
                supports_tools=True,
                supports_structured_output=True,
            )
        },
        default_model="probe",
    )

    registry = ModelRegistry(config)

    return (
        ModelProbe(registry),
        registry.get_model_definition("probe"),
    )
async def _run_probe(
    configured_probe: tuple[ModelProbe, ModelExecutionDefinition],
    *,
    level: ProbeLevel,
):
    probe, definition = configured_probe
    return await probe.run(definition, level=level)


@pytest.mark.anyio
@pytest.mark.parametrize(
    (
        "max_output_tokens",
        "expected_probe_limit",
    ),
    [
        (None, 32),
        (16, 16),
        (32_768, 32),
    ],
)
async def test_probe_output_limit_is_probe_policy_not_model_capability(
    provider_server: tuple[_ProviderState, str],
    monkeypatch: pytest.MonkeyPatch,
    max_output_tokens: int | None,
    expected_probe_limit: int,
) -> None:
    state, base_url = provider_server
    monkeypatch.setenv(
        "PROBE_API_KEY",
        "probe-secret",
    )

    evidence = await _run_probe(
        _probe(
            base_url,
            max_output_tokens=max_output_tokens,
        ),
        level=ProbeLevel.FULL,
    )

    assert evidence.completion_ok is True
    assert evidence.tool_call_ok is True
    assert evidence.structured_output_ok is True

    assert state.requests

    assert all(
        request["max_tokens"]
        == expected_probe_limit
        for request in state.requests
    )

@pytest.mark.anyio
async def test_connectivity_probe_verifies_advertised_model(
    provider_server: tuple[_ProviderState, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, base_url = provider_server
    monkeypatch.setenv("PROBE_API_KEY", "probe-secret")

    evidence = await _run_probe(_probe(base_url), level=ProbeLevel.CONNECTIVITY)

    assert evidence.connectivity_ok is True
    assert evidence.text_delta_count == 0
    assert evidence.completion_ok is False

    state.advertised_model = "different-model"
    with pytest.raises(ModelProbeError, match="model_identity"):
        await _run_probe(_probe(base_url), level=ProbeLevel.CONNECTIVITY)


@pytest.mark.anyio
async def test_stream_probe_observes_real_deltas_and_authoritative_completion(
    provider_server: tuple[_ProviderState, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, base_url = provider_server
    monkeypatch.setenv("PROBE_API_KEY", "probe-secret")

    evidence = await _run_probe(_probe(base_url), level=ProbeLevel.STREAM)

    assert evidence.connectivity_ok is True
    assert evidence.text_delta_count == 2
    assert evidence.completion_ok is True
    assert evidence.tool_call_ok is None


@pytest.mark.anyio
async def test_full_probe_forces_harmless_tool_without_executing_it_and_checks_structured_output(
    provider_server: tuple[_ProviderState, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, base_url = provider_server
    monkeypatch.setenv("PROBE_API_KEY", "probe-secret")

    evidence = await _run_probe(_probe(base_url), level=ProbeLevel.FULL)

    assert evidence.tool_call_ok is True
    assert evidence.structured_output_ok is True
    tool_request = next(request for request in state.requests if request.get("tools"))
    assert tool_request["tool_choice"] == {
        "type": "function",
        "function": {"name": "probe_capability"},
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    "mode, phase",
    [
        ("malformed_stream", "stream"),
        ("missing_tool", "tool_call"),
        ("invalid_tool", "tool_call"),
    ],
)
async def test_probe_rejects_malformed_stream_and_missing_forced_tool(
    provider_server: tuple[_ProviderState, str],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    phase: str,
) -> None:
    state, base_url = provider_server
    state.mode = mode
    monkeypatch.setenv("PROBE_API_KEY", "probe-secret")

    with pytest.raises(ModelProbeError, match=phase):
        await _run_probe(
            _probe(base_url),
            level=ProbeLevel.STREAM if phase == "stream" else ProbeLevel.FULL,
        )


@pytest.mark.anyio
async def test_probe_redacts_credentials_and_provider_exception_chain(
    provider_server: tuple[_ProviderState, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, base_url = provider_server
    secret = "do-not-leak-this-secret"
    monkeypatch.setenv("PROBE_API_KEY", secret)

    with pytest.raises(ModelProbeError) as caught:
        await _run_probe(_probe(base_url), level=ProbeLevel.CONNECTIVITY)

    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.anyio
async def test_probe_rejects_missing_credential_before_provider_io(
    provider_server: tuple[_ProviderState, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, base_url = provider_server
    monkeypatch.delenv("PROBE_API_KEY", raising=False)

    with pytest.raises(ModelProbeError, match="authentication.*PROBE_API_KEY"):
        await _run_probe(_probe(base_url), level=ProbeLevel.CONNECTIVITY)

    assert state.model_list_requests == 0
    assert state.requests == []


@pytest.mark.anyio
async def test_probe_honors_definition_timeout(
    provider_server: tuple[_ProviderState, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, base_url = provider_server
    state.mode = "timeout"
    monkeypatch.setenv("PROBE_API_KEY", "probe-secret")

    with pytest.raises(ModelProbeError, match="connectivity"):
        await _run_probe(
            _probe(base_url, timeout_seconds=0.05),
            level=ProbeLevel.CONNECTIVITY,
        )


@pytest.mark.anyio
async def test_probe_cancellation_closes_the_provider_stream_promptly(
    provider_server: tuple[_ProviderState, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, base_url = provider_server
    state.mode = "blocked_stream"
    monkeypatch.setenv("PROBE_API_KEY", "probe-secret")
    task = asyncio.create_task(_run_probe(_probe(base_url), level=ProbeLevel.STREAM))
    assert await asyncio.to_thread(state.stream_started.wait, 2)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.5)
