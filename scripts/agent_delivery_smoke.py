#!/usr/bin/env python
"""Exercise the stable product agent path with live or deterministic models."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

if TYPE_CHECKING:
    from agent_runtime.core.model_request import ModelCallRecord
    from agent_runtime.tools.tool import Tool

DEFAULT_MODEL = "groq_gpt_oss_120b"
RESIDENT_TOOL_NAMES = (
    "search_text",
    "list_files",
    "read_file",
    "inspect_data_file",
    "apply_patch",
    "run_command",
    "execute_python",
    "update_plan",
)


@dataclass(frozen=True)
class SmokeCase:
    name: str
    task: str
    expected_answer_contains: tuple[str, ...] = ()
    expected_answer_exact: str | None = None
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    expected_initial_tools: tuple[str, ...] = ()
    expected_tool_errors: tuple[str, ...] = ()
    tools: tuple[str, ...] | None = None
    disabled_tools: tuple[str, ...] = ()
    allow_write_tools: bool = False
    allow_execute_tools: bool = False
    allow_discovery_tools: bool = False
    auto_approve: bool = False
    expect_origin_retained: bool = False
    install_hidden_mcp: bool = False
    provider: Literal["openai-compatible", "mlx", "ollama"] = "openai-compatible"
    workspace_files: Mapping[str, str] = field(default_factory=dict)
    workspace_assertions: Mapping[str, str] = field(default_factory=dict)
    public_agent: bool = False
    max_turns: int = 12


@dataclass(frozen=True)
class SmokeResult:
    name: str
    passed: bool
    status: str
    answer: str | None
    tools: tuple[str, ...]
    visible_tools: tuple[tuple[str, ...], ...]
    workspace_path: str | None
    error: str = ""
    stop_reason: str | None = None
    diagnostics: tuple[str, ...] = ()
    schema_bytes: int = 0
    tool_errors: tuple[str, ...] = ()
    request_ids: tuple[str, ...] = ()
    prompt_revisions: tuple[str, ...] = ()
    toolset_revisions: tuple[str, ...] = ()
    provider_wire_hashes: tuple[str, ...] = ()
    provider_wire_kind: str = ""
    serializer_revision: str = ""
    usage_source: str | None = None
    cache_read_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    approval_count: int = 0
    origin_toolset_revisions: tuple[str, ...] = ()
    origin_retained: bool | None = None
    workspace_assertions_passed: bool = True
    result_content_kinds: tuple[str, ...] = ()
    event_lines: tuple[str, ...] = ()
    workspace_diff: str = ""


@dataclass(frozen=True)
class DeliveryMetricEvidence:
    """Measured delivery facts reused by the deterministic ACI evaluation."""

    schema_bytes: int
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    cache_usage_source: str | None
    recovery_successes: int
    recovery_cases: int


@dataclass(frozen=True)
class _FakeTurn:
    text: str = ""
    tool_name: str | None = None
    arguments: Mapping[str, object] = field(default_factory=dict)


def build_cases() -> tuple[SmokeCase, ...]:
    service_source = (Path(__file__).parents[1] / "agent_runtime" / "service.py").read_text(encoding="utf-8")
    return (
        SmokeCase(
            name="direct_answer",
            task="What is 2+2? Answer with exactly the number.",
            expected_answer_exact="4",
            expected_initial_tools=RESIDENT_TOOL_NAMES,
        ),
        SmokeCase(
            name="find_agent_service",
            task=(
                "Find class AgentService with search_text, then call read_file on "
                "the matching file. Do not answer until read_file succeeds. Finally "
                "output exactly its workspace-relative path and no other text."
            ),
            expected_answer_exact="input_files/service.py",
            expected_tools=("search_text", "read_file"),
            expected_initial_tools=("search_text", "read_file"),
            tools=("search_text", "read_file"),
            workspace_files={"agent_runtime/service.py": service_source},
        ),
        SmokeCase(
            name="patch_fixture",
            task="Replace before with after in fixture.txt.",
            expected_tools=("apply_patch",),
            expected_initial_tools=("apply_patch",),
            tools=("apply_patch",),
            allow_write_tools=True,
            workspace_files={"fixture.txt": "before\n"},
            workspace_assertions={"input_files/fixture.txt": "after\n"},
        ),
        SmokeCase(
            name="echo_hello",
            task="Run echo hello and answer with stdout.",
            expected_answer_contains=("hello",),
            expected_tools=("run_command",),
            expected_initial_tools=("run_command",),
            tools=("run_command",),
            allow_write_tools=True,
            allow_execute_tools=True,
        ),
        SmokeCase(
            name="missing_file_recovery",
            task="Read missing.txt, recover from the error, and answer file_not_found.",
            expected_answer_contains=("file_not_found",),
            expected_tools=("read_file",),
            expected_initial_tools=("read_file",),
            expected_tool_errors=("read_file:runner_failed",),
            tools=("read_file",),
        ),
        SmokeCase(
            name="repeated_failure_circuit",
            task=("Try missing.txt until the runtime circuit opens, then answer circuit_open."),
            expected_answer_exact="circuit_open",
            expected_tools=("read_file", "read_file", "read_file"),
            expected_initial_tools=("read_file",),
            expected_tool_errors=(
                "read_file:runner_failed",
                "read_file:repeated_tool_failure",
            ),
            tools=("read_file",),
        ),
        SmokeCase(
            name="hidden_mcp_disabled",
            task=("Without calling a tool, output exactly hidden_disabled and no other text."),
            expected_answer_exact="hidden_disabled",
            expected_initial_tools=RESIDENT_TOOL_NAMES,
            forbidden_tools=("find_tools", "mcp__docs__search"),
            install_hidden_mcp=True,
        ),
        SmokeCase(
            name="hidden_mcp_discovery",
            task="Discover the external documentation search and use it once.",
            expected_answer_contains=("hidden docs",),
            expected_tools=("find_tools", "mcp__docs__search"),
            expected_initial_tools=(*RESIDENT_TOOL_NAMES, "find_tools"),
            allow_discovery_tools=True,
            auto_approve=True,
            install_hidden_mcp=True,
        ),
        SmokeCase(
            name="approval_resume",
            task="Patch approval.txt from before to approved, requesting approval.",
            expected_tools=("apply_patch",),
            expected_initial_tools=("apply_patch",),
            tools=("apply_patch",),
            auto_approve=True,
            expect_origin_retained=True,
            workspace_files={"approval.txt": "before\n"},
            workspace_assertions={"input_files/approval.txt": "approved\n"},
        ),
        SmokeCase(
            name="cache_usage",
            task=("Without calling a tool, output exactly cache_visible and no other text."),
            expected_answer_exact="cache_visible",
            expected_initial_tools=RESIDENT_TOOL_NAMES,
        ),
        SmokeCase(
            name="mlx_local_envelope",
            task="Read local.txt and answer mlx local.",
            expected_answer_contains=("mlx local",),
            expected_tools=("read_file",),
            expected_initial_tools=("read_file",),
            tools=("read_file",),
            provider="mlx",
            workspace_files={"local.txt": "local envelope\n"},
        ),
        SmokeCase(
            name="ollama_local_envelope",
            task="Read local.txt and answer ollama local.",
            expected_answer_contains=("ollama local",),
            expected_tools=("read_file",),
            expected_initial_tools=("read_file",),
            tools=("read_file",),
            provider="ollama",
            workspace_files={"local.txt": "local envelope\n"},
        ),
        SmokeCase(
            name="praxis_demo",
            task=(
                "Inspect praxis_demo.py, change STATUS from draft to ready, "
                "inspect the result, run the workspace verification, then "
                "output exactly praxis demo complete."
            ),
            expected_answer_exact="praxis demo complete",
            expected_tools=(
                "read_file",
                "apply_patch",
                "read_file",
                "run_command",
            ),
            allow_write_tools=True,
            allow_execute_tools=True,
            public_agent=True,
            workspace_files={
                "praxis_demo.py": 'STATUS = "draft"\n',
                "Makefile": (
                    ".PHONY: verify\n"
                    "verify:\n"
                    "\t@grep -Fqx 'STATUS = \"ready\"' praxis_demo.py\n"
                    "\t@printf 'verification passed\\n'\n"
                ),
            },
            workspace_assertions={
                "praxis_demo.py": 'STATUS = "ready"\n',
            },
        ),
    )


def _fake_turns(case: SmokeCase) -> tuple[_FakeTurn, ...]:
    turns: dict[str, tuple[_FakeTurn, ...]] = {
        "direct_answer": (_FakeTurn(text="4"),),
        "find_agent_service": (
            _FakeTurn(
                tool_name="search_text",
                arguments={
                    "pattern": "class AgentService",
                    "path": "input_files",
                    "glob": "service.py",
                    "max_results": 1,
                },
            ),
            _FakeTurn(
                tool_name="read_file",
                arguments={
                    "path": "input_files/service.py",
                    "max_bytes": 512,
                },
            ),
            _FakeTurn(text="input_files/service.py"),
        ),
        "patch_fixture": (
            _FakeTurn(
                tool_name="apply_patch",
                arguments={
                    "file_path": "input_files/fixture.txt",
                    "old_string": "before",
                    "new_string": "after",
                },
            ),
            _FakeTurn(text="patched"),
        ),
        "echo_hello": (
            _FakeTurn(
                tool_name="run_command",
                arguments={
                    "command": "echo hello",
                    "working_dir": ".",
                    "timeout_seconds": 3,
                },
            ),
            _FakeTurn(text="hello"),
        ),
        "missing_file_recovery": (
            _FakeTurn(
                tool_name="read_file",
                arguments={"path": "missing.txt"},
            ),
            _FakeTurn(text="file_not_found"),
        ),
        "repeated_failure_circuit": (
            _FakeTurn(
                tool_name="read_file",
                arguments={"path": "missing.txt"},
            ),
            _FakeTurn(
                tool_name="read_file",
                arguments={"path": "missing.txt"},
            ),
            _FakeTurn(
                tool_name="read_file",
                arguments={"path": "missing.txt"},
            ),
            _FakeTurn(text="circuit_open"),
        ),
        "hidden_mcp_disabled": (_FakeTurn(text="hidden_disabled"),),
        "hidden_mcp_discovery": (
            _FakeTurn(
                tool_name="find_tools",
                arguments={"query": "external documentation", "limit": 5},
            ),
            _FakeTurn(
                tool_name="mcp__docs__search",
                arguments={"query": "runtime"},
            ),
            _FakeTurn(text="hidden docs"),
        ),
        "approval_resume": (
            _FakeTurn(
                tool_name="apply_patch",
                arguments={
                    "file_path": "input_files/approval.txt",
                    "old_string": "before",
                    "new_string": "approved",
                },
            ),
            _FakeTurn(text="approved"),
        ),
        "cache_usage": (_FakeTurn(text="cache_visible"),),
        "mlx_local_envelope": (
            _FakeTurn(
                tool_name="read_file",
                arguments={"path": "input_files/local.txt"},
            ),
            _FakeTurn(text="mlx local"),
        ),
        "ollama_local_envelope": (
            _FakeTurn(
                tool_name="read_file",
                arguments={"path": "input_files/local.txt"},
            ),
            _FakeTurn(text="ollama local"),
        ),
        "praxis_demo": (
            _FakeTurn(
                tool_name="read_file",
                arguments={"path": "praxis_demo.py", "max_bytes": 512},
            ),
            _FakeTurn(
                tool_name="apply_patch",
                arguments={
                    "file_path": "praxis_demo.py",
                    "old_string": 'STATUS = "draft"',
                    "new_string": 'STATUS = "ready"',
                },
            ),
            _FakeTurn(
                tool_name="read_file",
                arguments={"path": "praxis_demo.py", "max_bytes": 512},
            ),
            _FakeTurn(
                tool_name="run_command",
                arguments={
                    "command": "make verify",
                    "working_dir": ".",
                    "timeout_seconds": 30,
                },
            ),
            _FakeTurn(text="praxis demo complete"),
        ),
    }
    return turns[case.name]


class _WordAccounting:
    def count(self, text: str) -> int:
        return max(len(text.split()), 1)


class _FakeGenerator:
    def __init__(self, case: SmokeCase) -> None:
        self.provider = case.provider
        self._turns = list(_fake_turns(case))
        self.visible_tools: list[tuple[str, ...]] = []

    def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: object,
    ) -> object:
        del messages, kwargs
        visible = tuple(str(item["function"]["name"]) for item in tools)
        turn = self._next_turn(visible)
        raw_calls = []
        if turn.tool_name is not None:
            raw_calls.append(
                {
                    "id": f"call_{len(self.visible_tools)}_{turn.tool_name}",
                    "type": "function",
                    "function": {
                        "name": turn.tool_name,
                        "arguments": json.dumps(
                            dict(turn.arguments),
                            ensure_ascii=False,
                        ),
                    },
                }
            )
        return self._provider_result(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls" if raw_calls else "stop",
                        "message": {
                            "role": "assistant",
                            "content": turn.text,
                            "tool_calls": raw_calls,
                        },
                    }
                ]
            }
        )

    def generate_text_with_usage(
        self,
        *,
        prompt: str,
        **kwargs: object,
    ) -> object:
        del kwargs
        visible = _local_prompt_tool_names(prompt)
        turn = self._next_turn(visible)
        calls = []
        if turn.tool_name is not None:
            calls.append(
                {
                    "id": f"call_{len(self.visible_tools)}_{turn.tool_name}",
                    "name": turn.tool_name,
                    "arguments": dict(turn.arguments),
                }
            )
        return self._provider_result(
            json.dumps(
                {"text": turn.text, "tool_calls": calls},
                ensure_ascii=False,
            )
        )

    def _next_turn(self, visible: tuple[str, ...]) -> _FakeTurn:
        if not self._turns:
            raise AssertionError("fake model received an unexpected extra turn")
        turn = self._turns.pop(0)
        self.visible_tools.append(visible)
        if turn.tool_name is not None and turn.tool_name not in visible:
            raise AssertionError(f"fake model tool {turn.tool_name!r} was not visible: {visible!r}")
        return turn

    def _provider_result(self, value: object) -> object:
        from agent_runtime.modeling.contracts import LLMProviderResult, normalize_llm_usage

        usage = normalize_llm_usage(
            input_tokens=40,
            output_tokens=5,
            cache_read_input_tokens=7,
            cache_write_input_tokens=3,
            input_tokens_include_cache=True,
            usage_source="provider",
            raw_provider_usage={
                "input_tokens": 40,
                "cache_read_input_tokens": 7,
                "cache_write_input_tokens": 3,
            },
        )
        return LLMProviderResult(value=value, usage=usage)


class _FakeModelResolver:
    default_model = "fake"
    fallback_model = "fake"

    def __init__(self, case: SmokeCase, generator: _FakeGenerator) -> None:
        from agent_runtime.core.llm_registry import ResolvedModel
        from agent_runtime.modeling.contracts import LLMCallStage, LLMStageBudget
        from agent_runtime.modeling.gateway import LLMGateway

        gateway = LLMGateway(
            generator=generator,
            token_accounting=_WordAccounting(),  # type: ignore[arg-type]
            model_context_tokens=120_000,
            stage_budgets={
                LLMCallStage.TOOL_DECISION: LLMStageBudget(
                    max_input_tokens=100_000,
                    max_output_tokens=2_000,
                    safety_margin_tokens=0,
                )
            },
        )
        self._resolved = ResolvedModel(
            generator=generator,
            kwargs={"max_tokens": 512, "temperature": 0.0},
            context_window_tokens=120_000,
            gateway=gateway,
            token_accounting=_WordAccounting(),
            provider=case.provider,
            model="fake-model",
            supports_native_tools=case.provider == "openai-compatible",
        )

    def resolve_for_node(
        self,
        *,
        node_model: str | None,
        node_name: str,
    ) -> object:
        del node_model, node_name
        return self._resolved

    def current_model(self) -> object:
        @dataclass(frozen=True)
        class _CurrentModel:
            id: str = "fake"

        return _CurrentModel()


def _local_prompt_tool_names(prompt: str) -> tuple[str, ...]:
    marker = "[Selected Tools]\n"
    if marker not in prompt:
        return ()
    encoded = prompt.split(marker, 1)[1].split("\n\n", 1)[0]
    definitions = json.loads(encoded)
    return tuple(str(item["name"]) for item in definitions)


def _hidden_mcp_tools() -> tuple[Tool, ...]:
    from agent_runtime.tools.integrations.mcp import (
        MCPToolDescriptor,
        create_mcp_tools,
    )

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    descriptor = MCPToolDescriptor(
        server_name="docs",
        tool_name="search",
        description="Search external documentation for runtime facts.",
        input_schema=input_schema,
        read_only_hint=True,
        idempotent_hint=True,
    )
    return create_mcp_tools(
        (descriptor,),
        lambda _server, _tool, _arguments: {
            "content": [{"type": "text", "text": "hidden docs"}],
            "structuredContent": {"answer": "hidden docs"},
        },
    )


class _DemoEventSink:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def emit(self, event: object) -> None:
        self.events.append(event)


async def _run_public_agent_case(
    case: SmokeCase,
    *,
    model: str,
    fake_model: bool,
) -> SmokeResult:
    from agent_runtime import Agent

    if not fake_model:
        return SmokeResult(
            name=case.name,
            passed=False,
            status="error",
            answer=None,
            tools=(),
            visible_tools=(),
            workspace_path=None,
            error="deterministic public demo requires --fake-model",
        )

    generator = _FakeGenerator(case)
    event_sink = _DemoEventSink()
    with tempfile.TemporaryDirectory(prefix="praxis_demo_") as raw_root:
        root = Path(raw_root)
        workspace = root / "workspace"
        workspace.mkdir()
        _write_workspace_files(workspace, case.workspace_files)
        agent = Agent(
            model=model,
            checkpoint_db=root / "checkpoints.sqlite",
            workspace_path=workspace,
            model_session_path=None,
        )
        agent._model_control_plane = _FakeModelResolver(  # type: ignore[assignment]
            case,
            generator,
        )
        try:
            public_result = await agent.arun(
                case.task,
                files=[
                    str(workspace / relative)
                    for relative in case.workspace_files
                ],
                max_turns=case.max_turns,
                allow_write_tools=case.allow_write_tools,
                allow_execute_tools=case.allow_execute_tools,
                event_sink=event_sink,
            )
            assertion_error = _workspace_assertion_error(
                workspace,
                case.workspace_assertions,
            )
            event_lines, workspace_diff = _demo_event_lines(
                event_sink.events,
                status=public_result.status,
                answer=public_result.answer,
            )
            candidate = SmokeResult(
                name=case.name,
                passed=True,
                status=public_result.status,
                answer=public_result.answer,
                tools=tuple(
                    call.tool_name for call in public_result.tool_calls
                ),
                visible_tools=tuple(generator.visible_tools),
                workspace_path="workspace",
                stop_reason=public_result.stop_reason,
                diagnostics=_diagnostic_lines(public_result.diagnostics),
                schema_bytes=public_result.usage.tool_schema_bytes,
                tool_errors=_tool_error_lines(public_result.tool_calls),
                provider_wire_kind="openai",
                usage_source=public_result.usage.usage_source,
                cache_read_input_tokens=(
                    public_result.usage.cache_read_input_tokens
                ),
                cache_write_input_tokens=(
                    public_result.usage.cache_write_input_tokens
                ),
                workspace_assertions_passed=not assertion_error,
                result_content_kinds=tuple(
                    "structured"
                    if call.structured_output is not None
                    else "empty"
                    for call in public_result.tool_calls
                ),
                event_lines=event_lines,
                workspace_diff=workspace_diff,
            )
            error = assertion_error or _validate_result(case, candidate)
            return replace(candidate, passed=not error, error=error)
        except Exception as exc:
            return SmokeResult(
                name=case.name,
                passed=False,
                status="error",
                answer=None,
                tools=(),
                visible_tools=tuple(generator.visible_tools),
                workspace_path=None,
                error=_sanitize_demo_fragment(
                    f"{type(exc).__name__}: {exc}"
                ),
                event_lines=_demo_event_lines(
                    event_sink.events,
                    status="error",
                    answer=None,
                )[0],
            )
        finally:
            turn_store = getattr(agent, "_turn_store", None)
            if turn_store is not None:
                turn_store.close()


def _demo_event_lines(
    events: Sequence[object],
    *,
    status: str,
    answer: str | None,
) -> tuple[tuple[str, ...], str]:
    from agent_runtime import EventType, ItemStatus, TurnItemKind

    lines: list[str] = []
    phases_by_tool_id: dict[str, str] = {}
    read_count = 0
    workspace_diff = ""
    for event in events:
        event_type = getattr(event, "type", None)
        data = getattr(event, "data", {})
        if not isinstance(data, Mapping):
            continue
        item_kind = getattr(event, "item_kind", None)
        if (
            event_type is EventType.ITEM_STARTED
            and item_kind in {TurnItemKind.TOOL, TurnItemKind.COMMAND}
        ):
            tool_name = str(data.get("tool_name", item_kind.value))
            tool_id = str(getattr(event, "item_id", "") or "")
            if tool_name == "read_file":
                phase = "inspect" if read_count == 0 else "verify"
                read_count += 1
            elif tool_name == "apply_patch":
                phase = "patch"
            else:
                phase = "verify"
            phases_by_tool_id[tool_id] = phase
            subject = _demo_event_subject(
                tool_name,
                str(data.get("input_preview", "")),
            )
            suffix = f" {subject}" if subject else ""
            lines.append(f"[{phase}] tool:start {tool_name}{suffix}")
            continue
        if (
            event_type is EventType.ITEM_COMPLETED
            and item_kind in {TurnItemKind.TOOL, TurnItemKind.COMMAND}
        ):
            tool_id = str(getattr(event, "item_id", "") or "")
            result = data.get("result")
            result_data = result if isinstance(result, Mapping) else {}
            tool_name = str(result_data.get("tool_name", item_kind.value))
            phase = phases_by_tool_id.get(tool_id, "verify")
            if getattr(event, "status", None) is ItemStatus.SUCCESS:
                lines.append(f"[{phase}] tool:ok {tool_name}")
                metadata = result_data.get("metadata")
                if tool_name == "apply_patch" and isinstance(metadata, Mapping):
                    raw_diff = metadata.get("diff")
                    if isinstance(raw_diff, str):
                        workspace_diff = _sanitize_demo_diff(raw_diff)
            else:
                lines.append(f"[{phase}] tool:error")
            continue
        if event_type is EventType.TURN_COMPLETED:
            reason = _sanitize_demo_fragment(data.get("reason", data.get("status", "unknown")))
            lines.append(f"[complete] turn:end reason={reason}")
            continue
        if event_type is EventType.TOOL_USE_START:
            tool_name = str(data.get("tool_name", "tool"))
            tool_id = str(data.get("tool_id", ""))
            if tool_name == "read_file":
                phase = "inspect" if read_count == 0 else "verify"
                read_count += 1
            elif tool_name == "apply_patch":
                phase = "patch"
            else:
                phase = "verify"
            phases_by_tool_id[tool_id] = phase
            subject = _demo_event_subject(
                tool_name,
                str(data.get("input_preview", "")),
            )
            suffix = f" {subject}" if subject else ""
            lines.append(
                f"[{phase}] tool:start {tool_name}{suffix}"
            )
            continue
        if event_type is EventType.TOOL_USE_RESULT:
            tool_id = str(data.get("tool_id", ""))
            tool_name = str(data.get("tool_name", "tool"))
            phase = phases_by_tool_id.get(tool_id, "verify")
            lines.append(f"[{phase}] tool:ok {tool_name}")
            details = data.get("details")
            if tool_name == "apply_patch" and isinstance(details, Mapping):
                raw_diff = details.get("diff")
                if isinstance(raw_diff, str):
                    workspace_diff = _sanitize_demo_diff(raw_diff)
            continue
        if event_type is EventType.TOOL_USE_ERROR:
            tool_id = str(data.get("tool_id", ""))
            phase = phases_by_tool_id.get(tool_id, "verify")
            lines.append(f"[{phase}] tool:error")
            continue
        if event_type is EventType.LOOP_END:
            reason = _sanitize_demo_fragment(data.get("reason", "unknown"))
            turns = data.get("total_turns", 0)
            lines.append(
                f"[complete] loop:end reason={reason} turns={turns}"
            )
    completion = f"[complete] status={status}"
    if answer:
        completion += f" answer={_sanitize_demo_fragment(answer)}"
    lines.append(completion)
    return tuple(lines), workspace_diff


def _demo_event_subject(tool_name: str, preview: str) -> str:
    key = "command" if tool_name == "run_command" else (
        "file_path" if tool_name == "apply_patch" else "path"
    )
    match = re.search(
        rf"(?:^|, ){re.escape(key)}='([^']*)'",
        preview,
    )
    return _sanitize_demo_fragment(match.group(1) if match else "")


_DEMO_ABSOLUTE_PATH_PATTERN = re.compile(
    r"""(?ix)
    (?:
        (?<![A-Za-z0-9_.:/\\])/(?!/)[^\s,'"<>]+
        |
        (?<![A-Za-z0-9_])[A-Z]:[\\/][^\s,'"<>]+
        |
        (?<![A-Za-z0-9_\\])\\\\[^\\/\s,'"<>]+\\[^\s,'"<>]+
    )
    """
)


def _contains_demo_absolute_path(value: str) -> bool:
    """Recognize POSIX, Windows drive, and UNC absolute paths in trace text."""

    return _DEMO_ABSOLUTE_PATH_PATTERN.search(value) is not None


def _redact_demo_absolute_paths(value: str) -> str:
    return _DEMO_ABSOLUTE_PATH_PATTERN.sub("[absolute-path]", value)


def _sanitize_demo_fragment(value: object) -> str:
    from agent_runtime.core.runtime_diagnostics import redact_sensitive_text

    redacted = redact_sensitive_text(value)
    redacted = _redact_demo_absolute_paths(redacted)
    return " ".join(redacted.split())[:240]


def _sanitize_demo_diff(value: str) -> str:
    from agent_runtime.core.runtime_diagnostics import redact_sensitive_text

    redacted = redact_sensitive_text(value)
    return _redact_demo_absolute_paths(redacted)[:2000]


async def run_case(
    case: SmokeCase,
    *,
    model: str,
    fake_model: bool = False,
) -> SmokeResult:
    if case.public_agent:
        return await _run_public_agent_case(
            case,
            model=model,
            fake_model=fake_model,
        )

    from agent_runtime.core.messages import canonical_json_text
    from agent_runtime.core.model_request import tool_definition_payload
    from agent_runtime.models import ModelControlPlane
    from agent_runtime.result import AgentResult
    from agent_runtime.runtime.builder import build_agent_service
    from agent_runtime.service import AgentRunRequest
    from agent_runtime.tools.selection import select_tools
    from agent_runtime.turns import RuntimeBinding, TurnStore

    turn_id = str(uuid4())
    service = None
    turn_store: TurnStore | None = None
    service_workspace = None
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    generator = _FakeGenerator(case) if fake_model else None
    try:
        model_registry: object = (
            _FakeModelResolver(case, generator)
            if generator is not None
            else ModelControlPlane.from_env(initial_model_id=model)
        )
        turn_store = TurnStore()
        service = build_agent_service(
            None,
            model_alias=model,
            model_control_plane=model_registry,  # type: ignore[arg-type]
            mcp_tools=(_hidden_mcp_tools() if case.install_hidden_mcp else ()),
            turn_store=turn_store,
            runtime_binding=RuntimeBinding(model_alias=model),
        )
        service_workspace = service._workspace
        if service_workspace is None:
            raise RuntimeError("agent service did not create a workspace")
        temp_dir = tempfile.TemporaryDirectory(prefix=f"delivery_smoke_{case.name}_")
        source_path = Path(temp_dir.name)
        _write_workspace_files(source_path, case.workspace_files)
        workspace_path = service_workspace.root
        request = AgentRunRequest(
            message=case.task,
            turn_id=turn_id,
            input_files=[str(source_path / relative) for relative in case.workspace_files],
            max_turns=case.max_turns,
            tools=case.tools,
            disabled_tools=case.disabled_tools,
            allow_write_tools=case.allow_write_tools,
            allow_execute_tools=case.allow_execute_tools,
            allow_discovery_tools=case.allow_discovery_tools,
        )
        initial_state = service.initial_state(request)
        initial_names = tuple(
            (
                *initial_state["resident_tool_names"],
                *initial_state["explicit_tool_names"],
            )
        )
        initial_tools = select_tools(
            service._tool_snapshot,
            resident_names=initial_names,
            disabled_names=initial_state["disabled_tool_names"],
        )
        schema_bytes = len(
            canonical_json_text(tuple(tool_definition_payload(tool.definition) for tool in initial_tools)).encode(
                "utf-8"
            )
        )

        result = await service.run(request)
        approvals = 0
        while result.status == "paused" and case.auto_approve and approvals < 5:
            human_request = service.pending_human_input_request(turn_id=turn_id)
            if human_request.kind != "tool_approval":
                break
            result = await service.resume_turn(
                turn_id=turn_id,
                action="allow_once",
            )
            approvals += 1

        origin_revisions, serializer_revision = await _checkpoint_evidence(
            service,
            turn_id=turn_id,
        )
        records = tuple(result.model_call_records)
        public_result = AgentResult._from_internal(result)
        tool_results = tuple(result.tool_results)
        tools = tuple(item.tool_name for item in public_result.tool_calls)
        origin_retained = (
            None if not origin_revisions or not records else origin_revisions[0] == records[0].toolset_revision
        )
        assertion_error = _workspace_assertion_error(
            workspace_path,
            case.workspace_assertions,
        )
        visible_tools = (
            tuple(generator.visible_tools)
            if generator is not None
            else (tuple(tool.definition.name for tool in initial_tools),)
        )
        diagnostics = (
            *_diagnostic_lines(public_result.diagnostics),
            *_model_record_lines(records),
        )
        candidate = SmokeResult(
            name=case.name,
            passed=True,
            status=public_result.status,
            answer=public_result.answer,
            tools=tools,
            visible_tools=visible_tools,
            workspace_path=public_result.workspace_path,
            stop_reason=public_result.stop_reason,
            diagnostics=diagnostics,
            schema_bytes=schema_bytes,
            tool_errors=_tool_error_lines(public_result.tool_calls),
            request_ids=tuple(record.request_id for record in records),
            prompt_revisions=tuple(record.prompt_revision for record in records),
            toolset_revisions=tuple(record.toolset_revision for record in records),
            provider_wire_hashes=tuple(record.provider_wire_hash for record in records),
            provider_wire_kind=(case.provider if case.provider in {"mlx", "ollama"} else "openai"),
            serializer_revision=serializer_revision,
            usage_source=public_result.usage.usage_source,
            cache_read_input_tokens=(public_result.usage.cache_read_input_tokens),
            cache_write_input_tokens=(public_result.usage.cache_write_input_tokens),
            approval_count=approvals,
            origin_toolset_revisions=origin_revisions,
            origin_retained=origin_retained,
            workspace_assertions_passed=not assertion_error,
            result_content_kinds=_result_content_kinds(tool_results),
        )
        error = assertion_error or _validate_result(case, candidate)
        return replace(candidate, passed=not error, error=error)
    except Exception as exc:
        return SmokeResult(
            name=case.name,
            passed=False,
            status="error",
            answer=None,
            tools=(),
            visible_tools=(tuple(generator.visible_tools) if generator is not None else ()),
            workspace_path=None,
            error=f"{type(exc).__name__}: {exc}",
            provider_wire_kind=(case.provider if case.provider in {"mlx", "ollama"} else "openai"),
        )
    finally:
        if service is not None:
            await service.aclose()
        if turn_store is not None:
            turn_store.close()
        if temp_dir is not None:
            temp_dir.cleanup()
        if service_workspace is not None and service_workspace.is_temporary:
            shutil.rmtree(service_workspace.root, ignore_errors=True)


async def _checkpoint_evidence(
    service: Any,
    *,
    turn_id: str,
) -> tuple[tuple[str, ...], str]:
    """Read canonical v2 evidence without materializing the whole loop state."""

    from agent_runtime.core.checkpointing import (
        LOOP_CHECKPOINT_NAMESPACE,
        LOOP_STATE_CHANNEL,
        decode_tool_checkpoint,
    )

    checkpoint_tuple = await service._checkpointer.aget_tuple(
        {
            "configurable": {
                # LangGraph calls its storage key thread_id; the domain ID is a Turn.
                "thread_id": turn_id,
                "checkpoint_ns": LOOP_CHECKPOINT_NAMESPACE,
            }
        }
    )
    if checkpoint_tuple is None:
        return (), ""
    raw_state = checkpoint_tuple.checkpoint["channel_values"].get(LOOP_STATE_CHANNEL)
    if not isinstance(raw_state, Mapping):
        return (), ""
    raw_tool_checkpoint = raw_state.get("tool_checkpoint")
    if raw_tool_checkpoint is None:
        return (), ""
    tool_checkpoint = decode_tool_checkpoint(raw_tool_checkpoint)
    return (
        tuple(call.origin.toolset_revision for call in tool_checkpoint.tool_calls),
        tool_checkpoint.manifest.provider_serializer_revision,
    )


def _write_workspace_files(
    root: Path,
    files: Mapping[str, str],
) -> None:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _workspace_assertion_error(
    root: Path,
    assertions: Mapping[str, str],
) -> str:
    for relative, expected in assertions.items():
        path = root / relative
        if not path.is_file():
            return f"missing workspace file {relative!r}"
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            return f"{relative!r} content mismatch: {actual!r}"
    return ""


def _validate_result(case: SmokeCase, result: SmokeResult) -> str:
    if result.status != "done":
        return f"expected status done, got {result.status}"
    raw_answer = result.answer or ""
    answer = raw_answer.strip().casefold()
    if case.expected_answer_exact is not None and answer != case.expected_answer_exact.strip().casefold():
        return f"expected exact answer {case.expected_answer_exact!r}, got {result.answer!r}"
    for expected in case.expected_answer_contains:
        if expected.casefold() not in answer:
            return f"answer missing {expected!r}: {result.answer!r}"
    if result.tools != case.expected_tools:
        return f"expected tools {case.expected_tools!r}, got {result.tools!r}"
    for forbidden in case.forbidden_tools:
        if forbidden in result.tools:
            return f"forbidden tool {forbidden!r} was called"
    if case.expected_initial_tools:
        if not result.visible_tools:
            return "model request did not expose a tool surface"
        if result.visible_tools[0] != case.expected_initial_tools:
            return f"expected initial tools {case.expected_initial_tools!r}, got {result.visible_tools[0]!r}"
    for expected_error in case.expected_tool_errors:
        if not any(line.startswith(expected_error) for line in result.tool_errors):
            return f"expected tool error {expected_error!r}, got {result.tool_errors!r}"
    if case.expect_origin_retained and result.origin_retained is not True:
        return "originating toolset revision was not retained across resume"
    if not result.workspace_assertions_passed:
        return "workspace assertions failed"
    if not case.public_agent:
        if not result.request_ids or not result.prompt_revisions:
            return "model request revisions were not recorded"
        if not result.toolset_revisions or not result.provider_wire_hashes:
            return "toolset revision or provider wire hash was not recorded"
    if not result.usage_source:
        return "usage source was not recorded"
    return ""


def _result_content_kinds(results: Sequence[object]) -> tuple[str, ...]:
    kinds: list[str] = []
    for result in results:
        if getattr(result, "structured_content", None) is not None:
            kinds.append("structured")
            continue
        content = tuple(getattr(result, "content", ()) or ())
        kinds.append(str(getattr(content[0], "type", "empty")) if content else "empty")
    return tuple(kinds)


def _diagnostic_lines(
    diagnostics: Sequence[object],
) -> tuple[str, ...]:
    lines: list[str] = []
    for diagnostic in diagnostics or ():
        code = str(getattr(diagnostic, "code", "diagnostic"))
        message = str(getattr(diagnostic, "message", ""))
        lines.append(f"{code}: {message}" if message else code)
    return tuple(lines)


def _model_record_lines(
    records: Sequence[ModelCallRecord],
) -> tuple[str, ...]:
    return tuple(
        " ".join(
            (
                f"request={record.request_id}",
                f"prompt={record.prompt_revision}",
                f"toolset={record.toolset_revision}",
                f"wire_hash={record.provider_wire_hash}",
                f"usage_source={record.usage.usage_source}",
                f"cache_read={record.usage.cache_read_input_tokens}",
                f"cache_write={record.usage.cache_write_input_tokens}",
            )
        )
        for record in records
    )


def _tool_error_lines(tool_results: Sequence[object]) -> tuple[str, ...]:
    lines: list[str] = []
    for result in tool_results:
        if not bool(getattr(result, "is_error", False)):
            continue
        tool_name = str(getattr(result, "tool_name", "tool"))
        code = str(getattr(result, "error_code", None) or "tool_error")
        message = str(getattr(result, "error_message", None) or "")
        lines.append(f"{tool_name}:{code}: {message}" if message else f"{tool_name}:{code}")
    return tuple(lines)


def _format_result(result: SmokeResult, *, verbose: bool) -> list[str]:
    marker = "PASS" if result.passed else "FAIL"
    tools = ",".join(result.tools) or "-"
    lines = [f"{marker} {result.name} status={result.status} tools={tools}"]
    if result.error:
        lines.append(f"  error: {result.error}")
    if result.answer:
        lines.append(f"  answer: {result.answer}")
    if result.workspace_path:
        lines.append(f"  workspace: {result.workspace_path}")
    if not verbose and result.passed:
        return lines

    lines.append(f"  schema_bytes: {result.schema_bytes}")
    if result.visible_tools:
        lines.append("  visible_tools: " + " | ".join(",".join(names) or "-" for names in result.visible_tools))
    for request_id, prompt, toolset, wire_hash in zip(
        result.request_ids,
        result.prompt_revisions,
        result.toolset_revisions,
        result.provider_wire_hashes,
        strict=True,
    ):
        lines.append(f"  revision: request={request_id} prompt={prompt} toolset={toolset} wire_hash={wire_hash}")
    lines.append(
        f"  provider: wire_kind={result.provider_wire_kind} serializer={result.serializer_revision or 'unknown'}"
    )
    lines.append(
        "  usage: "
        f"usage_source={result.usage_source or 'unknown'} "
        f"cache_read={result.cache_read_input_tokens} "
        f"cache_write={result.cache_write_input_tokens}"
    )
    if result.origin_retained is not None:
        lines.append(
            "  origin: "
            f"retained={str(result.origin_retained).lower()} "
            f"toolsets={','.join(result.origin_toolset_revisions)}"
        )
    if result.stop_reason:
        lines.append(f"  stop_reason: {result.stop_reason}")
    for diagnostic in result.diagnostics:
        lines.append(f"  diagnostic: {diagnostic}")
    for tool_error in result.tool_errors:
        lines.append(f"  tool_error: {tool_error}")
    return lines


async def run_matrix(
    *,
    model: str,
    fake_model: bool = False,
    only: set[str] | None = None,
) -> list[SmokeResult]:
    cases = [
        case
        for case in build_cases()
        if (only is None or case.name in only)
        and (not case.public_agent or only is not None)
    ]
    return [await run_case(case, model=model, fake_model=fake_model) for case in cases]


def delivery_metric_evidence(
    results: Sequence[SmokeResult],
) -> DeliveryMetricEvidence:
    """Extract cache and recovery evidence from completed delivery cases."""

    by_name = {result.name: result for result in results}
    required = {
        "cache_usage",
        "missing_file_recovery",
        "repeated_failure_circuit",
    }
    missing = required - set(by_name)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"delivery metric cases missing: {names}")

    cache = by_name["cache_usage"]
    recovery = by_name["missing_file_recovery"]
    circuit = by_name["repeated_failure_circuit"]
    recovered = (
        recovery.passed
        and recovery.status == "done"
        and any(error.startswith("read_file:runner_failed:") for error in recovery.tool_errors)
    )
    circuit_recovered = (
        circuit.passed
        and circuit.status == "done"
        and any(error.startswith("read_file:repeated_tool_failure:") for error in circuit.tool_errors)
    )
    return DeliveryMetricEvidence(
        schema_bytes=cache.schema_bytes,
        cache_read_tokens=cache.cache_read_input_tokens,
        cache_write_tokens=cache.cache_write_input_tokens,
        cache_usage_source=cache.usage_source,
        recovery_successes=int(recovered) + int(circuit_recovered),
        recovery_cases=2,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--fake-model",
        action="store_true",
        help="Use the deterministic provider-compatible fake model matrix.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run only this case. Can be provided multiple times.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=("Print revisions, schema bytes, provider wire data, usage, diagnostics, and tool errors."),
    )
    args = parser.parse_args()

    results = asyncio.run(
        run_matrix(
            model=args.model,
            fake_model=args.fake_model,
            only=set(args.cases) if args.cases else None,
        )
    )
    for result in results:
        for line in _format_result(result, verbose=args.verbose):
            print(line)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
