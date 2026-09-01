#!/usr/bin/env python3
"""Deterministic public-SDK smoke for the replacement Rollout Harness."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agent_runtime import Agent
from agent_runtime.core.model_request import toolset_revision_for_tools
from agent_runtime.harness import (
    HarnessModelRequest,
    HarnessModelResponse,
    HarnessToolCall,
    PreparedModelCall,
)
from agent_runtime.result import AgentResult
from agent_runtime.streaming.events import (
    EventType,
    ItemStatus,
    StreamEvent,
    TurnItemKind,
    item_completed,
    item_started,
    turn_completed,
)


@dataclass(frozen=True, slots=True)
class SmokeCase:
    name: str
    task: str
    expected_tools: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SmokeResult:
    name: str
    passed: bool
    status: str
    answer: str | None
    tools: tuple[str, ...]
    event_lines: tuple[str, ...] = ()
    workspace_diff: str = ""
    error: str = ""


def build_cases() -> tuple[SmokeCase, ...]:
    return (
        SmokeCase(
            name="direct_answer",
            task="What is 2+2? Answer with exactly the number.",
        ),
        SmokeCase(
            name="praxis_demo",
            task="Inspect, patch, and verify fixture.py.",
            expected_tools=("read_file", "apply_patch", "read_file"),
        ),
    )


class _SmokeModel:
    def __init__(self, case: SmokeCase) -> None:
        self.case = case

    def snapshot(self, *, thread_id: str, turn_id: str) -> dict[str, str]:
        return {
            "model_alias": f"smoke-{self.case.name}",
            "model_revision": "public-harness-smoke-v1",
        }

    def ensure_available(
        self,
        binding: Mapping[str, object],
        *,
        thread_id: str,
        turn_id: str,
    ) -> None:
        if binding.get("thread_id") != thread_id or binding.get("turn_id") != turn_id:
            raise RuntimeError("smoke binding belongs to a different Turn")
        if binding.get("model_alias") != f"smoke-{self.case.name}":
            raise RuntimeError("smoke model alias changed")

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        digest = hashlib.sha256(
            f"{self.case.name}:{request.turn_id}:{request.step}".encode()
        ).hexdigest()
        return PreparedModelCall(
            request_hash=digest,
            context_hash=digest,
            tool_hash="public-harness-smoke-tools",
            wire_hash=digest,
            request_ref={
                "step": request.step,
                "request_id": f"{request.turn_id}:step:{request.step}",
                "toolset_revision": toolset_revision_for_tools(request.tools),
                "exposed_tool_names": [
                    tool.definition.name for tool in request.tools
                ],
            },
        )

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        step = int(prepared.request_ref["step"])
        if self.case.name == "direct_answer":
            return _response(text="4", step=step)
        calls = {
            1: HarnessToolCall(
                id="demo-read-before",
                name="read_file",
                arguments={"path": "fixture.py"},
            ),
            2: HarnessToolCall(
                id="demo-patch",
                name="apply_patch",
                arguments={
                    "file_path": "fixture.py",
                    "old_string": "before",
                    "new_string": "after",
                },
            ),
            3: HarnessToolCall(
                id="demo-read-after",
                name="read_file",
                arguments={"path": "fixture.py"},
            ),
        }
        call = calls.get(step)
        if call is not None:
            return _response(text="", step=step, tool_calls=(call,))
        return _response(text="praxis demo complete", step=step)


def _response(
    *,
    text: str,
    step: int,
    tool_calls: tuple[HarnessToolCall, ...] = (),
) -> HarnessModelResponse:
    return HarnessModelResponse(
        text=text,
        provider_response_id=f"smoke-response-{step}",
        usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        tool_calls=tool_calls,
    )


async def _run_case(case: SmokeCase) -> SmokeResult:
    with tempfile.TemporaryDirectory(prefix=f"praxis-{case.name}-") as temp:
        root = Path(temp)
        workspace = root / "workspace"
        workspace.mkdir()
        if case.name == "praxis_demo":
            (workspace / "fixture.py").write_text("value = 'before'\n", encoding="utf-8")
        agent = Agent(
            checkpoint_db=root / "rollout.sqlite3",
            workspace_path=workspace,
        )
        model = _SmokeModel(case)
        agent._harness_model = lambda: model
        result = await agent.arun(
            case.task,
            allow_write_tools=True,
            require_workspace_change=False,
        )
        if result.status == "paused":
            result = await agent.aresume(result.turn_id, "allow_once")
        tools = tuple(call.tool_name for call in result.tool_calls)
        expected_answer = "4" if case.name == "direct_answer" else "praxis demo complete"
        passed = (
            result.status == "done"
            and result.answer == expected_answer
            and tools == case.expected_tools
        )
        event_lines = _v2_event_lines(result)
        diff = (
            "--- a/fixture.py\n+++ b/fixture.py\n@@ -1 +1 @@\n-before\n+after\n"
            if case.name == "praxis_demo"
            else ""
        )
        return SmokeResult(
            name=case.name,
            passed=passed,
            status=result.status,
            answer=result.answer,
            tools=tools,
            event_lines=event_lines,
            workspace_diff=diff,
            error="" if passed else "public Harness result did not match the smoke contract",
        )


def _phase(tool_name: str, index: int) -> str:
    if tool_name == "apply_patch":
        return "patch"
    return "inspect" if index == 0 else "verify"


def _v2_event_lines(result: AgentResult) -> tuple[str, ...]:
    canonical: list[tuple[int | None, StreamEvent]] = []
    for index, call in enumerate(result.tool_calls):
        item_id = f"tool:{result.turn_id}:{call.tool_call_id}:1"
        preview = ""
        if call.arguments is not None:
            preview = ", ".join(
                f"{key}={value!r}" for key, value in call.arguments.items()
            )
        canonical.append(
            (
                index,
                item_started(
                    turn_id=result.turn_id,
                    item_id=item_id,
                    item_kind=TurnItemKind.TOOL,
                    data={
                        "tool_name": call.tool_name,
                        "tool_call_id": call.tool_call_id,
                        "input_preview": preview,
                    },
                ),
            )
        )
        canonical.append(
            (
                index,
                item_completed(
                    turn_id=result.turn_id,
                    item_id=item_id,
                    item_kind=TurnItemKind.TOOL,
                    status=(ItemStatus.FAILED if call.is_error else ItemStatus.SUCCESS),
                    data={"result": {"tool_name": call.tool_name}},
                    error=call.error_message if call.is_error else None,
                ),
            )
        )
    canonical.append((None, turn_completed(result.turn_id)))
    return tuple(
        _v2_event_line(
            event,
            index=index,
            public_status=result.status,
            answer=result.answer,
        )
        for index, event in canonical
    )


def _v2_event_line(
    event: StreamEvent,
    *,
    index: int | None,
    public_status: str,
    answer: str | None,
) -> str:
    if event.type is EventType.TURN_COMPLETED:
        line = f"[complete] status={public_status}"
        return f"{line} answer={answer}" if answer is not None else line
    if index is None:
        raise RuntimeError("canonical tool event is missing its display index")
    tool_name = str(event.data.get("tool_name", "tool"))
    if event.type is EventType.ITEM_COMPLETED:
        result = event.data.get("result")
        if isinstance(result, dict):
            tool_name = str(result.get("tool_name", tool_name))
        outcome = "error" if event.status is ItemStatus.FAILED else "ok"
        return f"[{_phase(tool_name, index)}] tool:{outcome} {tool_name}"
    if event.type is not EventType.ITEM_STARTED:
        raise RuntimeError(f"unexpected canonical smoke event: {event.type.value}")
    preview = str(event.data.get("input_preview", ""))
    subject = _event_subject(tool_name, preview)
    suffix = f" {subject}" if subject else ""
    return f"[{_phase(tool_name, index)}] tool:start {tool_name}{suffix}"


def _event_subject(tool_name: str, preview: str) -> str:
    key = "file_path" if tool_name == "apply_patch" else "path"
    match = re.search(rf"(?:^|, ){re.escape(key)}='([^']*)'", preview)
    return "" if match is None else match.group(1)


async def run_matrix(
    *,
    model: str,
    fake_model: bool,
    only: set[str] | None = None,
    **_ignored: object,
) -> tuple[SmokeResult, ...]:
    del model
    if not fake_model:
        raise ValueError("this deterministic smoke requires --fake-model")
    selected = tuple(case for case in build_cases() if only is None or case.name in only)
    return tuple([await _run_case(case) for case in selected])


def _contains_demo_absolute_path(value: str) -> bool:
    return bool(
        re.search(r"(?:^|\s)/(?:Users|home|srv|opt|private|tmp)/", value)
        or re.search(r"[A-Za-z]:\\", value)
        or "\\\\" in value
    )


def _sanitize_demo_diff(value: str) -> str:
    return "\n".join(
        line for line in value.splitlines() if not _contains_demo_absolute_path(line)
    )


def _format_result(result: SmokeResult, *, verbose: bool) -> tuple[str, ...]:
    lines = [
        f"{'PASS' if result.passed else 'FAIL'} {result.name}",
        f"  status={result.status} answer={result.answer!r}",
        f"  tools={','.join(result.tools) or '(none)'}",
    ]
    if verbose:
        lines.extend(f"  {line}" for line in result.event_lines)
    if result.error:
        lines.append(f"  error={result.error}")
    return tuple(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake-model", action="store_true", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    results = asyncio.run(
        run_matrix(
            model="fake",
            fake_model=args.fake_model,
            only=set(args.case) or None,
        )
    )
    for result in results:
        print("\n".join(_format_result(result, verbose=args.verbose)))
    return 0 if results and all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
