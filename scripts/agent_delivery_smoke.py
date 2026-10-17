#!/usr/bin/env python3
"""Deterministic public-SDK smoke for the replacement Rollout Harness."""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agent_runtime import Agent
from agent_runtime.core.model_request import toolset_revision_for_tools
from agent_runtime.harness import (
    HarnessModelDeltaSink,
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
    validation_status: str = "not_evaluated"
    validation_scope: str | None = None
    check_outputs: tuple[str, ...] = ()


CHECKOUT_BEFORE = (
    "def total_after_discount(subtotal, discount_percent):\n"
    "    return round(subtotal * (1 - discount_percent), 2)\n"
)
CHECKOUT_AFTER = CHECKOUT_BEFORE.replace("1 - discount_percent", "1 - discount_percent / 100")
CHECKOUT_TESTS = '''import unittest
from pricing import total_after_discount

class PricingTests(unittest.TestCase):
    def test_no_discount(self):
        self.assertEqual(total_after_discount(100, 0), 100)

    def test_twenty_percent(self):
        self.assertEqual(total_after_discount(100, 20), 80)

    def test_full_discount(self):
        self.assertEqual(total_after_discount(100, 100), 0)
'''
CHECKOUT_COMMAND = "python3 -m unittest -v"


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
        SmokeCase(
            name="checkout_demo",
            task="修复 pricing.py 中的折扣计算：100 元打八折应为 80 元。先复现，再修改并运行测试。",
            expected_tools=("read_file", "run_command", "apply_patch", "run_command"),
        ),
    )


class _SmokeModel:
    def __init__(self, case: SmokeCase) -> None:
        self.case = case

    def snapshot(self, *, thread_id: str, turn_id: str) -> dict[str, str]:
        return {
            "model_id": f"smoke-{self.case.name}",
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
        if "model_alias" in binding:
            raise RuntimeError("legacy model_alias is unsupported")
        if binding.get("model_id") != f"smoke-{self.case.name}":
            raise RuntimeError("smoke model ID changed")

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

    async def dispatch(
        self,
        prepared: PreparedModelCall,
        *,
        delta_sink: HarnessModelDeltaSink | None = None,
    ) -> HarnessModelResponse:
        del delta_sink
        step = int(prepared.request_ref["step"])
        if self.case.name == "direct_answer":
            return _response(text="4", step=step)
        if self.case.name == "checkout_demo":
            calls = {
                1: HarnessToolCall(id="checkout-read", name="read_file", arguments={"path": "pricing.py"}),
                2: HarnessToolCall(
                    id="checkout-reproduce", name="run_command", arguments={"command": CHECKOUT_COMMAND},
                ),
                3: HarnessToolCall(id="checkout-patch", name="apply_patch", arguments={
                    "file_path": "pricing.py", "old_string": CHECKOUT_BEFORE, "new_string": CHECKOUT_AFTER,
                }),
                4: HarnessToolCall(id="checkout-verify", name="run_command", arguments={"command": CHECKOUT_COMMAND}),
            }
            call = calls.get(step)
            if call is not None:
                return _response(text="", step=step, tool_calls=(call,))
            return _response(text="折扣计算已修复，三项测试通过。", step=step)
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


async def _run_case(case: SmokeCase, *, model: _SmokeModel | None = None) -> SmokeResult:
    with tempfile.TemporaryDirectory(prefix=f"praxis-{case.name}-") as temp:
        root = Path(temp)
        workspace = root / "workspace"
        workspace.mkdir()
        if case.name == "praxis_demo":
            (workspace / "fixture.py").write_text("value = 'before'\n", encoding="utf-8")
        if case.name == "checkout_demo":
            (workspace / "pricing.py").write_text(CHECKOUT_BEFORE, encoding="utf-8")
            (workspace / "test_pricing.py").write_text(CHECKOUT_TESTS, encoding="utf-8")
        filename = "pricing.py" if case.name == "checkout_demo" else "fixture.py"
        before = (workspace / filename).read_text() if case.name != "direct_answer" else ""
        agent = Agent(
            checkpoint_db=root / "rollout.sqlite3",
            workspace_path=workspace,
        )
        model = model or _SmokeModel(case)
        agent.__dict__["_model_factory"] = lambda: model
        result = await agent.run(
            case.task,
            allow_write_tools=True,
            require_workspace_change=case.name != "direct_answer",
            acceptance_command=CHECKOUT_COMMAND if case.name == "checkout_demo" else None,
            max_turns=6,
        )
        while result.status == "paused":
            result = await agent.resume(result.turn_id, "allow_once")
        after = (workspace / filename).read_text() if case.name != "direct_answer" else ""
        tools = tuple(call.tool_name for call in result.tool_calls)
        expected_answer = "4" if case.name == "direct_answer" else "praxis demo complete"
        if case.name == "checkout_demo":
            expected_answer = "折扣计算已修复，三项测试通过。"
        expected_after = CHECKOUT_AFTER if case.name == "checkout_demo" else "value = 'after'\n"
        checks = tuple(
            str(call.structured_output.get("stderr", ""))
            for call in result.tool_calls
            if call.tool_name == "run_command" and isinstance(call.structured_output, Mapping)
        )
        passed = (
            result.status == "done"
            and result.answer == expected_answer
            and tools == case.expected_tools
            and (case.name == "direct_answer" or (
                after == expected_after and after != before
                and result.validation_status == "passed"
                and result.validation_scope == (
                    "acceptance_command" if case.name == "checkout_demo" else "workspace_change_recorded"
                )
            ))
        )
        if case.name == "checkout_demo":
            passed = (
                passed and len(checks) == 2 and "FAILED (failures=2)" in checks[0]
                and "Ran 3 tests" in checks[1] and "\nOK\n" in checks[1]
            )
        event_lines = _v2_event_lines(result)
        diff = "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                          fromfile=f"a/{filename}", tofile=f"b/{filename}"))
        return SmokeResult(
            name=case.name,
            passed=passed,
            status=result.status,
            answer=result.answer,
            tools=tools,
            event_lines=event_lines,
            workspace_diff=diff,
            error="" if passed else "public Harness result did not match the smoke contract",
            validation_status=result.validation_status,
            validation_scope=result.validation_scope,
            check_outputs=checks,
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
