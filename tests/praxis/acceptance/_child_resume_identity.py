"""R-FINAL-C1 kill -9 端到端子进程:spawn 一个 read_only 子并让它持续 running,
等着被 SIGKILL——父测试随后经裸 open_session(resume_turn_id) 重建子身份续跑。

    python _child_resume_identity.py <workspace> <responses.json> <parent_thread> <out_file>

responses.json = {"parent": [spec...], "child": [spec...]}。父剧本首帧
spawn_subagent(task 含 ``[child]`` 标记);路由按标记把子 turn 的模型请求交给
child 剧本。子首帧 child_hold(测试缝工具,轮询永不出现的 hold-marker,
idempotent=False)→ 子持续 running(tool_call 已写)。父 spawn 后即收尾
(COMPLETED),进程经长睡保持存活,子 driver 在同一事件循环里持续 running,
直到父测试 SIGKILL 整个进程(无任何用户态清理)。

子身份(thread/turn)经 out_file JSON 报给父测试,供其定位子根 turn 续跑。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from praxis.adapters.models.scripted import ScriptedAdapter  # noqa: E402
from praxis.core.statemachine import TurnStatus  # noqa: E402
from praxis.interfaces.sdk import open_session  # noqa: E402
from praxis.protocol.items import ItemKind  # noqa: E402
from praxis.protocol.json import JsonValue  # noqa: E402
from praxis.protocol.models import (  # noqa: E402
    ModelEvent,
    ModelMessage,
    ModelResponse,
    ModelSettings,
    ModelToolCall,
    Usage,
    WireRequest,
    assemble_model_events,
    collect_events,
)
from praxis.protocol.policy import NetworkPolicy, PermissionPolicy  # noqa: E402
from praxis.protocol.sandbox import ToolExecContext  # noqa: E402
from praxis.protocol.tools import (  # noqa: E402
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolResult,
)

_CHILD_MARKER = "[child]"


def _to_response(spec: Mapping[str, Any]) -> ModelResponse:
    calls = tuple(
        ModelToolCall(call_id=c["call_id"], name=c["name"], arguments=dict(c["arguments"]))
        for c in spec.get("tool_calls", [])
    )
    usage = spec.get("usage", {"input_tokens": 1, "output_tokens": 1})
    return ModelResponse(
        content=spec["content"],
        tool_calls=calls,
        usage=Usage(input_tokens=int(usage["input_tokens"]), output_tokens=int(usage["output_tokens"])),
        finish_reason=str(spec.get("finish_reason", "tool_calls" if calls else "stop")),
    )


class _Router:
    """父/子双剧本路由:子 user_message 含 [child] 标记 → child 剧本。"""

    def __init__(self, parent: Sequence[ModelResponse], child: Sequence[ModelResponse]) -> None:
        self._parent = ScriptedAdapter(parent)
        self._child = ScriptedAdapter(child)

    def build_request(
        self, messages: Sequence[ModelMessage], tools: Sequence[Any], settings: ModelSettings
    ) -> WireRequest:
        return self._parent.build_request(messages, tools, settings)

    async def stream(self, req: WireRequest) -> AsyncIterator[ModelEvent]:
        text = " ".join(message.content for message in req.messages)
        adapter = self._child if _CHILD_MARKER in text else self._parent
        async for event in adapter.stream(req):
            yield event

    async def complete(self, req: WireRequest) -> ModelResponse:
        return assemble_model_events(await collect_events(self.stream(req)))


def _hold_tool(workspace: Path) -> Tool:
    """子保持缝:轮询永不出现的 hold-marker,子持续 running(idempotent=False)。"""

    async def run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        marker = workspace / "hold-marker"
        deadline = time.monotonic() + 50.0
        while time.monotonic() < deadline:
            if marker.exists():
                break
            await asyncio.sleep(0.05)
        return ToolResult(
            tool_call_id="pending",
            tool_name="child_hold",
            content=(ToolContentBlock(type="text", data={"text": "hold finished"}),),
            status=ResultStatus.OK,
        )

    return Tool(
        definition=ToolDefinition(
            name="child_hold",
            description="TEST SEAM: hold the child running until a marker file appears.",
            input_schema={"type": "object"},
        ),
        validate_input=lambda arguments: arguments,
        run=run,
        static_effects=frozenset(),
        resolve_use=lambda arguments: ResolvedUse(frozenset(), ()),
        idempotent=False,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=60,
        max_model_output_bytes=4_000,
    )


async def _amain(workspace: Path, scripts: Mapping[str, Any], thread_id: str, out_file: Path) -> int:
    router = _Router(
        [_to_response(spec) for spec in scripts["parent"]],
        [_to_response(spec) for spec in scripts["child"]],
    )
    session = open_session(
        workspace,
        router,
        thread_id=thread_id,
        enable_subagents=True,
        extra_tools=(_hold_tool(workspace),),
        policy=PermissionPolicy(network=NetworkPolicy(decision="allow")),
    )
    try:
        result = await session.submit("acceptance: kill -9 child resume identity rebuild")
        if result.status is not TurnStatus.COMPLETED:
            print(json.dumps({"stage": "submit", "status": result.status.value, "reason": result.reason}))
            return 1
        # 报子身份给父测试(从 allocation 事实折叠)。
        child_thread_id = ""
        child_root_turn_id = ""
        for item in session.store.read_items(thread_id):
            if item.kind == ItemKind.CHILD_ALLOCATION:
                child_thread_id = str(item.payload.get("child_thread_id") or "")
                child_root_turn_id = str(item.payload.get("child_root_turn_id") or "")
        out_file.write_text(
            json.dumps({"child_thread_id": child_thread_id, "child_root_turn_id": child_root_turn_id}),
            encoding="utf-8",
        )
        # 保持进程存活:子 driver 在同一循环持续 running,直到被 SIGKILL。
        await asyncio.sleep(60.0)
    finally:
        session.close()
    return 1


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(f"usage: {sys.argv[0]} <workspace> <responses.json> <parent_thread> <out_file>")
    workspace = Path(sys.argv[1])
    scripts = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    raise SystemExit(asyncio.run(_amain(workspace, scripts, sys.argv[3], Path(sys.argv[4]))))


if __name__ == "__main__":
    main()