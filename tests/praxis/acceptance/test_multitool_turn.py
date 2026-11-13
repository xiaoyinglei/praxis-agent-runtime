"""Task 13 阶段验收:多工具 turn 端到端跑通(设计 §5.4/§5.12;硬规则 4)。

不是模拟:ScriptedAdapter 只是确定性的模型替身,循环是真实的
``run_turn``,工具是真实 builtins,文件落真实文件系统——
write 建文件 → bash ``cat`` 验证 → read 读回 → 模型收尾。

断言三件事:文件系统真实变化、items 全序列无缺无重、终态 COMPLETED。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.core.events import EventBus
from praxis.core.statemachine import TurnStatus
from praxis.core.turn import TurnDeps, TurnState, run_turn
from praxis.protocol.items import ItemKind
from praxis.protocol.models import ModelResponse, ModelToolCall, Usage
from praxis.protocol.permission import PermissionContext, SandboxAvailability
from praxis.protocol.policy import NetworkPolicy, PermissionPolicy
from praxis.protocol.tools import Tool
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import create_turn, get_turn
from praxis.tools.builtins.command import create_bash_tool
from praxis.tools.builtins.filesystem import create_read_tool, create_write_tool

THREAD_ID = "accept-multitool"
TURN_ID = "accept-turn"
NOTE = "note.txt"
CONTENT = "praxis phase1 acceptance\n"


def _resp(content: str, calls: tuple[tuple[str, str, dict[str, Any]], ...] = ()) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tuple(ModelToolCall(call_id=c, name=n, arguments=a) for c, n, a in calls),
        usage=Usage(input_tokens=3, output_tokens=5),
        finish_reason="tool_calls" if calls else "stop",
    )


def test_multitool_turn_end_to_end(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = Store(workspace / "store.db")
    try:
        bus = EventBus()
        store.ensure_thread(THREAD_ID)
        create_turn(store, turn_id=TURN_ID, thread_id=THREAD_ID, binding={})
        store.append_items(
            THREAD_ID,
            TURN_ID,
            [ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "建一个笔记文件并验证其内容"})],
        )
        tools: dict[str, Tool] = {
            "write": create_write_tool(workspace),
            "bash": create_bash_tool(workspace),
            "read": create_read_tool(workspace),
        }
        adapter = ScriptedAdapter(
            [
                _resp("", (("c1", "write", {"path": NOTE, "content": CONTENT}),)),
                _resp("", (("c2", "bash", {"command": f"cat {NOTE}"}),)),
                _resp("", (("c3", "read", {"path": NOTE}),)),
                _resp("文件已写入并验证:内容即 praxis phase1 acceptance"),
            ]
        )
        deps = TurnDeps(
            store=store,
            bus=bus,
            adapter=adapter,
            tools=tools,
            permissions=PermissionContext(
                workspace=str(workspace),
                # Task 4 迁移:真实 decide 下 bash 需 network allow(默认 ask 会第 2 步暂停)
                # + 沙箱可用占位(Task 5/7 接真实 probe/compiler)。
                policy=PermissionPolicy(network=NetworkPolicy(decision="allow")),
                mode="on-request",
                sandbox=SandboxAvailability(backend="fake", available=True),
            ),
            workspace=workspace,
            system_prompt="你是验收 agent:按 write → bash → read 的顺序完成任务。",
        )

        result = asyncio.run(run_turn(deps, TurnState(TURN_ID, THREAD_ID)))

        # ── 终态 ──
        assert result.status is TurnStatus.COMPLETED
        assert result.reason == "model_finished"
        assert "praxis phase1 acceptance" in result.output
        assert result.pending_approvals == () and result.unknown_calls == ()
        turn = get_turn(store, TURN_ID)
        assert turn is not None and turn.status == "completed"

        # ── 文件系统真实变化 ──
        assert (workspace / NOTE).read_text() == CONTENT

        # ── items 全序列:user → 4×(model_response [→ call+result]) → turn_status ──
        items = store.read_items(THREAD_ID)
        assert [i.kind for i in items] == [
            ItemKind.USER_MESSAGE,
            ItemKind.MODEL_RESPONSE,
            ItemKind.TOOL_CALL,
            ItemKind.TOOL_RESULT,
            ItemKind.MODEL_RESPONSE,
            ItemKind.TOOL_CALL,
            ItemKind.TOOL_RESULT,
            ItemKind.MODEL_RESPONSE,
            ItemKind.TOOL_CALL,
            ItemKind.TOOL_RESULT,
            ItemKind.MODEL_RESPONSE,
            ItemKind.TURN_STATUS,
        ]

        # ── 工具结果是真实执行产物:bash cat 输出了文件内容,read 原样读回 ──
        results_by_call = {i.payload["call_id"]: i for i in items if i.kind == ItemKind.TOOL_RESULT}
        bash_text: str = results_by_call["c2"].payload["content"][0]["data"]["text"]  # type: ignore[index]
        read_text: str = results_by_call["c3"].payload["content"][0]["data"]["text"]  # type: ignore[index]
        assert CONTENT in bash_text
        assert read_text == CONTENT
    finally:
        store.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
