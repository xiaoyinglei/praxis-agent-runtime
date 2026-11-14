"""Task 14: update_plan 工具与无截断计划投影（B16 回归）。

旧实现中 execute → SDK 视图静默切 12 步（只取前 12），findings B16。
本文件验证：20 步完整回写、超限拒绝、重复 in_progress 拒绝、
step_id 自动补齐确定性、畸形 payload 不抛。

仓库无 pytest-asyncio：使用 asyncio.run() 包装，同 test_pipeline.py 风格。
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, execute
from praxis.protocol.items import Item
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PermissionPolicy
from praxis.protocol.tools import ToolCall
from praxis.store.log import Store
from praxis.protocol.plan import latest_plan
from praxis.tools.builtins.planning import create_update_plan_tool


# --- Fixture helpers (parallel to test_pipeline.py pattern) -----------------

def make_ctx(
    store: Store,
    bus: EventBus,
    workspace: Path,
    tools: dict[str, object],
) -> ExecCtx:
    """Build an ExecCtx with a running turn so that PAUSED transition is legal."""
    policy = PermissionPolicy()
    from praxis.store.turns import create_turn as _create_turn

    turn = _create_turn(store, turn_id=f"turn-plan-{id(store)}", thread_id="thread-plan", binding={})
    return ExecCtx(
        store=store,
        bus=bus,
        workspace=workspace,
        tools=tools,
        permissions=PermissionContext(
            workspace=str(workspace.resolve()),
            policy=policy,
            mode="on-request",
        ),
        turn_id=turn.turn_id,
        thread_id="thread-plan",
        policy_revision=policy.revision,
    )


@pytest.fixture
def plan_ctx(store: Store, workspace: Path) -> ExecCtx:
    tool = create_update_plan_tool(workspace)
    return make_ctx(store, EventBus(), workspace, {"update_plan": tool})


# --- Test functions ----------------------------------------------------------


def test_twenty_steps_project_all_twenty(plan_ctx: ExecCtx) -> None:
    """B16 直接回归:execute 写 20 步 ⇒ latest_plan 读到 len(steps)==20。"""
    steps = [{"step": f"s{i}", "status": "pending"} for i in range(20)]
    result = asyncio.run(execute(ToolCall("c1", "update_plan", {"plan": steps}), plan_ctx))
    assert result.status.value == "ok"
    items = plan_ctx.store.read_items("thread-plan")
    proj = latest_plan(items)
    assert proj is not None
    assert len(proj.steps) == 20  # OLD impl had only 12


def test_twenty_one_steps_is_invalid_input(workspace: Path) -> None:
    """超出 MAX_PLAN_STEPS(20) → invalid_input。"""
    store = Store(workspace / "db.db")
    try:
        tool = create_update_plan_tool(workspace)
        cctx = make_ctx(store, EventBus(), workspace, {"update_plan": tool})

        steps = [{"step": f"s{i}", "status": "pending"} for i in range(21)]
        result = asyncio.run(execute(ToolCall("c1", "update_plan", {"plan": steps}), cctx))
        assert result.error_code == "invalid_input"
    finally:
        store.close()


def test_two_in_progress_rejected(workspace: Path) -> None:
    """两个 in_progress → invalid_input，消息含 'At most one step'。"""
    store = Store(workspace / "db.db")
    try:
        tool = create_update_plan_tool(workspace)
        cctx = make_ctx(store, EventBus(), workspace, {"update_plan": tool})

        steps = [
            {"step": "first", "status": "in_progress"},
            {"step": "second", "status": "in_progress"},
        ]
        result = asyncio.run(execute(ToolCall("c1", "update_plan", {"plan": steps}), cctx))
        assert result.error_code == "invalid_input"
        assert "At most one step" in result.error_message
    finally:
        store.close()


def test_step_id_auto_filled_deterministic(workspace: Path) -> None:
    """省略 step_id → 被自动补齐为 step_1, step_2, ...; 两次同输入结果一致。"""
    tool = create_update_plan_tool(workspace)

    steps = [
        {"step": "alpha", "status": "pending"},
        {"step": "beta", "status": "in_progress"},
    ]

    # First execution on fresh store
    store1 = Store(workspace / "db1.db")
    try:
        cctx1 = make_ctx(store1, EventBus(), workspace, {"update_plan": tool})
        asyncio.run(execute(ToolCall("c1", "update_plan", {"plan": steps}), cctx1))
        plan1 = latest_plan(store1.read_items("thread-plan"))
        assert plan1 is not None
        ids1 = [s.step_id for s in plan1.steps]
        assert ids1 == ["step_1", "step_2"]
    finally:
        store1.close()

    # Second execution on fresh store with same input → deterministic
    store2 = Store(workspace / "db2.db")
    try:
        cctx2 = make_ctx(store2, EventBus(), workspace, {"update_plan": tool})
        asyncio.run(execute(ToolCall("c1", "update_plan", {"plan": steps}), cctx2))
        plan2 = latest_plan(store2.read_items("thread-plan"))
        assert plan2 is not None
        ids2 = [s.step_id for s in plan2.steps]
        assert ids2 == ids1
    finally:
        store2.close()


def test_malformed_plan_payload_is_fail_open(workspace: Path) -> None:
    """畸形 metadata（非 list、条目缺键、类型不对）→ latest_plan 返回 None，永不抛。"""
    payloads = [
        # plan 不是 list
        {"metadata": {"plan": "not-a-list", "revision": 1}},
        # plan 是 list 但条目缺 status
        {"metadata": {"plan": [{"step": "x"}], "revision": 1}},
        # plan 是 list 但条目缺少 step
        {"metadata": {"plan": [{"status": "pending"}], "revision": 1}},
        # 非法 status 值
        {"metadata": {"plan": [{"step": "x", "status": "bogus"}], "revision": 1}},
        # plan 缺失
        {"metadata": {"revision": 1}},
        # plan 不是 mapping 元素
        {"metadata": {"plan": ["just-string"], "revision": 1}},
        # revision 缺失
        {"metadata": {"plan": [{"step": "x", "status": "pending"}]}},
        # 空 plan
        {"metadata": {"plan": [], "revision": 1}},
        # error item should also return None
        {
            "metadata": {"plan": [{"step": "x", "status": "pending"}], "revision": 1},
            "_error": True,
        },
    ]

    for idx, meta in enumerate(payloads):
        error_flag = meta.pop("_error", False) if isinstance(meta, dict) else False
        if isinstance(meta, dict) and error_flag:
            payload_meta = meta.copy()
            item_meta = {"metadata": payload_meta.get("metadata", {}), "is_error": True}
        else:
            item_meta = meta

        item = Item(
            item_id=f"mal-{idx}",
            thread_id="thread-mal",
            turn_id="turn-mal",
            seq=idx + 1,
            kind="tool_result",
            payload={
                "call_id": f"cc{idx}",
                "tool_name": "update_plan",
                "status": "ok" if not error_flag else "error",
                "content": [],
                "is_error": error_flag,
                "error_code": None,
                "error_message": None,
                "truncated": False,
                "metadata": item_meta.get("metadata", item_meta.get("metadata", {})),
            },
            created_at=0.0,
        )
        result = latest_plan([item])
        assert result is None, f"payload {idx}: expected None, got {result}"


def test_no_second_plan_ledger() -> None:
    """结构钉子：不存在第二份计划账本。

    - `plan_state` 不在 items.py 中；
    - MAX_PLAN_STEPS 在 praxis/ 里只有一处定义。
    """
    from pathlib import Path as _P  # noqa: PLC0415

    items_text = _P(__file__).parent.parent.parent / "praxis" / "protocol" / "items.py"
    assert "plan_state" not in items_text.read_text(), (
        "No second plan ledger in items.py"
    )

    # Count lines defining MAX_PLAN_STEPS via rgrep
    praxis_dir = _P(__file__).parent.parent.parent / "praxis"
    result = subprocess.run(
        ["rg", "-n", r"^MAX_PLAN_STEPS\s*=\s*20$"],
        capture_output=True, text=True, cwd=str(praxis_dir),
    )
    lines = [l for l in result.stdout.splitlines() if l]
    assert len(lines) == 1, (
        f"MAX_PLAN_STEPS should be defined exactly once; found:\n" + "\n".join(lines)
    )
