"""Task 14: update_plan 工具与无截断计划投影（B16 回归）。

旧实现中 execute → SDK 视图静默切 12 步（只取前 12），findings B16。
本文件验证：20 步完整回写、超限拒绝、重复 in_progress 拒绝、
step_id 自动补齐确定性、畸形 payload 不抛。

仓库无 pytest-asyncio：使用 asyncio.run() 包装，同 test_pipeline.py 风格。
"""

from __future__ import annotations

import ast
import asyncio
import json
import time
from pathlib import Path

import pytest

from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, execute
from praxis.core.planning import latest_plan
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PermissionPolicy
from praxis.protocol.tools import ToolCall
from praxis.store.log import Store
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
    tool = create_update_plan_tool()
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
        tool = create_update_plan_tool()
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
        tool = create_update_plan_tool()
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
    tool = create_update_plan_tool()

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


def test_update_plan_override_first(workspace: Path) -> None:
    """第二次 update_plan 覆盖第一次：latest_plan 返回后写入的步骤集。"""
    tool = create_update_plan_tool()

    store = Store(workspace / "db.db")
    try:
        cctx = make_ctx(store, EventBus(), workspace, {"update_plan": tool})

        # First: 3 steps
        steps_a = [{"step": f"a{i}", "status": "pending"} for i in range(3)]
        r1 = asyncio.run(execute(ToolCall("c1", "update_plan", {"plan": steps_a}), cctx))
        assert r1.status.value == "ok"

        # Second: 7 steps — should override
        steps_b = [{"step": f"b{i}", "status": "pending"} for i in range(7)]
        r2 = asyncio.run(execute(ToolCall("c2", "update_plan", {"plan": steps_b}), cctx))
        assert r2.status.value == "ok"

        items = store.read_items("thread-plan")
        proj = latest_plan(items)
        assert proj is not None
        assert len(proj.steps) == 7
        assert proj.steps[0].step == "b0"  # overwritten, not appended
    finally:
        store.close()


def test_error_does_not_change_existing_plan(workspace: Path) -> None:
    """出错/被拒的 update_plan 不改变既有计划。"""
    tool = create_update_plan_tool()

    store = Store(workspace / "db.db")
    try:
        cctx = make_ctx(store, EventBus(), workspace, {"update_plan": tool})

        # First: valid 5-step plan
        steps_a = [{"step": f"a{i}", "status": "pending"} for i in range(5)]
        r1 = asyncio.run(execute(ToolCall("c1", "update_plan", {"plan": steps_a}), cctx))
        assert r1.status.value == "ok"

        # Second: rejected (two in_progress)
        steps_bad = [
            {"step": "x", "status": "in_progress"},
            {"step": "y", "status": "in_progress"},
        ]
        r2 = asyncio.run(execute(ToolCall("c2", "update_plan", {"plan": steps_bad}), cctx))
        assert r2.error_code == "invalid_input"

        # Plan should still be 5 steps
        items = store.read_items("thread-plan")
        proj = latest_plan(items)
        assert proj is not None
        assert len(proj.steps) == 5
        assert proj.steps[0].step == "a0"
    finally:
        store.close()


def test_authoritative_malformed_plan_does_not_fall_back(workspace: Path) -> None:
    """最后一条成功的 update_plan 项畸形 → None，不回落到更旧的计划。"""
    store = Store(workspace / "db.db")
    try:
        tool = create_update_plan_tool()
        cctx = make_ctx(store, EventBus(), workspace, {"update_plan": tool})

        # First: a valid 3-step plan exists and is readable.
        steps = [{"step": f"a{i}", "status": "pending"} for i in range(3)]
        r1 = asyncio.run(execute(ToolCall("c1", "update_plan", {"plan": steps}), cctx))
        assert r1.status.value == "ok"
        assert latest_plan(store.read_items("thread-plan")) is not None

        # Second: an ok update_plan result whose payload is malformed, at a newer seq.
        def _write_bad(conn):
            conn.execute(
                "INSERT INTO items VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "bad-0", "thread-plan", "turn-bad", 999, "tool_result",
                    json.dumps({
                        "call_id": "c9",
                        "content": [],
                        "error_code": None,
                        "error_message": None,
                        "is_error": False,
                        "metadata": {"plan": "not-a-list", "revision": 3},
                        "status": "ok",
                        "tool_name": "update_plan",
                        "truncated": False,
                    }, sort_keys=True),
                    time.time(),
                ),
            )

        store.transact(_write_bad)

        # The newest success is authoritative; malformed → None, no old-plan fallback.
        assert latest_plan(store.read_items("thread-plan")) is None
    finally:
        store.close()


def test_malformed_store_payload_is_fail_open(workspace: Path) -> None:
    """通过真 store 往返后的畸形 payload → latest_plan 返回 None，永不抛。

    绕过手搓 Item 直达 freeze/thaw 层（已知坑）。
    """
    store = Store(workspace / "db.db")
    try:
        # Manually append malformed tool_result entries via execute pipeline path
        # We use a simpler approach: create valid items, mutate their serialized payload.
        # Actually simplest: use store transact + raw SQL for exact control.
        def _write_malware(conn):
            """Write a raw tool_result with known bad metadata into sqlite."""
            now = time.time()
            conn.execute(
                "INSERT INTO items VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "mal-0", "thread-plan", "turn-fail-open", 1, "tool_result",
                    json.dumps({
                        "call_id": "c0",
                        "content": [],
                        "error_code": None,
                        "error_message": None,
                        "is_error": False,
                        "metadata": {"plan": "not-a-list", "revision": 1},
                        "status": "ok",
                        "tool_name": "update_plan",
                        "truncated": False,
                    }, sort_keys=True),
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO items VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "mal-1", "thread-plan", "turn-fail-open", 2, "tool_result",
                    json.dumps({
                        "call_id": "c1",
                        "content": [],
                        "error_code": None,
                        "error_message": None,
                        "is_error": False,
                        "metadata": {"plan": [], "revision": 1},
                        "status": "ok",
                        "tool_name": "update_plan",
                        "truncated": False,
                    }, sort_keys=True),
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO items VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "mal-2", "thread-plan", "turn-fail-open", 3, "tool_result",
                    json.dumps({
                        "call_id": "c2",
                        "content": [],
                        "error_code": None,
                        "error_message": None,
                        "is_error": False,
                        "metadata": {"plan": [{"step": "x", "status": "bogus"}], "revision": True},
                        "status": "ok",
                        "tool_name": "update_plan",
                        "truncated": False,
                    }, sort_keys=True),
                    now,
                ),
            )

        store.transact(_write_malware)

        items = store.read_items("thread-plan")
        for item in items:
            result = latest_plan([item])
            assert result is None, (
                f"expected None for malformed item seq={item.seq}, got {result}"
            )
    finally:
        store.close()


def test_no_second_plan_ledger() -> None:
    """结构钉子：不存在第二份计划账本。

    - `plan_state` 不在 items.py 中；
    - MAX_PLAN_STEPS 在 praxis/ 里只有一处模块级定义，且字面值为 20。
    """
    repo_root = Path(__file__).parent.parent.parent
    items_text = repo_root / "praxis" / "protocol" / "items.py"
    assert "plan_state" not in items_text.read_text(), (
        "No second plan ledger in items.py"
    )

    # AST scan: module-level Assign/AnnAssign targeting MAX_PLAN_STEPS.
    definitions: list[ast.expr | None] = []
    for py_file in sorted((repo_root / "praxis").rglob("*.py")):
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            if any(
                isinstance(t, ast.Name) and t.id == "MAX_PLAN_STEPS" for t in targets
            ):
                definitions.append(node.value)

    assert len(definitions) == 1, (
        f"MAX_PLAN_STEPS must be defined exactly once under praxis/, "
        f"found {len(definitions)} definitions"
    )
    value = definitions[0]
    assert isinstance(value, ast.Constant) and type(value.value) is int, (
        f"MAX_PLAN_STEPS must be an int literal, got {ast.dump(value) if value else None}"
    )
    assert value.value == 20, f"MAX_PLAN_STEPS must be 20, got {value.value}"
