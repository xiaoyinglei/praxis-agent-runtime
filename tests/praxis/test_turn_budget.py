"""阶段4 T3:turn/session 预算接线——记账、跨 resume steps、耗尽→PAUSED+补预算续跑。

控制器裁决在此钉死:
- R-T3-1 steps 双闸:``limits.steps``(树维度)耗尽 → PAUSED "budget_exhausted"
  (约定预算,可 resume+extend 续跑);``max_steps`` 硬护栏 → FAILED
  "step_limit_exceeded"(失控防御,保持 FAILED)。护栏比较基数改 R-PF-4 口径
  (fold 声明轮数 + 本次重放轮自增),跨 pause/resume 累计;
- R-T3-2 binding 容错:键缺失/None → UNLIMITED(历史 turn 零扰动);畸形值 →
  UNLIMITED + 一次性诊断(fail-open,不炸既有 turn);
- R-T3-3 extend 校验先于 transition:负值/bool/未知维度 → 拒绝,零写入零 transition;
- R-T3-4 unit_costs 由 composition 从 ModelSpec.cost 注入;无单价而 cost 限额
  生效 → pricing_unavailable fail-closed(T2 admit 判据兜底)。

无限预算零扰动是主证据之一:UNLIMITED tracker 与无 tracker 的 item 序列逐值一致。
仓库无 pytest-asyncio:与 test_turn.py 同风格,asyncio.run 包装。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from praxis.adapters.config.store import ConnectionEntry, ModelConfigStore, ModelEntry
from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.core import session as session_module
from praxis.core.budget import BudgetTracker
from praxis.core.events import EventBus
from praxis.core.recovery import sweep
from praxis.core.statemachine import TurnStatus, transition
from praxis.core.turn import TurnDeps, TurnResult, TurnState, run_turn
from praxis.interfaces import sdk as sdk_pkg
from praxis.interfaces.cli import app
from praxis.interfaces.sdk import Limits, open_session
from praxis.protocol.budget import UNLIMITED
from praxis.protocol.items import ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import CostMicros, ModelResponse, ModelSpec, ModelToolCall, Usage, WireRequest
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PermissionPolicy
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolResult,
    json_schema_input,
)
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import create_turn, get_turn
from tests.praxis.conftest import make_minimal_tool

THREAD = "thread-1"
TURN = "turn-1"
_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}


# ---------------------------------------------------------------------------
# turn 级脚手架(与 test_turn.py 同风格)
# ---------------------------------------------------------------------------


def _turn(store: Store, *, thread_id: str = THREAD, turn_id: str = TURN) -> None:
    store.ensure_thread(thread_id)
    create_turn(store, turn_id=turn_id, thread_id=thread_id, binding={})


def _resp(
    content: str,
    calls: tuple[tuple[str, str, dict[str, Any]], ...] = (),
    *,
    usage: Usage = Usage(input_tokens=3, output_tokens=5),
) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tuple(ModelToolCall(call_id=c, name=n, arguments=a) for c, n, a in calls),
        usage=usage,
        finish_reason="tool_calls" if calls else "stop",
    )


def _toy_tool(name: str, *, text: str = "ok") -> Tool:
    async def run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        return ToolResult(
            tool_call_id="pending",
            tool_name=name,
            content=(ToolContentBlock(type="text", data={"text": text}),),
            status=ResultStatus.OK,
        )

    return Tool(
        definition=ToolDefinition(name=name, description=f"{name} toy", input_schema=_EMPTY_SCHEMA),
        validate_input=json_schema_input(_EMPTY_SCHEMA),
        run=run,
        static_effects=frozenset(),
        resolve_use=lambda arguments: ResolvedUse(effects=frozenset(), targets=()),
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30,
        max_model_output_bytes=64_000,
    )


def _deps(
    store: Store,
    bus: Any,
    workspace: Path,
    adapter: Any,
    tools: Mapping[str, Tool],
    *,
    budget: BudgetTracker | None = None,
    max_steps: int = 50,
    compact_limit_tokens: int | None = None,
    measure_tokens: Callable[[WireRequest], int] | None = None,
) -> TurnDeps:
    return TurnDeps(
        store=store,
        bus=bus,
        adapter=adapter,
        tools=tools,
        permissions=PermissionContext(workspace=str(workspace), policy=PermissionPolicy(), mode="on-request"),
        workspace=workspace,
        system_prompt="test-agent system prompt",
        max_steps=max_steps,
        budget=budget,
        compact_limit_tokens=compact_limit_tokens,
        measure_tokens=measure_tokens,
    )


def _tracker(store: Store, limits: Limits, *, unit_costs: CostMicros | None = None, turn_id: str = TURN) -> BudgetTracker:
    return BudgetTracker(
        store=store, thread_id=THREAD, turn_id=turn_id, root_limits=limits, unit_costs=unit_costs
    )


def _kinds(store: Store, turn_id: str = TURN) -> list[str]:
    return [item.kind for item in store.read_turn_items(turn_id)]


# ---------------------------------------------------------------------------
# brief ①:steps 树维度耗尽 → PAUSED(非 FAILED),pending_approvals 为空
# ---------------------------------------------------------------------------


def test_steps_budget_exhaustion_pauses_not_fails(store: Store, bus: Any, workspace: Path) -> None:
    _turn(store)
    adapter = ScriptedAdapter(
        [
            _resp("", (("c1", "noop", {}),)),
            _resp("", (("c2", "noop", {}),)),
            _resp("", (("c3", "noop", {}),)),
            _resp("done"),
        ]
    )
    deps = _deps(
        store, bus, workspace, adapter, {"noop": _toy_tool("noop")}, budget=_tracker(store, Limits(steps=2))
    )

    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.PAUSED  # 树预算是约定,不是失控:绝不 FAILED
    assert result.reason == "budget_exhausted"
    assert result.budget_dimension == "steps"  # R-T3-6:耗尽维透出,CLI 据此点名出路
    assert result.pending_approvals == ()
    turn = get_turn(store, TURN)
    assert turn is not None and turn.status == "paused"
    kinds = _kinds(store)
    assert kinds.count(ItemKind.MODEL_RESPONSE) == 2  # 第 3 次 dispatch 被挡在门前
    assert kinds.count(ItemKind.TOOL_CALL) == 2


# ---------------------------------------------------------------------------
# brief ②:token 维耗尽 → 下一次 dispatch 前 PAUSED;settle 不截断事实
# ---------------------------------------------------------------------------


def test_token_exhaustion_pauses_before_next_dispatch_settle_not_truncated(
    store: Store, bus: Any, workspace: Path
) -> None:
    """limits.tokens=100 → 保护尾 10,普通有效上限 90。resp2 结算后 used=98:
    事实如实落库(不截断),第 3 次 dispatch 前被拒。"""
    _turn(store)
    adapter = ScriptedAdapter(
        [
            _resp("", (("c1", "noop", {}),), usage=Usage(input_tokens=3, output_tokens=5)),
            _resp("", (("c2", "noop", {}),), usage=Usage(input_tokens=85, output_tokens=5)),
            _resp("不可达"),
        ]
    )
    tracker = _tracker(store, Limits(tokens=100))
    deps = _deps(store, bus, workspace, adapter, {"noop": _toy_tool("noop")}, budget=tracker)

    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.PAUSED and result.reason == "budget_exhausted"
    assert result.budget_dimension == "tokens"
    responses = [i for i in store.read_turn_items(TURN) if i.kind == ItemKind.MODEL_RESPONSE]
    assert len(responses) == 2  # 第 3 次请求从未发出
    last_usage = responses[-1].payload["usage"]
    assert isinstance(last_usage, Mapping)
    assert last_usage["input_tokens"] == 85 and last_usage["output_tokens"] == 5  # settle 不截断
    node = tracker.refresh()
    assert node.used.tokens == 98  # 超额诚实暴露在 exposure,不下钳事实
    turn = get_turn(store, TURN)
    assert turn is not None and turn.status == "paused"


# ---------------------------------------------------------------------------
# brief ③:compaction 逃生——逼近限额时压缩续作可入保护尾(真实压缩一次)
# ---------------------------------------------------------------------------


def _compaction_scenario(
    store: Store, bus: Any, workspace: Path, *, compact: bool
) -> tuple[TurnResult, BudgetTracker]:
    """tokens=1000(尾 100,普通有效 900):两轮各 450 → used=900 恰好触顶。
    工具输出 3000 字符使第 3 轮出站请求超压缩阈值(1000):压缩摘要(usage 10)
    与续作(usage 20)都走保护尾逃生面;无压缩时第 3 轮门前即 PAUSED。"""
    _turn(store)
    store.append_items(
        THREAD,
        TURN,
        [ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "hist"}) for _ in range(3)],
    )
    adapter = ScriptedAdapter(
        [
            _resp("", (("c1", "big", {}),), usage=Usage(input_tokens=400, output_tokens=50)),
            _resp("", (("c2", "big", {}),), usage=Usage(input_tokens=400, output_tokens=50)),
            _resp("摘要正文", usage=Usage(input_tokens=5, output_tokens=5)),
            _resp("done", usage=Usage(input_tokens=10, output_tokens=10)),
        ]
    )
    tracker = _tracker(store, Limits(tokens=1000))
    deps = _deps(
        store,
        bus,
        workspace,
        adapter,
        {"big": _toy_tool("big", text="x" * 3000)},
        budget=tracker,
        compact_limit_tokens=1000 if compact else None,
        measure_tokens=(lambda req: sum(len(m.content) for m in req.messages) // 4) if compact else None,
    )
    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))
    return result, tracker


def test_compaction_continuation_enters_protected_tail(store: Store, bus: Any, workspace: Path) -> None:
    result, tracker = _compaction_scenario(store, bus, workspace, compact=True)

    assert result.status is TurnStatus.COMPLETED and result.output == "done"
    kinds = _kinds(store)
    assert kinds.count(ItemKind.SUMMARY) == 1  # 真实压缩恰好一次
    assert kinds.count(ItemKind.COMPACTION_BOUNDARY) == 1
    assert tracker.refresh().used.tokens == 930  # 900 + 摘要 10 + 续作 20,全部如实入账


def test_compaction_escape_is_load_bearing(store: Store, bus: Any, workspace: Path) -> None:
    """同一剧本关闭压缩:used=900 ≥ 普通有效上限 900 → 第 3 轮门前 PAUSED。
    与上一测试对照,证明保护尾逃生是续作完成的唯一原因。"""
    result, tracker = _compaction_scenario(store, bus, workspace, compact=False)

    assert result.status is TurnStatus.PAUSED and result.reason == "budget_exhausted"
    assert _kinds(store).count(ItemKind.MODEL_RESPONSE) == 2
    assert _kinds(store).count(ItemKind.SUMMARY) == 0
    assert tracker.refresh().used.tokens == 900


def test_normal_dispatch_after_continuation_cannot_eat_protected_tail(
    store: Store, bus: Any, workspace: Path
) -> None:
    """R-T3-7 负例:保护尾豁免逐次 dispatch——压缩成功置位、下一次 admit 消费
    即复位;续作之后第 2+ 轮普通调用不得入尾。tokens=1000(有效 900):两轮 900
    → 摘要 10 → 续作工具轮 20 → used=930;再下一轮普通准入 930≥900 → PAUSED
    (锁存实现会带 needs_compaction=True 放行到 COMPLETED——本测试钉死危险方向)。"""
    _turn(store)
    store.append_items(
        THREAD,
        TURN,
        [ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "hist"}) for _ in range(3)],
    )
    adapter = ScriptedAdapter(
        [
            _resp("", (("c1", "big", {}),), usage=Usage(input_tokens=400, output_tokens=50)),
            _resp("", (("c2", "big", {}),), usage=Usage(input_tokens=400, output_tokens=50)),
            _resp("摘要正文", usage=Usage(input_tokens=5, output_tokens=5)),
            _resp("", (("c3", "big", {}),), usage=Usage(input_tokens=10, output_tokens=10)),
            _resp("不可达"),
        ]
    )
    tracker = _tracker(store, Limits(tokens=1000))
    deps = _deps(
        store,
        bus,
        workspace,
        adapter,
        {"big": _toy_tool("big", text="x" * 3000)},
        budget=tracker,
        compact_limit_tokens=1000,
        measure_tokens=lambda req: sum(len(m.content) for m in req.messages) // 4,
    )

    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.PAUSED and result.reason == "budget_exhausted"
    assert result.budget_dimension == "tokens"
    kinds = _kinds(store)
    assert kinds.count(ItemKind.MODEL_RESPONSE) == 3  # r1/r2/续作(摘要落 summary item);第 5 条剧本未消费
    assert kinds.count(ItemKind.SUMMARY) == 1
    assert tracker.refresh().used.tokens == 930


def test_over_threshold_without_compaction_readmits_as_normal(store: Store, bus: Any, workspace: Path) -> None:
    """R-T3-7 同源边界:超阈值预授豁免只给摘要调用;压缩未产出摘要(无可覆盖
    区间)时,同轮普通 dispatch 必须按 needs_compaction=False 重新准入——
    used=900≥有效 900 → PAUSED,不得借预授放行(旧实现会 COMPLETED)。"""
    _turn(store)  # 不预置历史:6 条可压缩 item < keep_tail 8 → 无覆盖区间
    adapter = ScriptedAdapter(
        [
            _resp("", (("c1", "big", {}),), usage=Usage(input_tokens=400, output_tokens=50)),
            _resp("", (("c2", "big", {}),), usage=Usage(input_tokens=400, output_tokens=50)),
            _resp("不可达"),
        ]
    )
    tracker = _tracker(store, Limits(tokens=1000))
    deps = _deps(
        store,
        bus,
        workspace,
        adapter,
        {"big": _toy_tool("big", text="x" * 3000)},
        budget=tracker,
        compact_limit_tokens=1000,
        measure_tokens=lambda req: sum(len(m.content) for m in req.messages) // 4,
    )

    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.PAUSED and result.reason == "budget_exhausted"
    kinds = _kinds(store)
    assert kinds.count(ItemKind.MODEL_RESPONSE) == 2  # 普通 dispatch 从未发出
    assert kinds.count(ItemKind.SUMMARY) == 0
    assert tracker.refresh().used.tokens == 900


# ---------------------------------------------------------------------------
# brief ④:pricing_unavailable fail-closed(cost 限额生效而单价缺失/全零)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit_costs", [None, CostMicros()], ids=["missing", "all-zero"])
def test_pricing_unavailable_pauses_before_dispatch(
    store: Store, bus: Any, workspace: Path, unit_costs: CostMicros | None
) -> None:
    _turn(store)
    adapter = ScriptedAdapter([_resp("不可达")])
    deps = _deps(
        store, bus, workspace, adapter, {}, budget=_tracker(store, Limits(cost_micros=1), unit_costs=unit_costs)
    )

    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.PAUSED and result.reason == "pricing_unavailable"
    assert result.budget_dimension is None  # 只有 budget_exhausted 填耗尽维
    assert _kinds(store).count(ItemKind.MODEL_RESPONSE) == 0  # 绝不按 0 计价后放行
    turn = get_turn(store, TURN)
    assert turn is not None and turn.status == "paused"


def test_cost_exhaustion_pauses_with_budget_exhausted(store: Store, bus: Any, workspace: Path) -> None:
    """单价 1 micro/token:resp1 cost 8、resp2 cost 13 → 累计 21 ≥ 限额 20。"""
    _turn(store)
    adapter = ScriptedAdapter(
        [
            _resp("", (("c1", "noop", {}),), usage=Usage(input_tokens=3, output_tokens=5)),
            _resp("", (("c2", "noop", {}),), usage=Usage(input_tokens=8, output_tokens=5)),
            _resp("不可达"),
        ]
    )
    tracker = _tracker(
        store,
        Limits(cost_micros=20),
        unit_costs=CostMicros(input_per_1m=1_000_000, output_per_1m=1_000_000),
    )
    deps = _deps(store, bus, workspace, adapter, {"noop": _toy_tool("noop")}, budget=tracker)

    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.PAUSED and result.reason == "budget_exhausted"
    assert result.budget_dimension == "cost_micros"
    assert tracker.refresh().used.cost_micros == 21


# ---------------------------------------------------------------------------
# brief ⑤:无限预算零扰动 + R-T3-1 双闸并存
# ---------------------------------------------------------------------------


def test_unlimited_tracker_item_sequence_identical_to_no_budget(store: Store, bus: Any, workspace: Path) -> None:
    script = [_resp("", (("c1", "noop", {}),)), _resp("done")]
    tools = {"noop": _toy_tool("noop")}
    _turn(store)
    baseline = asyncio.run(run_turn(_deps(store, bus, workspace, ScriptedAdapter(script), tools), TurnState(TURN, THREAD)))
    baseline_kinds = _kinds(store)

    _turn(store, turn_id="turn-2")
    adapter = ScriptedAdapter(script)
    deps = _deps(store, bus, workspace, adapter, tools, budget=_tracker(store, UNLIMITED, turn_id="turn-2"))
    result = asyncio.run(run_turn(deps, TurnState("turn-2", THREAD)))

    assert baseline.status is TurnStatus.COMPLETED
    assert result.status is TurnStatus.COMPLETED and result.output == "done"
    assert _kinds(store, "turn-2") == baseline_kinds  # item 序列逐值一致


def test_unlimited_budget_keeps_max_steps_guardrail_failed(store: Store, bus: Any, workspace: Path) -> None:
    """R-T3-1 双闸:树无限不改变硬护栏语义——max_steps 耗尽仍是 FAILED。"""
    _turn(store)
    adapter = ScriptedAdapter([_resp("", (("c1", "noop", {}),)), _resp("", (("c2", "noop", {}),))])
    deps = _deps(
        store, bus, workspace, adapter, {"noop": _toy_tool("noop")}, budget=_tracker(store, UNLIMITED), max_steps=2
    )

    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.FAILED and result.reason == "step_limit_exceeded"


# ---------------------------------------------------------------------------
# R-PF-4 / R-T3-1:steps 初值 = fold 声明轮数;护栏基数跨 resume 累计
# ---------------------------------------------------------------------------


def _seed_round(store: Store, call_id: str) -> None:
    """预置一个完整声明轮(model_response + tool_call + tool_result),usage 0。"""
    store.append_items(
        THREAD,
        TURN,
        [
            ItemEntry(
                kind=ItemKind.MODEL_RESPONSE,
                payload={
                    "content": "",
                    "tool_calls": ({"call_id": call_id, "name": "noop", "arguments": {}},),
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "finish_reason": "tool_calls",
                },
            ),
            ItemEntry(kind=ItemKind.TOOL_CALL, payload={"call_id": call_id, "tool_name": "noop"}),
            ItemEntry(
                kind=ItemKind.TOOL_RESULT,
                payload={
                    "call_id": call_id,
                    "tool_name": "noop",
                    "status": "ok",
                    "content": ({"type": "text", "data": {"text": "ok"}},),
                },
            ),
        ],
    )


def test_state_steps_initialized_from_fold(store: Store, bus: Any, workspace: Path) -> None:
    _turn(store)
    _seed_round(store, "p1")
    _seed_round(store, "p2")
    adapter = ScriptedAdapter([_resp("done")])
    deps = _deps(store, bus, workspace, adapter, {"noop": _toy_tool("noop")})
    state = TurnState(TURN, THREAD)

    result = asyncio.run(run_turn(deps, state))

    assert result.status is TurnStatus.COMPLETED
    assert state.steps == 2  # 初值来自 fold 声明轮数,本次无工具轮不自增


def test_step_guardrail_accumulates_across_resume(store: Store, bus: Any, workspace: Path) -> None:
    """R-T3-1 行为变化钉:limits.steps=2 先 PAUSED;extend 后续跑,护栏基数
    从 fold 的 2 起算(旧"每次 run_turn 从 0 计"废除)——max_steps=3 在第 3 轮
    完成后即 FAILED,而不是重置后跑完剧本。"""
    _turn(store)
    adapter = ScriptedAdapter(
        [
            _resp("", (("c1", "noop", {}),)),
            _resp("", (("c2", "noop", {}),)),
            _resp("", (("c3", "noop", {}),)),
            _resp("done"),
        ]
    )
    deps = _deps(
        store, bus, workspace, adapter, {"noop": _toy_tool("noop")}, budget=_tracker(store, Limits(steps=2)), max_steps=3
    )
    state = TurnState(TURN, THREAD)

    paused = asyncio.run(run_turn(deps, state))
    assert paused.status is TurnStatus.PAUSED and paused.reason == "budget_exhausted"
    assert state.steps == 2

    store.append_items(
        THREAD, TURN, [ItemEntry(kind=ItemKind.BUDGET_EXTENSION, payload={"limits_delta": {"steps": 5}})]
    )
    transition(store, bus, TURN, TurnStatus.RUNNING, reason="budget_extended")
    result = asyncio.run(run_turn(deps, state))

    assert result.status is TurnStatus.FAILED and result.reason == "step_limit_exceeded"
    assert _kinds(store).count(ItemKind.MODEL_RESPONSE) == 3  # 第 4 条剧本从未消费


# ---------------------------------------------------------------------------
# session 级脚手架(composition 装配;Session 构造只发生在 composition)
# ---------------------------------------------------------------------------


@pytest.fixture
def open_ws(workspace: Path) -> Iterator[Callable[..., Any]]:
    made: list[Any] = []

    def _open(adapter: Any, **kwargs: Any) -> Any:
        session = open_session(workspace, adapter, **kwargs)
        made.append(session)
        return session

    yield _open
    for session in reversed(made):
        session.close()


def _sresp(
    content: str, calls: tuple[tuple[str, str, dict[str, Any]], ...] = ()
) -> ModelResponse:
    return _resp(content, calls)


def _other_store(workspace: Path, fn: Callable[[Store], Any]) -> Any:
    other = Store(workspace / ".praxis" / "praxis.sqlite")
    try:
        return fn(other)
    finally:
        other.close()


# ---------------------------------------------------------------------------
# submit binding 携带 budget payload(R-T3-2 写端)
# ---------------------------------------------------------------------------


def test_submit_binding_carries_budget_payload(workspace: Path, open_ws: Callable[..., Any]) -> None:
    session = open_ws(ScriptedAdapter([_sresp("hi")]), budget=Limits(tokens=100, steps=3))

    result = asyncio.run(session.submit("go"))

    assert result.status is TurnStatus.COMPLETED
    assert session.status(result.turn_id).binding == {
        "model": "scripted",
        "budget": {"tokens": 100, "steps": 3},
    }


def test_submit_default_binding_unchanged(workspace: Path, open_ws: Callable[..., Any]) -> None:
    """UNLIMITED(缺省)不写 budget 键——历史 turn 形状零扰动。"""
    session = open_ws(ScriptedAdapter([_sresp("hi")]))

    result = asyncio.run(session.submit("go"))

    assert session.status(result.turn_id).binding == {"model": "scripted"}


# ---------------------------------------------------------------------------
# brief ①:耗尽 → PAUSED;resume(extend) 续跑成功(R-T3-3 校验序)
# ---------------------------------------------------------------------------


def test_budget_exhausted_resume_extend_continues(workspace: Path, open_ws: Callable[..., Any]) -> None:
    adapter = ScriptedAdapter(
        [
            _sresp("", (("c1", "minimal", {}),)),
            _sresp("", (("c2", "minimal", {}),)),
            _sresp("", (("c3", "minimal", {}),)),
            _sresp("done"),
        ]
    )
    session = open_ws(adapter, budget=Limits(steps=2), tools=[make_minimal_tool()])

    paused = asyncio.run(session.submit("go"))
    assert paused.status is TurnStatus.PAUSED and paused.reason == "budget_exhausted"
    assert paused.pending_approvals == ()

    # 无 approval/reconcile/extend 的裸 resume 仍被守卫拒绝(零写入)。
    with pytest.raises(ValueError, match="approval or reconcile"):
        asyncio.run(session.resume(paused.turn_id))
    assert session.status(paused.turn_id).status == "paused"

    done = asyncio.run(session.resume(paused.turn_id, extend={"steps": 5}))

    assert done.status is TurnStatus.COMPLETED and done.output == "done"
    items = session.items()
    extensions = [i for i in items if i.kind == ItemKind.BUDGET_EXTENSION]
    assert len(extensions) == 1
    assert extensions[0].payload == {"limits_delta": {"steps": 5}}  # EXTENSION_KEYS 常量键
    reasons = [i.payload.get("reason") for i in items if i.kind == ItemKind.TURN_STATUS]
    assert "budget_extended" in reasons


def test_extend_validation_before_transition_zero_writes(workspace: Path, open_ws: Callable[..., Any]) -> None:
    """R-T3-3:负值/bool/未知维度一律拒绝,且拒绝发生在任何 transition/写入之前。"""
    adapter = ScriptedAdapter([_sresp("", (("c1", "minimal", {}),)), _sresp("done")])
    session = open_ws(adapter, budget=Limits(steps=1), tools=[make_minimal_tool()])
    paused = asyncio.run(session.submit("go"))
    assert paused.status is TurnStatus.PAUSED and paused.reason == "budget_exhausted"
    before = [(i.seq, i.kind) for i in session.items()]

    for bad in ({"steps": -1}, {"tokens": True}, {"bogus": 5}, {"cost_micros": -3}):
        with pytest.raises(ValueError):
            asyncio.run(session.resume(paused.turn_id, extend=bad))
        assert session.status(paused.turn_id).status == "paused"
        assert [(i.seq, i.kind) for i in session.items()] == before  # 零 transition、零 item

    # 0 是合法非负增量:写入后续跑,仍耗尽则如实再 PAUSED(不假装续跑成功)。
    again = asyncio.run(session.resume(paused.turn_id, extend={"steps": 0}))
    assert again.status is TurnStatus.PAUSED and again.reason == "budget_exhausted"
    extensions = [i for i in session.items() if i.kind == ItemKind.BUDGET_EXTENSION]
    assert extensions[-1].payload == {"limits_delta": {"steps": 0}}

    done = asyncio.run(session.resume(paused.turn_id, extend={"steps": 2}))
    assert done.status is TurnStatus.COMPLETED and done.output == "done"


# ---------------------------------------------------------------------------
# R-T3-2:binding 缺失/畸形 → UNLIMITED fail-open(历史 turn 零扰动)
# ---------------------------------------------------------------------------


def _dead_turn(workspace: Path, thread_id: str, binding: Mapping[str, JsonValue]) -> str:
    def _make(other: Store) -> str:
        turn_id = f"dead-{len(binding)}-budget"
        create_turn(other, turn_id=turn_id, thread_id=thread_id, binding=binding)
        other.append_items(thread_id, turn_id, [ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "old"})])
        assert turn_id in sweep(other, EventBus(), workspace)
        return turn_id

    return _other_store(workspace, _make)


@pytest.mark.parametrize(
    "budget_value",
    [{"tokens": "lots"}, "junk", {"steps": True}, {"steps": 1.5}],
    ids=["non-int", "non-mapping", "bool", "float"],
)
def test_malformed_budget_binding_fail_open(
    workspace: Path, open_ws: Callable[..., Any], caplog: pytest.LogCaptureFixture, budget_value: Any
) -> None:
    first = open_ws(ScriptedAdapter([]), thread_id="budget-thread")
    turn_id = _dead_turn(workspace, first.thread_id, {"model": "x", "budget": budget_value})
    first.close()

    second = open_ws(ScriptedAdapter([_sresp("back")]), thread_id="budget-thread")
    with caplog.at_level(logging.WARNING, logger="praxis.core.session"):
        result = asyncio.run(second.resume(turn_id, confirm=True))

    assert result.status is TurnStatus.COMPLETED and result.output == "back"  # 不炸既有 turn
    assert "malformed budget binding" in caplog.text  # 一次性诊断


def test_missing_budget_binding_no_warning(workspace: Path, open_ws: Callable[..., Any], caplog: pytest.LogCaptureFixture) -> None:
    first = open_ws(ScriptedAdapter([]), thread_id="budget-thread")
    turn_id = _dead_turn(workspace, first.thread_id, {"model": "x"})
    first.close()

    second = open_ws(ScriptedAdapter([_sresp("back")]), thread_id="budget-thread")
    with caplog.at_level(logging.WARNING, logger="praxis.core.session"):
        result = asyncio.run(second.resume(turn_id, confirm=True))

    assert result.status is TurnStatus.COMPLETED
    assert "malformed budget binding" not in caplog.text


# ---------------------------------------------------------------------------
# R-T3-4:unit_costs 注入(composition 从 ModelSpec.cost 取一次)
# ---------------------------------------------------------------------------


def test_explicit_adapter_without_pricing_fail_closed(workspace: Path, open_ws: Callable[..., Any]) -> None:
    """显式 adapter 无 ModelSpec → unit_costs None → cost 限额生效即 fail-closed。"""
    session = open_ws(ScriptedAdapter([_sresp("hi")]), budget=Limits(cost_micros=10))

    result = asyncio.run(session.submit("go"))

    assert result.status is TurnStatus.PAUSED and result.reason == "pricing_unavailable"


def test_composition_injects_unit_costs_from_config_store(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ModelConfigStore(tmp_path / "models.db")
    try:
        connection = ConnectionEntry("local", "ollama", "http://127.0.0.1:11434")
        spec = ModelSpec(
            model_id="priced", provider="ollama", cost=CostMicros(input_per_1m=100, output_per_1m=200)
        )
        config.upsert_connection(connection)
        revision = config.upsert_model(ModelEntry(spec=spec, connection=connection, revision=0))
        config.set_default_model("priced", expected_revision=revision)

        captured: list[TurnDeps] = []

        async def fake_run_turn(deps: TurnDeps, state: TurnState, *, resume_calls: Any = ()) -> TurnResult:
            captured.append(deps)
            return TurnResult(
                turn_id=state.turn_id, status=TurnStatus.COMPLETED, output="ok", reason="model_finished"
            )

        monkeypatch.setattr(session_module, "run_turn", fake_run_turn)
        session = open_session(workspace, model="priced", config_store=config, budget=Limits(cost_micros=10_000))
        try:
            asyncio.run(session.submit("hi"))
            assert len(captured) == 1
            tracker = captured[0].budget
            assert tracker is not None
            node = tracker.refresh()  # store 关闭前读(store 生命周期归 session)
        finally:
            session.close()

        assert node.unit_costs == CostMicros(input_per_1m=100, output_per_1m=200)
        assert node.limits.cost_micros == 10_000  # root limits 来自 turn binding
    finally:
        config.close()


# ---------------------------------------------------------------------------
# CLI 面:run --max-* / resume --extend-* / status 文案
# ---------------------------------------------------------------------------

runner = CliRunner()
_CLEAN_ENV = {
    "PRAXIS_MODEL_ID": None,
    "PRAXIS_MODEL": None,
    "PRAXIS_BASE_URL": None,
    "PRAXIS_API_KEY": None,
    "OPENAI_API_KEY": None,
}


def _write_script(tmp_path: Path, name: str, entries: list[dict[str, Any]]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _turn_id_from_output(output: str) -> str:
    match = re.search(r"turn ([0-9a-f]{32})", output)
    assert match is not None, output
    return match.group(1)


def _list_call(call_id: str) -> dict[str, Any]:
    return {"content": "", "tool_calls": [{"call_id": call_id, "name": "list_files", "arguments": {}}]}


def test_cli_run_max_steps_pauses_with_resume_hint(workspace: Path, tmp_path: Path) -> None:
    script = _write_script(tmp_path, "steps.json", [_list_call("c1"), _list_call("c2"), {"content": "done"}])
    paused = runner.invoke(
        app,
        ["run", "go", "--workspace", str(workspace), "--thread", "b1", "--scripted", str(script), "--max-steps", "1"],
        env=_CLEAN_ENV,
    )
    assert paused.exit_code == 2, paused.output
    assert "budget_exhausted" in paused.output
    assert "budget exhausted" in paused.output  # 一行文案增强:点名可续跑出路
    # R-T3-6:steps 耗尽必须点名 --extend-steps(旧文案恒印 tokens 是死路误导)。
    assert "--extend-steps" in paused.output
    assert "steps" in paused.output
    turn_id = _turn_id_from_output(paused.output)

    detail = runner.invoke(
        app, ["status", turn_id, "--workspace", str(workspace), "--thread", "b1"], env=_CLEAN_ENV
    )
    assert detail.exit_code == 0, detail.output
    assert "budget_exhausted" in detail.output
    assert "budget exhausted" in detail.output
    assert "--extend-steps" in detail.output  # status 重建无维度 → 通用文案列全三旗标

    reader = open_session(workspace, ScriptedAdapter([]), thread_id="b1")
    try:
        assert reader.status(turn_id).binding["budget"] == {"steps": 1}
    finally:
        reader.close()


def test_cli_cost_flag_pricing_pause_then_extend_resumes(workspace: Path, tmp_path: Path) -> None:
    """scripted 无单价:--max-cost-micros → pricing_unavailable PAUSED(fail-closed
    端到端);resume --extend-tokens 写 extension item 后自动续跑一步(仍如实再暂停)。"""
    script = _write_script(tmp_path, "final.json", [{"content": "done"}])
    paused = runner.invoke(
        app,
        [
            "run", "go", "--workspace", str(workspace), "--thread", "b2", "--scripted", str(script),
            "--max-tokens", "500", "--max-cost-micros", "1000",
        ],
        env=_CLEAN_ENV,
    )
    assert paused.exit_code == 2, paused.output
    assert "pricing_unavailable" in paused.output
    turn_id = _turn_id_from_output(paused.output)

    resumed = runner.invoke(
        app,
        ["resume", turn_id, "--workspace", str(workspace), "--thread", "b2", "--scripted", str(script),
         "--extend-tokens", "250"],
        env=_CLEAN_ENV,
    )
    assert resumed.exit_code == 2, resumed.output  # 单价仍缺:续跑一步后如实再暂停
    assert "pricing_unavailable" in resumed.output

    reader = open_session(workspace, ScriptedAdapter([]), thread_id="b2")
    try:
        assert reader.status(turn_id).binding["budget"] == {"tokens": 500, "cost_micros": 1000}
        extensions = [i for i in reader.items() if i.kind == ItemKind.BUDGET_EXTENSION]
        assert len(extensions) == 1
        assert extensions[0].payload == {"limits_delta": {"tokens": 250}}
    finally:
        reader.close()


def test_cli_extend_steps_resumes_steps_exhausted_turn(workspace: Path, tmp_path: Path) -> None:
    """R-T3-6 端到端:--max-steps 1 耗尽 PAUSED → resume --extend-steps 5 续跑
    成功(CLI steps 死路关闭),extension item 落 steps 增量。"""
    script = _write_script(tmp_path, "steps2.json", [_list_call("c1"), _list_call("c2"), {"content": "done"}])
    paused = runner.invoke(
        app,
        ["run", "go", "--workspace", str(workspace), "--thread", "b3", "--scripted", str(script), "--max-steps", "1"],
        env=_CLEAN_ENV,
    )
    assert paused.exit_code == 2, paused.output
    turn_id = _turn_id_from_output(paused.output)

    done = runner.invoke(
        app,
        ["resume", turn_id, "--workspace", str(workspace), "--thread", "b3", "--scripted", str(script),
         "--extend-steps", "5"],
        env=_CLEAN_ENV,
    )
    assert done.exit_code == 0, done.output
    assert "done" in done.output

    reader = open_session(workspace, ScriptedAdapter([]), thread_id="b3")
    try:
        extensions = [i for i in reader.items() if i.kind == ItemKind.BUDGET_EXTENSION]
        assert len(extensions) == 1
        assert extensions[0].payload == {"limits_delta": {"steps": 5}}
    finally:
        reader.close()


@pytest.mark.parametrize(
    ("flags", "flag_name"),
    [
        (["run", "go", "--max-tokens", "-1"], "--max-tokens"),
        (["run", "go", "--max-cost-micros", "-5"], "--max-cost-micros"),
        (["run", "go", "--max-steps", "-2"], "--max-steps"),
        (["resume", "abc", "--extend-tokens", "-5"], "--extend-tokens"),
        (["resume", "abc", "--extend-cost-micros", "-1"], "--extend-cost-micros"),
        (["resume", "abc", "--extend-steps", "-1"], "--extend-steps"),
    ],
)
def test_cli_negative_budget_flags_exit_3(
    workspace: Path, tmp_path: Path, flags: list[str], flag_name: str
) -> None:
    """负值在边界拒绝(BadParameter 点名旗标),先于剧本加载/turn 查询。"""
    script = _write_script(tmp_path, "ok.json", [{"content": "done"}])
    result = runner.invoke(
        app, [*flags, "--workspace", str(workspace), "--scripted", str(script)], env=_CLEAN_ENV
    )
    assert result.exit_code == 3, result.output
    assert flag_name in result.output


def test_sdk_reexports_limits() -> None:
    for name in ("Limits", "UNLIMITED"):
        assert name in sdk_pkg.__all__, f"sdk.__all__ missing {name}"
        assert hasattr(sdk_pkg, name), f"sdk missing attribute {name}"
