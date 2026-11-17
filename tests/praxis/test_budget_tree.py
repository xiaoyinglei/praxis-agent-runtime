"""core.budget:预算树事实折叠、份额占用与 admission 判据(设计 §5.5;阶段4 T2)。

本文件钉住四组机制:
1. ``fold_turn_facts`` —— 唯一事实账本:usage 求和(estimated 照计)、steps 声明轮口径
   (R-PF-4)、web/browser 请求计数(含 dispatch 后中断的保守面)、cost 向上取整;
2. ``protected_tail`` / ``normal_token_remaining`` / ``admit_dispatch`` —— 保护尾与
   fail-closed(cost 限额而单价缺失 ⇒ 拒,绝不按 0 计);
3. ``build_tree`` —— 父 turn items + ``child_allocation`` 子线程事实折叠;子终态
   ⇒ 未用份额**推导式**归还(settle 不是写);子事实不可读 ⇒ 保守 active 全额占用;
4. 守恒属性(hypothesis):``exposure = Σ已用 + Σactive 子未用``,扩展只增 limits。

对照出处:
- 保护尾 / normal remaining: ``agent_runtime/budget/accounting.py:187-195,205-212``
  (逐字移植,禁止 import);
- 子分配保守计数: ``agent_runtime/harness/budget_ops.py:284-322`` 的 reserve-before-IO
  语义(一旦 tool_call 落库即视为已占用;禁止 import);
- 份额公式本身是 T1 ``protocol/budget.py`` 的唯一表达,本文件只验证折叠消费。
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from praxis.core.budget import (
    Admission,
    BudgetNode,
    BudgetTracker,
    ChildAllocation,
    UsageFacts,
    admit_dispatch,
    build_tree,
    estimate_cost,
    exposure,
    fold_turn_facts,
    normal_token_remaining,
    protected_tail,
    remaining,
)
from praxis.protocol.budget import Limits
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import CostMicros, Usage
from praxis.store.log import ItemEntry, Store

# ---------------------------------------------------------------------------
# 构造助手
# ---------------------------------------------------------------------------

PARENT_THREAD = "parent-thread"
TURN = "turn-1"


def _item(
    seq: int,
    kind: str,
    payload: Mapping[str, JsonValue],
    *,
    thread_id: str = PARENT_THREAD,
    turn_id: str | None = TURN,
) -> Item:
    return Item(
        item_id=f"{thread_id}:{seq}",
        thread_id=thread_id,
        turn_id=turn_id,
        seq=seq,
        kind=kind,
        payload=payload,
        created_at=0.0,
    )


def _usage_payload(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    reasoning_tokens: int = 0,
    source: str = "provider",
) -> dict[str, JsonValue]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "reasoning_tokens": reasoning_tokens,
        "source": source,
    }


def _model_response(
    seq: int,
    *,
    usage: dict[str, JsonValue] | None = None,
    tool_calls: Sequence[Mapping[str, JsonValue]] = (),
    thread_id: str = PARENT_THREAD,
    turn_id: str | None = TURN,
) -> Item:
    return _item(
        seq,
        ItemKind.MODEL_RESPONSE,
        {
            "content": "",
            "tool_calls": tuple(tool_calls),
            "usage": usage if usage is not None else _usage_payload(),
            "finish_reason": "stop",
            "reasoning": "",
        },
        thread_id=thread_id,
        turn_id=turn_id,
    )


def _web_result(
    seq: int,
    *,
    tool_name: str = "web_fetch",
    status: str = "ok",
    call_id: str | None = "call-1",
    metadata: Mapping[str, JsonValue] | None = None,
    thread_id: str = PARENT_THREAD,
    turn_id: str | None = TURN,
) -> Item:
    payload: dict[str, JsonValue] = {
        "call_id": call_id,
        "tool_name": tool_name,
        "status": status,
        "content": (),
        "is_error": status != "ok",
        "error_code": None,
        "error_message": None,
        "truncated": False,
        "metadata": dict(metadata or {}),
    }
    return _item(seq, ItemKind.TOOL_RESULT, payload, thread_id=thread_id, turn_id=turn_id)


def _tool_call(
    seq: int,
    *,
    tool_name: str = "web_fetch",
    call_id: str | None = "call-1",
    thread_id: str = PARENT_THREAD,
    turn_id: str | None = TURN,
) -> Item:
    return _item(
        seq,
        ItemKind.TOOL_CALL,
        {
            "call_id": call_id,
            "tool_name": tool_name,
            "arguments": {},
            "effects": (),
            "targets": (),
            "args_digest": "digest",
            "idempotent": True,
            "source": "builtin",
        },
        thread_id=thread_id,
        turn_id=turn_id,
    )


def _turn_status(
    seq: int,
    to: str,
    *,
    thread_id: str,
    turn_id: str | None,
) -> Item:
    return _item(
        seq,
        ItemKind.TURN_STATUS,
        {"from": "running", "to": to, "reason": "test"},
        thread_id=thread_id,
        turn_id=turn_id,
    )


def _allocation(
    seq: int,
    *,
    allocation_id: str = "alloc-1",
    child_thread_id: str = "child-thread",
    child_root_turn_id: str = "child-turn",
    grant: dict[str, JsonValue] | None = None,
    source: str = "model",
    thread_id: str = PARENT_THREAD,
    turn_id: str | None = TURN,
) -> Item:
    return _item(
        seq,
        ItemKind.CHILD_ALLOCATION,
        {
            "allocation_id": allocation_id,
            "parent_turn_id": turn_id,
            "child_thread_id": child_thread_id,
            "child_root_turn_id": child_root_turn_id,
            "workspace_mode": "read_only",
            "workspace": None,
            "grant": dict(grant or {}),
            "requested": {},
            "source": source,
        },
        thread_id=thread_id,
        turn_id=turn_id,
    )


def _extension(
    seq: int,
    delta: Mapping[str, JsonValue],
    *,
    thread_id: str = PARENT_THREAD,
    turn_id: str | None = TURN,
) -> Item:
    return _item(
        seq,
        ItemKind.BUDGET_EXTENSION,
        {"limits_delta": dict(delta)},
        thread_id=thread_id,
        turn_id=turn_id,
    )


def _node(
    *,
    limits: Limits = Limits(),
    used: UsageFacts | None = None,
    children: dict[str, ChildAllocation] | None = None,
    unit_costs: CostMicros | None = None,
) -> BudgetNode:
    return BudgetNode(
        turn_id=TURN,
        limits=limits,
        used=used or UsageFacts(),
        children=children or {},
        unit_costs=unit_costs,
    )


def _child(
    *,
    grant: Limits = Limits(),
    used: UsageFacts | None = None,
    state: str = "active",
) -> ChildAllocation:
    return ChildAllocation(
        allocation_id="alloc",
        child_thread_id="child",
        child_root_turn_id="child-turn",
        grant=grant,
        source="model",
        state=state,  # type: ignore[arg-type]
        used=used or UsageFacts(),
    )


class _ThreadStore:
    """build_tree 只依赖 read_items(thread_id);属性测试用无 IO 的假 store。"""

    def __init__(self, threads: Mapping[str, Sequence[Item]]) -> None:
        self._threads = threads

    def read_items(self, thread_id: str, *, after_seq: int = 0, limit: int | None = None) -> list[Item]:
        del after_seq, limit
        return list(self._threads.get(thread_id, ()))


# ---------------------------------------------------------------------------
# estimate_cost:旧 pricing.py ``_ceil_cost`` 语义(向上取整,绝不按 0)
# ---------------------------------------------------------------------------


def test_estimate_cost_ceils_once_over_all_components() -> None:
    usage = Usage(input_tokens=1, output_tokens=1)
    cost = CostMicros(input_per_1m=1_000_000, output_per_1m=500_000)
    # 1 µ$ + 0.5 µ$ = 1.5 µ$ → 向上取整 2(逐分量取整会得到 1+1=2,用 3 分量区分口径)
    assert estimate_cost(usage, cost) == 2


def test_estimate_cost_ceil_is_once_not_per_component() -> None:
    # 两个 0.1 µ$ 分量:一次 ceil(0.2)=1;逐分量 ceil 会得到 2。
    cost = CostMicros(input_per_1m=100_000, output_per_1m=100_000)
    usage = Usage(input_tokens=1, output_tokens=1)
    assert estimate_cost(usage, cost) == 1


def test_estimate_cost_uncaches_cached_input_and_falls_back_to_input_rate() -> None:
    # input 10 含 cached 4:未缓存 6 × 1µ$/tok + cached 4 × 1µ$/tok(0=未配置 ⇒ 回落 input)
    cost = CostMicros(input_per_1m=1_000_000, cache_read_per_1m=0)
    assert estimate_cost(Usage(input_tokens=10, output_tokens=0, cached_input_tokens=4), cost) == 10


def test_estimate_cost_uses_configured_cache_read_rate() -> None:
    cost = CostMicros(input_per_1m=1_000_000, cache_read_per_1m=250_000)
    usage = Usage(input_tokens=10, output_tokens=0, cached_input_tokens=4)
    # 6 × 1 + 4 × 0.25 = 7 µ$
    assert estimate_cost(usage, cost) == 7


def test_estimate_cost_zero_rate_is_zero_but_not_a_hiding_path() -> None:
    # 全零 = 未配置;estimate 本身返回 0,admission 层必须 fail-closed(见下)。
    assert estimate_cost(Usage(input_tokens=1_000_000, output_tokens=0), CostMicros()) == 0


def test_estimate_cost_reasoning_not_double_counted() -> None:
    # reasoning 是 output 的子集(models.Usage 契约),只按 output 计一次。
    cost = CostMicros(output_per_1m=1_000_000)
    usage = Usage(input_tokens=0, output_tokens=10, reasoning_tokens=7)
    assert estimate_cost(usage, cost) == 10


# ---------------------------------------------------------------------------
# fold_turn_facts:usage / steps / web / browser / cost
# ---------------------------------------------------------------------------


def test_fold_tokens_sums_model_response_and_summary_usage() -> None:
    items = [
        _model_response(1, usage=_usage_payload(input_tokens=100, output_tokens=20, cached_input_tokens=40)),
        _item(
            2,
            ItemKind.SUMMARY,
            {"text": "s", "usage": _usage_payload(input_tokens=10, output_tokens=5)},
        ),
    ]
    facts = fold_turn_facts(items, unit_costs=None)
    # cached/reasoning 是 input/output 子集,不重复相加;estimated 来源照单计入。
    assert facts.tokens == 135


def test_fold_treats_estimated_usage_as_a_fact() -> None:
    items = [_model_response(1, usage=_usage_payload(input_tokens=7, output_tokens=3, source="estimated"))]
    assert fold_turn_facts(items, unit_costs=None).tokens == 10


def test_fold_malformed_usage_fails_open_per_key() -> None:
    items = [
        _model_response(
            1,
            usage={"input_tokens": "oops", "output_tokens": 5, "cached_input_tokens": True},
        ),
        _item(2, ItemKind.SUMMARY, {"text": "s", "usage": "not-a-mapping"}),
    ]
    assert fold_turn_facts(items, unit_costs=None).tokens == 5


def test_fold_cost_micros_uses_unit_costs_ceil() -> None:
    items = [_model_response(1, usage=_usage_payload(input_tokens=1, output_tokens=1))]
    cost = CostMicros(input_per_1m=1_000_000, output_per_1m=500_000)
    assert fold_turn_facts(items, unit_costs=cost).cost_micros == 2


def test_fold_cost_is_zero_when_pricing_absent() -> None:
    items = [_model_response(1, usage=_usage_payload(input_tokens=1_000_000))]
    assert fold_turn_facts(items, unit_costs=None).cost_micros == 0


def test_fold_steps_is_declaration_rounds_not_completed_tool_batches() -> None:
    # R-PF-4:steps = tool_calls 非空的 model_response 条数(声明轮口径)。
    # 与旧 turn.py 每完成一轮“请求+全部工具”自增的关系:resume 重放轮的声明事实
    # 不新增,由 run_turn 进程内 +1 补计;二者相加逐值一致(T3 消费面)。
    calls = ({"call_id": "c1", "name": "read", "arguments": {}},)
    items = [
        _model_response(1, tool_calls=calls),
        _model_response(2, tool_calls=()),
        _model_response(3, tool_calls=calls),
        _model_response(4, tool_calls=calls),
        # 非 model_response 的工具事实不额外计步。
        _tool_call(5),
        _web_result(6),
    ]
    assert fold_turn_facts(items, unit_costs=None).steps == 3


def test_fold_steps_ignores_malformed_tool_calls() -> None:
    items = [
        _model_response(1, tool_calls=()),
        _item(2, ItemKind.MODEL_RESPONSE, {"content": "", "tool_calls": "oops"}),
        _item(3, ItemKind.MODEL_RESPONSE, {"content": ""}),
    ]
    assert fold_turn_facts(items, unit_costs=None).steps == 0


def test_fold_web_counts_only_completed_web_tool_dispatches() -> None:
    items = [
        _web_result(1, tool_name="web_fetch", status="ok", call_id="a"),
        _web_result(2, tool_name="web_search", status="error", call_id="b"),
        _web_result(3, tool_name="web_fetch", status="timeout", call_id="c"),
        _web_result(4, tool_name="web_search", status="cancelled", call_id="d"),
        # 非计费状态:denied 从未 dispatch(管道在 tool_call 之前就落结果)。
        _web_result(5, tool_name="web_fetch", status="denied", call_id="e"),
        # 非 web 工具不计。
        _web_result(6, tool_name="read", status="ok", call_id="f"),
    ]
    facts = fold_turn_facts(items, unit_costs=None)
    assert facts.web_requests == 4
    assert facts.browser_requests == 0


def test_fold_outcome_unknown_web_result_counts_conservatively() -> None:
    # R-T2-1:dispatch 后失联(outcome_unknown)按旧 reserve-before-IO 语义计 1。
    items = [_web_result(1, status="outcome_unknown", call_id="lost")]
    assert fold_turn_facts(items, unit_costs=None).web_requests == 1


def test_fold_resume_compensated_unknown_web_counts_once() -> None:
    # 真实 resume 形状:tool_call 已落库 + 补偿结果 outcome_unknown ⇒ 恰好 1 次。
    items = [
        _tool_call(1, call_id="lost"),
        _web_result(2, status="outcome_unknown", call_id="lost"),
    ]
    assert fold_turn_facts(items, unit_costs=None).web_requests == 1


def test_fold_not_executed_web_result_not_billed() -> None:
    # not_executed 是"从未 dispatch"的补偿结论,不占请求配额。
    items = [_web_result(1, status="not_executed", call_id="never")]
    assert fold_turn_facts(items, unit_costs=None).web_requests == 0


def test_fold_web_result_without_tool_call_still_counts() -> None:
    # 防御面:旧日志/外部写入可能只有结果事实;结果本身即证明请求发生。
    items = [_web_result(1, status="ok", call_id="solo", metadata={"browser_render": True})]
    facts = fold_turn_facts(items, unit_costs=None)
    assert (facts.web_requests, facts.browser_requests) == (1, 1)


def test_fold_browser_flag_requires_exact_true_and_web_tool() -> None:
    items = [
        _web_result(1, call_id="a", metadata={"browser_render": True}),
        _web_result(2, call_id="b", metadata={"browser_render": 1}),
        _web_result(3, call_id="c", metadata={"browser_render": "yes"}),
        _web_result(4, tool_name="read", call_id="d", metadata={"browser_render": True}),
        _web_result(5, call_id="e", metadata={}),
    ]
    facts = fold_turn_facts(items, unit_costs=None)
    assert facts.web_requests == 4  # 4 条 web 工具结果;read 不计
    assert facts.browser_requests == 1


def test_fold_unpaired_web_tool_call_is_counted_conservatively() -> None:
    # dispatch 后中断:tool_call 已落库(= reserve-before-IO 语义),结果缺失 ⇒ 保守 1。
    items = [_tool_call(1, call_id="crashed")]
    assert fold_turn_facts(items, unit_costs=None).web_requests == 1


def test_fold_paired_web_tool_call_not_double_counted() -> None:
    items = [
        _tool_call(1, call_id="c1"),
        _web_result(2, call_id="c1", status="ok"),
        _tool_call(3, call_id="c2"),
        _web_result(4, call_id="c2", status="error"),
    ]
    assert fold_turn_facts(items, unit_costs=None).web_requests == 2


def test_fold_non_web_unpaired_tool_call_not_counted() -> None:
    items = [_tool_call(1, tool_name="read", call_id="r1")]
    assert fold_turn_facts(items, unit_costs=None).web_requests == 0


def test_fold_web_tool_call_without_call_id_counted_conservatively() -> None:
    items = [_tool_call(1, call_id=None)]
    assert fold_turn_facts(items, unit_costs=None).web_requests == 1


def test_fold_duplicate_unpaired_call_id_counted_once() -> None:
    items = [_tool_call(1, call_id="dup"), _tool_call(2, call_id="dup")]
    assert fold_turn_facts(items, unit_costs=None).web_requests == 1


# ---------------------------------------------------------------------------
# R-FINAL-I1:B31 真实 provider 每轮重编号 call_0——跨声明窗口复用 call_id
# 必须按窗口键 (turn_id, 声明 seq, call_id) 归属计数,不得按裸 call_id 去重。
# ---------------------------------------------------------------------------


def _declares(seq: int, call_id: str, tool_name: str = "web_fetch") -> Item:
    return _model_response(
        seq, tool_calls=({"call_id": call_id, "name": tool_name, "arguments": {}},)
    )


def test_fold_b31_cross_window_reused_call_id_counts_each_window() -> None:
    # 两声明窗口各发一次 web_fetch,模型每轮重编号 call_0 → 两次真实请求计 2。
    items = [
        _declares(1, "call_0"),
        _tool_call(2, call_id="call_0"),
        _web_result(3, call_id="call_0", status="ok"),
        _declares(4, "call_0"),
        _tool_call(5, call_id="call_0"),
        _web_result(6, call_id="call_0", status="ok"),
    ]
    assert fold_turn_facts(items, unit_costs=None).web_requests == 2


def test_fold_b31_cross_window_crash_conservative_per_window() -> None:
    # 窗口 1 完成 + 窗口 2 dispatch 后失联(无结果)→ 完成 1 + 保守 1 = 2。
    items = [
        _declares(1, "call_0"),
        _tool_call(2, call_id="call_0"),
        _web_result(3, call_id="call_0", status="ok"),
        _declares(4, "call_0"),
        _tool_call(5, call_id="call_0"),
    ]
    assert fold_turn_facts(items, unit_costs=None).web_requests == 2


def test_fold_b31_cross_window_browser_reused_call_id_counts_each_window() -> None:
    items = [
        _declares(1, "call_0"),
        _web_result(2, call_id="call_0", status="ok", metadata={"browser_render": True}),
        _declares(3, "call_0"),
        _web_result(4, call_id="call_0", status="ok", metadata={"browser_render": True}),
    ]
    facts = fold_turn_facts(items, unit_costs=None)
    assert (facts.web_requests, facts.browser_requests) == (2, 2)


def test_fold_b31_cross_window_non_billable_latest_not_double_counted() -> None:
    # 窗口 1 ok(计费)+ 窗口 2 denied(未 dispatch 不计)→ 恰 1,不因复用 id 漏计窗口 1。
    items = [
        _declares(1, "call_0"),
        _web_result(2, call_id="call_0", status="ok"),
        _declares(3, "call_0"),
        _web_result(4, call_id="call_0", status="denied"),
    ]
    assert fold_turn_facts(items, unit_costs=None).web_requests == 1


# ---------------------------------------------------------------------------
# protected_tail / normal_token_remaining:旧 accounting.py 逐字移植
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        (None, 0),
        (0, 0),
        (1, 0),  # limit ≤ 1 下限:保护尾不得吞掉整个预算
        (2, 1),
        (9, 1),  # limit//10 = 0 → max(1, 0) = 1
        (10, 1),
        (20, 2),
        (100, 10),  # 10% 界生效
        (1_000, 100),
        (81_920, 8_192),  # 8192 界与 10% 相等
        (100_000, 8_192),  # 8192 上界生效
        (1_000_000, 8_192),
    ],
)
def test_protected_tail_table(limit: int | None, expected: int) -> None:
    assert protected_tail(limit) == expected


def test_protected_tail_rejects_bool_and_negative() -> None:
    with pytest.raises(ValueError):
        protected_tail(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        protected_tail(-1)


def test_normal_token_remaining_excludes_protected_tail() -> None:
    assert normal_token_remaining(_node(limits=Limits())) is None
    assert normal_token_remaining(_node(limits=Limits(tokens=1_000), used=UsageFacts(tokens=100))) == 800
    # 剩余 50 < 保护尾 100:普通调用已无 headroom。
    assert normal_token_remaining(_node(limits=Limits(tokens=1_000), used=UsageFacts(tokens=950))) == 0


# ---------------------------------------------------------------------------
# exposure / remaining / 子份额占用
# ---------------------------------------------------------------------------


def test_exposure_adds_active_unused_grants() -> None:
    child = _child(grant=Limits(tokens=500), used=UsageFacts(tokens=200), state="active")
    # node.used 契约:含子树已用(守恒 exposure = used + reserved)。
    node = _node(limits=Limits(tokens=1_000), used=UsageFacts(tokens=250), children={"a": child})
    assert node.reserved.tokens == 300
    assert exposure(node).tokens == 550
    assert remaining(node).tokens == 450


def test_exposure_terminal_child_releases_only_unused_share() -> None:
    for terminal in ("done", "cancelled", "interrupted"):
        child = _child(grant=Limits(tokens=500), used=UsageFacts(tokens=200), state=terminal)
        node = _node(limits=Limits(tokens=1_000), used=UsageFacts(tokens=250), children={"a": child})
        assert node.reserved.tokens == 0
        # 已用 200 是事实,不随 settle 消失;只归还 grant-used 的 300。
        assert exposure(node).tokens == 250
        assert remaining(node).tokens == 750


def test_exposure_clamps_overused_active_child_and_flags() -> None:
    child = _child(grant=Limits(tokens=500), used=UsageFacts(tokens=600), state="active")
    node = _node(limits=Limits(tokens=1_000), used=UsageFacts(tokens=600), children={"a": child})
    assert node.reserved.tokens == 0
    assert exposure(node).tokens == 600  # 超额诚实暴露


def test_remaining_unbounded_dimension_is_none() -> None:
    rem = remaining(_node(limits=Limits(tokens=100)))
    assert rem.tokens == 100
    assert rem.cost_micros is None
    assert rem.steps is None
    assert rem.web_requests is None
    assert rem.browser_requests is None


def test_remaining_clamps_at_zero_but_exposure_stays_honest() -> None:
    node = _node(limits=Limits(tokens=10), used=UsageFacts(tokens=25))
    assert remaining(node).tokens == 0
    assert exposure(node).tokens == 25


# ---------------------------------------------------------------------------
# build_tree:扩展折叠 + 子线程事实 + settle 推导
# ---------------------------------------------------------------------------


def _build(
    store: Any,
    *,
    root_limits: Limits = Limits(tokens=1_000),
    unit_costs: CostMicros | None = None,
    child_reader: Any = None,
    thread_id: str = PARENT_THREAD,
    turn_id: str = TURN,
) -> BudgetNode:
    return build_tree(
        store,
        thread_id=thread_id,
        turn_id=turn_id,
        root_limits=root_limits,
        unit_costs=unit_costs,
        child_reader=child_reader,
    )


def test_build_tree_filters_parent_items_by_turn(store: Store) -> None:
    store.append_items(
        PARENT_THREAD,
        TURN,
        [ItemEntry(ItemKind.MODEL_RESPONSE, _model_response(1, usage=_usage_payload(input_tokens=10)).payload)],
    )
    store.append_items(
        PARENT_THREAD,
        "turn-other",
        [ItemEntry(ItemKind.MODEL_RESPONSE, _model_response(1, usage=_usage_payload(input_tokens=999)).payload)],
    )
    node = _build(store)
    assert node.used.tokens == 10
    assert node.used.steps == 0


def test_build_tree_folds_multiple_extensions_additively(store: Store) -> None:
    store.append_items(
        PARENT_THREAD,
        TURN,
        [
            ItemEntry(ItemKind.BUDGET_EXTENSION, _extension(1, {"tokens": 100}).payload),
            ItemEntry(ItemKind.BUDGET_EXTENSION, _extension(2, {"tokens": 50, "steps": 2}).payload),
            ItemEntry(ItemKind.BUDGET_EXTENSION, _extension(3, {"tokens": -25}).payload),
        ],
    )
    store.append_items(
        PARENT_THREAD,
        "turn-other",
        [ItemEntry(ItemKind.BUDGET_EXTENSION, _extension(1, {"tokens": 9_999}).payload)],
    )
    node = _build(store, root_limits=Limits(tokens=1_000, cost_micros=7, steps=1))
    assert node.limits.tokens == 1_150  # 负增量不缩预算
    assert node.limits.steps == 3  # 有界基线维 + 扩展
    assert node.limits.cost_micros == 7  # 未扩展维不变


def test_build_tree_extension_cannot_bound_an_unlimited_dimension(store: Store) -> None:
    store.append_items(
        PARENT_THREAD,
        TURN,
        [ItemEntry(ItemKind.BUDGET_EXTENSION, _extension(1, {"tokens": 100}).payload)],
    )
    node = _build(store, root_limits=Limits())
    assert node.limits.tokens is None


def test_build_tree_extension_malformed_delta_is_skipped(store: Store) -> None:
    store.append_items(
        PARENT_THREAD,
        TURN,
        [
            ItemEntry(
                ItemKind.BUDGET_EXTENSION,
                {"limits_delta": {"tokens": "oops"}},
            ),
            ItemEntry(ItemKind.BUDGET_EXTENSION, {"limits_delta": "not-a-mapping"}),
        ],
    )
    assert _build(store, root_limits=Limits(tokens=100)).limits.tokens == 100


def test_build_tree_child_active_holds_full_grant_unused_part(store: Store) -> None:
    store.append_items(
        PARENT_THREAD,
        TURN,
        [
            ItemEntry(ItemKind.MODEL_RESPONSE, _model_response(1, usage=_usage_payload(input_tokens=50)).payload),
            ItemEntry(
                ItemKind.CHILD_ALLOCATION,
                _allocation(2, grant={"tokens": 500}).payload,
            ),
        ],
    )
    store.append_items(
        "child-thread",
        "child-turn",
        [
            ItemEntry(
                ItemKind.MODEL_RESPONSE,
                _model_response(1, usage=_usage_payload(input_tokens=200, output_tokens=0), thread_id="child-thread", turn_id="child-turn").payload,
            ),
            ItemEntry(ItemKind.TURN_STATUS, _turn_status(2, "running", thread_id="child-thread", turn_id="child-turn").payload),
        ],
    )
    node = _build(store)
    child = node.children["alloc-1"]
    assert child.state == "active"
    assert child.used.tokens == 200
    # node.used 含子树已用;reserved 只含 active 子未用份额。
    assert node.used.tokens == 250
    assert node.reserved.tokens == 300
    assert exposure(node).tokens == 550
    assert remaining(node).tokens == 450


@pytest.mark.parametrize(
    ("turn_result", "expected_state", "exposure_tokens"),
    [
        ("completed", "done", 250),
        ("failed", "done", 250),
        ("cancelled", "cancelled", 250),
        ("running", "active", 550),
        ("paused", "active", 550),
        ("interrupted", "active", 550),
    ],
)
def test_build_tree_child_state_and_settle_derivation(
    store: Store,
    turn_result: str,
    expected_state: str,
    exposure_tokens: int,
) -> None:
    store.append_items(
        PARENT_THREAD,
        TURN,
        [
            ItemEntry(ItemKind.MODEL_RESPONSE, _model_response(1, usage=_usage_payload(input_tokens=50)).payload),
            ItemEntry(ItemKind.CHILD_ALLOCATION, _allocation(2, grant={"tokens": 500}).payload),
        ],
    )
    store.append_items(
        "child-thread",
        "child-turn",
        [
            ItemEntry(
                ItemKind.MODEL_RESPONSE,
                _model_response(1, usage=_usage_payload(input_tokens=200), thread_id="child-thread", turn_id="child-turn").payload,
            ),
            ItemEntry(ItemKind.TURN_STATUS, _turn_status(2, turn_result, thread_id="child-thread", turn_id="child-turn").payload),
        ],
    )
    node = _build(store)
    assert node.children["alloc-1"].state == expected_state
    assert exposure(node).tokens == exposure_tokens
    # settle 是推导:父线程没有任何 settle item。
    assert [i.kind for i in store.read_items(PARENT_THREAD)] == [
        ItemKind.MODEL_RESPONSE,
        ItemKind.CHILD_ALLOCATION,
    ]


def test_build_tree_uses_last_turn_status_by_seq_not_input_order(store: Store) -> None:
    # reader 顺序不可依赖:终态判据取最大 seq 的 turn_status。
    child_items = [
        _turn_status(9, "completed", thread_id="child-thread", turn_id="child-turn"),
        _turn_status(3, "running", thread_id="child-thread", turn_id="child-turn"),
    ]

    def reader(thread_id: str) -> Sequence[Item]:
        assert thread_id == "child-thread"
        return child_items

    store.append_items(
        PARENT_THREAD,
        TURN,
        [ItemEntry(ItemKind.CHILD_ALLOCATION, _allocation(1, grant={"tokens": 500}).payload)],
    )
    node = _build(store, child_reader=reader)
    assert node.children["alloc-1"].state == "done"


def test_build_tree_empty_or_unreadable_child_is_conservatively_active(store: Store) -> None:
    store.append_items(
        PARENT_THREAD,
        TURN,
        [ItemEntry(ItemKind.CHILD_ALLOCATION, _allocation(1, grant={"tokens": 500}).payload)],
    )

    def exploding_reader(thread_id: str) -> Sequence[Item]:
        raise RuntimeError(f"thread vanished: {thread_id}")

    node = _build(store, child_reader=exploding_reader)
    child = node.children["alloc-1"]
    assert child.state == "active"
    assert child.used == UsageFacts()
    assert node.reserved.tokens == 500
    assert exposure(node).tokens == 500  # 线程读不到 ⇒ 全额占用,绝不白送预算


def test_build_tree_active_child_overspend_sets_under_accounted(store: Store) -> None:
    store.append_items(
        PARENT_THREAD,
        TURN,
        [ItemEntry(ItemKind.CHILD_ALLOCATION, _allocation(1, grant={"tokens": 100}).payload)],
    )
    store.append_items(
        "child-thread",
        "child-turn",
        [
            ItemEntry(
                ItemKind.MODEL_RESPONSE,
                _model_response(1, usage=_usage_payload(input_tokens=150), thread_id="child-thread", turn_id="child-turn").payload,
            ),
            ItemEntry(ItemKind.TURN_STATUS, _turn_status(2, "running", thread_id="child-thread", turn_id="child-turn").payload),
        ],
    )
    node = _build(store)
    assert node.under_accounted is True
    assert node.reserved.tokens == 0
    assert exposure(node).tokens == 150


def test_build_tree_recurses_grandchildren_into_child_used(store: Store) -> None:
    # 父 grant 500 给子;子用 50 并再分配 200 给孙;孙用 100。
    store.append_items(
        PARENT_THREAD,
        TURN,
        [ItemEntry(ItemKind.CHILD_ALLOCATION, _allocation(1, grant={"tokens": 500}).payload)],
    )
    store.append_items(
        "child-thread",
        "child-turn",
        [
            ItemEntry(
                ItemKind.MODEL_RESPONSE,
                _model_response(1, usage=_usage_payload(input_tokens=50), thread_id="child-thread", turn_id="child-turn").payload,
            ),
            ItemEntry(
                ItemKind.CHILD_ALLOCATION,
                _allocation(
                    2,
                    allocation_id="grand-alloc",
                    child_thread_id="grand-thread",
                    child_root_turn_id="grand-turn",
                    grant={"tokens": 200},
                    thread_id="child-thread",
                    turn_id="child-turn",
                ).payload,
            ),
            ItemEntry(ItemKind.TURN_STATUS, _turn_status(3, "completed", thread_id="child-thread", turn_id="child-turn").payload),
        ],
    )
    store.append_items(
        "grand-thread",
        "grand-turn",
        [
            ItemEntry(
                ItemKind.MODEL_RESPONSE,
                _model_response(1, usage=_usage_payload(input_tokens=100), thread_id="grand-thread", turn_id="grand-turn").payload,
            ),
            ItemEntry(ItemKind.TURN_STATUS, _turn_status(2, "completed", thread_id="grand-thread", turn_id="grand-turn").payload),
        ],
    )
    node = _build(store)
    child = node.children["alloc-1"]
    assert child.used.tokens == 150  # 子自身 50 + 孙 100
    assert child.state == "done"
    assert node.used.tokens == 150
    assert exposure(node).tokens == 150


def test_build_tree_cycle_is_bounded_and_conservative(store: Store) -> None:
    # 病态数据:分配指向父线程自身。visited 守卫必须终止并保守占用。
    store.append_items(
        PARENT_THREAD,
        TURN,
        [
            ItemEntry(
                ItemKind.CHILD_ALLOCATION,
                _allocation(1, child_thread_id=PARENT_THREAD, child_root_turn_id=TURN, grant={"tokens": 500}).payload,
            )
        ],
    )
    node = _build(store)
    assert node.children["alloc-1"].state == "active"
    assert node.reserved.tokens == 500


def test_build_tree_default_child_reader_is_store_read_items(store: Store) -> None:
    store.append_items(
        PARENT_THREAD,
        TURN,
        [ItemEntry(ItemKind.CHILD_ALLOCATION, _allocation(1, grant={"tokens": 300}).payload)],
    )
    store.append_items(
        "child-thread",
        "child-turn",
        [
            ItemEntry(
                ItemKind.MODEL_RESPONSE,
                _model_response(1, usage=_usage_payload(input_tokens=100), thread_id="child-thread", turn_id="child-turn").payload,
            )
        ],
    )
    assert _build(store).children["alloc-1"].used.tokens == 100


def test_build_tree_malformed_allocation_is_skipped_fail_open(store: Store) -> None:
    store.append_items(
        PARENT_THREAD,
        TURN,
        [
            ItemEntry(ItemKind.CHILD_ALLOCATION, {"allocation_id": "bad"}),
            ItemEntry(ItemKind.CHILD_ALLOCATION, _allocation(2, grant={"tokens": 100}).payload),
        ],
    )
    node = _build(store)
    assert set(node.children) == {"alloc-1"}


# ---------------------------------------------------------------------------
# admit_dispatch:保护尾矩阵 + pricing fail-closed
# ---------------------------------------------------------------------------


def test_admit_dispatch_unlimited_always_allows() -> None:
    node = _node(limits=Limits(), unit_costs=None)
    assert admit_dispatch(node, needs_compaction=False) == Admission(allowed=True, reason="")
    assert admit_dispatch(node, needs_compaction=True).allowed


def test_admit_dispatch_normal_call_cannot_eat_protected_tail() -> None:
    costs = CostMicros(input_per_1m=1)
    # 限额 1000,保护尾 100:普通调用有效上限 900。
    under = _node(limits=Limits(tokens=1_000), used=UsageFacts(tokens=899), unit_costs=None)
    assert admit_dispatch(under, needs_compaction=False).allowed

    at_edge = _node(limits=Limits(tokens=1_000), used=UsageFacts(tokens=900), unit_costs=None)
    assert admit_dispatch(at_edge, needs_compaction=False) == Admission(
        allowed=False, reason="budget_exhausted", dimension="tokens"
    )

    in_tail = _node(limits=Limits(tokens=1_000), used=UsageFacts(tokens=950), unit_costs=None)
    assert admit_dispatch(in_tail, needs_compaction=False).reason == "budget_exhausted"
    # compaction 续作可以进入保护尾。
    assert admit_dispatch(in_tail, needs_compaction=True).allowed

    fully_spent = _node(limits=Limits(tokens=1_000), used=UsageFacts(tokens=1_000), unit_costs=None)
    assert admit_dispatch(fully_spent, needs_compaction=True).reason == "budget_exhausted"


def test_admit_dispatch_steps_exhaustion() -> None:
    node = _node(limits=Limits(steps=3), used=UsageFacts(steps=2))
    assert admit_dispatch(node, needs_compaction=False).allowed
    node_exhausted = _node(limits=Limits(steps=3), used=UsageFacts(steps=3))
    assert admit_dispatch(node_exhausted, needs_compaction=False) == Admission(
        allowed=False, reason="budget_exhausted", dimension="steps"
    )


def test_admit_dispatch_pricing_unavailable_fails_closed() -> None:
    limits = Limits(cost_micros=1_000)
    assert admit_dispatch(_node(limits=limits, unit_costs=None), needs_compaction=False) == Admission(
        allowed=False, reason="pricing_unavailable", dimension="cost_micros"
    )
    assert admit_dispatch(_node(limits=limits, unit_costs=CostMicros()), needs_compaction=False) == Admission(
        allowed=False, reason="pricing_unavailable", dimension="cost_micros"
    )


def test_admit_dispatch_cost_dimension_uses_estimated_facts() -> None:
    pricing = CostMicros(input_per_1m=1_000_000)  # 1µ$/token
    under = _node(
        limits=Limits(cost_micros=101),
        used=UsageFacts(cost_micros=100),
        unit_costs=pricing,
    )
    assert admit_dispatch(under, needs_compaction=False).allowed

    at_limit = _node(
        limits=Limits(cost_micros=100),
        used=UsageFacts(cost_micros=100),
        unit_costs=pricing,
    )
    assert admit_dispatch(at_limit, needs_compaction=False) == Admission(
        allowed=False, reason="budget_exhausted", dimension="cost_micros"
    )


def test_admit_dispatch_does_not_gate_web_dimensions() -> None:
    # web 维由 admit_web_request 在工具 dispatch 前判;模型调用不受它阻塞。
    node = _node(limits=Limits(web_requests=0, browser_requests=0))
    assert admit_dispatch(node, needs_compaction=False).allowed


# ---------------------------------------------------------------------------
# BudgetTracker:refresh / admission / 在途 web 计数去重
# ---------------------------------------------------------------------------


def _tracker(
    store: Store,
    *,
    limits: Limits,
    unit_costs: CostMicros | None = None,
    child_reader: Any = None,
    thread_id: str = PARENT_THREAD,
    turn_id: str = TURN,
) -> BudgetTracker:
    return BudgetTracker(
        store=store,
        thread_id=thread_id,
        turn_id=turn_id,
        root_limits=limits,
        unit_costs=unit_costs,
        child_reader=child_reader,
    )


def test_tracker_refresh_rebuilds_fold_from_store(store: Store) -> None:
    tracker = _tracker(store, limits=Limits(tokens=1_000))
    assert tracker.refresh().used.tokens == 0
    store.append_items(
        PARENT_THREAD,
        TURN,
        [ItemEntry(ItemKind.MODEL_RESPONSE, _model_response(1, usage=_usage_payload(input_tokens=42)).payload)],
    )
    node = tracker.refresh()
    assert node.used.tokens == 42
    assert tracker.node.used.tokens == 42


def test_tracker_node_lazily_rebuilds_before_first_refresh(store: Store) -> None:
    store.append_items(
        PARENT_THREAD,
        TURN,
        [ItemEntry(ItemKind.MODEL_RESPONSE, _model_response(1, usage=_usage_payload(input_tokens=7)).payload)],
    )
    tracker = _tracker(store, limits=Limits(tokens=100))
    assert tracker.node.used.tokens == 7


def test_tracker_admit_dispatch_uses_latest_fold(store: Store) -> None:
    store.append_items(
        PARENT_THREAD,
        TURN,
        [ItemEntry(ItemKind.MODEL_RESPONSE, _model_response(1, usage=_usage_payload(input_tokens=900)).payload)],
    )
    tracker = _tracker(store, limits=Limits(tokens=1_000))
    assert tracker.admit_dispatch(needs_compaction=False).reason == "budget_exhausted"
    assert tracker.admit_dispatch(needs_compaction=True).allowed


def test_tracker_web_in_flight_counts_and_note_result_releases(store: Store) -> None:
    tracker = _tracker(store, limits=Limits(web_requests=2))
    assert tracker.admit_web_request("web_fetch", browser=False).allowed
    assert tracker.admit_web_request("web_search", browser=False).allowed
    assert tracker.admit_web_request("web_fetch", browser=False) == Admission(
        allowed=False, reason="budget_exhausted", dimension="web_requests"
    )
    tracker.note_result("web_fetch", browser=False)
    assert tracker.admit_web_request("web_fetch", browser=False).allowed


def test_tracker_web_count_merges_fold_and_in_flight_without_double_count(store: Store) -> None:
    tracker = _tracker(store, limits=Limits(web_requests=2))
    # 第一批:admit → 落库 tool_call/tool_result → note_result(结果已入事实)。
    assert tracker.admit_web_request("web_fetch", browser=False).allowed
    store.append_items(
        PARENT_THREAD,
        TURN,
        [
            ItemEntry(ItemKind.TOOL_CALL, _tool_call(1, call_id="c1").payload),
            ItemEntry(ItemKind.TOOL_RESULT, _web_result(2, call_id="c1").payload),
        ],
    )
    tracker.note_result("web_fetch", browser=False)
    assert tracker.refresh().used.web_requests == 1  # 事实折叠
    # 在途再加 1:总占用 2 ⇒ 下一次拒。若 note_result 没释放,这里会提前拒。
    assert tracker.admit_web_request("web_search", browser=False).allowed
    assert not tracker.admit_web_request("web_search", browser=False).allowed


def test_tracker_web_browser_dimension_and_web_subset(store: Store) -> None:
    tracker = _tracker(store, limits=Limits(web_requests=10, browser_requests=1))
    assert tracker.admit_web_request("web_fetch", browser=True).allowed
    # browser 维已满;非 browser 的 web 请求仍可走 web 维。
    assert not tracker.admit_web_request("web_fetch", browser=True).allowed
    assert tracker.admit_web_request("web_search", browser=False).allowed


def test_tracker_non_web_name_is_not_gated_or_counted(store: Store) -> None:
    tracker = _tracker(store, limits=Limits(web_requests=0, browser_requests=0))
    assert tracker.admit_web_request("read", browser=False).allowed
    tracker.note_result("read", browser=False)  # 不抛、不出现负计数
    assert tracker.admit_web_request("read", browser=False).allowed


def test_tracker_unlimited_web_always_allows(store: Store) -> None:
    tracker = _tracker(store, limits=Limits())
    for _ in range(50):
        assert tracker.admit_web_request("web_fetch", browser=True).allowed
    assert tracker.admit_web_request("web_fetch", browser=True).allowed


def test_tracker_note_result_without_admit_never_goes_negative(store: Store) -> None:
    tracker = _tracker(store, limits=Limits(web_requests=1))
    tracker.note_result("web_fetch", browser=False)
    tracker.note_result("web_fetch", browser=False)
    assert tracker.admit_web_request("web_fetch", browser=False).allowed


def test_tracker_web_gate_sees_active_child_reserved_share(store: Store) -> None:
    # R-T2-1:active 子分配的 web 未用份额在 exposure 里,父 gate 必须看见,
    # 否则父会超发把总 exposure 推过限额。
    store.append_items(
        PARENT_THREAD,
        TURN,
        [ItemEntry(ItemKind.CHILD_ALLOCATION, _allocation(1, grant={"web_requests": 5}).payload)],
    )
    store.append_items(
        "child-thread",
        "child-turn",
        [
            ItemEntry(
                ItemKind.TURN_STATUS,
                _turn_status(1, "running", thread_id="child-thread", turn_id="child-turn").payload,
            )
        ],
    )
    tracker = _tracker(store, limits=Limits(web_requests=10))
    # exposure.web = 0 已用 + 5 active 子未用份额 = 5;父只剩 5 个请求额度。
    for _ in range(5):
        assert tracker.admit_web_request("web_fetch", browser=False).allowed
    assert tracker.admit_web_request("web_fetch", browser=False) == Admission(
        allowed=False, reason="budget_exhausted", dimension="web_requests"
    )


# ---------------------------------------------------------------------------
# 守恒属性(hypothesis)
# ---------------------------------------------------------------------------


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    limit_tokens=st.integers(min_value=0, max_value=5_000),
    parent_tokens=st.integers(min_value=0, max_value=2_000),
    children=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=1_000),  # grant tokens
            st.integers(min_value=0, max_value=1_000),  # 子已用 tokens
            st.sampled_from(["completed", "failed", "cancelled", "running", "paused", "interrupted"]),
        ),
        max_size=3,
    ),
    extensions=st.lists(st.integers(min_value=0, max_value=500), max_size=3),
    reverse_facts=st.booleans(),
)
def test_conservation_exposure_equals_used_plus_active_unused(
    limit_tokens: int,
    parent_tokens: int,
    children: list[tuple[int, int, str]],
    extensions: list[int],
    reverse_facts: bool,
) -> None:
    """守恒:exposure.tokens = Σ已用事实 + Σactive 子未用份额。

    假设空间(写清以便定位反例):
    - 单层树:1 个父 turn + 0..3 个直接子分配,无孙代;tokens 维;
    - 每个子的 grant/used ∈ [0,1000],state ∈ 六种 turn_status(终态/非终态);
    - 扩展增量 ∈ [0,500],只增不减;事实序列可能整体倒序(折叠必须与顺序无关);
    - 不做畸形 payload(畸形由上面示例测试覆盖),allocation_id 唯一。
    """
    parent_items: list[Item] = [
        _model_response(1, usage=_usage_payload(input_tokens=parent_tokens)),
    ]
    threads: dict[str, list[Item]] = {PARENT_THREAD: parent_items}
    seq = 2
    expected_used = parent_tokens
    expected_exposure = parent_tokens
    expected_under = False
    active_states = {"running", "paused", "interrupted"}

    for index, (grant, used, turn_result) in enumerate(children):
        allocation_id = f"alloc-{index}"
        child_thread = f"child-{index}"
        child_turn = f"child-turn-{index}"
        parent_items.append(_allocation(seq, allocation_id=allocation_id, child_thread_id=child_thread, child_root_turn_id=child_turn, grant={"tokens": grant}))
        seq += 1
        threads[child_thread] = [
            _model_response(
                1,
                usage=_usage_payload(input_tokens=used),
                thread_id=child_thread,
                turn_id=child_turn,
            ),
            _turn_status(2, turn_result, thread_id=child_thread, turn_id=child_turn),
        ]
        expected_used += used
        if turn_result in active_states:
            # active:已用 + (grant - 已用) 下钳 0 = max(grant, used)
            expected_exposure += used + max(grant - used, 0)
            expected_under = expected_under or used > grant
        else:
            expected_exposure += used

    for delta in extensions:
        parent_items.append(_extension(seq, {"tokens": delta}))
        seq += 1

    ordered = list(reversed(parent_items)) if reverse_facts else parent_items
    store = _ThreadStore({**threads, PARENT_THREAD: ordered})
    node = build_tree(
        store,
        thread_id=PARENT_THREAD,
        turn_id=TURN,
        root_limits=Limits(tokens=limit_tokens),
        unit_costs=None,
    )

    effective_limit = limit_tokens + sum(extensions)
    assert node.used.tokens == expected_used
    assert exposure(node).tokens == expected_exposure
    assert remaining(node).tokens == max(effective_limit - expected_exposure, 0)
    assert node.limits.tokens == effective_limit
    assert node.under_accounted is expected_under
    # 预算不凭空归零:已发生的事实永远在 exposure 里。
    assert exposure(node).tokens >= expected_used


# ---------------------------------------------------------------------------
# 守恒属性补强(阶段4 T9 / T2 minor②):kill 点前缀截断 + 孙代嵌套
# ---------------------------------------------------------------------------


@settings(
    max_examples=60,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    limit_tokens=st.integers(min_value=0, max_value=5_000),
    web_limit=st.integers(min_value=0, max_value=10),
    parent_tokens=st.integers(min_value=0, max_value=500),
    parent_declares_calls=st.booleans(),
    children=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=300),  # grant tokens
            st.integers(min_value=0, max_value=5),  # grant web_requests
            st.integers(min_value=0, max_value=300),  # 子已用 tokens
            st.booleans(),  # 子是否声明并 dispatch 一个 web 调用(未闭合)
            st.sampled_from(["completed", "failed", "cancelled", "running", "paused", "interrupted"]),
        ),
        min_size=1,
        max_size=2,
    ),
    extensions=st.lists(st.integers(min_value=0, max_value=200), max_size=2),
)
def test_conservation_every_prefix_truncation_rebuilds(
    limit_tokens: int,
    web_limit: int,
    parent_tokens: int,
    parent_declares_calls: bool,
    children: list[tuple[int, int, int, bool, str]],
    extensions: list[int],
) -> None:
    """kill 点前缀截断:父日志在**每个 seq 切一刀**、子日志各自任意前缀,
    重建的预算树仍守恒(exposure = Σ可见事实 + Σactive 子未用份额)。

    假设空间(写清以便定位反例):
    - 父 = 1 条 model_response(可选声明轮)+ 1..2 条 allocation + 0..2 条扩展;
    - 子 = model_response(used tokens,可选声明 web 轮)+ 可选未闭合 web tool_call
      + turn_status;子的 grant 覆盖 tokens/web_requests 两维(steps 无限);
    - 截断 = 崩溃现场:父前缀 k ∈ [0, len],每个可见子的前缀 j ∈ [0, len] 全组合
      (allocation 不可见 ⇒ 子线程根本不进折叠;allocation 可见但子前缀为空 ⇒
      保守 active 全额占用——与 build_tree fail-open 同向);
    - 期望值全部**手算**(不调 fold_turn_facts),属性钉的是折叠本身而非同义反复。
    """
    parent_items: list[Item] = [
        _model_response(
            1,
            usage=_usage_payload(input_tokens=parent_tokens),
            tool_calls=({"call_id": "p1", "name": "read", "arguments": {}},) if parent_declares_calls else (),
        )
    ]
    child_threads: dict[str, list[Item]] = {}
    seq = 2
    for index, (grant, grant_web, used, has_web, state) in enumerate(children):
        child_thread = f"child-{index}"
        child_turn = f"child-turn-{index}"
        parent_items.append(
            _allocation(
                seq,
                allocation_id=f"alloc-{index}",
                child_thread_id=child_thread,
                child_root_turn_id=child_turn,
                grant={"tokens": grant, "web_requests": grant_web},
            )
        )
        seq += 1
        items = [
            _model_response(
                1,
                usage=_usage_payload(input_tokens=used),
                tool_calls=({"call_id": "w1", "name": "web_fetch", "arguments": {}},) if has_web else (),
                thread_id=child_thread,
                turn_id=child_turn,
            )
        ]
        if has_web:
            items.append(_tool_call(2, call_id="w1", thread_id=child_thread, turn_id=child_turn))
        items.append(_turn_status(3 if has_web else 2, state, thread_id=child_thread, turn_id=child_turn))
        child_threads[child_thread] = items
    for delta in extensions:
        parent_items.append(_extension(seq, {"tokens": delta}))
        seq += 1

    active_states = {"running", "paused", "interrupted"}
    cuts = 0
    for k in range(len(parent_items) + 1):
        visible = parent_items[:k]
        visible_allocs = [i for i in visible if i.kind == ItemKind.CHILD_ALLOCATION]
        exp_limit = limit_tokens + sum(
            int(i.payload["limits_delta"]["tokens"])  # type: ignore[index]
            for i in visible
            if i.kind == ItemKind.BUDGET_EXTENSION
        )
        # 可见子的前缀全组合(每个子独立切一刀)。
        visible_children = [
            (i, child_threads[str(i.payload["child_thread_id"])]) for i in visible_allocs
        ]
        for combo in itertools.product(*(range(len(items) + 1) for _j, items in visible_children)):
            # 手算期望:每个组合独立重算(父可见事实 + 各子前缀事实)。
            exp_tokens = parent_tokens if k >= 1 else 0
            exp_steps = 1 if (k >= 1 and parent_declares_calls) else 0
            exp_web = 0
            exp_reserved_tokens = 0
            exp_reserved_web = 0
            exp_under = False
            threads: dict[str, list[Item]] = {PARENT_THREAD: visible}
            for (alloc, items), j in zip(visible_children, combo, strict=True):
                threads[str(alloc.payload["child_thread_id"])] = items[:j]
                grant = int(alloc.payload["grant"]["tokens"])  # type: ignore[index]
                grant_web = int(alloc.payload["grant"]["web_requests"])  # type: ignore[index]
                index = int(str(alloc.payload["allocation_id"]).rsplit("-", 1)[1])
                _g, _gw, used, has_web, state = children[index]
                child_tokens = used if j >= 1 else 0
                child_steps = 1 if (j >= 1 and has_web) else 0
                child_web = 1 if (j >= 2 and has_web) else 0
                status_seq = 3 if has_web else 2
                # 子终态事实可见才结算;前缀截断可能在 turn_status 之前(保守 active)。
                active = state in active_states or j < status_seq
                exp_tokens += child_tokens
                exp_steps += child_steps
                exp_web += child_web
                if active:
                    exp_reserved_tokens += max(grant - child_tokens, 0)
                    exp_reserved_web += max(grant_web - child_web, 0)
                    exp_under = exp_under or child_tokens > grant or child_web > grant_web
            node = build_tree(
                _ThreadStore(threads),
                thread_id=PARENT_THREAD,
                turn_id=TURN,
                root_limits=Limits(tokens=limit_tokens, web_requests=web_limit),
                unit_costs=None,
            )
            expected_used = UsageFacts(tokens=exp_tokens, steps=exp_steps, web_requests=exp_web)
            expected_reserved = UsageFacts(tokens=exp_reserved_tokens, web_requests=exp_reserved_web)
            assert node.used == expected_used, (k, combo)
            assert node.reserved == expected_reserved, (k, combo)
            assert exposure(node) == expected_used + expected_reserved, (k, combo)
            assert node.under_accounted is exp_under, (k, combo)
            assert node.limits.tokens == exp_limit, (k, combo)
            assert remaining(node).tokens == max(exp_limit - exposure(node).tokens, 0)
            assert remaining(node).web_requests == max(web_limit - exposure(node).web_requests, 0)
            cuts += 1
    assert cuts > len(parent_items)  # 非空泛:每个父 seq 切点都真实重建过


@settings(
    max_examples=80,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    limit_tokens=st.integers(min_value=0, max_value=5_000),
    parent_tokens=st.integers(min_value=0, max_value=500),
    child_grant=st.integers(min_value=0, max_value=600),
    child_tokens=st.integers(min_value=0, max_value=300),
    child_state=st.sampled_from(["completed", "failed", "cancelled", "running", "paused", "interrupted"]),
    grand_grant=st.integers(min_value=0, max_value=300),
    grand_tokens=st.integers(min_value=0, max_value=300),
    grand_state=st.sampled_from(["completed", "failed", "cancelled", "running", "paused", "interrupted"]),
    extension=st.integers(min_value=0, max_value=200),
)
def test_conservation_nested_grandchild_allocation(
    limit_tokens: int,
    parent_tokens: int,
    child_grant: int,
    child_tokens: int,
    child_state: str,
    grand_grant: int,
    grand_tokens: int,
    grand_state: str,
    extension: int,
) -> None:
    """孙代嵌套守恒:父→子→孙三层 allocation,每层 exposure = 本层已用(含全部
    后代事实)+ **本层** active 直接子的未用份额。

    承重区分:孙代的未用份额只在子层 reserved(子的 grant 已在父层占用);
    父层 reserved 绝不重复计孙代 grant。子终态 ⇒ 父层份额推导式归还,即便孙
    仍 active(子的预算节点随子终态不再准入,孙占用留在子层账上)。
    """
    parent_items = [
        _model_response(1, usage=_usage_payload(input_tokens=parent_tokens)),
        _allocation(
            2, allocation_id="alloc-c", child_thread_id="child", child_root_turn_id="child-turn",
            grant={"tokens": child_grant},
        ),
        _extension(3, {"tokens": extension}),
    ]
    child_items = [
        _model_response(
            1, usage=_usage_payload(input_tokens=child_tokens), thread_id="child", turn_id="child-turn"
        ),
        _allocation(
            2, allocation_id="alloc-g", child_thread_id="grand", child_root_turn_id="grand-turn",
            grant={"tokens": grand_grant}, thread_id="child", turn_id="child-turn",
        ),
        _turn_status(3, child_state, thread_id="child", turn_id="child-turn"),
    ]
    grand_items = [
        _model_response(1, usage=_usage_payload(input_tokens=grand_tokens), thread_id="grand", turn_id="grand-turn"),
        _turn_status(2, grand_state, thread_id="grand", turn_id="grand-turn"),
    ]
    store = _ThreadStore({PARENT_THREAD: parent_items, "child": child_items, "grand": grand_items})
    active_states = {"running", "paused", "interrupted"}
    child_active = child_state in active_states
    grand_active = grand_state in active_states

    root = build_tree(
        store, thread_id=PARENT_THREAD, turn_id=TURN, root_limits=Limits(tokens=limit_tokens), unit_costs=None
    )
    subtree_used = parent_tokens + child_tokens + grand_tokens
    root_reserved = max(child_grant - (child_tokens + grand_tokens), 0) if child_active else 0
    assert root.used.tokens == subtree_used
    assert root.reserved.tokens == root_reserved  # 孙代 grant 不在父层重复占用
    assert exposure(root).tokens == subtree_used + root_reserved
    assert root.under_accounted is (child_active and child_tokens + grand_tokens > child_grant)
    assert root.limits.tokens == limit_tokens + extension
    assert root.children["alloc-c"].used.tokens == child_tokens + grand_tokens

    child_node = build_tree(
        store, thread_id="child", turn_id="child-turn", root_limits=Limits(tokens=child_grant), unit_costs=None
    )
    child_reserved = max(grand_grant - grand_tokens, 0) if grand_active else 0
    assert child_node.used.tokens == child_tokens + grand_tokens
    assert child_node.reserved.tokens == child_reserved
    assert exposure(child_node).tokens == child_tokens + grand_tokens + child_reserved
    assert child_node.under_accounted is (grand_active and grand_tokens > grand_grant)
    # 子终态 + 孙 active:父层已归还(settle 推导),孙占用仍留在子层账上。
    if not child_active:
        assert root.reserved.tokens == 0
        assert exposure(root).tokens == subtree_used
