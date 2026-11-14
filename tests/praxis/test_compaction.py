"""core.compaction:自动压缩(Task 9 / D11/D15;B22 读侧收口;R-T9-2 声明窗口保护)。

钉死:
- 未超阈值 → 不写任何 item;
- 超阈值 → 只写 ``summary`` + ``compaction_boundary`` 两条,正文绝不复制(B22);
- **R-T9-2**:覆盖上界不得切断未闭合的声明窗口(否则后续结果会成为孤儿);
- 摘要模型异常/空摘要 → 返回 ``None`` 且绝不 fail turn;
- turn 接入每 turn 最多压缩一次(R-P4)。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.adapters.models.tokenization import measure_request
from praxis.core.compaction import (
    COMPACT_VERSION,
    SUMMARY_MAX_OUTPUT_TOKENS,
    SUMMARY_PROMPT_MAX_CHARS,
    CompactionOutcome,
    default_compact_limit,
    default_summary_max_output_tokens,
    maybe_compact,
)
from praxis.core.context import build_context, to_messages
from praxis.core.events import EventBus
from praxis.core.recovery import classify_pending
from praxis.core.statemachine import TurnStatus
from praxis.core.turn import TurnDeps, TurnState, run_turn
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.models import (
    ModelResponse,
    ModelSpec,
    Usage,
    WireRequest,
)
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PermissionPolicy
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import create_turn

THREAD = "thread-1"
TURN = "turn-1"
LONG_BODY = "LONG-TOOL-OUTPUT-BODY " + "x" * 4096


def _measure(req: WireRequest) -> int:
    return measure_request(req).count


def _response(text: str) -> ModelResponse:
    return ModelResponse(
        content=text,
        tool_calls=(),
        usage=Usage(input_tokens=11, output_tokens=7),
        finish_reason="stop",
    )


def _summary_adapter(text: str = "SUMMARY-TEXT") -> ScriptedAdapter:
    return ScriptedAdapter([_response(text)])


def _seed(store: Store, entries: Sequence[ItemEntry], *, thread_id: str = THREAD) -> list[Item]:
    store.ensure_thread(thread_id)
    store.append_items(thread_id, TURN, entries)
    return store.read_items(thread_id)


def _basic_history() -> list[ItemEntry]:
    return [
        ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "hello"}),
        ItemEntry(
            kind=ItemKind.MODEL_RESPONSE,
            payload={
                "content": "use c1",
                "tool_calls": [{"call_id": "c1", "name": "bash", "arguments": {}}],
            },
        ),
        ItemEntry(
            kind=ItemKind.TOOL_RESULT,
            payload={
                "call_id": "c1",
                "tool_name": "bash",
                "content": [{"type": "text", "data": {"text": LONG_BODY}}],
            },
        ),
        ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "again"}),
        ItemEntry(
            kind=ItemKind.MODEL_RESPONSE,
            payload={
                "content": "use c2",
                "tool_calls": [{"call_id": "c2", "name": "read", "arguments": {}}],
            },
        ),
        ItemEntry(
            kind=ItemKind.TOOL_RESULT,
            payload={
                "call_id": "c2",
                "tool_name": "read",
                "content": [{"type": "text", "data": {"text": "short result"}}],
            },
        ),
        ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "third"}),
        ItemEntry(kind=ItemKind.MODEL_RESPONSE, payload={"content": "final answer"}),
    ]


# ---------------------------------------------------------------------------
# 阈值(tests below limit)与 payload 形状
# ---------------------------------------------------------------------------


def test_below_limit_writes_nothing(store: Store, bus: EventBus) -> None:
    before = _seed(store, _basic_history())

    async def run() -> CompactionOutcome | None:
        return await maybe_compact(
            store=store,
            adapter=_summary_adapter(),
            thread_id=THREAD,
            turn_id=TURN,
            items=before,
            limit_tokens=10**9,
            measure_tokens=_measure,
        )

    assert asyncio.run(run()) is None
    assert store.read_items(THREAD) == before


def test_writes_summary_and_boundary_only_and_never_copies_body(
    store: Store, bus: EventBus
) -> None:
    before = _seed(store, _basic_history())
    adapter = _summary_adapter("SUMMARY-TEXT")

    async def run() -> CompactionOutcome | None:
        return await maybe_compact(
            store=store,
            adapter=adapter,
            thread_id=THREAD,
            turn_id=TURN,
            items=before,
            limit_tokens=10,
            keep_tail_items=2,
            measure_tokens=_measure,
        )

    outcome = asyncio.run(run())
    assert outcome is not None
    assert outcome.covered_from == 1
    assert outcome.covered_to == 6
    assert outcome.covered_from < outcome.covered_to
    assert outcome.summary_chars == len("SUMMARY-TEXT")
    assert outcome.usage is not None and outcome.usage.output_tokens == 7

    after = store.read_items(THREAD)
    assert len(after) == len(before) + 2
    summary, boundary = after[-2], after[-1]
    assert (summary.kind, boundary.kind) == (ItemKind.SUMMARY, ItemKind.COMPACTION_BOUNDARY)
    assert summary.payload == {
        "text": "SUMMARY-TEXT",
        "covered_from": 1,
        "covered_to": 6,
        "compact_version": COMPACT_VERSION,
        "model_id": "scripted",
    }
    assert boundary.payload == {
        "algorithm_version": "v1",
        "covered_from": 1,
        "covered_to": 6,
    }

    # B22:压缩只新增 summary + boundary;长正文只留在原 item,绝不复制进新条目。
    new_payloads = json.dumps(
        [dict(summary.payload), dict(boundary.payload)], ensure_ascii=False
    )
    assert "LONG-TOOL-OUTPUT-BODY" not in new_payloads
    assert before[2].payload["content"][0]["data"]["text"] == LONG_BODY  # 原 item 未改

    # 读端折叠后:被覆盖的 1..6 不可见,tail 7/8 与 summary/boundary 保留。
    assert [item.seq for item in build_context(after)] == [7, 8, 9, 10]


def test_default_compact_limit_from_spec_or_override() -> None:
    spec = ModelSpec(model_id="m", context_window_tokens=100_000, max_output_tokens=8_000)
    assert default_compact_limit(spec) == 100_000 - 8_000 - 8192
    assert default_compact_limit(spec, reserve_output=1_000) == 100_000 - 8_000 - 1_000
    overridden = ModelSpec(
        model_id="m",
        context_window_tokens=100_000,
        max_output_tokens=8_000,
        auto_compact_token_limit=50_000,
    )
    assert default_compact_limit(overridden) == 50_000
    # 显式 0 是合法值("预算耗尽"),不得被 `or` 当成"没配"回落到窗口值。
    disabled = ModelSpec(
        model_id="m",
        context_window_tokens=100_000,
        max_output_tokens=8_000,
        auto_compact_token_limit=0,
    )
    assert default_compact_limit(disabled) == 0
    # 窗口预算耗尽 → 0(调用方把 0/None 都视为"不自动压缩")
    tiny = ModelSpec(model_id="m", context_window_tokens=10_000, max_output_tokens=9_000)
    assert default_compact_limit(tiny) == 0


def test_default_summary_max_output_tokens_respects_small_windows() -> None:
    big = ModelSpec(model_id="m", context_window_tokens=100_000, max_output_tokens=8_000)
    assert default_summary_max_output_tokens(big) == SUMMARY_MAX_OUTPUT_TOKENS
    small = ModelSpec(model_id="m", context_window_tokens=4_000, max_output_tokens=1_024)
    assert default_summary_max_output_tokens(small) == 1_024


def test_summary_request_is_bounded_and_elides_old_tool_bodies(store: Store) -> None:
    # 压缩恰在"旧输出很多"时触发:摘要请求必须过 elide + 硬字符上限,
    # 否则摘要自身会被撑爆、静默失败,自动压缩对长历史失效。
    before = _seed(
        store,
        [
            ItemEntry(
                kind=ItemKind.TOOL_RESULT,
                payload={
                    "call_id": "c1",
                    "tool_name": "bash",
                    "content": [{"type": "text", "data": {"text": LONG_BODY}}],
                },
            ),
            # 用户正文本身无界:只有 elide 不够,必须还有硬上限。
            ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "Z" * 50_000}),
            ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "after"}),
        ],
    )
    adapter = _summary_adapter()

    async def run() -> CompactionOutcome | None:
        return await maybe_compact(
            store=store,
            adapter=adapter,
            thread_id=THREAD,
            turn_id=TURN,
            items=before,
            limit_tokens=10,
            keep_tail_items=0,
            summary_max_output_tokens=64,
            measure_tokens=_measure,
        )

    assert asyncio.run(run()) is not None
    assert adapter.last_request is not None
    prompt = adapter.last_request.messages[0].content
    assert len(prompt) <= SUMMARY_PROMPT_MAX_CHARS
    assert "LONG-TOOL-OUTPUT-BODY" not in prompt
    assert "[elided tool output:" in prompt
    assert "更近的内容已截断" in prompt
    # 方向断言:保留的是最早那段(elided 工具输出在投影开头),丢的是更近的内容
    # (末尾 USER_MESSAGE "after")。把 head/tail 调换时这两条必须失败。
    assert "after" not in prompt
    assert prompt.rstrip().endswith("[... 更近的内容已截断 ...]")
    # composition 注入的输出预算原样进入摘要请求(小窗口模型不会被 2048 撞拒)。
    assert adapter.last_request.max_output_tokens == 64


def test_second_compaction_folds_previous_summary_before_tail(store: Store) -> None:
    store.ensure_thread(THREAD)
    store.append_items(
        THREAD,
        TURN,
        [ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": f"m{i}"}) for i in range(10)],
    )
    adapter = ScriptedAdapter([_response("S1"), _response("S2")])

    async def compact(items: Sequence[Item]) -> CompactionOutcome | None:
        return await maybe_compact(
            store=store,
            adapter=adapter,
            thread_id=THREAD,
            turn_id=TURN,
            items=items,
            limit_tokens=10,
            keep_tail_items=4,
            measure_tokens=_measure,
        )

    assert asyncio.run(compact(store.read_items(THREAD))) is not None
    store.append_items(
        THREAD,
        TURN,
        [ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": f"n{i}"}) for i in range(4)],
    )
    assert asyncio.run(compact(store.read_items(THREAD))) is not None

    visible = build_context(store.read_items(THREAD))
    summaries = [item for item in visible if item.kind == ItemKind.SUMMARY]
    assert len(summaries) == 1  # 旧摘要被新 boundary 覆盖并折叠,不累积
    assert summaries[0].payload["text"] == "S2"

    messages = to_messages(visible)
    summary_indexes = [
        index
        for index, message in enumerate(messages)
        if message.content.startswith("[conversation summary]")
    ]
    assert summary_indexes == [0]  # 存活摘要投影在保留尾部之前
    assert any(message.role == "user" and message.content.startswith("n") for message in messages)


# ---------------------------------------------------------------------------
# R-T9-2:覆盖区间不得切断未闭合的声明窗口
# ---------------------------------------------------------------------------


def test_does_not_cut_open_declaration_window(store: Store, bus: EventBus) -> None:
    before = _seed(
        store,
        [
            ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "hello"}),
            ItemEntry(
                kind=ItemKind.MODEL_RESPONSE,
                payload={
                    "content": "closed",
                    "tool_calls": [{"call_id": "c1", "name": "bash", "arguments": {}}],
                },
            ),
            ItemEntry(
                kind=ItemKind.TOOL_RESULT,
                payload={
                    "call_id": "c1",
                    "tool_name": "bash",
                    "content": [{"type": "text", "data": {"text": "ok"}}],
                },
            ),
            ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "next"}),
            # 最后一轮声明未闭合:没有 c2 的 tool_result。
            ItemEntry(
                kind=ItemKind.MODEL_RESPONSE,
                payload={
                    "content": "pending",
                    "tool_calls": [{"call_id": "c2", "name": "read", "arguments": {}}],
                },
            ),
        ],
    )

    async def run() -> CompactionOutcome | None:
        return await maybe_compact(
            store=store,
            adapter=_summary_adapter(),
            thread_id=THREAD,
            turn_id=TURN,
            items=before,
            limit_tokens=10,
            keep_tail_items=0,  # 迫使未闭合声明进入候选区间,验证上界回退
            measure_tokens=_measure,
        )

    outcome = asyncio.run(run())
    assert outcome is not None
    # 若上界停在 5,seq5 的声明会被折叠,后续 c2 结果将成孤儿。
    assert outcome.covered_to == 4
    assert outcome.covered_from == 1

    after = store.read_items(THREAD)
    visible = build_context(after)
    assert [item.seq for item in visible] == [5, 6, 7]

    messages = to_messages(visible)
    roles = [message.role for message in messages]
    # summary 投影在保留尾部之前(seq 上它更晚,语义上是"更早历史的摘要")。
    assert roles == ["user", "assistant", "tool"]
    assert messages[0].content.startswith("[conversation summary]")
    tool_messages = [message for message in messages if message.role == "tool"]
    assert len(tool_messages) == 1  # 无孤儿、无重复
    assert tool_messages[0].tool_call_id == "c2"
    assert messages[1].tool_calls and messages[1].tool_calls[0].call_id == "c2"

    pending = classify_pending(visible, {})
    assert [call.call.call_id for call in pending] == ["c2"]


def test_retreats_iteratively_when_first_cut_exposes_an_older_pair(store: Store, bus: EventBus) -> None:
    # decl c1 @2 → result @6;更晚的未闭合声明 @4 先逼上界退到 3,
    # 但 3 会切断 2..6 这一对 → 必须继续退到 1(before 2),否则 seq6 成孤儿。
    before = _seed(
        store,
        [
            ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "hello"}),
            ItemEntry(
                kind=ItemKind.MODEL_RESPONSE,
                payload={
                    "content": "open c1",
                    "tool_calls": [{"call_id": "c1", "name": "bash", "arguments": {}}],
                },
            ),
            ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "mid"}),
            ItemEntry(
                kind=ItemKind.MODEL_RESPONSE,
                payload={
                    "content": "open c2",
                    "tool_calls": [{"call_id": "c2", "name": "read", "arguments": {}}],
                },
            ),
            ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "later"}),
            ItemEntry(
                kind=ItemKind.TOOL_RESULT,
                payload={
                    "call_id": "c1",
                    "tool_name": "bash",
                    "content": [{"type": "text", "data": {"text": "ok"}}],
                },
            ),
        ],
    )

    async def run() -> CompactionOutcome | None:
        return await maybe_compact(
            store=store,
            adapter=_summary_adapter(),
            thread_id=THREAD,
            turn_id=TURN,
            items=before,
            limit_tokens=10,
            keep_tail_items=0,
            measure_tokens=_measure,
        )

    outcome = asyncio.run(run())
    assert outcome is not None
    assert outcome.covered_to == 1  # 回退到声明 2 之前,保住 2→6 的完整配对

    visible = build_context(store.read_items(THREAD))
    messages = to_messages(visible)
    # 声明 c1 与它的真实结果都在可见区;c2 未闭合补合成;summary 在最前。
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert [m.content for m in messages if m.role == "tool"] == [
        "ok",
        "[tool result unavailable: turn ended before execution]",
    ]


# ---------------------------------------------------------------------------
# 失败绝不 fail turn
# ---------------------------------------------------------------------------


def test_model_exception_returns_none_and_writes_nothing(
    store: Store, bus: EventBus, caplog: pytest.LogCaptureFixture
) -> None:
    before = _seed(store, _basic_history())
    adapter = ScriptedAdapter([])  # 剧本耗尽 → complete() 抛 ScriptExhausted

    async def run() -> CompactionOutcome | None:
        return await maybe_compact(
            store=store,
            adapter=adapter,
            thread_id=THREAD,
            turn_id=TURN,
            items=before,
            limit_tokens=10,
            keep_tail_items=2,
            measure_tokens=_measure,
        )

    with caplog.at_level("WARNING", logger="praxis.core.compaction"):
        assert asyncio.run(run()) is None
    assert store.read_items(THREAD) == before
    assert any("compaction" in record.message.lower() for record in caplog.records)


def test_empty_summary_returns_none_and_writes_nothing(store: Store, bus: EventBus) -> None:
    before = _seed(store, _basic_history())

    async def run() -> CompactionOutcome | None:
        return await maybe_compact(
            store=store,
            adapter=_summary_adapter("   \n  "),
            thread_id=THREAD,
            turn_id=TURN,
            items=before,
            limit_tokens=10,
            keep_tail_items=2,
            measure_tokens=_measure,
        )

    assert asyncio.run(run()) is None
    assert store.read_items(THREAD) == before


# ---------------------------------------------------------------------------
# turn 接入:每 turn 最多压缩一次(R-P4),且压缩后继续正常完成
# ---------------------------------------------------------------------------


def _turn_deps(
    store: Store,
    bus: EventBus,
    workspace: Path,
    adapter: ScriptedAdapter,
    *,
    compact_limit_tokens: int | None,
) -> TurnDeps:
    return TurnDeps(
        store=store,
        bus=bus,
        adapter=adapter,
        tools={},
        permissions=PermissionContext(
            workspace=str(workspace), policy=PermissionPolicy(), mode="never"
        ),
        workspace=workspace,
        system_prompt="test system prompt",
        compact_limit_tokens=compact_limit_tokens,
        measure_tokens=_measure,
    )


def test_run_turn_compacts_at_most_once_then_completes(
    store: Store, bus: EventBus, workspace: Path
) -> None:
    store.ensure_thread(THREAD)
    store.append_items(
        THREAD,
        TURN,
        [ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": f"message {i} " + "z" * 40}) for i in range(12)],
    )
    create_turn(store, turn_id=TURN, thread_id=THREAD, binding={})
    adapter = ScriptedAdapter([_response("COMPACTED"), _response("done")])

    deps = _turn_deps(store, bus, workspace, adapter, compact_limit_tokens=1)
    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.COMPLETED
    assert result.output == "done"

    items = store.read_items(THREAD)
    assert [item.kind for item in items].count(ItemKind.SUMMARY) == 1
    assert [item.kind for item in items].count(ItemKind.COMPACTION_BOUNDARY) == 1
    summary = next(item for item in items if item.kind == ItemKind.SUMMARY)
    assert summary.payload["text"] == "COMPACTED"  # 摘要来自同一 adapter.complete()


def test_run_turn_without_compact_limit_keeps_item_sequence(
    store: Store, bus: EventBus, workspace: Path
) -> None:
    store.ensure_thread(THREAD)
    store.append_items(
        THREAD, TURN, [ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "hello"})]
    )
    create_turn(store, turn_id=TURN, thread_id=THREAD, binding={})
    adapter = ScriptedAdapter([_response("done")])

    deps = _turn_deps(store, bus, workspace, adapter, compact_limit_tokens=None)
    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.COMPLETED
    kinds = [item.kind for item in store.read_items(THREAD)]
    assert ItemKind.SUMMARY not in kinds and ItemKind.COMPACTION_BOUNDARY not in kinds
    assert kinds == [
        ItemKind.USER_MESSAGE,
        ItemKind.MODEL_RESPONSE,
        ItemKind.TURN_STATUS,
    ]


def test_turn_read_chain_elides_old_tool_outputs(
    store: Store, bus: EventBus, workspace: Path
) -> None:
    # D11 读侧链:turn 的真实请求必须经过 elide(最后 12 条完整,更早的大输出省略)。
    store.ensure_thread(THREAD)
    store.append_items(
        THREAD,
        TURN,
        [
            ItemEntry(
                kind=ItemKind.TOOL_RESULT,
                payload={
                    "call_id": f"c{i}",
                    "tool_name": "bash",
                    "content": [
                        {"type": "text", "data": {"text": f"BODY-{i}-" + "x" * 3000}}
                    ],
                },
            )
            for i in range(1, 15)
        ],
    )
    create_turn(store, turn_id=TURN, thread_id=THREAD, binding={})
    adapter = ScriptedAdapter([_response("done")])

    deps = _turn_deps(store, bus, workspace, adapter, compact_limit_tokens=None)
    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.COMPLETED
    assert adapter.last_request is not None
    tool_texts = [m.content for m in adapter.last_request.messages if m.role == "tool"]
    assert len(tool_texts) == 14
    assert sum("[elided tool output:" in text for text in tool_texts) == 2
    assert any("BODY-14-" in text for text in tool_texts)  # 最新的保持完整


# ---------------------------------------------------------------------------
# 覆盖区间选择器(经 maybe_compact 的公开行为观测,不做私有 import)
# ---------------------------------------------------------------------------


def test_keep_tail_excludes_non_projecting_status_items(
    store: Store, bus: EventBus
) -> None:
    entries = [ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": f"m{i}"}) for i in range(4)]
    entries.append(ItemEntry(kind=ItemKind.TURN_STATUS, payload={"status": "running"}))
    before = _seed(store, entries)

    async def run() -> CompactionOutcome | None:
        return await maybe_compact(
            store=store,
            adapter=_summary_adapter(),
            thread_id=THREAD,
            turn_id=TURN,
            items=before,
            limit_tokens=10,
            keep_tail_items=2,
            measure_tokens=_measure,
        )

    outcome = asyncio.run(run())
    assert outcome is not None
    # 可压缩 item = 4 条 user_message;tail=最后 2 条;覆盖上界 = seq2。
    assert (outcome.covered_from, outcome.covered_to) == (1, 2)


def test_to_messages_after_compaction_has_no_orphan_tool_result(
    store: Store, bus: EventBus
) -> None:
    before = _seed(store, _basic_history())

    async def run() -> CompactionOutcome | None:
        return await maybe_compact(
            store=store,
            adapter=_summary_adapter(),
            thread_id=THREAD,
            turn_id=TURN,
            items=before,
            limit_tokens=10,
            keep_tail_items=2,
            measure_tokens=_measure,
        )

    assert asyncio.run(run()) is not None
    messages = to_messages(build_context(store.read_items(THREAD)))
    tool_ids = [m.tool_call_id for m in messages if m.role == "tool"]
    # 两个声明窗口都落在覆盖区间内、且各自结果随声明一起折叠 → 不产生孤儿。
    assert tool_ids == []
    assistant_call_ids = {
        call.call_id
        for message in messages
        if message.role == "assistant"
        for call in message.tool_calls
    }
    assert set(tool_ids) <= assistant_call_ids
