"""B31 回归:同 turn 内复用 call_id 的声明窗口配对(投影)与 classify_pending。

本地小模型/网关每轮从 ``call_0`` 重新起号是真实形状。旧读路径把调用身份当成
``(turn_id, call_id)``,同一 turn 第二轮复用 id 时第一轮的真实结果被判"过期"
丢弃,而键存在又抑制了第一轮的合成 → 第一轮 assistant 成为孤儿 tool_call,
真实 provider 拒收整个请求(投影侧 B31);恢复侧同一错误假设会让第二轮尚未
执行的调用被第一轮结果"吞掉"(静默丢失一次调用,R-T7-1)。

本文件钉死修复后的判据:配对键 = ``(turn_id, 声明它的 model_response.seq,
call_id)``;结果配给它之前最近的同 ``(turn_id, call_id)`` 声明;配不上的孤儿
结果原样投影(fail-open);未闭合声明补合成(R9);同窗口多条结果最新 seq 胜
(R10)。``classify_pending`` 与 ``to_messages`` 共用 ``pair_tool_results``。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from hypothesis import given, settings
from hypothesis import strategies as st

from praxis.core.context import build_context, to_messages
from praxis.core.recovery import PendingCall, classify_pending
from praxis.interfaces.sdk import (
    Item,
    ItemKind,
    ModelResponse,
    ModelToolCall,
    ScriptedAdapter,
    Usage,
    open_session,
)
from praxis.protocol.models import ModelMessage
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    Tool,
    ToolDefinition,
    ToolEffect,
    ToolResult,
)
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import create_turn

_UNPAIRED_NOTE = "[tool result unavailable: turn ended before execution]"
_TURN = "turn-1"


def mk(
    seq: int,
    kind: str,
    payload: Mapping[str, Any],
    turn_id: str | None = _TURN,
) -> Item:
    return Item(
        item_id=f"i{seq}",
        thread_id="t",
        turn_id=turn_id,
        seq=seq,
        kind=kind,
        payload=payload,
        created_at=float(seq),
    )


def assistant(seq: int, calls: Sequence[str], turn_id: str = _TURN) -> Item:
    return mk(
        seq,
        ItemKind.MODEL_RESPONSE,
        {
            "content": "resp",
            "tool_calls": [
                {"call_id": call_id, "name": "t", "arguments": {}} for call_id in calls
            ],
        },
        turn_id,
    )


def result(seq: int, call_id: str, text: str, turn_id: str = _TURN) -> Item:
    return mk(
        seq,
        ItemKind.TOOL_RESULT,
        {
            "call_id": call_id,
            "status": "ok",
            "content": [{"type": "text", "data": {"text": text}}],
        },
        turn_id,
    )


# ---------------------------------------------------------------------------
# brief Step 1:B31 投影用例
# ---------------------------------------------------------------------------


def test_same_turn_reused_call_ids_both_project() -> None:
    """B31:本地小模型每轮从 call_0 重新起号 → 早期轮结果不得被判过期。"""
    items = [
        mk(1, ItemKind.USER_MESSAGE, {"text": "go"}),
        assistant(2, ["call_0"]),
        result(3, "call_0", "A-CONTENT"),
        assistant(4, ["call_0"]),
        result(5, "call_0", "B-CONTENT"),
    ]
    msgs = to_messages(build_context(items))
    assert [m.content for m in msgs if m.role == "tool"] == ["A-CONTENT", "B-CONTENT"]
    # 每组 assistant 后恰好一条 tool 消息;两轮都是真实结果,没有任何合成
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant", "tool"]
    assert all(_UNPAIRED_NOTE not in m.content for m in msgs)
    assert [m.tool_call_id for m in msgs if m.role == "tool"] == ["call_0", "call_0"]


def test_duplicate_within_one_response_still_last_wins() -> None:
    """同一条 model_response 声明两次 call_0(真畸形)→ 一条真实结果 + 一条合成,不抛。

    两次声明落在同一声明窗口(同 seq + 同 call_id),窗口内最新结果只投影一次;
    第二次声明没有独立结果可配 → R9 合成。provider 视角这是已畸形的输入,要求
    的只是"不丢正文、不抛、每个声明都有应答"。
    """
    items = [assistant(1, ["call_0", "call_0"]), result(2, "call_0", "ONLY-RESULT")]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["assistant", "tool", "tool"]
    assert [m.content for m in msgs if m.role == "tool"] == ["ONLY-RESULT", _UNPAIRED_NOTE]
    assert [m.tool_call_id for m in msgs if m.role == "tool"] == ["call_0", "call_0"]


def test_orphan_result_without_declaration_projects() -> None:
    """只有 tool_result(CLI 直调遗留)→ 原样在自身位置投影(fail-open 读,不静默丢)。"""
    items = [
        mk(1, ItemKind.USER_MESSAGE, {"text": "before"}),
        result(2, "ghost", "LEGACY-RESULT"),
        mk(3, ItemKind.USER_MESSAGE, {"text": "after"}),
    ]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["user", "tool", "user"]
    assert msgs[1].content == "LEGACY-RESULT" and msgs[1].tool_call_id == "ghost"


def test_orphan_result_with_malformed_payload_is_skipped_not_raised() -> None:
    items = [
        mk(1, ItemKind.TOOL_RESULT, {"call_id": "ghost", "content": "not-a-list"}),
        result(2, "ghost2", "OK"),
    ]
    assert [m.content for m in to_messages(items)] == ["OK"]


def test_multiple_orphan_results_same_id_only_latest_projects() -> None:
    """同 (turn_id, call_id) 的多条孤儿结果仍最新 seq 胜(R10 精神)。

    全部投影会产出多条同 tool_call_id 的 tool 消息——provider 直接拒收。
    """
    items = [
        mk(1, ItemKind.USER_MESSAGE, {"text": "before"}),
        result(2, "ghost", "OLD"),
        result(3, "ghost", "NEW"),
        mk(4, ItemKind.USER_MESSAGE, {"text": "after"}),
    ]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["user", "tool", "user"]
    assert msgs[1].content == "NEW" and msgs[1].tool_call_id == "ghost"


def test_partially_malformed_declaration_does_not_swallow_real_result() -> None:
    """I-1:声明项整条不可重建(如缺 name)→ 不开窗,真实结果不得被静默吞掉。

    开窗谓词与 ``_reconstruct_tool_calls`` 同严格度:assistant 渲染不出来时,
    配对结果降级为孤儿、在自身位置 fail-open 投影(旧实现会因键在 paired 中
    被抑制、assistant 又被跳过,结果从消息里消失)。
    """
    items = [
        mk(1, ItemKind.MODEL_RESPONSE, {"content": "", "tool_calls": [{"call_id": "c1"}]}),
        result(2, "c1", "REAL-RESULT"),
    ]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["tool"]
    assert msgs[0].content == "REAL-RESULT" and msgs[0].tool_call_id == "c1"


def test_pairable_result_is_not_projected_at_its_own_position() -> None:
    """配对成功的结果统一跟在声明 assistant 之后;自身位置不再重复投影。"""
    items = [
        assistant(1, ["c1"]),
        mk(2, ItemKind.USER_MESSAGE, {"text": "interleaved"}),
        result(3, "c1", "PAIRED"),
    ]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["assistant", "tool", "user"]
    assert msgs[1].content == "PAIRED"


def test_malformed_paired_result_falls_back_to_synthesis() -> None:
    """声明的结果不可重建时补合成:绝不把 assistant 留成孤儿 tool_call。"""
    items = [
        assistant(1, ["c1"]),
        mk(2, ItemKind.TOOL_RESULT, {"call_id": "c1", "content": "not-a-list"}),
    ]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["assistant", "tool"]
    assert msgs[1] == ModelMessage(role="tool", content=_UNPAIRED_NOTE, tool_call_id="c1")


# ---------------------------------------------------------------------------
# 属性:任意(声明/结果/乱序/重复/畸形)组合下投影不变式
# ---------------------------------------------------------------------------


@st.composite
def declaration_streams(draw: st.DrawFn) -> tuple[list[Item], list[list[str]]]:
    """随机声明窗口 + 结果(0-2 条/声明)+ 孤儿结果 + 畸形/未知 kind,乱序返回。

    同时返回"每个可渲染 assistant 的 tool 消息期望内容(按声明顺序)"——属性
    不只校验条数,还钉死内容归属:声明的投影必须等于该窗口内最新 seq 的真实
    结果;无结果(或重复声明)才是合成注记。这样"第一轮结果串到第二轮窗口"
    或"合成盖掉真实结果"都会被抓住。
    """
    items: list[Item] = []
    seq = 0
    declared: set[str] = set()
    pool = ["c0", "c1", "c2"]
    junk_payloads = st.one_of(
        st.just({}),
        st.just({"content": None}),
        st.just({"content": "", "tool_calls": "nope"}),
        st.just({"tool_calls": [{"call_id": 7, "name": "t"}]}),
        st.dictionaries(st.text(max_size=4), st.integers(-3, 3), max_size=3),
    )
    # call_id -> (当前窗口声明 seq, 窗口内最新真实文本);声明换代时冻结旧窗口。
    windows: dict[str, tuple[int, str | None]] = {}
    frozen: dict[tuple[int, str], str | None] = {}
    declarations_by_seq: dict[int, list[str]] = {}

    def freeze(call_id: str) -> None:
        previous_seq, latest = windows.pop(call_id)
        frozen[(previous_seq, call_id)] = latest

    for _ in range(draw(st.integers(0, 3))):
        for _ in range(draw(st.integers(0, 1))):
            seq += 1
            items.append(
                mk(
                    seq,
                    draw(st.sampled_from(["user_message", "turn_status", "mystery-kind"])),
                    draw(junk_payloads),
                )
            )
        # 孤儿结果:只从"此前未声明"的 id 池取,保证真的配不上任何声明
        unseen = [call_id for call_id in pool if call_id not in declared]
        if unseen and draw(st.booleans()):
            seq += 1
            items.append(result(seq, draw(st.sampled_from(unseen)), f"orphan-{seq}"))

        declarations = draw(st.lists(st.sampled_from(pool), min_size=1, max_size=3))
        seq += 1
        if draw(st.booleans()):
            # 整条不可重建的畸形响应:不渲染、不产生窗口(其结果由别的轮次覆盖)
            items.append(mk(seq, ItemKind.MODEL_RESPONSE, draw(junk_payloads)))
            continue
        for call_id in dict.fromkeys(declarations):
            if call_id in windows:
                freeze(call_id)
            windows[call_id] = (seq, None)
        items.append(assistant(seq, declarations))
        declared.update(declarations)
        declarations_by_seq[seq] = declarations
        for call_id in declarations:
            for _ in range(draw(st.integers(0, 2))):
                if draw(st.booleans()):
                    seq += 1
                    text = draw(st.sampled_from(["a", "b", "c"]))
                    items.append(result(seq, call_id, text))
                    declare_seq, _previous = windows[call_id]
                    windows[call_id] = (declare_seq, text)
    for call_id in list(windows):
        freeze(call_id)

    expected: list[list[str]] = []
    for declare_seq in sorted(declarations_by_seq):
        contents: list[str] = []
        seen: set[str] = set()
        for call_id in declarations_by_seq[declare_seq]:
            latest = frozen.get((declare_seq, call_id))
            if call_id not in seen and latest is not None:
                contents.append(latest)  # 该窗口最新真实结果
            else:
                contents.append(_UNPAIRED_NOTE)  # 未闭合/重复声明 → 合成
            seen.add(call_id)
        expected.append(contents)

    shuffled = draw(st.permutations(items))
    return list(shuffled), expected


@given(declaration_streams())
@settings(deadline=None, max_examples=200)
def test_pairing_invariant_holds_for_arbitrary_streams(
    data: tuple[list[Item], list[list[str]]],
) -> None:
    items, expected = data
    msgs = to_messages(items)  # 永不抛
    assert all(isinstance(m, ModelMessage) for m in msgs)
    assistants: list[tuple[ModelMessage, list[ModelMessage]]] = []
    for index, message in enumerate(msgs):
        if message.role != "assistant" or not message.tool_calls:
            continue
        next_assistant = next(
            (j for j in range(index + 1, len(msgs)) if msgs[j].role == "assistant"),
            len(msgs),
        )
        group = [m for m in msgs[index + 1 : next_assistant] if m.role == "tool"]
        assistants.append((message, group))

    assert len(assistants) == len(expected)  # 每个可渲染声明恰好一个 assistant 组
    for (message, group), wanted in zip(assistants, expected, strict=True):
        declared = [call.call_id for call in message.tool_calls]
        # 内容归属:组内每条声明的 tool 消息逐条等于"该窗口最新真实内容/合成"
        paired_messages = [m for m in group if m.tool_call_id in set(declared)]
        assert [m.content for m in paired_messages] == wanted, (declared, group)
        # 每条声明(按出现次数计)恰有一条 tool 消息(真实或合成),且保声明序
        for call_id in set(declared):
            following = [m for m in group if m.tool_call_id == call_id]
            assert len(following) == declared.count(call_id), (call_id, declared, group)


@st.composite
def junk_streams(draw: st.DrawFn) -> list[Item]:
    kinds = st.sampled_from(
        [
            "user_message",
            "steering",
            "model_response",
            "tool_call",
            "tool_result",
            "approval_request",
            "approval_decision",
            "turn_status",
            "summary",
            "compaction_boundary",
            "mystery-kind",
        ]
    )
    values = st.recursive(
        st.none()
        | st.booleans()
        | st.integers(-5, 5)
        | st.text(max_size=5),
        lambda children: st.lists(children, max_size=3)
        | st.dictionaries(st.text(max_size=4), children, max_size=3),
        max_leaves=6,
    )
    payload = st.dictionaries(st.text(max_size=8), values, max_size=4)
    size = draw(st.integers(0, 12))
    items = [
        mk(seq, draw(kinds), draw(payload), turn_id=draw(st.sampled_from([_TURN, None])))
        for seq in range(1, size + 1)
    ]
    return list(draw(st.permutations(items)))


@given(junk_streams())
@settings(deadline=None, max_examples=200)
def test_to_messages_never_raises_on_arbitrary_junk(items: list[Item]) -> None:
    assert all(isinstance(m, ModelMessage) for m in to_messages(items))


# ---------------------------------------------------------------------------
# R9 配对修复:未配对的 assistant tool_calls 合成 tool 结果(真实 provider 配对规则)
# ---------------------------------------------------------------------------


def test_pairing_repair_synthesizes_missing_result_in_declaration_order() -> None:
    # (a) assistant 两个 call,只有 c1 有真实结果:配对修复后两条 tool 消息紧跟
    # assistant、按声明顺序(c1 真实在前,c2 合成在后),先于后续 user item。
    # B31 迁移:旧实现把合成消息插在 assistant 紧后、真实结果留在自身位置,
    # 顺序是 c2→c1;声明窗口配对按 tool_calls 顺序输出(provider 组内顺序契约)。
    items = [assistant(1, ["c1", "c2"]), result(2, "c1", "real"), mk(3, ItemKind.USER_MESSAGE, {"text": "next"})]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["assistant", "tool", "tool", "user"]
    assert msgs[1].content == "real" and msgs[1].tool_call_id == "c1"
    assert msgs[2] == ModelMessage(role="tool", content=_UNPAIRED_NOTE, tool_call_id="c2")


def test_all_calls_unpaired_get_synthetic_results() -> None:
    # (b) 零结果:全部合成,按 tool_calls 顺序
    msgs = to_messages([assistant(1, ["c1", "c2"])])
    assert msgs == [
        ModelMessage(
            role="assistant",
            content="resp",
            tool_calls=(
                ModelToolCall(call_id="c1", name="t", arguments={}),
                ModelToolCall(call_id="c2", name="t", arguments={}),
            ),
        ),
        ModelMessage(role="tool", content=_UNPAIRED_NOTE, tool_call_id="c1"),
        ModelMessage(role="tool", content=_UNPAIRED_NOTE, tool_call_id="c2"),
    ]


def test_fully_paired_history_has_no_synthesis() -> None:
    # (c) 全配对:投影与旧行为逐字节一致——无任何合成消息
    items = [
        mk(1, ItemKind.USER_MESSAGE, {"text": "go"}),
        assistant(2, ["c1"]),
        result(3, "c1", "done"),
        mk(4, ItemKind.MODEL_RESPONSE, {"content": "fin"}),
    ]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert all(_UNPAIRED_NOTE not in m.content for m in msgs)
    assert msgs[2].content == "done"


def test_denied_or_not_executed_result_counts_as_paired() -> None:
    # (d) 存在性按 kind+call_id 判定,与 status/文本无关:denied/not_executed 不触发合成
    for status in ("denied", "not_executed", "outcome_unknown", "error"):
        items = [assistant(1, ["c1"]), _result_status(2, "c1", "x", status)]
        msgs = to_messages(items)
        assert [m.role for m in msgs] == ["assistant", "tool"], status
        assert len(msgs) == 2, status


@st.composite
def pairing_items(draw: st.DrawFn) -> list[Item]:
    # (e) 变体:每 turn 1-3 个 call id,从固定小池抽取——跨 turn 复用同一 id 是
    # 真实形状(local provider 每轮从 call_0 重新起号;CLI 累积 default 线程);
    # 单 turn 内 id 唯一,每个 id 随机有/无真实结果
    items: list[Item] = []
    seq = 0
    for turn_idx in range(1, draw(st.integers(1, 4)) + 1):
        turn_id = f"turn-{turn_idx}"
        calls = draw(st.lists(st.sampled_from(["tc0", "tc1", "tc2"]), min_size=1, max_size=3, unique=True))
        seq += 1
        items.append(assistant(seq, calls, turn_id))
        for call_id in calls:
            if draw(st.booleans()):
                seq += 1
                items.append(result(seq, call_id, "ok", turn_id))
        if draw(st.booleans()):
            seq += 1
            items.append(mk(seq, ItemKind.USER_MESSAGE, {"text": "next"}, turn_id))
    return items


@given(pairing_items())
@settings(deadline=None, max_examples=150)
def test_pairing_invariant_holds_for_every_assistant(items: list[Item]) -> None:
    msgs = to_messages(items)
    for idx, message in enumerate(msgs):
        if message.role != "assistant" or not message.tool_calls:
            continue
        next_assistant = next(
            (j for j in range(idx + 1, len(msgs)) if msgs[j].role == "assistant"),
            len(msgs),
        )
        for call in message.tool_calls:
            following = [
                m
                for m in msgs[idx + 1 : next_assistant]
                if m.role == "tool" and m.tool_call_id == call.call_id
            ]
            assert len(following) == 1, call.call_id


# ---------------------------------------------------------------------------
# R10:同一 call_id 只投影最后一条 tool_result(最新 seq 胜;旧结果留库审计)
# ---------------------------------------------------------------------------


def _result_status(seq: int, call_id: str, text: str, status: str, turn_id: str = _TURN) -> Item:
    return mk(
        seq,
        ItemKind.TOOL_RESULT,
        {
            "call_id": call_id,
            "status": status,
            "content": [{"type": "text", "data": {"text": text}}],
        },
        turn_id,
    )


def test_r10_only_last_result_for_a_call_is_projected() -> None:
    items = [assistant(1, ["c1"]), _result_status(2, "c1", "first answer", "ok"),
             _result_status(3, "c1", "second answer", "ok")]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["assistant", "tool"]
    assert msgs[1].content == "second answer"
    assert msgs[1].tool_call_id == "c1"


def test_r10_outcome_unknown_then_reconciled_projects_ok_only() -> None:
    # reconcile 的真实序列:崩溃续跑写 outcome_unknown,人工对账再写 ok——只有最后一条进投影
    items = [
        assistant(1, ["c1"]),
        _result_status(2, "c1", "[outcome unknown: process died during execution]", "outcome_unknown"),
        _result_status(3, "c1", "reconciled by user: ok", "ok"),
    ]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["assistant", "tool"]
    assert msgs[1].content == "reconciled by user: ok"
    assert "outcome unknown" not in msgs[1].content
    # failed 对账同理
    items2 = [items[0], items[1], _result_status(3, "c1", "reconciled by user: failed", "error")]
    msgs2 = to_messages(items2)
    assert msgs2[1].content == "reconciled by user: failed"


def test_r10_single_result_behavior_unchanged() -> None:
    items = [assistant(1, ["c1"]), _result_status(2, "c1", "the only one", "ok"),
             mk(3, ItemKind.USER_MESSAGE, {"text": "next"})]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["assistant", "tool", "user"]
    assert msgs[1].content == "the only one"


def test_r10_pairing_invariant_holds_with_duplicate_results() -> None:
    # 每 call_id 恰好一条 tool 消息(c1 双结果取最后,c2 走 R9 合成),且顺序不变
    items = [
        assistant(1, ["c1", "c2"]),
        _result_status(2, "c1", "old", "outcome_unknown"),
        _result_status(3, "c1", "new", "ok"),
    ]
    msgs = to_messages(items)
    tool_msgs = [m for m in msgs if m.role == "tool"]
    # B31 迁移:按声明顺序(c1 真实结果在前,c2 合成在后);仍是每 id 恰一条。
    assert [m.tool_call_id for m in tool_msgs] == ["c1", "c2"]
    assert tool_msgs[0].content == "new"
    assert tool_msgs[1] == ModelMessage(role="tool", content=_UNPAIRED_NOTE, tool_call_id="c2")
    assert "old" not in "".join(m.content for m in msgs)


def test_r10_last_by_seq_wins_even_if_list_order_differs() -> None:
    # 判据是 seq(最新 seq 胜),不是列表位置——乱序输入仍投影 seq 最大的一条
    items = [assistant(1, ["c1"]), _result_status(4, "c1", "later seq wins", "ok"),
             _result_status(2, "c1", "earlier seq loses", "ok")]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["assistant", "tool"]
    assert msgs[1].content == "later seq wins"


# ---------------------------------------------------------------------------
# 跨 turn 复用 call_id 的回归(整分支评审 Important #1):R9/R10 的存在性与
# 覆盖键必须按 (turn_id, call_id) 分键——call_id 只在单个 turn 内是调用身份,
# local provider 每轮从 call_0 重新起号、CLI 在 default 线程上多轮累积都会复用。
# 全局键会把前一 turn 的真实结果判为"非最新"丢弃、同时不再为它合成 → 孤儿
# assistant tool_call → 真实 provider 直接拒收请求。
# ---------------------------------------------------------------------------


def test_cross_turn_repeated_call_id_keeps_both_real_results() -> None:
    # 两个 turn 都用 c1、各有真实结果:两个结果都必须投影——前一 turn 的结果
    # 不得被跨 turn 的全局 R10 "最新 seq 胜" 判为旧结果而丢弃。
    items = [
        assistant(1, ["c1"], "turn-A"),
        result(2, "c1", "from-turn-A", "turn-A"),
        mk(3, ItemKind.USER_MESSAGE, {"text": "next"}, "turn-B"),
        assistant(4, ["c1"], "turn-B"),
        result(5, "c1", "from-turn-B", "turn-B"),
    ]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["assistant", "tool", "user", "assistant", "tool"]
    assert msgs[1].content == "from-turn-A" and msgs[1].tool_call_id == "c1"
    assert msgs[4].content == "from-turn-B" and msgs[4].tool_call_id == "c1"
    assert all(_UNPAIRED_NOTE not in m.content for m in msgs)  # 两边都是真实结果,无合成


def test_cross_turn_reuse_missing_result_synthesizes_only_that_turn() -> None:
    # turn-A 的 c1 有真实结果、turn-B 的 c1 无结果:B 必须合成配对结果——
    # A 的存在不得让 B 的 R9 合成被抑制(B 的 assistant 成为孤儿)。
    items = [
        assistant(1, ["c1"], "turn-A"),
        result(2, "c1", "real-A", "turn-A"),
        assistant(3, ["c1"], "turn-B"),
    ]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["assistant", "tool", "assistant", "tool"]
    assert msgs[1].content == "real-A" and msgs[1].tool_call_id == "c1"
    assert msgs[3] == ModelMessage(role="tool", content=_UNPAIRED_NOTE, tool_call_id="c1")


def test_cross_turn_reuse_real_result_not_masked_by_later_turn_gap() -> None:
    # 反向:turn-B 有结果不得掩盖 turn-A 缺结果——A 合成,B 投影真实结果。
    items = [
        assistant(1, ["c1"], "turn-A"),
        assistant(2, ["c1"], "turn-B"),
        result(3, "c1", "real-B", "turn-B"),
    ]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["assistant", "tool", "assistant", "tool"]
    assert msgs[1].content == _UNPAIRED_NOTE and msgs[1].tool_call_id == "c1"
    assert msgs[3].content == "real-B"


def test_within_turn_r10_last_wins_holds_with_real_turn_ids() -> None:
    # turn_id 分键后,单 turn 内 reconcile 覆盖语义不变:outcome_unknown 之后
    # 的对账结果仍"最后一条胜",旧结果只留审计不进投影。
    items = [
        assistant(1, ["c1"], "turn-A"),
        _result_status(
            2, "c1", "[outcome unknown: process died during execution]", "outcome_unknown", "turn-A"
        ),
        _result_status(3, "c1", "reconciled by user: ok", "ok", "turn-A"),
    ]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["assistant", "tool"]
    assert msgs[1].content == "reconciled by user: ok"


# ---------------------------------------------------------------------------
# R-T7-1:classify_pending 的声明窗口配对(恢复路径不得静默丢调用)
# ---------------------------------------------------------------------------


def _tool(name: str, *, idempotent: bool) -> Tool:
    return Tool(
        definition=ToolDefinition(name=name, description=f"{name} tool", input_schema={"type": "object"}),
        validate_input=lambda arguments: arguments,
        run=lambda arguments, exec_ctx: ToolResult(tool_call_id="c", tool_name=name),
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda arguments: ResolvedUse(effects=frozenset({ToolEffect.READ_WORKSPACE}), targets=()),
        idempotent=idempotent,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30.0,
        max_model_output_bytes=4096,
    )


_TOOLS: Mapping[str, Tool] = {
    "read": _tool("read", idempotent=True),
    "bash": _tool("bash", idempotent=False),
}


def _call(call_id: str, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"call_id": call_id, "name": name, "arguments": dict(arguments or {})}


def _two_round_items(*, second_started: bool, tool_name: str = "read") -> list[Item]:
    """同 turn 两轮复用 call_0:第一轮已解析,第二轮(status 见参数)是崩溃现场。"""
    items = [
        mk(1, ItemKind.USER_MESSAGE, {"text": "go"}),
        mk(2, ItemKind.MODEL_RESPONSE, {"content": "", "tool_calls": [_call("call_0", tool_name, {"path": "a"})]}),
        result(3, "call_0", "FIRST-ROUND-RESULT"),
        mk(4, ItemKind.MODEL_RESPONSE, {"content": "", "tool_calls": [_call("call_0", tool_name, {"path": "b"})]}),
    ]
    if second_started:
        items.append(
            mk(5, ItemKind.TOOL_CALL, {"call_id": "call_0", "tool_name": tool_name, "arguments": {"path": "b"}})
        )
    return items


def test_classify_pending_second_round_started_not_swallowed_by_first_result() -> None:
    """第二轮 tool_call 已落库、结果未写(kill -9 落在执行中)。

    第一轮的 call_0 结果不得让第二轮的同 id 调用被判"已解决"——它必须出现在
    待处理清单里(started=True),否则 resume 不写 not_executed、也不重放,
    请求里就出现孤儿 tool_call,且这次调用从账面上消失。
    """
    pending = classify_pending(_two_round_items(second_started=True, tool_name="bash"), _TOOLS)
    assert pending == [
        PendingCall(
            call=ModelToolCall(call_id="call_0", name="bash", arguments={"path": "b"}),
            started=True,
            idempotent=False,
        )
    ]


def test_classify_pending_second_round_not_started_not_swallowed_by_first_result() -> None:
    """第二轮 model_response 已落、dispatch 之前崩溃:同样不得被第一轮结果吞掉。"""
    pending = classify_pending(_two_round_items(second_started=False, tool_name="read"), _TOOLS)
    assert pending == [
        PendingCall(
            call=ModelToolCall(call_id="call_0", name="read", arguments={"path": "b"}),
            started=False,
            idempotent=True,
        )
    ]


def test_classify_pending_second_round_own_result_resolves_it() -> None:
    """第二轮自己的结果在场 → 非 pending;第一轮结果不是判据。"""
    items = _two_round_items(second_started=True)
    items.append(result(6, "call_0", "SECOND-ROUND-RESULT"))
    # seq 顺序修正:结果必须晚于第二轮声明(seq 4);此处 6 > 4
    assert classify_pending(items, _TOOLS) == []


# ---------------------------------------------------------------------------
# R-T7-1 端到端:同形状走真实 resume(confirm=True),provider 请求不得有孤儿
# ---------------------------------------------------------------------------


def _response(content: str, calls: Sequence[tuple[str, str, dict[str, Any]]] = ()) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tuple(ModelToolCall(call_id=c, name=n, arguments=a) for c, n, a in calls),
        usage=Usage(input_tokens=1, output_tokens=1),
        finish_reason="tool_calls" if calls else "stop",
    )


def _assistant_entry(call_id: str, name: str, arguments: Mapping[str, Any]) -> ItemEntry:
    return ItemEntry(
        kind=ItemKind.MODEL_RESPONSE,
        payload={
            "content": "",
            "tool_calls": ({"call_id": call_id, "name": name, "arguments": dict(arguments)},),
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "finish_reason": "tool_calls",
        },
    )


def _result_entry(call_id: str, name: str, text: str) -> ItemEntry:
    return ItemEntry(
        kind=ItemKind.TOOL_RESULT,
        payload={
            "call_id": call_id,
            "tool_name": name,
            "status": "ok",
            "content": ({"type": "text", "data": {"text": text}},),
            "is_error": False,
            "error_code": None,
            "error_message": None,
            "truncated": False,
            "metadata": {},
        },
    )


def _tool_call_entry(call_id: str, name: str, arguments: Mapping[str, Any]) -> ItemEntry:
    return ItemEntry(
        kind=ItemKind.TOOL_CALL,
        payload={
            "call_id": call_id,
            "tool_name": name,
            "arguments": dict(arguments),
            "effects": (),
            "targets": (),
            "args_digest": "0" * 64,
            "idempotent": name != "bash",
        },
    )


def _seed_dead_turn(thread_id: str, workspace: Path, entries: Sequence[ItemEntry]) -> str:
    """另一"已死进程"视角:running turn + items;open_session 的 sweep 负责收编。"""
    praxis_dir = workspace / ".praxis"
    praxis_dir.mkdir(parents=True, exist_ok=True)
    store = Store(praxis_dir / "praxis.sqlite")
    try:
        turn_id = uuid4().hex
        create_turn(store, turn_id=turn_id, thread_id=thread_id, binding={})
        store.append_items(thread_id, turn_id, list(entries))
        return turn_id
    finally:
        store.close()


def _assert_every_declaration_paired(messages: Sequence[Any]) -> None:
    for index, message in enumerate(messages):
        if message.role != "assistant" or not message.tool_calls:
            continue
        next_assistant = next(
            (j for j in range(index + 1, len(messages)) if messages[j].role == "assistant"),
            len(messages),
        )
        for call in message.tool_calls:
            following = [
                m
                for m in messages[index + 1 : next_assistant]
                if m.role == "tool" and m.tool_call_id == call.call_id
            ]
            assert len(following) == 1, (call.call_id, message, following)


def test_resume_reused_call_id_provider_request_has_no_orphan(
    workspace: Path,
) -> None:
    """同 turn 两轮 call_0,第二轮 read 已 started 未结果 → 续跑重放后请求全配对。

    第一轮真实结果 + 第二轮重放结果都必须投影;每个声明恰有一条 tool 消息,
    既无孤儿也无重复(旧实现会把第一轮结果判过期、且不为第二轮合成)。
    """
    thread_id = "pairing-resume"
    (workspace / "a.txt").write_text("A-CONTENT", encoding="utf-8")
    (workspace / "b.txt").write_text("B-CONTENT", encoding="utf-8")
    turn_id = _seed_dead_turn(
        thread_id,
        workspace,
        [
            _assistant_entry("call_0", "read", {"path": "a.txt"}),
            _result_entry("call_0", "read", "A-CONTENT"),
            _assistant_entry("call_0", "read", {"path": "b.txt"}),
            _tool_call_entry("call_0", "read", {"path": "b.txt"}),
        ],
    )

    adapter = ScriptedAdapter([_response("done")])
    session = open_session(workspace, adapter, thread_id=thread_id)
    try:
        result = asyncio.run(session.resume(turn_id, confirm=True))
        assert result.status.value == "completed" and result.output == "done"
        request = adapter.last_request
        assert request is not None
        _assert_every_declaration_paired(request.messages)
        tool_messages = [m for m in request.messages if m.role == "tool"]
        assert [m.content for m in tool_messages] == ["A-CONTENT", "B-CONTENT"]
        assert all(_UNPAIRED_NOTE not in m.content for m in request.messages)
        assert session.status(turn_id).status == "completed"
    finally:
        session.close()
