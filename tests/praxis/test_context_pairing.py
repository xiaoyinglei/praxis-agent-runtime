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

from hypothesis import HealthCheck, given, settings
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
def declaration_streams(draw: st.DrawFn) -> list[Item]:
    """随机声明窗口 + 结果(0-2 条/声明)+ 孤儿结果 + 畸形/未知 kind,返回乱序列表。"""
    items: list[Item] = []
    seq = 0
    declared: set[str] = set()
    pool = ["c0", "c1", "c2", "c3"]
    junk_payloads = st.one_of(
        st.just({}),
        st.just({"content": None}),
        st.just({"content": "", "tool_calls": "nope"}),
        st.just({"tool_calls": [{"call_id": 7, "name": "t"}]}),
        st.dictionaries(st.text(max_size=4), st.integers(-3, 3), max_size=3),
    )

    for _ in range(draw(st.integers(0, 5))):
        for _ in range(draw(st.integers(0, 2))):
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
            for _ in range(draw(st.integers(1, 2))):
                seq += 1
                items.append(result(seq, draw(st.sampled_from(unseen)), f"orphan-{seq}"))

        declarations = draw(st.lists(st.sampled_from(pool), min_size=1, max_size=4))
        seq += 1
        if draw(st.booleans()):
            items.append(mk(seq, ItemKind.MODEL_RESPONSE, draw(junk_payloads)))
        else:
            items.append(assistant(seq, declarations))
            declared.update(declarations)

        for call_id in declarations:
            for _ in range(draw(st.integers(0, 2))):
                if draw(st.booleans()):
                    seq += 1
                    items.append(result(seq, call_id, draw(st.sampled_from(["a", "b", "c"]))))

    shuffled = draw(st.permutations(items))
    return list(shuffled)


@given(declaration_streams())
@settings(deadline=None, max_examples=400, suppress_health_check=[HealthCheck.too_slow])
def test_pairing_invariant_holds_for_arbitrary_streams(items: list[Item]) -> None:
    msgs = to_messages(items)  # 永不抛
    assert all(isinstance(m, ModelMessage) for m in msgs)
    for index, message in enumerate(msgs):
        if message.role != "assistant" or not message.tool_calls:
            continue
        next_assistant = next(
            (j for j in range(index + 1, len(msgs)) if msgs[j].role == "assistant"),
            len(msgs),
        )
        group = [m for m in msgs[index + 1 : next_assistant] if m.role == "tool"]
        declared = [call.call_id for call in message.tool_calls]
        # 每条声明(按出现次数计)在其 assistant 之后恰有一条 tool 消息(真实或合成)
        for call_id in set(declared):
            following = [m for m in group if m.tool_call_id == call_id]
            assert len(following) == declared.count(call_id), (call_id, declared, group)
        # 组内配对 tool 消息保持声明顺序(provider 组内顺序契约)
        assert [m.tool_call_id for m in group if m.tool_call_id in set(declared)] == declared


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
@settings(deadline=None, max_examples=300, suppress_health_check=[HealthCheck.too_slow])
def test_to_messages_never_raises_on_arbitrary_junk(items: list[Item]) -> None:
    assert all(isinstance(m, ModelMessage) for m in to_messages(items))


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