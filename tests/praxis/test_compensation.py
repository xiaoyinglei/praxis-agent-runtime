"""Task 4:唯一清算协议——unclosed_windows 判据层 + close_unresolved 包装层(R-PF-2)。

B29 回归钉(旧 agent_runtime/harness/ownership.py:181-190 病灶):同窗口第 1 个
缺结果的调用触发早退,整组其余调用永不闭合。这里断言**全组**清算——判据基于
``pair_tool_results`` 的声明窗口,覆盖**所有**未闭合窗口(含跨窗口孤儿声明),
不限最后一条 model_response(与 recovery.classify_pending 的"最后窗口"职责分清)。

幂等判据 = pair 已配:补过结果的窗口第二次清算零产出(brief ⑥)。
interrupted 文案逐字保留旧 session 常量(提取零行为变化的文案面钉)。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from praxis.core.compensation import (
    NOT_EXECUTED_TEXTS,
    OUTCOME_UNKNOWN_TEXTS,
    UnresolvedCall,
    close_unresolved,
    unclosed_windows,
)
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.models import ModelToolCall
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolDefinition,
    ToolEffect,
    ToolResult,
)
from praxis.store.log import ItemEntry, Store

# 旧 session._NOT_EXECUTED_TEXT / _OUTCOME_UNKNOWN_TEXT 逐字(提取零行为变化)。
_LEGACY_NOT_EXECUTED = "[not executed: turn interrupted before dispatch]"
_LEGACY_OUTCOME_UNKNOWN = "[outcome unknown: process died during execution]"


def _tool(name: str, *, idempotent: bool) -> Tool:
    return Tool(
        definition=ToolDefinition(name=name, description=f"{name} toy", input_schema={"type": "object"}),
        validate_input=lambda arguments: arguments,
        run=lambda arguments, exec_ctx: ToolResult(tool_call_id="pending", tool_name=name),
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
    "read_file": _tool("read_file", idempotent=True),
    "write_file": _tool("write_file", idempotent=False),
}


def _item(seq: int, kind: str, payload: Mapping[str, object], *, turn_id: str = "turn-1") -> Item:
    return Item(
        item_id=f"item-{seq}",
        thread_id="t",
        turn_id=turn_id,
        seq=seq,
        kind=kind,
        payload=payload,
        created_at=float(seq),
    )


def _call(call_id: str, name: str, arguments: Mapping[str, object] | None = None) -> dict[str, object]:
    return {"call_id": call_id, "name": name, "arguments": arguments or {}}


def _response(seq: int, calls: object) -> Item:
    return _item(seq, ItemKind.MODEL_RESPONSE.value, {"content": "", "tool_calls": calls})


def _tool_call(seq: int, call_id: str, name: str) -> Item:
    return _item(seq, ItemKind.TOOL_CALL.value, {"call_id": call_id, "tool_name": name, "arguments": {}})


def _tool_result(seq: int, call_id: str, name: str, status: str = "ok") -> Item:
    return _item(seq, ItemKind.TOOL_RESULT.value, {"call_id": call_id, "tool_name": name, "status": status})


# ---------------------------------------------------------------------------
# B29 全组清算:第 1 个缺结果不得早退(ownership.py:181-190 病灶回归)
# ---------------------------------------------------------------------------


def test_b29_first_missing_result_does_not_abort_group() -> None:
    """c1 缺结果(started)、c2 已有结果、c3 缺结果(未 started)——旧实现处理 c1
    后早退,c3 永不闭合;新判据整组枚举:c1/c3 都在,各判正确。"""
    items = [
        _response(1, [_call("c1", "write_file"), _call("c2", "read_file"), _call("c3", "read_file")]),
        _tool_call(2, "c1", "write_file"),
        _tool_result(3, "c2", "read_file"),
    ]
    unresolved = unclosed_windows(items, _TOOLS)
    assert [(u.call.call_id, u.started, u.idempotent) for u in unresolved] == [
        ("c1", True, False),
        ("c3", False, True),
    ]
    entries, unknown = close_unresolved(items, _TOOLS, text_for="cancelled")
    assert [e.payload["call_id"] for e in entries] == ["c1", "c3"]
    assert [e.payload["status"] for e in entries] == ["outcome_unknown", "not_executed"]
    assert unknown == ["c1"]


def test_b29_brief_case_first_has_result_rest_missing() -> None:
    """brief ①字面:同窗口 3 调用、第 1 个已有结果第 2/3 无 → 补齐 2 条各判正确。"""
    items = [
        _response(1, [_call("c1", "read_file"), _call("c2", "write_file"), _call("c3", "read_file")]),
        _tool_result(2, "c1", "read_file"),
        _tool_call(3, "c2", "write_file"),
    ]
    entries, unknown = close_unresolved(items, _TOOLS, text_for="cancelled")
    assert len(entries) == 2
    assert entries[0].payload["call_id"] == "c2"
    assert entries[0].payload["status"] == "outcome_unknown"  # started → 未知副作用
    assert entries[1].payload["call_id"] == "c3"
    assert entries[1].payload["status"] == "not_executed"  # 未 dispatch
    assert unknown == ["c2"]


def test_cross_window_orphan_declarations_also_settled() -> None:
    """更早轮次的未闭合声明(跨窗口孤儿)同样补齐——不限最后一条 model_response。"""
    items = [
        _response(1, [_call("c1", "read_file")]),
        _response(3, [_call("c2", "write_file")]),
        _tool_call(4, "c2", "write_file"),
    ]
    unresolved = unclosed_windows(items, _TOOLS)
    assert [(u.call.call_id, u.started, u.declaration_seq) for u in unresolved] == [
        ("c1", False, 1),
        ("c2", True, 3),
    ]
    entries, unknown = close_unresolved(items, _TOOLS, text_for="interrupted")
    assert [e.payload["status"] for e in entries] == ["not_executed", "outcome_unknown"]
    assert unknown == ["c2"]


# ---------------------------------------------------------------------------
# 声明窗口判据(与 pair_tool_results 同口径)
# ---------------------------------------------------------------------------


def test_reused_call_id_earlier_result_does_not_close_later_window() -> None:
    items = [
        _response(1, [_call("c1", "read_file")]),
        _tool_result(2, "c1", "read_file"),
        _response(3, [_call("c1", "read_file")]),
    ]
    unresolved = unclosed_windows(items, _TOOLS)
    assert [(u.call.call_id, u.declaration_seq, u.started) for u in unresolved] == [("c1", 3, False)]


def test_started_attribution_is_per_window() -> None:
    """更早窗口同 id 的 tool_call 不是本窗口的 dispatch(与 classify_pending 同纪律)。"""
    items = [
        _response(1, [_call("c1", "read_file")]),
        _tool_call(2, "c1", "read_file"),
        _tool_result(3, "c1", "read_file"),
        _response(4, [_call("c1", "read_file")]),
    ]
    unresolved = unclosed_windows(items, _TOOLS)
    assert [(u.declaration_seq, u.started) for u in unresolved] == [(4, False)]


def test_duplicate_declaration_in_one_response_yields_one_entry() -> None:
    """同一响应重复声明同 id 折叠进同一窗口键——清算一条(投影侧为第二次声明补合成)。"""
    items = [_response(1, [_call("c1", "read_file"), _call("c1", "read_file")])]
    entries, unknown = close_unresolved(items, _TOOLS, text_for="cancelled")
    assert len(entries) == 1 and unknown == []


def test_unknown_tool_is_non_idempotent() -> None:
    items = [_response(1, [_call("c1", "no_such_tool")])]
    unresolved = unclosed_windows(items, _TOOLS)
    assert unresolved[0].idempotent is False


def test_no_model_response_returns_empty() -> None:
    assert unclosed_windows([_item(1, ItemKind.USER_MESSAGE.value, {"text": "hi"})], _TOOLS) == []
    assert close_unresolved([], _TOOLS, text_for="cancelled") == ([], [])


def test_malformed_response_opens_no_window_fail_open() -> None:
    """部分畸形的响应不开窗(与 pair_tool_results 同严格度):不抛、不补,
    其结果保持孤儿身份由读侧兜底投影。"""
    items = [
        _response(1, [_call("c1", "read_file"), {"call_id": 7, "name": "read_file"}]),
        _tool_result(2, "c1", "read_file"),
    ]
    assert unclosed_windows(items, _TOOLS) == []
    assert close_unresolved(items, _TOOLS, text_for="cancelled") == ([], [])


def _response_entry(calls: tuple[dict[str, object], ...]) -> ItemEntry:
    return ItemEntry(kind=ItemKind.MODEL_RESPONSE, payload={"content": "", "tool_calls": calls})


def _tool_call_entry(call_id: str, name: str) -> ItemEntry:
    return ItemEntry(kind=ItemKind.TOOL_CALL, payload={"call_id": call_id, "tool_name": name, "arguments": {}})


def test_frozen_store_shapes_round_trip(store: Store) -> None:
    """Store 读出的冻结 payload(tuple/mappingproxy)与内存形状同判。"""
    store.ensure_thread("t")
    store.append_items(
        "t", "turn-1", [_response_entry((_call("c1", "write_file"),)), _tool_call_entry("c1", "write_file")]
    )
    items = store.read_turn_items("turn-1")
    assert isinstance(items[0].payload["tool_calls"], tuple)
    unresolved = unclosed_windows(items, _TOOLS)
    assert [(u.call.call_id, u.started) for u in unresolved] == [("c1", True)]
    assert unresolved[0].call == ModelToolCall(call_id="c1", name="write_file", arguments={})


def test_mappingproxy_payload_accepted() -> None:
    frozen = MappingProxyType({"call_id": "c1", "name": "read_file", "arguments": MappingProxyType({"path": "x"})})
    items = [_response(1, (frozen,))]
    unresolved = unclosed_windows(items, _TOOLS)
    assert unresolved[0].call == ModelToolCall(call_id="c1", name="read_file", arguments={"path": "x"})


# ---------------------------------------------------------------------------
# 幂等(brief ⑥):pair 已配即不再补
# ---------------------------------------------------------------------------


def test_second_settlement_after_entries_appended_is_empty(store: Store) -> None:
    store.ensure_thread("t")
    store.append_items(
        "t",
        "turn-1",
        [
            _response_entry((_call("c1", "write_file"), _call("c2", "read_file"))),
            _tool_call_entry("c1", "write_file"),  # c1 已 dispatch;c2 未 dispatch
        ],
    )
    items = store.read_turn_items("turn-1")
    entries, unknown = close_unresolved(items, _TOOLS, text_for="cancelled")
    assert len(entries) == 2 and unknown == ["c1"]
    store.append_items("t", "turn-1", entries)
    again = store.read_turn_items("turn-1")
    assert unclosed_windows(again, _TOOLS) == []
    assert close_unresolved(again, _TOOLS, text_for="cancelled") == ([], [])


# ---------------------------------------------------------------------------
# 文案与 entry 形状
# ---------------------------------------------------------------------------


def test_interrupted_texts_are_legacy_verbatim() -> None:
    """提取零行为变化:interrupted 文案与旧 session 常量逐字一致。"""
    assert NOT_EXECUTED_TEXTS["interrupted"] == _LEGACY_NOT_EXECUTED
    assert OUTCOME_UNKNOWN_TEXTS["interrupted"] == _LEGACY_OUTCOME_UNKNOWN
    items = [
        _response(1, [_call("c1", "write_file"), _call("c2", "read_file")]),
        _tool_call(2, "c1", "write_file"),
    ]
    entries, _ = close_unresolved(items, _TOOLS, text_for="interrupted")
    texts = [e.payload["content"][0]["data"]["text"] for e in entries]  # type: ignore[index]
    assert texts == [_LEGACY_OUTCOME_UNKNOWN, _LEGACY_NOT_EXECUTED]


def test_cancelled_texts_are_distinct() -> None:
    assert "cancelled" in NOT_EXECUTED_TEXTS["cancelled"]
    assert "cancelled" in OUTCOME_UNKNOWN_TEXTS["cancelled"]
    assert NOT_EXECUTED_TEXTS["cancelled"] != _LEGACY_NOT_EXECUTED


def test_entry_shape_matches_pipeline_convention() -> None:
    items = [_response(1, [_call("c1", "read_file")])]
    entries, _ = close_unresolved(items, _TOOLS, text_for="cancelled")
    payload = entries[0].payload
    assert entries[0].kind == ItemKind.TOOL_RESULT
    assert payload["call_id"] == "c1" and payload["tool_name"] == "read_file"
    assert payload["status"] == ResultStatus.NOT_EXECUTED.value
    assert payload["is_error"] is False
    assert payload["error_code"] is None and payload["error_message"] is None
    assert payload["truncated"] is False
    assert payload["metadata"] == {}
    assert payload["content"] == ({"type": "text", "data": {"text": NOT_EXECUTED_TEXTS["cancelled"]}},)


def test_unresolved_call_is_frozen() -> None:
    uc = UnresolvedCall(
        call=ModelToolCall(call_id="c", name="n", arguments={}), started=False, idempotent=False, declaration_seq=1
    )
    with pytest.raises(FrozenInstanceError):
        uc.started = True  # type: ignore[misc]