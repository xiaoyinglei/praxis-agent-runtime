"""core.context:boundary 折叠(§5.1/§5.7 · B22 回归)与 to_messages 投影。

钉死语义:压缩不复制正文——读端按 compaction_boundary 折叠,原 Item 留在 store;
未知 algorithm_version / 畸形 boundary → fail-open 不折叠;to_messages 永不抛。
"""

from __future__ import annotations

from typing import Any

from hypothesis import HealthCheck, given, settings, strategies as st

from praxis.core.context import build_context, to_messages
from praxis.protocol.items import Item
from praxis.protocol.models import ModelMessage, ModelToolCall
from praxis.protocol.tools import ToolContentBlock, ToolResult


_UNPAIRED_NOTE = "[tool result unavailable: turn ended before execution]"


def mk(seq: int, kind: str, payload: dict[str, Any]) -> Item:
    return Item(
        item_id=f"i{seq}",
        thread_id="t",
        turn_id=None,
        seq=seq,
        kind=kind,
        payload=payload,
        created_at=float(seq),
    )


# ---------------------------------------------------------------------------
# Step 1(brief verbatim):B22 回归钉死
# ---------------------------------------------------------------------------


def test_boundary_folds_without_copying() -> None:
    items = [mk(i, "user_message", {"text": f"msg{i}"}) for i in range(1, 6)]
    summary = mk(6, "summary", {"text": "短摘要", "algorithm_version": "v1"})
    boundary = mk(
        7,
        "compaction_boundary",
        {"covered_from": 1, "covered_to": 5, "algorithm_version": "v1"},
    )
    out = build_context(items + [summary, boundary])
    assert [i.seq for i in out] == [6, 7]
    # B22:新 item 不携带被覆盖正文——摘要长度远小于原文,且原 item 对象未被修改
    assert all("msg" not in str(i.payload) for i in out)
    assert items[0].payload == {"text": "msg1"}


def test_unknown_boundary_version_fail_open() -> None:
    boundary = mk(
        2,
        "compaction_boundary",
        {"covered_from": 1, "covered_to": 1, "algorithm_version": "v999-from-future"},
    )
    out = build_context([mk(1, "user_message", {"text": "keep"}), boundary])
    assert [i.seq for i in out] == [1, 2]  # 未知版本不折叠


# ---- brief 第 3、4 组(hypothesis 展开 + to_messages roles)----------------


@st.composite
def item_lists(draw: st.SearchStrategy) -> list[Item]:  # type: ignore[misc]
    kinds_st = st.sampled_from(
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
    junk_values = st.recursive(
        st.none() | st.booleans() | st.integers(-5, 5) | st.text(max_size=5),
        lambda children: st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=4), children, max_size=3),
        max_leaves=6,
    )
    junk_payload = st.dictionaries(st.text(max_size=8), junk_values, max_size=4)
    boundary_payload = st.fixed_dictionaries(
        {},
        optional={
            "covered_from": st.one_of(st.integers(-1, 11), st.none(), st.booleans(), st.text(max_size=3)),
            "covered_to": st.one_of(st.integers(-1, 11), st.none(), st.booleans(), st.text(max_size=3)),
            "algorithm_version": st.one_of(st.just("v1"), st.text(max_size=8), st.none(), st.integers()),
        },
    )
    n = draw(st.integers(0, 12))
    kinds = draw(st.lists(kinds_st, min_size=n, max_size=n))
    items: list[Item] = []
    for seq, kind in enumerate(kinds, start=1):
        payload = draw(boundary_payload if kind == "compaction_boundary" else junk_payload)
        items.append(mk(seq, kind, payload))
    return items


@given(item_lists())
@settings(deadline=None, max_examples=300, suppress_health_check=[HealthCheck.too_slow])
def test_build_context_idempotent_and_never_raises_on_unknown_kind(items: list[Item]) -> None:
    before = [(i.item_id, i.seq, i.kind, dict(i.payload)) for i in items]
    once = build_context(items)
    twice = build_context(once)
    assert twice == once  # 幂等
    assert [(i.item_id, i.seq, i.kind, dict(i.payload)) for i in items] == before  # 输入不被修改
    for out_item in once:  # 输出 item 与输入同一对象——纯函数,不复制正文
        assert any(out_item is i for i in items)
    msgs = to_messages(once)
    assert all(isinstance(m, ModelMessage) for m in msgs)  # 随机 kind/payload 永不抛
    assert all(m.role in {"system", "user", "assistant", "tool"} for m in msgs)


def test_to_messages_roles() -> None:
    items = [
        mk(1, "user_message", {"text": "hi"}),
        mk(2, "steering", {"text": "wait"}),
        mk(
            3,
            "model_response",
            {"content": "thinking", "tool_calls": [{"call_id": "c1", "name": "search", "arguments": {"q": "x"}}]},
        ),
        mk(
            4,
            "tool_result",
            {
                "call_id": "c1",
                "content": [
                    {"type": "text", "data": {"text": "result-A"}},
                    {"type": "text", "data": {"text": "result-B"}},
                ],
            },
        ),
        mk(5, "turn_status", {"status": "running"}),
        mk(6, "approval_request", {"tool_call_id": "c2"}),
        mk(7, "approval_decision", {"decision": "deny"}),
        mk(8, "tool_call", {"call_id": "c2", "tool_name": "x", "arguments": {}}),
        mk(9, "mystery-kind", {"whatever": True}),
    ]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["user", "user", "assistant", "tool"]
    assert msgs[0].content == "hi"
    assert msgs[1].content == "wait"
    assert msgs[2].content == "thinking"
    assert msgs[2].tool_calls == (ModelToolCall(call_id="c1", name="search", arguments={"q": "x"}),)
    assert msgs[3].content == "result-A\nresult-B"
    assert msgs[3].tool_call_id == "c1"


# ---------------------------------------------------------------------------
# 补充:多 boundary 并集 / 畸形 payload fail-open / summary 前缀 / ToolResult.text 对齐
# ---------------------------------------------------------------------------


def test_multiple_boundaries_union() -> None:
    items = [mk(s, "user_message", {"text": f"m{s}"}) for s in range(1, 11)]
    b1 = mk(11, "compaction_boundary", {"covered_from": 2, "covered_to": 4, "algorithm_version": "v1"})
    b2 = mk(12, "compaction_boundary", {"covered_from": 7, "covered_to": 8, "algorithm_version": "v1"})
    out = build_context(items + [b1, b2])
    assert [i.seq for i in out] == [1, 5, 6, 9, 10, 11, 12]
    assert build_context(out) == out  # 幂等:折叠后的重放不变


def test_boundary_keeps_summary_even_inside_covered_range() -> None:
    # summary/boundary 自身永远保留,即使 seq 落入覆盖区间
    summary = mk(3, "summary", {"text": "s"})
    boundary = mk(4, "compaction_boundary", {"covered_from": 1, "covered_to": 3, "algorithm_version": "v1"})
    out = build_context([mk(1, "user_message", {"text": "a"}), summary, mk(5, "steering", {"text": "b"}), boundary])
    assert [i.seq for i in out] == [3, 5, 4]  # 保留输入顺序;seq 1 被折叠


def test_malformed_boundary_payload_folds_nothing_never_raises() -> None:
    junk_boundaries = [
        mk(10, "compaction_boundary", {}),
        mk(11, "compaction_boundary", {"covered_from": 1, "covered_to": 2}),  # 缺 version
        mk(12, "compaction_boundary", {"covered_from": "1", "covered_to": 2, "algorithm_version": "v1"}),
        mk(13, "compaction_boundary", {"covered_from": 1, "covered_to": None, "algorithm_version": "v1"}),
        mk(14, "compaction_boundary", {"covered_from": True, "covered_to": True, "algorithm_version": "v1"}),
        mk(15, "compaction_boundary", {"covered_from": [1], "covered_to": {"a": 1}, "algorithm_version": "v1"}),
        mk(16, "compaction_boundary", {"algorithm_version": 1}),
    ]
    payload = [mk(1, "user_message", {"text": "a"}), mk(2, "turn_status", {"status": "running"})]
    out = build_context(payload + junk_boundaries)
    assert [i.seq for i in out] == [1, 2, 10, 11, 12, 13, 14, 15, 16]  # 全部 fail-open 保留


def test_summary_message_has_fixed_prefix() -> None:
    msg = to_messages([mk(1, "summary", {"text": "短摘要"})])[0]
    assert msg == ModelMessage(role="user", content="[conversation summary]\n短摘要")


def test_tool_result_text_matches_toolresult_convention() -> None:
    # 镜像 protocol.tools.ToolResult.text:仅 type=="text" 块、data["text"] 为 str 才计入,"\n".join
    blocks = (
        ToolContentBlock(type="text", data={"text": "a"}),
        ToolContentBlock(type="image", data={"url": "u"}),
        ToolContentBlock(type="text", data={"text": 3}),  # 非字符串跳过
        ToolContentBlock(type="text", data={"nope": 1}),  # 无 text 键跳过
        ToolContentBlock(type="text", data={"text": "b"}),
    )
    expected = ToolResult(tool_call_id="c1", tool_name="t", content=blocks).text
    payload = {
        "call_id": "c1",
        "content": [
            {"type": "text", "data": {"text": "a"}},
            {"type": "image", "data": {"url": "u"}},
            {"type": "text", "data": {"text": 3}},
            {"type": "text", "data": {"nope": 1}},
            {"type": "text", "data": {"text": "b"}},
        ],
    }
    msg = to_messages([mk(2, "tool_result", payload)])[0]
    assert msg.content == expected == "a\nb"
    assert msg.tool_call_id == "c1"


def test_tool_result_without_content_blocks_yields_empty_text() -> None:
    msgs = to_messages([mk(1, "tool_result", {"call_id": "c1"})])
    assert msgs == [ModelMessage(role="tool", content="", tool_call_id="c1")]


def test_to_messages_skips_malformed_items_never_raises() -> None:
    bad = [
        mk(1, "user_message", {}),  # 缺 text
        mk(2, "user_message", {"text": 42}),  # text 非 str
        mk(3, "steering", {"text": None}),
        mk(4, "model_response", {"content": None}),  # content 非 str
        mk(5, "model_response", {"content": "x", "tool_calls": "nope"}),  # tool_calls 非 list
        mk(6, "model_response", {"content": "x", "tool_calls": [{"call_id": 1, "name": "n"}]}),  # call_id 非 str
        mk(7, "model_response", {"content": "x", "tool_calls": [{"call_id": "c", "name": "n", "arguments": []}]}),
        mk(8, "tool_result", {"content": []}),  # 缺 call_id
        mk(9, "tool_result", {"call_id": "c", "content": "nope"}),  # content 非 list
        mk(10, "tool_result", {"call_id": "c", "content": ["not-a-block-mapping"]}),
        mk(11, "summary", {"text": None}),
    ]
    assert to_messages(bad) == []


def test_model_response_tool_calls_defaults() -> None:
    msgs = to_messages(
        [
            mk(1, "model_response", {"content": "plain"}),
            mk(2, "model_response", {"content": "", "tool_calls": [{"call_id": "c1", "name": "n"}]}),
        ]
    )
    assert msgs[0] == ModelMessage(role="assistant", content="plain")
    assert msgs[1].tool_calls == (ModelToolCall(call_id="c1", name="n", arguments={}),)


def test_model_response_without_content_salvages_tool_calls() -> None:
    # R7 读端 fail-open:tool-call-only 轮次(content 缺失/非 str)不丢 tool_calls
    call = {"call_id": "c1", "name": "search", "arguments": {"q": "x"}}
    expected = ModelMessage(
        role="assistant",
        content="",
        tool_calls=(ModelToolCall(call_id="c1", name="search", arguments={"q": "x"}),),
    )
    for payload in ({"content": None, "tool_calls": [call]}, {"tool_calls": [call]}):
        # R9:救回的 assistant 同样享受配对修复——缺结果的 c1 补合成 tool 消息
        assert to_messages([mk(1, "model_response", payload)]) == [
            expected,
            ModelMessage(role="tool", content=_UNPAIRED_NOTE, tool_call_id="c1"),
        ]


def test_tool_call_only_turn_keeps_paired_tool_result() -> None:
    # 回归钉死:assistant 消息不再消失,配对的 tool_result 不会在投影里成为孤儿
    msgs = to_messages(
        [
            mk(1, "model_response", {"content": None, "tool_calls": [{"call_id": "c1", "name": "t", "arguments": {}}]}),
            mk(2, "tool_result", {"call_id": "c1", "content": [{"type": "text", "data": {"text": "ok"}}]}),
        ]
    )
    assert [m.role for m in msgs] == ["assistant", "tool"]
    assert msgs[1].tool_call_id == msgs[0].tool_calls[0].call_id


def test_model_response_no_content_and_no_valid_calls_skipped() -> None:
    bad = [
        mk(1, "model_response", {"content": None, "tool_calls": []}),  # 空 tool_calls → 无正文可投,跳过
        mk(2, "model_response", {"content": 42}),  # 无 content 无 tool_calls → 跳过
        mk(3, "model_response", {"content": None, "tool_calls": "nope"}),  # 不可重建 → 跳过
        mk(4, "model_response", {"content": None, "tool_calls": [{"call_id": 1, "name": "n"}]}),  # 条目畸形 → 跳过
    ]
    assert to_messages(bad) == []


# ---------------------------------------------------------------------------
# R9 配对修复:未配对的 assistant tool_calls 合成 tool 结果(真实 provider 配对规则)
# ---------------------------------------------------------------------------


def test_pairing_repair_synthesizes_missing_result_immediately_after_assistant() -> None:
    # (a) assistant 两个 call,只有 c1 有真实结果:c2 的合成结果紧跟 assistant,先于后续 item
    items = [
        mk(
            1,
            "model_response",
            {
                "content": "",
                "tool_calls": [
                    {"call_id": "c1", "name": "t", "arguments": {}},
                    {"call_id": "c2", "name": "t", "arguments": {}},
                ],
            },
        ),
        mk(2, "tool_result", {"call_id": "c1", "content": [{"type": "text", "data": {"text": "real"}}]}),
        mk(3, "user_message", {"text": "next"}),
    ]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["assistant", "tool", "tool", "user"]
    assert msgs[1] == ModelMessage(role="tool", content=_UNPAIRED_NOTE, tool_call_id="c2")
    assert msgs[2].content == "real" and msgs[2].tool_call_id == "c1"


def test_all_calls_unpaired_get_synthetic_results() -> None:
    # (b) 零结果:全部合成,按 tool_calls 顺序
    items = [
        mk(
            1,
            "model_response",
            {
                "content": "calling",
                "tool_calls": [
                    {"call_id": "c1", "name": "t", "arguments": {}},
                    {"call_id": "c2", "name": "t", "arguments": {}},
                ],
            },
        )
    ]
    msgs = to_messages(items)
    assert msgs == [
        ModelMessage(
            role="assistant",
            content="calling",
            tool_calls=(ModelToolCall(call_id="c1", name="t", arguments={}), ModelToolCall(call_id="c2", name="t", arguments={})),
        ),
        ModelMessage(role="tool", content=_UNPAIRED_NOTE, tool_call_id="c1"),
        ModelMessage(role="tool", content=_UNPAIRED_NOTE, tool_call_id="c2"),
    ]


def test_fully_paired_history_has_no_synthesis() -> None:
    # (c) 全配对:投影与旧行为逐字节一致——无任何合成消息
    items = [
        mk(1, "user_message", {"text": "go"}),
        mk(2, "model_response", {"content": "", "tool_calls": [{"call_id": "c1", "name": "t", "arguments": {}}]}),
        mk(3, "tool_result", {"call_id": "c1", "content": [{"type": "text", "data": {"text": "done"}}]}),
        mk(4, "model_response", {"content": "fin"}),
    ]
    msgs = to_messages(items)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert all(_UNPAIRED_NOTE not in m.content for m in msgs)
    assert msgs[2].content == "done"


def test_denied_or_not_executed_result_counts_as_paired() -> None:
    # (d) 存在性按 kind+call_id 判定,与 status/文本无关:denied/not_executed 不触发合成
    for status in ("denied", "not_executed", "outcome_unknown", "error"):
        items = [
            mk(1, "model_response", {"content": "", "tool_calls": [{"call_id": "c1", "name": "t", "arguments": {}}]}),
            mk(2, "tool_result", {"call_id": "c1", "status": status, "content": [{"type": "text", "data": {"text": "x"}}]}),
        ]
        msgs = to_messages(items)
        assert [m.role for m in msgs] == ["assistant", "tool"], status
        assert len(msgs) == 2, status


@st.composite
def pairing_items(draw: st.SearchStrategy) -> list[Item]:  # type: ignore[misc]
    # (e) 变体:每轮 1-3 个全局唯一 call id,每个 id 随机有/无真实结果
    items: list[Item] = []
    seq = 0
    next_id = 0
    for _ in range(draw(st.integers(1, 5))):
        calls = []
        for _ in range(draw(st.integers(1, 3))):
            calls.append(f"tc{next_id}")
            next_id += 1
        seq += 1
        items.append(
            mk(
                seq,
                "model_response",
                {"content": "resp", "tool_calls": [{"call_id": c, "name": "tool", "arguments": {}} for c in calls]},
            )
        )
        for c in calls:
            if draw(st.booleans()):
                seq += 1
                items.append(mk(seq, "tool_result", {"call_id": c, "content": [{"type": "text", "data": {"text": "ok"}}]}))
        if draw(st.booleans()):
            seq += 1
            items.append(mk(seq, "user_message", {"text": "next"}))
    return items


@given(pairing_items())
@settings(deadline=None, max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_pairing_invariant_holds_for_every_assistant(items: list[Item]) -> None:
    msgs = to_messages(items)
    for idx, m in enumerate(msgs):
        if m.role != "assistant" or not m.tool_calls:
            continue
        next_assistant = len(msgs)
        for j in range(idx + 1, len(msgs)):
            if msgs[j].role == "assistant":
                next_assistant = j
                break
        for call in m.tool_calls:
            following = [k for k in range(idx + 1, next_assistant) if msgs[k].role == "tool" and msgs[k].tool_call_id == call.call_id]
            assert len(following) == 1, call.call_id


def test_build_context_empty_and_no_boundaries() -> None:
    assert build_context([]) == []
    items = [mk(1, "user_message", {"text": "a"})]
    out = build_context(items)
    assert out == items and out is not items  # 返回新 list,不改输入
