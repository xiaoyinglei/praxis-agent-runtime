"""读侧三层上下文链(Task 9 / D11):``build_context`` → ``elide_old_tool_outputs`` → ``to_messages``。

钉死:
- ``elide_old_tool_outputs`` 是纯函数、幂等、永不抛;只动 ``tool_result``;
- **R-T9-1**:省略旧工具输出时必须原样保留 ``artifact://`` 指针行,否则模型永久
  失去回读自己那份大输出的能力;
- 对任意 item 序列,``fold → elide → to_messages`` 不抛且二次 elide 幂等。
"""

from __future__ import annotations

from typing import Any

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from praxis.core.context import build_context, elide_old_tool_outputs, to_messages
from praxis.protocol.items import Item
from praxis.protocol.models import ModelMessage

ELIDED_MARKER = "[elided tool output:"
ARTIFACT_POINTER = (
    "[full output stored: artifact://"
    + "a" * 64
    + " (4096 bytes); read with read_artifact]"
)


def mk(seq: int, kind: str, payload: dict[str, Any], turn_id: str | None = None) -> Item:
    return Item(
        item_id=f"i{seq}",
        thread_id="t",
        turn_id=turn_id,
        seq=seq,
        kind=kind,
        payload=payload,
        created_at=float(seq),
    )


def tool_result(
    seq: int,
    text: str,
    *,
    call_id: str = "c1",
    tool_name: str = "bash",
    turn_id: str | None = None,
) -> Item:
    return mk(
        seq,
        "tool_result",
        {
            "call_id": call_id,
            "tool_name": tool_name,
            "content": [{"type": "text", "data": {"text": text}}],
        },
        turn_id=turn_id,
    )


# ---------------------------------------------------------------------------
# brief Step 1 钉死:纯函数 + 幂等 + 输入不变
# ---------------------------------------------------------------------------


def test_elide_is_pure_and_idempotent() -> None:
    items = [
        tool_result(1, "LONG-" + "x" * 200, tool_name="bash"),
        tool_result(2, "second-" + "y" * 200, call_id="c2", tool_name="read"),
        tool_result(3, "tail-" + "z" * 200, call_id="c3", tool_name="bash"),
    ]
    original_payloads = [dict(item.payload) for item in items]

    once = elide_old_tool_outputs(items, keep_last=1, min_bytes=10)
    assert elide_old_tool_outputs(once, keep_last=1, min_bytes=10) == once
    assert [dict(item.payload) for item in items] == original_payloads  # 输入不被修改

    assert once[0].payload["elided"] is True
    assert ELIDED_MARKER in once[0].payload["content"][0]["data"]["text"]
    assert once[1].payload["elided"] is True
    # 最后 keep_last(=1) 条完整
    assert "elided" not in once[2].payload
    assert once[2].payload["content"][0]["data"]["text"] == original_payloads[2]["content"][0]["data"]["text"]


def test_keep_last_zero_elides_every_eligible_result() -> None:
    items = [
        tool_result(1, "a" * 100),
        tool_result(2, "b" * 100, call_id="c2"),
    ]
    out = elide_old_tool_outputs(items, keep_last=0, min_bytes=10)
    assert all(item.payload["elided"] is True for item in out)


def test_small_outputs_and_non_tool_results_untouched() -> None:
    items = [
        tool_result(1, "short"),
        mk(2, "user_message", {"text": "x" * 1000}),
        mk(3, "model_response", {"content": "y" * 1000}),
    ]
    assert elide_old_tool_outputs(items, keep_last=0, min_bytes=10) == items


def test_non_text_blocks_do_not_count_toward_min_bytes() -> None:
    item = mk(
        1,
        "tool_result",
        {
            "call_id": "c1",
            "content": [
                {"type": "image", "data": {"url": "u" * 1000}},
                {"type": "text", "data": {"text": "tiny"}},
            ],
        },
    )
    out = elide_old_tool_outputs([item], keep_last=0, min_bytes=10)
    assert out == [item]  # 文本口径只有 4 字节,不触发省略


# ---------------------------------------------------------------------------
# R-T9-1:省略必须保留 artifact 指针
# ---------------------------------------------------------------------------


def test_elide_preserves_artifact_pointer() -> None:
    body = "HEADER-BODY\n" + "x" * 500 + "\n" + ARTIFACT_POINTER
    item = tool_result(1, body)
    [out] = elide_old_tool_outputs([item], keep_last=0, min_bytes=10)
    text: str = out.payload["content"][0]["data"]["text"]

    assert out.payload["elided"] is True
    assert ELIDED_MARKER in text
    # 指针行原样保留;正文被省略
    assert ARTIFACT_POINTER in text
    assert "HEADER-BODY" not in text
    assert "x" * 500 not in text
    # 省略后的文本显著短于原文(没有把正文换个地方复制)
    assert len(text.encode()) < len(body.encode())


def test_elide_keeps_every_pointer_line() -> None:
    second = ARTIFACT_POINTER.replace("artifact://" + "a" * 64, "artifact://" + "b" * 64)
    item = tool_result(1, "junk" * 100 + "\n" + ARTIFACT_POINTER + "\n" + second)
    [out] = elide_old_tool_outputs([item], keep_last=0, min_bytes=10)
    text: str = out.payload["content"][0]["data"]["text"]
    assert ARTIFACT_POINTER in text
    assert second in text


# ---------------------------------------------------------------------------
# fail-open:未知 boundary 版本不折叠(brief Step 1 第 4 组)
# ---------------------------------------------------------------------------


def test_unknown_boundary_version_still_fail_open() -> None:
    items = [
        mk(1, "user_message", {"text": "a"}),
        mk(
            9,
            "compaction_boundary",
            {"algorithm_version": "v99", "covered_from": 1, "covered_to": 9},
        ),
    ]
    assert build_context(items) == list(items)


def test_summary_projects_before_later_tail_messages() -> None:
    # 压缩时 summary 的 seq 天然晚于它概括的旧历史;投影必须把它放在保留尾部之前。
    items = [
        mk(1, "user_message", {"text": "tail1"}),
        mk(2, "user_message", {"text": "tail2"}),
        mk(3, "summary", {"text": "old history"}),
    ]
    assert [message.content for message in to_messages(items)] == [
        "[conversation summary]\nold history",
        "tail1",
        "tail2",
    ]


# ---------------------------------------------------------------------------
# hypothesis:fold → elide → to_messages 对任意序列不抛且 elide 幂等
# ---------------------------------------------------------------------------


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
        lambda children: st.lists(children, max_size=3)
        | st.dictionaries(st.text(max_size=4), children, max_size=3),
        max_leaves=6,
    )
    junk_payload = st.dictionaries(st.text(max_size=8), junk_values, max_size=4)
    # 合法 tool_result 形状 + 能真正越过大小的文本,让 elide 的替换分支被覆盖。
    tool_payload = st.fixed_dictionaries(
        {"call_id": st.text(min_size=1, max_size=6)},
        optional={
            "tool_name": st.text(max_size=8),
            "content": st.lists(
                st.fixed_dictionaries(
                    {
                        "type": st.just("text"),
                        "data": st.fixed_dictionaries(
                            {
                                "text": st.one_of(
                                    st.text(max_size=5),
                                    st.text(min_size=20, max_size=300),
                                )
                            }
                        ),
                    }
                ),
                max_size=3,
            ),
            "elided": st.booleans(),
        },
    )
    boundary_payload = st.fixed_dictionaries(
        {},
        optional={
            "covered_from": st.one_of(
                st.integers(-1, 11), st.none(), st.booleans(), st.text(max_size=3)
            ),
            "covered_to": st.one_of(
                st.integers(-1, 11), st.none(), st.booleans(), st.text(max_size=3)
            ),
            "algorithm_version": st.one_of(
                st.just("v1"), st.text(max_size=8), st.none(), st.integers()
            ),
        },
    )
    n = draw(st.integers(0, 12))
    kinds = draw(st.lists(kinds_st, min_size=n, max_size=n))
    items: list[Item] = []
    for seq, kind in enumerate(kinds, start=1):
        if kind == "compaction_boundary":
            payload = draw(boundary_payload)
        elif kind == "tool_result":
            payload = draw(st.one_of(junk_payload, tool_payload))
        else:
            payload = draw(junk_payload)
        items.append(mk(seq, kind, payload))
    return items


@given(item_lists())
# 固定的"必然触发替换"样本:属性测试整体跑 200 例,但显式钉一例保证分支被执行。
@example(
    [
        tool_result(1, "a" * 300),
        tool_result(2, "b" * 300, call_id="c2"),
        tool_result(3, "c" * 300, call_id="c3"),
    ]
)
@settings(deadline=None, max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_fold_elide_project_never_raises_and_elide_is_idempotent(items: list[Item]) -> None:
    before = [(i.item_id, i.seq, i.kind, dict(i.payload)) for i in items]
    once = elide_old_tool_outputs(build_context(items), keep_last=2, min_bytes=10)
    twice = elide_old_tool_outputs(once, keep_last=2, min_bytes=10)
    assert twice == once
    assert [(i.item_id, i.seq, i.kind, dict(i.payload)) for i in items] == before

    messages = to_messages(once)
    assert all(isinstance(message, ModelMessage) for message in messages)
    assert all(message.role in {"system", "user", "assistant", "tool"} for message in messages)
