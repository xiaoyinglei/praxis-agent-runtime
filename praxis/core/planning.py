"""计划投影：从 item 序列提取最新 update_plan 结果（B16：无截断）。

纯函数：最后一条成功的 tool_result（tool_name == "update_plan"，status == ok）
即权威计划；读 payload["metadata"]["plan"] / ["revision"] / ["explanation"]，
该权威项畸形 → None（不回落到更旧的计划），永不抛。

按 seq + index 排序保证 recency 不依赖列表顺序（同 core/context.py 纪律）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, get_args

from praxis.protocol.items import Item
from praxis.protocol.json import thaw_json
from praxis.protocol.plan import PlanSnapshot, PlanStatus, PlanStep

_VALID_STATUSES = frozenset(get_args(PlanStatus))


def latest_plan(items: Sequence[Item]) -> PlanSnapshot | None:
    """返回最近一次成功写入的计划快照（零截断）。

    按 (seq, list-index) 排序后找到最后一条成功的 ``update_plan`` 结果项——
    它就是权威计划，不假设调用方传入的列表已按时间排序。
    该权威项 payload 畸形 → ``None``：不回落到更旧的计划（旧计划已被模型
    本次提交取代，展示它是在陈述一个不再成立的事实）。
    """
    ordered = sorted(enumerate(items), key=lambda entry: (entry[1].seq, entry[0]))
    authoritative: Item | None = None
    for _idx, item in ordered:
        if item.kind != "tool_result":
            continue
        tn = item.payload.get("tool_name")
        st = item.payload.get("status")
        is_err = item.payload.get("is_error")
        if tn != "update_plan" or st != "ok" or is_err is True:
            continue
        authoritative = item
    if authoritative is None:
        return None
    return _try_build_snapshot(authoritative)


def _try_build_snapshot(item: Item) -> PlanSnapshot | None:
    """从单条 tool_result 构建 PlanSnapshot；畸形 → None。"""
    metadata = item.payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    raw_plan = metadata.get("plan")
    if not isinstance(raw_plan, (list, tuple)):
        return None
    # Thaw frozen plan entries and verify each is a mapping
    plan_entries: list[Mapping[str, Any]] = []
    for e in raw_plan:
        th = thaw_json(e)
        if not isinstance(th, Mapping):
            return None
        plan_entries.append(th)
    rev = metadata.get("revision")
    if not isinstance(rev, int) or isinstance(rev, bool) or rev < 0:
        return None
    explanation = metadata.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        return None
    steps: list[PlanStep] = []
    for entry in plan_entries:
        sid = entry.get("step_id")
        if sid is None or not isinstance(sid, str):
            return None
        step_text = entry.get("step")
        if not isinstance(step_text, str):
            return None
        status_val_raw = entry.get("status")
        if not isinstance(status_val_raw, str):
            return None
        if status_val_raw not in _VALID_STATUSES:
            return None
        steps.append(PlanStep(step_id=sid, step=step_text, status=status_val_raw))  # type: ignore[arg-type]
    if not steps:
        return None
    return PlanSnapshot(
        steps=tuple(steps),
        revision=rev,
        explanation=explanation if explanation else None,
    )
