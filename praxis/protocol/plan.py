"""计划协议：常量、数据类型 + 只读投影（设计 §X；B16 源头）。

旧规划实现把步骤静默截断到 12——所有约束在 protocol 层集中定义，
工具端和投影端都唯一定义 / 消费这些上限。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

from praxis.protocol.items import Item
from praxis.protocol.json import thaw_json

MAX_PLAN_STEPS = 20
MAX_STEP_TITLE_CHARS = 180
MAX_PLAN_SUMMARY_CHARS = 800

PlanStatus = Literal["pending", "in_progress", "completed"]

_VALID_STATUSES = frozenset(get_args(PlanStatus))


@dataclass(frozen=True, slots=True)
class PlanStep:
    """一个步骤的规范化投影。"""

    step_id: str
    step: str
    status: PlanStatus


@dataclass(frozen=True, slots=True)
class PlanSnapshot:
    """只读快照：从 tool_result item 投影而来。"""

    steps: tuple[PlanStep, ...]
    revision: int
    explanation: str | None


def latest_plan(items: Sequence[Item]) -> PlanSnapshot | None:
    """返回最近一次成功写入的计划快照（零截断）。

    取 last item with kind==tool_result AND tool_name=="update_plan"
    AND status=="ok"。然后从 payload.metadata 读 plan / revision / explanation。
    畸形 → None，永不抛。
    """
    for item in reversed(items):
        if item.kind != "tool_result":
            continue
        tn = item.payload.get("tool_name")
        st = item.payload.get("status")
        is_err = item.payload.get("is_error")
        if tn != "update_plan" or st != "ok" or is_err is True:
            continue
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
        if not isinstance(rev, int) or rev < 0:
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
    return None


def project_to_json(snapshot: PlanSnapshot) -> dict[str, object]:
    """将 PlanSnapshot 序列化为可存入 item metadata 的 JSON 兼容结构。

    供测试验证 round-trip 使用。
    """
    return {
        "plan": [
            {"step_id": s.step_id, "step": s.step, "status": s.status}
            for s in snapshot.steps
        ],
        "revision": snapshot.revision,
        "explanation": snapshot.explanation,
    }
