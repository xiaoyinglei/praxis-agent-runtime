"""计划协议：常量与数据类型（设计 §5.7；B16 源头）。

旧规划实现把步骤静默截断到 12——所有约束在 protocol 层集中定义，
工具端和投影端都唯一定义 / 消费这些上限。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MAX_PLAN_STEPS = 20
MAX_STEP_TITLE_CHARS = 180
MAX_PLAN_SUMMARY_CHARS = 800

PlanStatus = Literal["pending", "in_progress", "completed"]


@dataclass(frozen=True, slots=True)
class PlanStep:
    """一个步骤的规范化投影。"""

    step_id: str
    step: str
    status: PlanStatus


@dataclass(frozen=True, slots=True)
class PlanSnapshot:
    """只读快照：从 tool_result item 投影而来。

    ``revision`` 是本次提交中的步骤数快照；**新旧判定一律以 item 的 seq 为准**。
    """

    steps: tuple[PlanStep, ...]
    revision: int
    explanation: str | None
