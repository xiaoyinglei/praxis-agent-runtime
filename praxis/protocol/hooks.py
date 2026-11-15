"""hook 协议数据:command hook 的唯一表示与结算事实(设计 §5.14;3B 笔记 §4)。

旧实现有三份 hook 表示(dataclass / binding dict / rollout manifest)和
handler-callable 双入口两种信任模型;这里只保留 **command hook** 一种:
``HookSpec.argv`` 是绝对路径命令,运行器(T4)只 spawn 它。

- ``HookEvent`` 只有 PreToolUse/PostToolUse;``Stop`` 属阶段 6 完成门槛,
  在加载期明确拒绝(见 ``adapters/hooks/config.py``),不静默。
- ``HookSpec`` 校验移植旧 ``LifecycleHook.__post_init__``:name/revision 1–128、
  timeout ∈ (0,30] 默认 5.0、argv 非空且 argv[0] 为绝对路径、
  tool_names/effects 是精确非空字符串,effects 按 ``ToolEffect`` 规范化;
  两个选择器都为空 = 该轴上不筛选(Pre/Post 都按 tool name 或 effects 匹配)。
- ``HookOutcome`` 是运行期结算事实的唯一形状;feedback 在构造期截断到
  ``MAX_FEEDBACK_BYTES``(旧 ``HookDecision`` 是抛错;这里是"截断"语义,
  避免运行器把超长宿主输出变成异常)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from praxis.protocol.tools import ToolEffect

HookEvent = Literal["PreToolUse", "PostToolUse"]
HookStatus = Literal[
    "allowed",
    "blocked",
    "observed",
    "timeout",
    "error",
    "unavailable",
    "unknown",
]

MAX_HOOKS = 16
MAX_FEEDBACK_BYTES = 2048
MAX_HOOK_STDOUT_BYTES = 16_385
MAX_HOOKS_CONFIG_BYTES = 65_537

_EVENTS: tuple[HookEvent, ...] = ("PreToolUse", "PostToolUse")
_STATUSES: tuple[HookStatus, ...] = (
    "allowed",
    "blocked",
    "observed",
    "timeout",
    "error",
    "unavailable",
    "unknown",
)
_MAX_NAME_CHARS = 128
_MAX_TIMEOUT_SECONDS = 30.0
_DEFAULT_TIMEOUT_SECONDS = 5.0


def _truncate_utf8(value: str, limit: int) -> str:
    """按 UTF-8 字节数截断,不产生半个码点。"""
    data = value.encode("utf-8")
    if len(data) <= limit:
        return value
    return data[:limit].decode("utf-8", "ignore")


def _bounded_event(value: object) -> str:
    text = value if isinstance(value, str) else type(value).__name__
    text = " ".join(text.split())
    return text if len(text) <= 64 else f"{text[:63]}…"


@dataclass(frozen=True, slots=True)
class HookSpec:
    """一个已校验的 command hook;``revision`` 是配置内容的规范化摘要。"""

    name: str
    revision: str
    event: HookEvent
    argv: tuple[str, ...]
    tool_names: frozenset[str] = frozenset()
    effects: frozenset[str] = frozenset()
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.event not in _EVENTS:
            raise ValueError(f"invalid hook event: {_bounded_event(self.event)}")
        if any(
            not isinstance(value, str) or not value.strip() or len(value) > _MAX_NAME_CHARS
            for value in (self.name, self.revision)
        ):
            raise ValueError("hook name and revision must have 1–128 characters")

        argv = self.argv
        if not isinstance(argv, (list, tuple)) or not argv:
            raise ValueError("hook argv must be a non-empty tuple of strings")
        if any(not isinstance(argument, str) or "\0" in argument for argument in argv):
            raise ValueError("hook argv must contain strings without NUL bytes")
        if not Path(argv[0]).is_absolute():
            raise ValueError("hook argv[0] must be an absolute path")
        object.__setattr__(self, "argv", tuple(argv))

        seconds = self.timeout_seconds
        if (
            type(seconds) not in {int, float}
            or not math.isfinite(seconds)
            or not 0 < seconds <= _MAX_TIMEOUT_SECONDS
        ):
            raise ValueError("hook timeout must be greater than 0 and at most 30 seconds")
        object.__setattr__(self, "timeout_seconds", float(seconds))

        for field_name in ("tool_names", "effects"):
            values = getattr(self, field_name)
            if not isinstance(values, (list, tuple, set, frozenset)) or isinstance(
                values, (str, bytes)
            ):
                raise ValueError(f"hook {field_name} must contain exact non-empty names")
            if any(
                not isinstance(value, str) or not value.strip() or value.strip() != value
                for value in values
            ):
                raise ValueError(f"hook {field_name} must contain exact non-empty names")
            if field_name == "effects":
                normalized: frozenset[str] = frozenset(ToolEffect(value).value for value in values)
            else:
                normalized = frozenset(values)
            object.__setattr__(self, field_name, normalized)


@dataclass(frozen=True, slots=True)
class HookOutcome:
    """一次 hook 调用的结算事实;feedback 构造期截断到 2048 字节。"""

    status: HookStatus
    feedback: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError(f"invalid hook status: {self.status!r}")
        if not isinstance(self.feedback, str) or not isinstance(self.source, str):
            raise ValueError("hook outcome feedback and source must be strings")
        if len(self.feedback.encode("utf-8")) > MAX_FEEDBACK_BYTES:
            object.__setattr__(
                self, "feedback", _truncate_utf8(self.feedback, MAX_FEEDBACK_BYTES)
            )