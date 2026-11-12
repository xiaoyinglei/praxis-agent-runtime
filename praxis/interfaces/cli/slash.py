"""chat 的 slash 命令注册表(设计 §5.12;C1:查表分派,无 facade loop)。

REGISTRY 是唯一分派依据:chat 循环只做 `REGISTRY.get(name)` → 执行,
新增 slash 命令 = 新增一条 Command 记录,不改 chat 循环本体。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from praxis.interfaces.sdk import Session


class ExitChat(Exception):  # noqa: N818 — 控制流信号(与 ConversationBusy 同族),非错误语义
    """slash 命令请求正常退出 chat 循环(退出码 0)。"""


@dataclass(frozen=True, slots=True)
class Command:
    """一条 slash 命令:name 不含前导斜杠;run(session) 返回要打印的文本("" = 无输出)。"""

    name: str
    help: str
    run: Callable[[Session], str]


def _format_time(timestamp: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def _help(session: Session) -> str:
    lines = [f"  /{command.name:<7} {command.help}" for command in REGISTRY.values()]
    return "slash commands:\n" + "\n".join(lines)


def _status(session: Session) -> str:
    turns = session.turns()
    if not turns:
        return "no turns yet"
    rows = [f"  {turn.turn_id}  {turn.status:<12} {_format_time(turn.updated_at)}" for turn in turns]
    return "\n".join(rows)


def _exit(session: Session) -> str:
    raise ExitChat


REGISTRY: dict[str, Command] = {
    command.name: command
    for command in (
        Command(name="help", help="list slash commands", run=_help),
        Command(name="status", help="show turns in this thread", run=_status),
        Command(name="exit", help="leave chat (exit code 0)", run=_exit),
    )
}
