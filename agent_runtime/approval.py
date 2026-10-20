"""Bounded, process-local command grants. No shell execution or persistence."""
from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from agent_runtime.tools.tool import CancellationMode, JsonValue, ResolvedToolUse, Tool, ToolApprovalProfile, ToolEffect

if TYPE_CHECKING:
    from agent_runtime.tools.permissions import ToolExecutionContext

ApprovalMode = Literal["ask", "auto", "full"]
APPROVAL_MODES: dict[str, str] = {
    "ask": "请求批准 · 写入、执行和联网时询问",
    "auto": "帮我批准 · 自动处理工作区编辑、只读沙箱命令和公开网页",
    "full": "完全访问 · 权限范围内自动批准；仍受沙箱和明确禁令约束",
}
MODE_ORDER = {"ask": 0, "auto": 1, "full": 2}
APPROVAL_CHOICES = {"approve": "允许一次", "allow_once": "允许一次",
                    "allow_session": "本会话允许同类命令", "deny": "拒绝"}
_SHELL_SYNTAX = re.compile(r"[;&|<>$`\n\r(){}*?\[\]\\\x00]")
_INTERPRETER = re.compile(r"(?:.*sh|fish|python[\d.]*|node|ruby|perl|php|lua|eval|exec|env|xargs)")


@dataclass(frozen=True, slots=True)
class CommandGrant:
    argv: tuple[str, ...]
    parameters: str
    tool_revision: str
    permissions_revision: str | None
    workspace: str
    cwd: str
    targets: tuple[tuple[str, str], ...]

    @property
    def label(self) -> str:
        return f"{shlex.join(self.argv)} [可追加参数]；目录 {self.cwd}；只读沙箱、禁止联网"

    def to_json(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(json.dumps(asdict(self))))

    def matches(self, other: CommandGrant) -> bool:
        return (other.argv[:len(self.argv)] == self.argv
                and all(getattr(self, key) == getattr(other, key) for key in (
                    "parameters", "tool_revision", "permissions_revision", "workspace", "cwd", "targets",
                )))


def command_grant(
    tool: Tool, args: Mapping[str, JsonValue], resolved: ResolvedToolUse, context: ToolExecutionContext,
) -> CommandGrant | None:
    """Offer token-prefix authority only inside the same read-only, offline sandbox.

    Keep the entire approved argv as the prefix; never infer a broader executable
    or subcommand prefix. Shell programs, expansions and compound commands do not
    have an argv-prefix meaning here and cannot receive reusable authority.
    """
    if (tool.definition.name != "run_command"
            or tool.approval_profile is not ToolApprovalProfile.RESTRICTED_READ_ONLY_PROCESS
            or tool.cancellation_mode is not CancellationMode.MANAGED_PROCESS
            or resolved.effects != frozenset({ToolEffect.READ_WORKSPACE, ToolEffect.EXECUTE_PROCESS})
            or not any(t.kind == "execution_mode" and t.value == "restricted_sandbox" for t in resolved.targets)
            or tool.definition.name in context.require_confirmation_for):
        return None
    command = args.get("command")
    if not isinstance(command, str) or _SHELL_SYNTAX.search(command):
        return None
    try:
        argv = tuple(shlex.split(command))
    except ValueError:
        return None
    if (not argv or "=" in argv[0] or any(
            _INTERPRETER.fullmatch(Path(token).name) or token in {"-c", "--eval"}
            or token.endswith((".sh", ".py", ".js", ".rb", ".pl")) for token in argv)):
        return None
    permissions = context.effective_permissions
    if permissions is not None and permissions.tool_decision(tool.definition.name) in {"ask", "deny"}:
        return None
    return CommandGrant(
        argv=argv,
        parameters=json.dumps({key: value for key, value in args.items() if key != "command"}, sort_keys=True),
        tool_revision=tool.execution_revision,
        permissions_revision=None if permissions is None else permissions.revision,
        workspace=str(context.workspace_root), cwd=str(context.cwd),
        targets=tuple((target.kind, target.value) for target in resolved.targets),
    )
