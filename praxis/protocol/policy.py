"""协议层权限策略:冻结 pydantic 模型、校验器与规则匹配代数(设计 §5.3)。

移植自旧 agent_runtime/permission_policy.py:数据模型(25-132)更名而来
(_FrozenModel→FrozenPolicyModel,Decision→PolicyVerdict,Decision 这个名字
留给 permission.py 的判定结果数据类);解析与匹配代数(Task 2,134-414 的
resolve/path_reason/path_access/tool_decision/network_decision/
command_network_allowed/_lexical/_same_path/_under_path/_domain_matches)以本模块
纯函数形式重写,是全项目规则匹配的唯一实现(core 判定管线与沙箱 profile 编译器
共用,不得有第二套)。
刻意不移植:user_config 依赖、_PROTECTED 地板(阶段 2 Task 3 floor.py 正交机制)、
ceilings 嵌套与 intersection(ceiling 组合在 Task 4 判定层按最严者胜完成)、
with_read_files/with_read_references/explain(上层能力,非匹配代数)。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

PolicyVerdict = Literal["allow", "ask", "deny"]
Access = Literal["read", "write", "deny"]
Profile = Literal["workspace", "read-only", "trusted-local"]
Approval = Literal["on-request", "never"]


class FrozenPolicyModel(BaseModel):
    """extra=forbid + frozen + strict 的策略模型基类;revision 是规范化 JSON 的 sha256。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @property
    def revision(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class PathRule(FrozenPolicyModel):
    """Allow reads, allow reads and writes, or deny a path and its descendants."""

    path: str
    access: Access

    @field_validator("path")
    @classmethod
    def valid_path(cls, value: str) -> str:
        if not value or "\0" in value:
            raise ValueError("path must be nonempty and contain no NUL bytes")
        return value


class ToolRule(FrozenPolicyModel):
    """A case-sensitive fnmatch pattern over canonical tool names."""

    pattern: str
    decision: PolicyVerdict

    @field_validator("pattern")
    @classmethod
    def valid_pattern(cls, value: str) -> str:
        if not value:
            raise ValueError("tool pattern must not be empty")
        return value


class NetworkRule(FrozenPolicyModel):
    """A decision for one exact host or '*.domain' subdomains."""

    domain: str
    decision: PolicyVerdict

    @field_validator("domain")
    @classmethod
    def valid_domain(cls, value: str) -> str:
        return NetworkPolicy.valid_domains((value,))[0]


# 权限规则的联合类型:供 PermissionContext 规则层字段引用;解析代数在 core.permission(Task 2)。
type PolicyRule = ToolRule | PathRule | NetworkRule


class NetworkPolicy(FrozenPolicyModel):
    """Domain filters operate on exact hosts or '*.domain' subdomains only."""

    decision: PolicyVerdict = "ask"
    allowed_domains: tuple[str, ...] = ()
    denied_domains: tuple[str, ...] = ()
    rules: tuple[NetworkRule, ...] = ()

    @field_validator("allowed_domains", "denied_domains", mode="before")
    @classmethod
    def domain_sequence(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("rules", mode="before")
    @classmethod
    def rule_sequence(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("allowed_domains", "denied_domains")
    @classmethod
    def valid_domains(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = []
        for value in values:
            domain = value.lower().rstrip(".")
            host = domain[2:] if domain.startswith("*.") else domain
            if not host or not re.fullmatch(r"[a-z0-9_-]+(?:\.[a-z0-9_-]+)*", host):
                raise ValueError("domain must be a host name or '*.domain', without URL, port, or path")
            normalized.append(domain)
        return tuple(normalized)


class PermissionPolicy(FrozenPolicyModel):
    """Strict policy declaration; relative paths resolve against the workspace."""

    profile: Profile = "workspace"
    approval: Approval = "on-request"
    filesystem: tuple[PathRule, ...] = ()
    tools: tuple[ToolRule, ...] = ()
    network: NetworkPolicy = NetworkPolicy()

    @field_validator("filesystem", "tools", mode="before")
    @classmethod
    def rule_sequence(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def readonly_cannot_widen(self) -> PermissionPolicy:
        if self.profile == "read-only" and any(rule.access == "write" for rule in self.filesystem):
            raise ValueError("read-only profile cannot contain write grants")
        return self


# --- 解析与规则匹配代数(Task 2;唯一实现)------------------------------------

_DECISION_ORDER: dict[str | None, int] = {None: -1, "allow": 0, "ask": 1, "deny": 2}
_ACCESS_ORDER: dict[str, int] = {"read": 0, "write": 1, "deny": 2}


@dataclass(frozen=True, slots=True)
class ResolvedPolicy:
    """策略解析的纯数据成品:filesystem 全绝对路径(profile base 授权已展开)。

    刻意不含 ceilings 嵌套与 intersection():多策略组合由 core 判定层(Task 4)
    对各自 ResolvedPolicy 的结论按"最严者胜"完成,不建第二套交集机器。
    """

    workspace: str
    profile: Profile = "workspace"
    approval: Approval = "on-request"
    filesystem: tuple[PathRule, ...] = ()
    tools: tuple[ToolRule, ...] = ()
    network: NetworkPolicy = field(default_factory=NetworkPolicy)
    private_paths: tuple[str, ...] = ()


def resolve_policy(
    policy: PermissionPolicy, workspace: Path, *, private_paths: tuple[str, ...] = (),
) -> ResolvedPolicy:
    """把声明策略落到具体工作区:workspace 本体授权 + 声明规则绝对化。

    不加地板规则——地板(Task 3)是 bypass 免疫的正交机制,不是普通 PathRule;
    private_paths 原样携带,由 path_reason 单独拒绝。
    """
    root = workspace.expanduser().resolve()
    rules = [PathRule(path=str(root), access="read" if policy.profile == "read-only" else "write")]
    if policy.profile == "trusted-local":
        rules.append(PathRule(path="/", access="read"))
    rules.extend(PathRule(path=str(_lexical(rule.path, root)), access=rule.access) for rule in policy.filesystem)
    return ResolvedPolicy(
        workspace=str(root),
        profile=policy.profile,
        approval=policy.approval,
        filesystem=tuple(rules),
        tools=policy.tools,
        network=policy.network,
        private_paths=tuple(private_paths),
    )


def path_reason(policy: ResolvedPolicy, path: str | Path, *, write: bool = False) -> str:
    """可执行的拒绝理由;""=已授权。移植旧的双候选语义。

    lexical(expanduser+abspath,不解析 symlink)与 real(**non-strict** resolve(),
    断链/缺失不抛,B10)都要通过,任一不过即 deny;real 候选失败时理由加
    "symlink target" 前缀——symlink 不把授权转移到其它子树。
    """
    try:
        lexical = _lexical(path, Path(policy.workspace))
        real = lexical.resolve()
    except (OSError, RuntimeError, ValueError):
        return "path cannot be safely resolved"
    if write and policy.profile == "read-only":
        return "read-only profile denies writes"
    for candidate in dict.fromkeys((lexical, real)):
        if any(_under_path(candidate, Path(private)) for private in policy.private_paths):
            return "path is under a private path"
        matching = [rule for rule in policy.filesystem if _under_path(candidate, Path(rule.path))]
        if not matching:
            prefix = "symlink target" if candidate != lexical else "path"
            return f"{prefix} is outside permitted paths"
        # 最具体(parts 数)优先;同一路径同时被 read 与 write 规则覆盖时按 access 序取严。
        selected = max(matching, key=lambda rule: (len(Path(rule.path).parts), _ACCESS_ORDER[rule.access]))
        if selected.access == "deny" or (write and selected.access != "write"):
            prefix = "symlink target" if candidate != lexical else "path"
            return f"{prefix} denied by {selected.access} rule: {selected.path}"
    return ""


def path_access(policy: ResolvedPolicy, path: str | Path, *, write: bool = False) -> bool:
    return not path_reason(policy, path, write=write)


def tool_decision(policy: ResolvedPolicy, name: str) -> PolicyVerdict | None:
    """fnmatchcase 全匹配取最严(deny>ask>allow);无匹配返回 None 交给上层默认。"""
    decisions: list[PolicyVerdict] = [rule.decision for rule in policy.tools if fnmatchcase(name, rule.pattern)]
    return max(decisions, key=_DECISION_ORDER.__getitem__) if decisions else None


def network_decision(policy: ResolvedPolicy, url: str) -> PolicyVerdict:
    """http(s) 主机匹配:rules 取最严(base deny 不被覆盖),denied 压一切,白名单外一律 deny。"""
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return "deny"
    if parsed.scheme not in {"http", "https"} or not host:
        return "deny"
    decision = policy.network.decision
    matching = [rule.decision for rule in policy.network.rules if _domain_matches(host, rule.domain)]
    if matching and decision != "deny":
        decision = max(matching, key=_DECISION_ORDER.__getitem__)
    if any(_domain_matches(host, domain) for domain in policy.network.denied_domains):
        decision = "deny"
    elif policy.network.allowed_domains and not any(
        _domain_matches(host, domain) for domain in policy.network.allowed_domains
    ):
        decision = "deny"
    return decision


def command_network_decision(policy: ResolvedPolicy) -> PolicyVerdict:
    """D9:子进程目的地无代理不可逐域强制——存在任何域名过滤即 fail closed 为 deny。"""
    if policy.network.allowed_domains or policy.network.denied_domains or policy.network.rules:
        return "deny"
    return policy.network.decision


def _lexical(path: str | Path, base: Path) -> Path:
    """词法绝对化(expanduser+abspath),不解析 symlink——symlink 逃逸另行裁决。"""
    candidate = Path(path).expanduser()
    return Path(os.path.abspath(candidate if candidate.is_absolute() else base / candidate))


def _same_path(first: Path, second: Path) -> bool:
    if first == second:
        return True
    try:
        return first.samefile(second)
    except OSError:
        return False


def _under_path(candidate: Path, root: Path) -> bool:
    if candidate.is_relative_to(root):
        return True
    # Filesystem identity handles case and Unicode aliases on macOS without
    # assuming every filesystem is case-insensitive and widening Linux grants.
    key = unicodedata.normalize("NFC", str(root)).casefold()
    return any(
        unicodedata.normalize("NFC", str(ancestor)).casefold() == key and _same_path(ancestor, root)
        for ancestor in (candidate, *candidate.parents)
    )


def _domain_matches(host: str, pattern: str) -> bool:
    """'*.domain' 匹配子域但不匹配裸域;裸 pattern 只精确匹配。"""
    return host.endswith(pattern[1:]) and host != pattern[2:] if pattern.startswith("*.") else host == pattern
