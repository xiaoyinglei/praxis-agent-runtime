"""协议层权限策略:冻结 pydantic 模型与校验器(设计 §5.3)。

移植自旧 agent_runtime/permission_policy.py:25-132 的数据模型部分:
_FrozenModel 更名为公开的 FrozenPolicyModel,Decision 更名为 PolicyVerdict
(Decision 这个名字留给 permission.py 的判定结果数据类)。
刻意不移植:user_config 依赖、_PROTECTED 地板、resolve/加载逻辑——
那些是阶段 2 core 的策略解析,本模块只有数据模型。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

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
