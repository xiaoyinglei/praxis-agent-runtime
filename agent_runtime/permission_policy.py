"""Immutable permissions shared by tool checks, approval, and process sandboxes.

Path grants are subtree grants. Both the lexical path and its current real target
must be authorized; a symlink does not transfer authority to another subtree.
Intersections retain their inputs as ceilings rather than merging away denials.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
import unicodedata
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from agent_runtime.user_config import command_state_directory, user_config_directory

Decision = Literal["allow", "ask", "deny"]
Access = Literal["read", "write", "deny"]
Profile = Literal["workspace", "read-only", "trusted-local"]
Approval = Literal["on-request", "never"]
_PROTECTED = frozenset({".git", ".agents", ".codex", ".aws"})
_DECISION_ORDER = {None: -1, "allow": 0, "ask": 1, "deny": 2}
_ACCESS_ORDER = {"read": 0, "write": 1, "deny": 2}


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @property
    def revision(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class PathRule(_FrozenModel):
    """Allow reads, allow reads and writes, or deny a path and its descendants."""

    path: str
    access: Access

    @field_validator("path")
    @classmethod
    def valid_path(cls, value: str) -> str:
        if not value or "\0" in value:
            raise ValueError("path must be nonempty and contain no NUL bytes")
        return value


class ToolRule(_FrozenModel):
    """A case-sensitive fnmatch pattern over canonical tool names."""

    pattern: str
    decision: Decision

    @field_validator("pattern")
    @classmethod
    def valid_pattern(cls, value: str) -> str:
        if not value:
            raise ValueError("tool pattern must not be empty")
        return value


class NetworkRule(_FrozenModel):
    """A decision for one exact host or '*.domain' subdomains."""

    domain: str
    decision: Decision

    @field_validator("domain")
    @classmethod
    def valid_domain(cls, value: str) -> str:
        return NetworkPolicy.valid_domains((value,))[0]


class NetworkPolicy(_FrozenModel):
    """Domain filters operate on exact hosts or '*.domain' subdomains only."""

    decision: Decision = "ask"
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


class PermissionPolicy(_FrozenModel):
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

    def resolve(self, workspace: Path) -> ResolvedPermissions:
        root = workspace.expanduser().resolve()
        rules = [PathRule(path=str(root), access="read" if self.profile == "read-only" else "write")]
        if self.profile == "trusted-local":
            rules.append(PathRule(path="/", access="read"))
        rules.extend(PathRule(path=str(_lexical(rule.path, root)), access=rule.access) for rule in self.filesystem)
        resolved = ResolvedPermissions(
            workspace=str(root), profile=self.profile, approval=self.approval,
            filesystem=tuple(rules), tools=self.tools, network=self.network,
        )
        # Protective rules narrow inherited grants; they never grant outside reads.
        directories = {root, *(Path(rule.path) for rule in rules if rule.access != "deny")}
        for directory in sorted(directories):
            for name in sorted(_PROTECTED):
                path = directory / name
                rules.append(PathRule(path=str(path), access="read" if resolved.path_access(path) else "deny"))
        return resolved.model_copy(update={
            "filesystem": tuple(rules), "private_paths": (str(command_state_directory()),),
        })


class ResolvedPermissions(_FrozenModel):
    """Concrete immutable policy snapshot; ceilings must also be enforced by sandboxes."""

    workspace: str
    profile: Profile = "workspace"
    approval: Approval = "on-request"
    filesystem: tuple[PathRule, ...] = ()
    tools: tuple[ToolRule, ...] = ()
    network: NetworkPolicy = NetworkPolicy()
    sources: tuple[str, ...] = ("default",)
    private_paths: tuple[str, ...] = ()
    ceilings: tuple[ResolvedPermissions, ...] = ()

    @property
    def revision(self) -> str:
        atoms: set[str] = set()
        sources: set[str] = set()

        def collect(policy: ResolvedPermissions) -> None:
            value = policy.model_dump(mode="json", exclude={"ceilings", "sources"})
            for key in ("filesystem", "tools"):
                value[key] = sorted(value[key], key=lambda rule: json.dumps(rule, sort_keys=True))
            value["private_paths"] = sorted(value["private_paths"])
            for key in ("allowed_domains", "denied_domains"):
                value["network"][key] = sorted(value["network"][key])
            value["network"]["rules"] = sorted(value["network"]["rules"],
                                               key=lambda rule: json.dumps(rule, sort_keys=True))
            atoms.add(json.dumps(value, sort_keys=True, separators=(",", ":")))
            sources.update(source for source in policy.sources if Path(source).is_absolute())
            for ceiling in policy.ceilings:
                collect(ceiling)

        collect(self)
        payload = json.dumps({"policies": sorted(atoms), "protected_sources": sorted(sources)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def effective_approval(self) -> Approval:
        return "never" if self.approval == "never" or any(
            ceiling.effective_approval == "never" for ceiling in self.ceilings
        ) else "on-request"

    @field_validator("filesystem", "tools", "sources", "private_paths", "ceilings", mode="before")
    @classmethod
    def sequence(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def absolute_paths(self) -> ResolvedPermissions:
        if (not Path(self.workspace).is_absolute()
                or any(not Path(rule.path).is_absolute() for rule in self.filesystem)
                or any(not Path(path).is_absolute() for path in self.private_paths)):
            raise ValueError("resolved permissions require absolute filesystem paths")
        return self

    @property
    def sandbox_filesystem(self) -> tuple[PathRule, ...]:
        """Project each rule boundary through all ceilings for backend enforcement.

        Backends must preserve most-specific rule semantics, or fail closed when
        they cannot represent an exception below a denied/read-only subtree.
        Component metadata write protection remains a separate hard floor.
        """
        boundaries = {rule.path for rule in self.filesystem}
        boundaries.update(source for source in self.sources if Path(source).is_absolute())
        boundaries.update(self.private_paths)
        for ceiling in self.ceilings:
            boundaries.update(rule.path for rule in ceiling.sandbox_filesystem)
        return tuple(PathRule(
            path=path,
            access="write" if self.path_access(path, write=True) else "read" if self.path_access(path) else "deny",
        ) for path in sorted(boundaries, key=lambda value: (len(Path(value).parts), value)))

    def path_reason(self, path: str | Path, write: bool = False) -> str:
        """Return an actionable denial reason, or an empty string if authorized."""
        try:
            lexical = _lexical(path, Path(self.workspace))
            real = lexical.resolve()
        except (OSError, RuntimeError, ValueError):
            return "path cannot be safely resolved"
        if write and self.profile == "read-only":
            return "read-only profile denies writes"
        for candidate in dict.fromkeys((lexical, real)):
            if any(_under_path(candidate, Path(private)) for private in self.private_paths):
                return "host credential directory is private"
            if write and any(Path(source).is_absolute() and (
                    _under_path(candidate, Path(source)) or _same_path(candidate, Path(source)))
                             for source in self.sources):
                return "permission policy source is read-only"
            if write and _PROTECTED.intersection(part.casefold() for part in candidate.parts):
                return "protected metadata directory is read-only"
            matching = [rule for rule in self.filesystem if _under_path(candidate, Path(rule.path))]
            if not matching:
                prefix = "symlink target" if candidate != lexical else "path"
                return f"{prefix} is outside permitted paths"
            selected = max(matching, key=lambda rule: (len(Path(rule.path).parts), _ACCESS_ORDER[rule.access]))
            if selected.access == "deny" or (write and selected.access != "write"):
                prefix = "symlink target" if candidate != lexical else "path"
                return f"{prefix} denied by {selected.access} rule: {selected.path}"
        for ceiling in self.ceilings:
            reason = ceiling.path_reason(path, write=write)
            if reason:
                return f"permission ceiling: {reason}"
        return ""

    def path_access(self, path: str | Path, write: bool = False) -> bool:
        return not self.path_reason(path, write=write)

    @property
    def read_only(self) -> bool:
        return self.profile == "read-only" or any(ceiling.read_only for ceiling in self.ceilings)

    def tool_decision(self, name: str) -> Decision | None:
        decisions: list[Decision | None] = [rule.decision for rule in self.tools if fnmatchcase(name, rule.pattern)]
        decisions.extend(ceiling.tool_decision(name) for ceiling in self.ceilings)
        return max(decisions, key=_DECISION_ORDER.__getitem__) if decisions else None

    def network_decision(self, url: str) -> Decision:
        try:
            parsed = urlsplit(url)
            host = (parsed.hostname or "").lower().rstrip(".")
        except ValueError:
            return "deny"
        if parsed.scheme not in {"http", "https"} or not host:
            return "deny"
        decision = self.network.decision
        matching = [rule.decision for rule in self.network.rules if _domain_matches(host, rule.domain)]
        if matching and decision != "deny":
            decision = max(matching, key=_DECISION_ORDER.__getitem__)
        if any(_domain_matches(host, domain) for domain in self.network.denied_domains):
            decision = "deny"
        elif self.network.allowed_domains and not any(
            _domain_matches(host, domain) for domain in self.network.allowed_domains
        ):
            decision = "deny"
        return max((decision, *(ceiling.network_decision(url) for ceiling in self.ceilings)),
                   key=_DECISION_ORDER.__getitem__)

    def command_network_allowed(self) -> Decision:
        """Unknown subprocess destinations cannot satisfy domain filters without a proxy."""
        decision = "deny" if (self.network.allowed_domains or self.network.denied_domains
                              or self.network.rules) else self.network.decision
        return max((decision, *(ceiling.command_network_allowed() for ceiling in self.ceilings)),
                   key=_DECISION_ORDER.__getitem__)

    def intersection(self, other: ResolvedPermissions) -> ResolvedPermissions:
        """Restrict this policy with another live or saved policy without widening either."""
        if self.workspace != other.workspace:
            raise ValueError("cannot intersect permission policies from different workspaces")
        if self.revision == other.revision or any(ceiling.revision == other.revision for ceiling in self.ceilings):
            return self
        # Keep one copy of each declaration. Saved sessions may already contain
        # this policy; recursively nesting them on every reopen grows without bound.
        seen = {self.model_copy(update={"ceilings": (), "sources": ()}).revision}
        ceilings: list[ResolvedPermissions] = []
        sources = list(self.sources)

        def retain(policy: ResolvedPermissions) -> None:
            sources.extend(policy.sources)
            atom = policy.model_copy(update={"ceilings": ()})
            key = atom.model_copy(update={"sources": ()}).revision
            if key not in seen:
                seen.add(key)
                ceilings.append(atom)
            for ceiling in policy.ceilings:
                retain(ceiling)

        for ceiling in self.ceilings:
            retain(ceiling)
        retain(other)
        return self.model_copy(update={
            "ceilings": tuple(ceilings),
            "sources": tuple(dict.fromkeys(sources)),
        })

    def explain(self) -> dict[str, Any]:
        return {"sources": list(self.sources), "revision": self.revision, "approval": self.effective_approval,
                "effective": self.model_dump(mode="json")}


def _lexical(path: str | Path, base: Path) -> Path:
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
    return host.endswith(pattern[1:]) and host != pattern[2:] if pattern.startswith("*.") else host == pattern


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate permission policy key: {key}")
        result[key] = value
    return result


def load_permission_policy(path: str | Path, workspace: Path) -> ResolvedPermissions:
    """Load strict JSON/TOML, resolving declared paths relative to the config file."""
    config = Path(path).expanduser().resolve()
    if config.suffix.lower() == ".json":
        data = json.loads(config.read_text(), object_pairs_hook=_strict_json_object)
    elif config.suffix.lower() == ".toml":
        data = tomllib.loads(config.read_text())
    else:
        raise ValueError("permission policy file must have .json or .toml extension")
    policy = PermissionPolicy.model_validate(data)
    policy = policy.model_copy(update={"filesystem": tuple(
        PathRule(path=str(_lexical(rule.path, config.parent)), access=rule.access) for rule in policy.filesystem
    )})
    source = Path(path).expanduser().absolute()
    return policy.resolve(workspace).model_copy(update={"sources": tuple(dict.fromkeys((str(source), str(config))))})


def resolve_permission_sources(
    workspace: Path,
    permissions: PermissionPolicy | Path | str | None = None,
    managed_permissions: PermissionPolicy | Path | str | None = None,
) -> ResolvedPermissions:
    """Resolve user defaults and independent administrator ceilings without services."""
    def resolve(value: PermissionPolicy | Path | str | None) -> ResolvedPermissions:
        if value is None or isinstance(value, PermissionPolicy):
            return (value or PermissionPolicy()).resolve(workspace)
        return load_permission_policy(value, workspace)

    if permissions is None:
        for suffix in ("toml", "json"):
            candidate = user_config_directory() / f"permissions.{suffix}"
            if candidate.exists():
                permissions = candidate
                break
    effective = resolve(permissions)
    if managed_permissions is None:
        for suffix in ("toml", "json"):
            candidate = Path("/etc/praxis") / f"permissions.{suffix}"
            if candidate.exists():
                for path in (candidate, candidate.parent):
                    metadata = path.stat()
                    if metadata.st_uid != 0 or metadata.st_mode & 0o022:
                        raise ValueError("managed permissions must be root-owned and not group/world writable")
                managed_permissions = candidate
                break
    if managed_permissions is not None:
        effective = effective.intersection(resolve(managed_permissions))
    # Every automatically discovered directory is host configuration, including
    # absent higher-priority slots. Tools must not install next-session policy.
    automatic_directories = (user_config_directory(), Path("/etc/praxis"))
    protected = [str(directory.resolve()) for directory in automatic_directories]
    private = user_config_directory() / "credentials"
    return effective.model_copy(update={
        "sources": tuple(dict.fromkeys((*effective.sources, *protected))),
        "private_paths": tuple(dict.fromkeys((*effective.private_paths, str(private), str(private.resolve())))),
    })
