"""hooks.json 的唯一加载与校验入口(设计 §5.14;Q1 库外配置;3B 笔记 §4)。

路径来源序:**显式参数 > ``PRAXIS_HOOKS_CONFIG`` > ``host_config_dir()/hooks.json``**。
``workspace`` 是调用方给出的真实工作区(realpath 规范化);配置必须落在它之外,
**绝不**用 ``Path.cwd()`` 代替。缺失语义(裁决 R-T3-c):显式参数或环境变量
指向的文件不存在 ⇒ 带路径的 ``ValueError``(边界上尽早失败,避免拼写错误被
静默吞掉);只有隐式默认路径 ``host_config_dir()/hooks.json`` 不存在才返回 ``()``。
每个 ``HookSpec.directory`` 都是配置文件父目录(绝对路径),供 T4 作为 spawn cwd。

配置必须通过旧 ``decide_mcp_config_trust`` 的个人配置卫生标准:文件与其目录
都属当前 uid、非 group/world 可写(因此直接放在 world-writable 目录如 /tmp
会被拒绝)。

拒绝面逐条对齐旧 ``agent_runtime/hooks.py:110-153``:
- 绝对路径、``resolve(strict=True)``、workspace 之外、≤64 KiB(读 65_537 判越界);
- JSON 数组 ≤``MAX_HOOKS`` 项;条目必填 event/name/argv,可选
  revision/timeout_seconds/tool_names/effects,未知键直接报错;
- ``Stop`` 事件在加载期明确拒绝(阶段 6),不静默忽略;
- ``argv[0]`` 绝对可执行且不在 workspace 内;
- ``revision`` = ``sha256(canonical JSON({argv,directory,revision}))[:16]``
  ——本模块是唯一实现(旧 ``canonical_hash`` 的等价物);
- ``(event,name)`` 在事件内唯一。

错误消息固定、有界、可诊断;绝不回显配置 body 或底层异常文本。
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import cast

from praxis.protocol.floor import host_config_dir
from praxis.protocol.hooks import (
    MAX_HOOKS,
    MAX_HOOKS_CONFIG_BYTES,
    HookEvent,
    HookSpec,
)

_ENV_VAR = "PRAXIS_HOOKS_CONFIG"
_DEFAULT_FILENAME = "hooks.json"
_REQUIRED_KEYS = frozenset({"event", "name", "argv"})
_OPTIONAL_KEYS = frozenset({"revision", "timeout_seconds", "tool_names", "effects"})

_PATH_ERROR = "PRAXIS_HOOKS_CONFIG must name an absolute file outside the workspace"
_READ_ERROR = "hook configuration could not be read"
_JSON_ERROR = "host hook configuration must be valid JSON"
_SHAPE_ERROR = "host hook configuration must be an array of at most 16 entries"
_ENTRY_ERROR = "invalid host hook entry; event, name and argv are required"
_ARGV_ERROR = "hook argv requires an absolute executable outside the workspace"
_REVISION_ERROR = "command hook revision must have 1–128 characters"
_HYGIENE_ERROR = (
    "hook configuration must be owned by the current user and not group/world writable"
)
_STOP_ERROR = "stop hooks land in phase 6"
_UNIQUE_ERROR = "hook names must be unique within each event"
_HOOKS_ERROR = "hooks must contain at most 16 HookSpec registrations"

# 读上限哨兵:多读 1 字节用于判定 >64 KiB。
_MAX_CONTENT_BYTES = MAX_HOOKS_CONFIG_BYTES - 1

_EVENTS: tuple[str, ...] = ("PreToolUse", "PostToolUse")


def load_host_hooks(workspace: Path, path: Path | None = None) -> tuple[HookSpec, ...]:
    """加载并校验 command hook 配置。

    Parameters:
        workspace: 真实工作区(realpath 规范化)。配置与 ``argv[0]`` 都必须
            位于其之外;绝不用进程 cwd 代替。
        path: 显式配置路径(优先于环境变量)。``None`` 时依次看
            ``PRAXIS_HOOKS_CONFIG`` 与 ``host_config_dir()/hooks.json``。

    Missing-file semantics:
        显式 ``path`` / ``PRAXIS_HOOKS_CONFIG`` 指向不存在的文件 ⇒ 带路径的
        ``ValueError``;只有隐式默认 ``host_config_dir()/hooks.json`` 缺失
        才返回 ``()``。

    Raises:
        ValueError: 路径/缺失/大小/JSON 形状/条目/卫生任一校验失败。
    """
    root = _normalize_workspace(workspace)
    source = _select_source(path, root)
    if source is None:
        return ()
    return _load(source, root)


def _normalize_workspace(workspace: Path) -> Path:
    try:
        return Path(workspace).expanduser().resolve()
    except (OSError, RuntimeError):
        raise ValueError("workspace must be an absolute directory") from None


def _select_source(path: Path | None, workspace: Path) -> Path | None:
    if path is not None:
        return _prepare_source(Path(path), workspace, explicit=True)
    configured = os.environ.get(_ENV_VAR)
    if configured:
        return _prepare_source(Path(configured), workspace, explicit=True)
    return _prepare_source(host_config_dir() / _DEFAULT_FILENAME, workspace, explicit=False)


def _prepare_source(candidate: Path, workspace: Path, *, explicit: bool) -> Path | None:
    expanded = candidate.expanduser()
    if not expanded.is_absolute():
        raise ValueError(_PATH_ERROR)
    try:
        source = expanded.resolve(strict=True)
    except FileNotFoundError:
        if explicit:
            raise ValueError(
                f"hook configuration does not exist: {_bounded_path(expanded)}"
            ) from None
        return None
    except (OSError, RuntimeError):
        raise ValueError(_PATH_ERROR) from None
    if source.is_relative_to(workspace):
        raise ValueError(_PATH_ERROR)
    _check_hygiene(source)
    return source


def _check_hygiene(source: Path) -> None:
    """文件与其目录都必须是当前用户所有且非 group/world 可写。"""
    try:
        metadata = source.stat()
        parent = source.parent.stat()
    except OSError:
        raise ValueError(_HYGIENE_ERROR) from None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o022
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
        or parent.st_mode & 0o022
    ):
        raise ValueError(_HYGIENE_ERROR)


def _load(source: Path, workspace: Path) -> tuple[HookSpec, ...]:
    try:
        with source.open("rb") as handle:
            raw = handle.read(MAX_HOOKS_CONFIG_BYTES)
    except OSError:
        raise ValueError(_READ_ERROR) from None
    if len(raw) > _MAX_CONTENT_BYTES:
        raise ValueError("host hook configuration exceeds 64 KiB")

    try:
        entries = json.loads(raw)
    except ValueError:
        raise ValueError(_JSON_ERROR) from None
    if not isinstance(entries, list) or len(entries) > MAX_HOOKS:
        raise ValueError(_SHAPE_ERROR)

    hooks = tuple(_build_spec(entry, source.parent, workspace) for entry in entries)
    return _validate_hooks(hooks)


def _build_spec(entry: object, directory: Path, workspace: Path) -> HookSpec:
    if not isinstance(entry, dict):
        raise ValueError(_ENTRY_ERROR)
    keys = entry.keys()
    if not _REQUIRED_KEYS <= keys or keys - (_REQUIRED_KEYS | _OPTIONAL_KEYS):
        raise ValueError(_ENTRY_ERROR)

    event = entry["event"]
    if event == "Stop":
        raise ValueError(_STOP_ERROR)
    if not isinstance(event, str) or event not in _EVENTS:
        raise ValueError(f"invalid hook event: {_bounded(event)}")

    argv = _validate_argv(entry["argv"], workspace)
    declared_revision = entry.get("revision", "1")
    if (
        not isinstance(declared_revision, str)
        or not declared_revision.strip()
        or len(declared_revision) > 128
    ):
        raise ValueError(_REVISION_ERROR)

    try:
        executable = str(Path(argv[0]).resolve(strict=True))
    except (OSError, RuntimeError):
        raise ValueError(_ARGV_ERROR) from None
    command = (executable, *argv[1:])

    return HookSpec(
        name=entry["name"],
        revision=_canonical_revision(command, directory, declared_revision),
        event=cast(HookEvent, event),
        argv=command,
        directory=str(directory),
        tool_names=entry.get("tool_names", ()),
        effects=entry.get("effects", ()),
        timeout_seconds=entry.get("timeout_seconds", 5.0),
    )


def _validate_argv(value: object, workspace: Path) -> tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(argument, str) or "\0" in argument for argument in value)
        or not Path(value[0]).is_absolute()
        or Path(value[0]).resolve().is_relative_to(workspace)
    ):
        raise ValueError(_ARGV_ERROR)
    return tuple(value)


def _canonical_revision(
    command: tuple[str, ...], directory: Path, declared_revision: str
) -> str:
    """``sha256(canonical JSON({argv,directory,revision}))`` 的前 16 位。"""
    payload = json.dumps(
        {"argv": list(command), "directory": str(directory), "revision": declared_revision},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _validate_hooks(hooks: tuple[HookSpec, ...]) -> tuple[HookSpec, ...]:
    if len(hooks) > MAX_HOOKS or any(not isinstance(hook, HookSpec) for hook in hooks):
        raise ValueError(_HOOKS_ERROR)
    keys = {(hook.event, hook.name) for hook in hooks}
    if len(keys) != len(hooks):
        raise ValueError(_UNIQUE_ERROR)
    return hooks


def _bounded(value: object) -> str:
    text = value if isinstance(value, str) else type(value).__name__
    text = " ".join(text.split())
    return text if len(text) <= 64 else f"{text[:63]}…"


def _bounded_path(path: Path, limit: int = 256) -> str:
    text = " ".join(str(path).split())
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


__all__ = ["load_host_hooks"]
