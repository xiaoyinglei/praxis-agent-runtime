"""``mcp_servers.yaml`` 的唯一加载、校验与信任门入口(3B 笔记 §2.1;Q1)。

**来源序(有意收缩,旧四源链 → 1+1)**:显式 ``path`` > ``PRAXIS_MCP_CONFIG`` >
``host_config_dir()/mcp_servers.yaml``。旧 ``resolve_product_mcp_config`` 的
个人配置 → workspace ``configs/`` → 产品内置链在本阶段坍缩为宿主配置目录:
workspace 与产品目录都不是 Praxis 的授权来源,因此 ``trust_workspace`` 与
"外部白名单"开关一并删除。显式参数/环境变量指向的文件缺失 ⇒ 带路径的
``ValueError``(裁决 R-T3-c 先例:拼写错误不得被静默吞掉);只有隐式默认
``host_config_dir()/mcp_servers.yaml`` 缺失才返回 ``()``。

**信任门**(旧 ``decide_mcp_config_trust`` 的个人配置卫生标准,与 T3 hooks
一致):文件与其父目录都属当前 uid、非 group/world 可写;配置必须位于
workspace 之外。来源分类只剩 ``user``(在 ``host_config_dir()`` 之内)与
``external``(显式参数/``PRAXIS_MCP_CONFIG`` 指向的其它宿主文件);两者
要求同样的卫生。``config_sha256`` 是配置字节漂移的唯一实现(T16/T17 复用)。

**默认值/校验语义**逐条对齐旧 ``_load_enabled_servers``(``runtime/mcp.py:
702-814``):``enabled`` 默认 False(disabled 条目在字段校验前跳过,与旧一致;
``include_disabled=True`` 供 CLI list/show,此时所有条目逐条完整校验)、
transport 默认 stdio、``tools_allowlist``/``allow_all_tools`` 至少其一、
``startup_timeout_seconds`` 默认 20.0 且正有限、HTTP ``read_only`` 默认
True、HTTP URL 仅 https 且无凭证/query/fragment(``validate_public_url``
已把端口限制在 80/443,与旧 ``validate_http_mcp_url`` 的 ``validate_public_url``
腿完全一致——不存在 8443 分歧)。``${VAR}`` 在加载期用 ``os.environ`` 展开
(stdio ``env`` 与 HTTP ``headers`` 值),缺失即抛且**只点名变量**;展开值
只进内存,任何错误消息都不得包含展开值。

**有意差异**:
- (R-PF-4)stdio ``cwd`` 必须落在 workspace 之内(相对路径按 workspace
  解析,不再相对配置文件目录):沙箱编译只授权 workspace+scratch。
- (Q6)OAuth 族键(``oauth``/``oauth_*``)在 3B 明确拒绝,不静默忽略。
- HTTP ``headers`` 值也支持 ``${VAR}`` 展开(加载期一次性),使密钥不必
  写进 YAML;展开值绝不进入消息,且 ``env``/``headers`` 字段 ``repr=False``
  (展开后的秘密不可被 repr 带出)。``${VAR}`` 与旧实现一样**没有转义形态**
  (在 ``$`` 前加反斜杠仍会被展开;要写字面量就不要使用该形状)。
- URL 存储形态取旧实现同源:``str(httpx.URL(...))``(host 小写/IDNA、
  IPv6 保留方括号、非 ASCII path 百分号编码、省略 443);``validate_public_url``
  仍只承担拒绝语义。T18 的 endpoint 逐字相等守卫因此与请求侧同源。
- (M3)server 名必须能规范化为非空串(全 unicode/全标点 ⇒ 加载期可诊断
  拒绝);这是相对旧实现"到运行期才拼出 ``mcp____*`` 垃圾"的有意收紧。
- ``StdioServerConfig``/``HttpServerConfig`` 字段逐一对齐旧两型,唯末位
  新增 ``enabled``(默认 True):默认加载仍只返回 enabled 条目,CLI 需要
  disabled 的可见状态时经 ``include_disabled=True`` 取完整记录。
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
import yaml

from praxis.adapters.mcp.names import normalize_mcp_name
from praxis.adapters.web.urls import PublicWebError, validate_public_url
from praxis.protocol.floor import host_config_dir

_ENV_VAR = "PRAXIS_MCP_CONFIG"
_DEFAULT_FILENAME = "mcp_servers.yaml"
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_PATH_ERROR = "MCP configuration must name an absolute file outside the workspace"
_HYGIENE_ERROR = (
    "MCP configuration must be owned by the current user and not group/world writable"
)
_READ_ERROR = "MCP configuration could not be read"
_YAML_ERROR = "MCP configuration must be valid YAML"
_SHAPE_ERROR = "MCP config must be a mapping"
_SERVERS_ERROR = "MCP config servers must be a list"
_ENTRY_ERROR = "MCP server entries must be mappings"
_OAUTH_ERROR = "MCP OAuth lands in phase 5; remove oauth/oauth_* keys from this server"
_URL_ERROR = "remote MCP requires a public HTTPS URL without credentials, query, or fragment"
_HEADERS_ERROR = "MCP HTTP headers must map strings to strings"
_CREDENTIAL_ERROR = "choose one MCP credential source"
_TOKEN_ENV_ERROR = "bearer_token_env_var must name an environment variable"


@dataclass(frozen=True, slots=True)
class StdioServerConfig:
    """一个 stdio MCP server 的已校验配置(旧 ``_StdioServerConfig`` 等价物)。"""

    name: str
    command: str
    args: tuple[str, ...]
    # 展开后的值可能含秘密:不得被 repr 带出(加载期值只进内存)。
    env: Mapping[str, str] = field(repr=False)
    cwd: str | None
    tools_allowlist: frozenset[str]
    allow_all_tools: bool
    startup_timeout_seconds: float
    # 解析层默认 enabled=False(旧语义);直接构造(测试/T16-T18)默认 True。
    # 经 load_mcp_configs 的条目永远携带显式状态。
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class HttpServerConfig:
    """一个 HTTP MCP server 的已校验配置(旧 ``_HTTPServerConfig`` 去掉 OAuth 族)。"""

    name: str
    url: str
    # 展开后的值可能含秘密:不得被 repr 带出(加载期值只进内存)。
    headers: Mapping[str, str] = field(repr=False)
    bearer_token_env_var: str | None
    keychain: bool
    read_only: bool
    tools_allowlist: frozenset[str]
    allow_all_tools: bool
    startup_timeout_seconds: float
    # 解析层默认 enabled=False(旧语义);直接构造(测试/T17/T18)默认 True。
    # 经 load_mcp_configs 的条目永远携带显式状态。
    enabled: bool = True


type McpServerConfig = StdioServerConfig | HttpServerConfig


@dataclass(frozen=True, slots=True)
class McpConfigTrustDecision:
    """对确切配置字节的授权结果(启动前后漂移对比的基线)。"""

    config_path: Path
    source: Literal["user", "external"]
    config_sha256: str


def config_sha256(path: Path) -> str:
    """配置字节的 sha256(调用期漂移比较的唯一实现)。

    读失败收敛为有界 ``ValueError``(权限/IO 错误不裸泄为原异常)。
    """
    try:
        data = Path(path).read_bytes()
    except OSError:
        raise ValueError(_READ_ERROR) from None
    return hashlib.sha256(data).hexdigest()


def decide_mcp_config_trust(
    config_path: Path,
    *,
    workspace: Path,
) -> McpConfigTrustDecision:
    """在加载任何配置字节之前授权确切路径与卫生。

    旧 ``decide_mcp_config_trust`` 的来源分类是 product/workspace/external/user;
    本阶段只保留 user/external(见模块 docstring),且不再有 ``trust_workspace``
    开关——workspace 内的配置文件一律拒绝。
    """
    root = _normalize_workspace(workspace)
    expanded = Path(config_path).expanduser()
    if not expanded.is_absolute():
        raise ValueError(_PATH_ERROR)
    try:
        source = expanded.resolve(strict=True)
    except FileNotFoundError:
        raise ValueError(
            f"MCP configuration does not exist: {_bounded_path(expanded)}"
        ) from None
    except (OSError, RuntimeError):
        raise ValueError(_PATH_ERROR) from None
    if source.is_relative_to(root):
        raise ValueError(_PATH_ERROR)
    _check_hygiene(source)
    host_root = host_config_dir().expanduser().resolve()
    kind: Literal["user", "external"] = (
        "user" if source.is_relative_to(host_root) else "external"
    )
    return McpConfigTrustDecision(
        config_path=source,
        source=kind,
        config_sha256=config_sha256(source),
    )


def resolve_mcp_config_path(
    workspace: Path,
    path: Path | None = None,
) -> Path | None:
    """按来源序返回已通过信任门的配置路径;隐式默认缺失返回 ``None``。

    显式 ``path``/``PRAXIS_MCP_CONFIG`` 指向不存在的文件 ⇒ ``ValueError``;
    ``host_config_dir()/mcp_servers.yaml`` 不存在 ⇒ ``None``。
    """
    root = _normalize_workspace(workspace)
    candidate = _select_candidate(path)
    if candidate is None:
        return None
    return decide_mcp_config_trust(candidate, workspace=root).config_path


def load_mcp_configs(
    workspace: Path,
    path: Path | None = None,
    *,
    include_disabled: bool = False,
) -> tuple[McpServerConfig, ...]:
    """加载、校验并返回 MCP server 配置。

    Parameters:
        workspace: 真实工作区(realpath 规范化)。配置与 stdio ``cwd`` 都必须
            落在其之外/之内(见模块 docstring);绝不用进程 cwd 代替。
        path: 显式配置路径(优先于 ``PRAXIS_MCP_CONFIG``)。
        include_disabled: ``False``(默认)逐字保留旧语义——disabled 条目在
            字段校验前跳过,只返回 enabled 配置;``True`` 时所有条目完整校验
            并保留 ``enabled`` 状态(CLI list/show 用)。

    Missing-file semantics:
        显式 ``path``/``PRAXIS_MCP_CONFIG`` 指向不存在的文件 ⇒ 带路径
        ``ValueError``;只有隐式默认 ``host_config_dir()/mcp_servers.yaml``
        缺失才返回 ``()``。

    Raises:
        ValueError: 路径/缺失/信任门/YAML 形状/条目校验任一失败。
    """
    root = _normalize_workspace(workspace)
    source = resolve_mcp_config_path(root, path)
    if source is None:
        return ()
    return _parse_servers(_read_payload(source), source, root, include_disabled)


def _normalize_workspace(workspace: Path) -> Path:
    try:
        return Path(workspace).expanduser().resolve()
    except (OSError, RuntimeError):
        raise ValueError("workspace must be an absolute directory") from None


def _select_candidate(path: Path | None) -> Path | None:
    if path is not None:
        return _prepare_candidate(Path(path), explicit=True)
    configured = os.environ.get(_ENV_VAR, "").strip()
    if configured:
        return _prepare_candidate(Path(configured), explicit=True)
    return _prepare_candidate(host_config_dir() / _DEFAULT_FILENAME, explicit=False)


def _prepare_candidate(candidate: Path, *, explicit: bool) -> Path | None:
    expanded = candidate.expanduser()
    if not expanded.is_absolute():
        raise ValueError(_PATH_ERROR)
    if not expanded.is_file():
        if explicit:
            raise ValueError(
                f"MCP configuration does not exist: {_bounded_path(expanded)}"
            )
        return None
    return expanded


def _check_hygiene(source: Path) -> None:
    """文件与其父目录都必须是当前用户所有且非 group/world 可写。"""
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


def _read_payload(source: Path) -> object:
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raise ValueError(_READ_ERROR) from None
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError:
        # 固定文案:YAML 解析错误会带出文件片段,可能包含展开前的敏感值。
        raise ValueError(_YAML_ERROR) from None
    # 旧 ``yaml.safe_load(...) or {}``:falsy 但非 None 的 payload(``[]``/``""``)
    # 也归一到 ``{}``,由同一 "servers must be a list" 文案拒绝。
    return payload or {}


def _parse_servers(
    payload: object,
    source: Path,
    workspace: Path,
    include_disabled: bool,
) -> tuple[McpServerConfig, ...]:
    if not isinstance(payload, Mapping):
        raise ValueError(_SHAPE_ERROR)
    raw_servers = payload.get("servers", ())
    if not isinstance(raw_servers, list):
        raise ValueError(_SERVERS_ERROR)

    configs: list[McpServerConfig] = []
    names: set[str] = set()
    for raw in raw_servers:
        if not isinstance(raw, Mapping):
            raise ValueError(_ENTRY_ERROR)
        enabled = _boolean(raw, "enabled", default=False)
        if not enabled and not include_disabled:
            continue
        name = _required_text(raw, "name")
        if not normalize_mcp_name(name):
            # M3:边界拒绝,不让运行期拼出空名字的通配/工具名。
            raise ValueError(
                f"MCP server name must normalize to a non-empty value: {_bounded(name)}"
            )
        if name in names:
            raise ValueError(f"duplicate MCP server: {_bounded(name)}")
        names.add(name)
        _reject_oauth_keys(raw)
        transport = str(raw.get("transport", "stdio"))
        if transport not in {"stdio", "http"}:
            raise ValueError(
                f"unsupported MCP transport for {_bounded(name)}: {_bounded(transport)}"
            )
        allowlist = frozenset(
            _string_sequence(raw.get("tools_allowlist", ()), field="tools_allowlist")
        )
        allow_all = _boolean(raw, "allow_all_tools", default=False)
        if not allowlist and not allow_all:
            raise ValueError(
                f"MCP server {_bounded(name)!r} requires tools_allowlist or allow_all_tools"
            )
        timeout = _timeout(raw)
        if transport == "http":
            configs.append(
                _http_config(
                    raw,
                    name=name,
                    enabled=enabled,
                    allowlist=allowlist,
                    allow_all=allow_all,
                    timeout=timeout,
                )
            )
            continue
        configs.append(
            _stdio_config(
                raw,
                name=name,
                enabled=enabled,
                allowlist=allowlist,
                allow_all=allow_all,
                timeout=timeout,
                workspace=workspace,
            )
        )
    return tuple(configs)


def _reject_oauth_keys(raw: Mapping[object, object]) -> None:
    """Q6:OAuth 族键在 3B 明确拒绝(不是静默忽略)。"""
    if any(str(key).casefold().startswith("oauth") for key in raw):
        raise ValueError(_OAUTH_ERROR)


def _stdio_config(
    raw: Mapping[object, object],
    *,
    name: str,
    enabled: bool,
    allowlist: frozenset[str],
    allow_all: bool,
    timeout: float,
    workspace: Path,
) -> StdioServerConfig:
    return StdioServerConfig(
        name=name,
        command=_required_text(raw, "command"),
        args=_string_sequence(raw.get("args", ()), field="args"),
        env=_environment(raw.get("env", {}), server_name=name),
        cwd=_cwd(raw.get("cwd"), workspace=workspace, server_name=name),
        tools_allowlist=allowlist,
        allow_all_tools=allow_all,
        startup_timeout_seconds=timeout,
        enabled=enabled,
    )


def _http_config(
    raw: Mapping[object, object],
    *,
    name: str,
    enabled: bool,
    allowlist: frozenset[str],
    allow_all: bool,
    timeout: float,
) -> HttpServerConfig:
    url = _validate_http_url(_required_text(raw, "url"))
    headers = _headers(raw.get("headers", {}), server_name=name)
    token_env = raw.get("bearer_token_env_var")
    if token_env is not None and (
        not isinstance(token_env, str) or not _ENV_NAME.fullmatch(token_env)
    ):
        raise ValueError(_TOKEN_ENV_ERROR)
    keychain = _boolean(raw, "keychain", default=False)
    if sum((token_env is not None, keychain)) > 1:
        raise ValueError(_CREDENTIAL_ERROR)
    return HttpServerConfig(
        name=name,
        url=url,
        headers=headers,
        bearer_token_env_var=token_env,
        keychain=keychain,
        read_only=_boolean(raw, "read_only", default=True),
        tools_allowlist=allowlist,
        allow_all_tools=allow_all,
        startup_timeout_seconds=timeout,
        enabled=enabled,
    )


def _validate_http_url(value: str) -> str:
    """旧 ``validate_http_mcp_url`` 的等价物:仅 https、无凭证/query/fragment。

    端口限制在 ``validate_public_url`` 内部(``{None,80,443}``),与旧实现
    所调的同一函数一致;存储形态直接取 ``str(httpx.URL(...))``(旧实现同源):
    host 小写/IDNA、IPv6 保留方括号、非 ASCII path 百分号编码、省略 443。
    """
    try:
        raw = httpx.URL(value)
        if raw.fragment or raw.query:
            raise ValueError
        _host, _port, _target = validate_public_url(value)
        if raw.scheme != "https":
            raise ValueError
        return str(raw)
    except (PublicWebError, ValueError, httpx.InvalidURL):
        raise ValueError(_URL_ERROR) from None


def _cwd(value: object, *, workspace: Path, server_name: str) -> str | None:
    """R-PF-4:stdio cwd 必须在 workspace 之内(相对路径按 workspace 解析)。"""
    if value is None:
        return None
    cwd = Path(str(value)).expanduser()
    if not cwd.is_absolute():
        cwd = workspace / cwd
    try:
        resolved = cwd.resolve()
    except (OSError, RuntimeError):
        raise ValueError(_cwd_error(server_name)) from None
    if not resolved.is_relative_to(workspace):
        raise ValueError(_cwd_error(server_name))
    return str(resolved)


def _cwd_error(server_name: str) -> str:
    return f"MCP server {_bounded(server_name)!r} cwd must be inside the workspace"


def _timeout(raw: Mapping[object, object]) -> float:
    value = raw.get("startup_timeout_seconds", 20.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("startup_timeout_seconds must be numeric")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("startup_timeout_seconds must be positive")
    return timeout


def _required_text(raw: Mapping[object, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MCP server {field} must be a non-empty string")
    return value.strip()


def _boolean(
    raw: Mapping[object, object],
    field: str,
    *,
    default: bool,
) -> bool:
    value = raw.get(field, default)
    if type(value) is not bool:
        raise ValueError(f"MCP server {field} must be a boolean")
    return value


def _string_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"MCP server {field} must be a list")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"MCP server {field} contains an empty value")
    return result


def _environment(value: object, *, server_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"MCP server {_bounded(server_name)!r} env must be a mapping"
        )
    return {
        str(key): _expand_env(str(raw_value), server_name=server_name)
        for key, raw_value in value.items()
    }


def _headers(value: object, *, server_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not isinstance(raw_value, str)
        for key, raw_value in value.items()
    ):
        raise ValueError(_HEADERS_ERROR)
    return {
        key: _expand_env(raw_value, server_name=server_name)
        for key, raw_value in value.items()
    }


def _expand_env(text: str, *, server_name: str) -> str:
    """加载期展开 ``${VAR}``;缺失只点名变量,绝不回显任何值。"""

    def replace(match: re.Match[str]) -> str:
        env_name = match.group(1)
        if env_name not in os.environ:
            raise ValueError(
                f"MCP server {_bounded(server_name)!r} requires environment variable {env_name}"
            )
        return os.environ[env_name]

    return _ENV_REFERENCE.sub(replace, text)


def _bounded(value: object, limit: int = 64) -> str:
    text = value if isinstance(value, str) else type(value).__name__
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _bounded_path(path: Path, limit: int = 256) -> str:
    text = " ".join(str(path).split())
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


__all__ = [
    "HttpServerConfig",
    "McpConfigTrustDecision",
    "McpServerConfig",
    "StdioServerConfig",
    "config_sha256",
    "decide_mcp_config_trust",
    "load_mcp_configs",
    "resolve_mcp_config_path",
]
