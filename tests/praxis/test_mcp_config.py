"""T15:MCP 名字唯一拼装 + ``mcp_servers.yaml`` 加载/信任门/漂移。

钉住:
1. ``normalize_mcp_name``/``canonical_tool_name``/``server_wildcard`` 与旧
   ``tools/integrations/mcp.py:70-83`` 及旧 6 处拼装调用点(``runtime/mcp.py:
   219/441``、``runtime/mcp_catalog.py:256``、``harness/session.py:141/1367``)
   的等价形状;
2. 默认值逐条(``enabled`` 默认 False、transport 默认 stdio、allowlist/
   allow_all_tools 至少其一、``startup_timeout_seconds`` 默认 20.0 正有限、
   HTTP ``read_only`` 默认 True);
3. ``${VAR}`` 加载期展开:缺失只点名变量、展开值绝不进入消息/文件外泄面;
4. HTTP URL 仅 https、无凭证/query/fragment、端口 80/443 纪律;
5. 信任门(属主 uid、非 group/world 可写、workspace 外)与来源分类
   user/external;来源序只有显式 > ``PRAXIS_MCP_CONFIG`` > 宿主默认文件;
6. ``config_sha256`` 的稳定性与字节漂移;
7. OAuth 族键的明确阶段 5 拒绝(Q6)。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from praxis.adapters.mcp.config import (
    HttpServerConfig,
    StdioServerConfig,
    config_sha256,
    decide_mcp_config_trust,
    load_mcp_configs,
)
from praxis.adapters.mcp.names import (
    canonical_tool_name,
    normalize_mcp_name,
    server_wildcard,
)
from praxis.protocol.floor import host_config_dir

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

_ENV = "PRAXIS_MCP_CONFIG"


@pytest.fixture(autouse=True)
def _isolated_host_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """绝不读写真实宿主目录;``PRAXIS_MCP_CONFIG`` 必须为空。"""
    config_dir = tmp_path / "praxis-config"
    config_dir.mkdir(mode=0o700, exist_ok=True)
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv(_ENV, raising=False)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    return root


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    root = tmp_path / "outside"
    root.mkdir(mode=0o700, exist_ok=True)
    root.chmod(0o700)
    return root


def _write_config(
    directory: Path,
    payload: object,
    *,
    mode: int = 0o600,
    filename: str = "mcp_servers.yaml",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    path = directory / filename
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    path.chmod(mode)
    return path


def _write_host_config(payload: object) -> Path:
    return _write_config(host_config_dir(), payload)


def _stdio(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": "files",
        "enabled": True,
        "command": "/usr/bin/true",
        "tools_allowlist": ["read_file"],
    }
    entry.update(overrides)
    return entry


def _http(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": "remote",
        "enabled": True,
        "transport": "http",
        "url": "https://example.com/mcp",
        "allow_all_tools": True,
    }
    entry.update(overrides)
    return entry


def _single(workspace: Path, entry: dict[str, Any]) -> Any:
    path = _write_config(workspace.parent / "outside", {"servers": [entry]})
    configs = load_mcp_configs(workspace, path)
    assert len(configs) == 1
    return configs[0]


# ---------------------------------------------------------------------------
# names:唯一拼装点
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Filesystem Server", "filesystem_server"),
        ("  GitHub-API  ", "github_api"),
        ("my mcp: tool (v2)", "my_mcp_tool_v2"),
        ("a--b", "a_b"),
        ("a__b", "a_b"),
        ("a___b___c", "a_b_c"),
        ("UPPER.Case/Mixed", "upper_case_mixed"),
        ("___", ""),
        ("", ""),
        ("名字", ""),
    ],
)
def test_normalize_mcp_name_vectors(raw: str, expected: str) -> None:
    assert normalize_mcp_name(raw) == expected


def test_normalize_mcp_name_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        normalize_mcp_name(1)  # type: ignore[arg-type]


def test_canonical_tool_name_matches_legacy_assembly_site() -> None:
    # 旧 tools/integrations/mcp.py:78-83 的形状。
    assert (
        canonical_tool_name("Filesystem Server", "Read File")
        == "mcp__filesystem_server__read_file"
    )
    assert canonical_tool_name("github-api", "search_repos") == "mcp__github_api__search_repos"


def test_canonical_tool_name_rejects_empty_normalization() -> None:
    with pytest.raises(ValueError):
        canonical_tool_name("!!!", "read")
    with pytest.raises(ValueError):
        canonical_tool_name("files", "???")


def test_server_wildcard_matches_legacy_assembly_sites() -> None:
    # 旧 runtime/mcp.py:219/441 与 runtime/mcp_catalog.py:256 的形状。
    assert server_wildcard("Filesystem Server") == "mcp__filesystem_server__*"
    assert server_wildcard("github-api") == "mcp__github_api__*"


def test_server_wildcard_rejects_empty_normalization() -> None:
    with pytest.raises(ValueError):
        server_wildcard("---")


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------


def test_stdio_defaults(workspace: Path) -> None:
    config = _single(workspace, _stdio())
    assert isinstance(config, StdioServerConfig)
    assert config.name == "files"
    assert config.command == "/usr/bin/true"
    assert config.args == ()
    assert config.env == {}
    assert config.cwd is None
    assert config.tools_allowlist == frozenset({"read_file"})
    assert config.allow_all_tools is False
    assert config.startup_timeout_seconds == 20.0
    assert config.enabled is True


def test_http_defaults(workspace: Path) -> None:
    config = _single(workspace, _http())
    assert isinstance(config, HttpServerConfig)
    assert config.name == "remote"
    assert config.url == "https://example.com/mcp"
    assert config.headers == {}
    assert config.bearer_token_env_var is None
    assert config.keychain is False
    assert config.read_only is True
    assert config.tools_allowlist == frozenset()
    assert config.allow_all_tools is True
    assert config.startup_timeout_seconds == 20.0
    assert config.enabled is True


def test_enabled_defaults_false_and_disabled_entries_are_filtered(workspace: Path) -> None:
    entry = _stdio()
    del entry["enabled"]
    path = _write_config(workspace.parent / "outside", {"servers": [entry]})
    assert load_mcp_configs(workspace, path) == ()

    (config,) = load_mcp_configs(workspace, path, include_disabled=True)
    assert config.enabled is False


def test_include_disabled_preserves_order_and_flags(workspace: Path) -> None:
    payload = {
        "servers": [
            _http(enabled=False, name="off"),
            _stdio(name="on"),
        ]
    }
    path = _write_config(workspace.parent / "outside", payload)
    (enabled,) = load_mcp_configs(workspace, path)
    assert enabled.name == "on"

    all_configs = load_mcp_configs(workspace, path, include_disabled=True)
    assert [(c.name, c.enabled) for c in all_configs] == [("off", False), ("on", True)]


def test_disabled_entries_skip_field_validation_by_default(workspace: Path) -> None:
    """旧语义:disabled 在字段校验前跳过;include_disabled 则完整校验。"""
    broken = {"name": "later", "enabled": False, "transport": "sse"}
    path = _write_config(workspace.parent / "outside", {"servers": [broken]})
    assert load_mcp_configs(workspace, path) == ()
    with pytest.raises(ValueError, match="unsupported MCP transport"):
        load_mcp_configs(workspace, path, include_disabled=True)


# ---------------------------------------------------------------------------
# per-entry validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"name": ""}, "name must be a non-empty string"),
        ({"name": "   "}, "name must be a non-empty string"),
        ({"name": None}, "name must be a non-empty string"),
        ({"enabled": "yes"}, "enabled must be a boolean"),
        ({"transport": "sse"}, "unsupported MCP transport"),
        ({"transport": 3}, "unsupported MCP transport"),
        ({"tools_allowlist": (), "allow_all_tools": False}, "requires tools_allowlist"),
        ({"tools_allowlist": "read_file"}, "tools_allowlist must be a list"),
        ({"tools_allowlist": [""]}, "tools_allowlist contains an empty value"),
        ({"args": "one"}, "args must be a list"),
        ({"env": ["A=1"]}, "env must be a mapping"),
        ({"startup_timeout_seconds": 0}, "startup_timeout_seconds must be positive"),
        ({"startup_timeout_seconds": -1.5}, "startup_timeout_seconds must be positive"),
        ({"startup_timeout_seconds": True}, "startup_timeout_seconds must be numeric"),
        ({"startup_timeout_seconds": "20"}, "startup_timeout_seconds must be numeric"),
        ({"startup_timeout_seconds": float("inf")}, "startup_timeout_seconds must be positive"),
        ({"startup_timeout_seconds": float("nan")}, "startup_timeout_seconds must be positive"),
    ],
)
def test_stdio_entry_validation(
    workspace: Path, overrides: dict[str, Any], match: str
) -> None:
    path = _write_config(workspace.parent / "outside", {"servers": [_stdio(**overrides)]})
    with pytest.raises(ValueError, match=match):
        load_mcp_configs(workspace, path)


def test_duplicate_names_are_rejected(workspace: Path) -> None:
    payload = {"servers": [_stdio(), _stdio()]}
    path = _write_config(workspace.parent / "outside", payload)
    with pytest.raises(ValueError, match="duplicate MCP server: files"):
        load_mcp_configs(workspace, path)


def test_duplicate_disabled_name_only_conflicts_when_included(workspace: Path) -> None:
    payload = {"servers": [_stdio(), _http(name="files", enabled=False)]}
    path = _write_config(workspace.parent / "outside", payload)
    assert [c.name for c in load_mcp_configs(workspace, path)] == ["files"]
    with pytest.raises(ValueError, match="duplicate MCP server"):
        load_mcp_configs(workspace, path, include_disabled=True)


def test_startup_timeout_accepts_positive_numbers(workspace: Path) -> None:
    config = _single(workspace, _stdio(startup_timeout_seconds=5))
    assert config.startup_timeout_seconds == 5.0


def test_relative_cwd_resolves_inside_workspace(workspace: Path) -> None:
    config = _single(workspace, _stdio(cwd="sub"))
    assert config.cwd == str((workspace / "sub").resolve())


def test_absolute_cwd_inside_workspace_is_accepted(workspace: Path) -> None:
    inside = workspace / "tools"
    inside.mkdir()
    config = _single(workspace, _stdio(cwd=str(inside)))
    assert config.cwd == str(inside.resolve())


def test_cwd_outside_workspace_is_rejected(workspace: Path, outside: Path) -> None:
    with pytest.raises(ValueError, match="cwd must be inside the workspace"):
        _single(workspace, _stdio(cwd=str(outside)))
    with pytest.raises(ValueError, match="cwd must be inside the workspace"):
        _single(workspace, _stdio(cwd="../outside"))


# ---------------------------------------------------------------------------
# HTTP URL / headers / credential sources
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/mcp",
        "https://example.com:8443/mcp",
        "https://user:secret@example.com/mcp",
        "https://example.com/mcp?x=1",
        "https://example.com/mcp#fragment",
        "http://localhost/mcp",
        "https://127.0.0.1/mcp",
        "https://[::1]/mcp",
        "not a url",
    ],
)
def test_http_url_rejections(workspace: Path, url: str) -> None:
    path = _write_config(workspace.parent / "outside", {"servers": [_http(url=url)]})
    with pytest.raises(ValueError, match="public HTTPS URL"):
        load_mcp_configs(workspace, path)


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("https://example.com/mcp", "https://example.com/mcp"),
        ("https://example.com", "https://example.com"),
        ("https://example.com/", "https://example.com/"),
        ("https://example.com:443/mcp", "https://example.com/mcp"),
        ("https://example.com:80/mcp", "https://example.com:80/mcp"),
        ("HTTPS://Example.COM/mcp", "https://example.com/mcp"),
        ("https://bücher.example/mcp", "https://xn--bcher-kva.example/mcp"),
    ],
)
def test_http_url_normalization(workspace: Path, raw: str, normalized: str) -> None:
    config = _single(workspace, _http(url=raw))
    assert config.url == normalized


def test_http_headers_must_map_strings_to_strings(workspace: Path) -> None:
    path = _write_config(
        workspace.parent / "outside", {"servers": [_http(headers={"X-Count": 1})]}
    )
    with pytest.raises(ValueError, match="headers must map strings to strings"):
        load_mcp_configs(workspace, path)


def test_http_headers_expand_env_references(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRAXIS_T15_AUTH", "Bearer token-123")
    config = _single(
        workspace,
        _http(headers={"Authorization": "${PRAXIS_T15_AUTH}", "X-Static": "plain"}),
    )
    assert config.headers == {"Authorization": "Bearer token-123", "X-Static": "plain"}


@pytest.mark.parametrize("bad", ["1BAD", "BAD-NAME", "", 7])
def test_bearer_token_env_var_must_be_an_env_name(workspace: Path, bad: object) -> None:
    path = _write_config(workspace.parent / "outside", {"servers": [_http(bearer_token_env_var=bad)]})
    with pytest.raises(ValueError, match="bearer_token_env_var must name an environment variable"):
        load_mcp_configs(workspace, path)


def test_credential_sources_are_mutually_exclusive(workspace: Path) -> None:
    path = _write_config(
        workspace.parent / "outside",
        {"servers": [_http(bearer_token_env_var="MCP_TOKEN", keychain=True)]},
    )
    with pytest.raises(ValueError, match="choose one MCP credential source"):
        load_mcp_configs(workspace, path)


def test_keychain_and_read_only_flags(workspace: Path) -> None:
    config = _single(workspace, _http(keychain=True, read_only=False))
    assert config.keychain is True
    assert config.read_only is False


# ---------------------------------------------------------------------------
# ${VAR} expansion:加载期、缺名报错、值不外泄
# ---------------------------------------------------------------------------


def test_env_values_expand_at_load_time_into_memory(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRAXIS_T15_HOME", "/opt/tool")
    config = _single(
        workspace,
        _stdio(env={"BIN": "${PRAXIS_T15_HOME}/bin", "LITERAL": "x"}),
    )
    assert config.env == {"BIN": "/opt/tool/bin", "LITERAL": "x"}
    # 加载期一次性展开:之后删掉环境变量,内存值不变(值只进内存)。
    monkeypatch.delenv("PRAXIS_T15_HOME")
    assert config.env["BIN"] == "/opt/tool/bin"


def test_missing_env_reference_names_variable_but_not_values(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRAXIS_T15_PRESENT", "supersecret-value")
    path = _write_config(
        workspace.parent / "outside",
        {
            "servers": [
                _stdio(env={"TOKEN": "${PRAXIS_T15_PRESENT}/${PRAXIS_T15_ABSENT}"})
            ]
        },
    )
    with pytest.raises(ValueError) as excinfo:
        load_mcp_configs(workspace, path)
    message = str(excinfo.value)
    assert "PRAXIS_T15_ABSENT" in message
    assert "supersecret-value" not in message


def test_missing_header_env_reference_names_variable(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRAXIS_T15_PRESENT", "supersecret-value")
    path = _write_config(
        workspace.parent / "outside",
        {
            "servers": [
                _http(headers={"Authorization": "${PRAXIS_T15_PRESENT}${PRAXIS_T15_ABSENT}"})
            ]
        },
    )
    with pytest.raises(ValueError) as excinfo:
        load_mcp_configs(workspace, path)
    message = str(excinfo.value)
    assert "PRAXIS_T15_ABSENT" in message
    assert "supersecret-value" not in message


def test_expanded_values_never_appear_in_later_errors(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRAXIS_T15_SECRET", "supersecret-value")
    payload = {
        "servers": [
            _stdio(env={"TOKEN": "${PRAXIS_T15_SECRET}"}),
            _stdio(),  # 同名 ⇒ 重复错误发生在第一个条目展开之后
        ]
    }
    path = _write_config(workspace.parent / "outside", payload)
    with pytest.raises(ValueError) as excinfo:
        load_mcp_configs(workspace, path)
    message = str(excinfo.value)
    assert "duplicate MCP server" in message
    assert "supersecret-value" not in message


def test_oauth_fields_are_explicitly_rejected(workspace: Path) -> None:
    for overrides in (
        {"oauth": True},
        {"oauth_client_id": "client"},
        {"oauth_scopes": ["read"]},
        {"oauth_allowed_domains": ["example.com"]},
    ):
        path = _write_config(workspace.parent / "outside", {"servers": [_http(**overrides)]})
        with pytest.raises(ValueError, match="MCP OAuth lands in phase 5"):
            load_mcp_configs(workspace, path)


# ---------------------------------------------------------------------------
# payload shape / YAML hygiene
# ---------------------------------------------------------------------------


def test_payload_must_be_mapping_and_servers_list(workspace: Path, outside: Path) -> None:
    path = _write_config(outside, ["not", "a", "mapping"])
    with pytest.raises(ValueError, match="must be a mapping"):
        load_mcp_configs(workspace, path)

    path = _write_config(outside, {"servers": {}})
    with pytest.raises(ValueError, match="servers must be a list"):
        load_mcp_configs(workspace, path)

    path = _write_config(outside, {})
    with pytest.raises(ValueError, match="servers must be a list"):
        load_mcp_configs(workspace, path)

    path = _write_config(outside, {"servers": ["entry"]})
    with pytest.raises(ValueError, match="entries must be mappings"):
        load_mcp_configs(workspace, path)


def test_invalid_yaml_does_not_echo_file_body(workspace: Path, outside: Path) -> None:
    outside.mkdir(parents=True, exist_ok=True)
    outside.chmod(0o700)
    path = outside / "mcp_servers.yaml"
    path.write_text("servers: [TOPSECRET\n", encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError) as excinfo:
        load_mcp_configs(workspace, path)
    assert "valid YAML" in str(excinfo.value)
    assert "TOPSECRET" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# discovery / trust gate / drift
# ---------------------------------------------------------------------------


def test_default_missing_config_returns_empty(workspace: Path) -> None:
    assert load_mcp_configs(workspace) == ()


def test_workspace_product_fallbacks_do_not_exist(workspace: Path) -> None:
    """Q1 有意收缩:workspace ``configs/`` 与产品内置目录都不是来源。"""
    configs_dir = workspace / "configs"
    configs_dir.mkdir()
    _write_config(configs_dir, {"servers": [_stdio()]})
    assert load_mcp_configs(workspace) == ()


def test_host_default_config_is_loaded(workspace: Path) -> None:
    _write_host_config({"servers": [_stdio()]})
    (config,) = load_mcp_configs(workspace)
    assert config.name == "files"


def test_explicit_missing_config_raises(workspace: Path, outside: Path) -> None:
    missing = outside / "nope.yaml"
    with pytest.raises(ValueError) as excinfo:
        load_mcp_configs(workspace, missing)
    assert "does not exist" in str(excinfo.value)
    assert "nope.yaml" in str(excinfo.value)


def test_env_missing_config_raises(
    workspace: Path, outside: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV, str(outside / "nope.yaml"))
    with pytest.raises(ValueError, match="does not exist"):
        load_mcp_configs(workspace)


def test_env_config_is_loaded(
    workspace: Path, outside: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_config(outside, {"servers": [_stdio()]})
    monkeypatch.setenv(_ENV, str(path))
    (config,) = load_mcp_configs(workspace)
    assert config.name == "files"


def test_explicit_path_beats_environment(
    workspace: Path, outside: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = _write_config(outside / "env", {"servers": [_stdio(name="from-env")]})
    explicit = _write_config(outside, {"servers": [_stdio(name="from-arg")]})
    monkeypatch.setenv(_ENV, str(env_path))
    (config,) = load_mcp_configs(workspace, explicit)
    assert config.name == "from-arg"


def test_relative_paths_are_rejected(
    workspace: Path, outside: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="absolute file outside the workspace"):
        load_mcp_configs(workspace, Path("mcp_servers.yaml"))
    monkeypatch.setenv(_ENV, "relative.yaml")
    with pytest.raises(ValueError, match="absolute file outside the workspace"):
        load_mcp_configs(workspace)


def test_workspace_owned_config_is_rejected(workspace: Path) -> None:
    path = _write_config(workspace, {"servers": [_stdio()]})
    with pytest.raises(ValueError, match="absolute file outside the workspace"):
        load_mcp_configs(workspace, path)


def test_group_or_world_writable_file_is_rejected(
    workspace: Path, outside: Path
) -> None:
    path = _write_config(outside, {"servers": [_stdio()]}, mode=0o666)
    with pytest.raises(ValueError, match="owned by the current user"):
        load_mcp_configs(workspace, path)


def test_group_or_world_writable_parent_is_rejected(
    workspace: Path, outside: Path
) -> None:
    path = _write_config(outside, {"servers": [_stdio()]})
    outside.chmod(0o777)
    with pytest.raises(ValueError, match="owned by the current user"):
        load_mcp_configs(workspace, path)


def test_trust_decision_classifies_user_and_external(workspace: Path, outside: Path) -> None:
    host_path = _write_host_config({"servers": [_stdio()]})
    user = decide_mcp_config_trust(host_path, workspace=workspace)
    assert user.source == "user"
    assert user.config_path == host_path.resolve()
    assert user.config_sha256 == hashlib.sha256(host_path.read_bytes()).hexdigest()

    external_path = _write_config(outside, {"servers": [_stdio()]})
    external = decide_mcp_config_trust(external_path, workspace=workspace)
    assert external.source == "external"


def test_trust_decision_rejects_workspace_path(workspace: Path) -> None:
    path = _write_config(workspace, {"servers": [_stdio()]})
    with pytest.raises(ValueError, match="absolute file outside the workspace"):
        decide_mcp_config_trust(path, workspace=workspace)


def test_config_sha256_tracks_bytes(outside: Path) -> None:
    path = _write_config(outside, {"servers": [_stdio()]})
    baseline = config_sha256(path)
    assert baseline == hashlib.sha256(path.read_bytes()).hexdigest()
    assert config_sha256(path) == baseline
    path.write_bytes(path.read_bytes() + b"\n")
    os.chmod(path, 0o600)
    assert config_sha256(path) != baseline