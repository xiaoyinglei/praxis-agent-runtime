"""Task 21 Step 1:阶段 3B 旧工具行为对照矩阵(web/hooks/skills/MCP)。

沿用 3A ``acceptance/test_tool_parity.py`` 口径:

- 每个数值/错误码/消息都带**旧代码 file:line**出处(相对仓库根的旧
  ``agent_runtime/``,只读参照,绝不 import);阈值以 praxis 的**公开契约**
  钉住——工具 ``validate_input``/``definition``/``timeout_seconds``/
  ``max_model_output_bytes``、配置加载器的真实拒绝、真实子进程/真实文件系统
  行为。个别构造点(``_stdio_parameters``/``_ConnectionCandidates``)按派单
  显式直取:护栏必须钉在"重新引入旧病灶的那一行"上(T16 证据分工先例)。
- **有意差异显式断言而非省略**:每处 divergence 都有一条"不存在面"或
  "新形状"断言 + 理由注释(schema 键缺席/错误码缺席/落点/预算口径等)。

旧代码出处总目(逐节另有行内锚点):
- hooks:``agent_runtime/hooks.py``、``agent_runtime/harness/lifecycle.py``;
- skills:``agent_runtime/skills/{models,loader,runtime}.py``、
  ``agent_runtime/tools/integrations/skills.py``、``agent_runtime/workspace.py``;
- web:``agent_runtime/tools/web_http.py``、``tools/web_limits.py``、
  ``tools/web_browser.py``、``tools/web_content.py``、``tools/builtins/web.py``;
- MCP:``agent_runtime/runtime/mcp.py``、``tools/integrations/mcp.py``、
  ``runtime/mcp_catalog.py``。
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import gzip
import hashlib
import inspect
import json
import os
import signal
import stat
import subprocess
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from mcp.types import CallToolResult, ListToolsResult, TextContent
from mcp.types import Tool as McpTool

from praxis.adapters.hooks.config import load_host_hooks
from praxis.adapters.hooks.runner import spawn_hook
from praxis.adapters.mcp.catalog import (
    _MAX_CATALOG_BYTES,
    _MAX_CATALOG_PAGES,
    _MAX_CATALOG_TOOLS,
    list_all_tools,
)
from praxis.adapters.mcp.config import (
    HttpServerConfig,
    StdioServerConfig,
    config_sha256,
    load_mcp_configs,
)
from praxis.adapters.mcp.descriptors import mcp_tools_for_connection
from praxis.adapters.mcp.http_transport import (
    GuardedMcpTransport,
    McpTransportError,
    credential_revision,
)
from praxis.adapters.mcp.lifecycle import McpConnection, _stdio_parameters
from praxis.adapters.mcp.names import (
    canonical_tool_name,
    global_wildcard,
    normalize_mcp_name,
    server_wildcard,
)
from praxis.adapters.mcp.output import normalize_mcp_output
from praxis.adapters.web.client import (
    CLIENT_ERROR_CODES,
    PublicHttpClient,
    PublicWebError,
    _ConnectionCandidates,
)
from praxis.adapters.web.extract import MAX_SOURCE_CHARACTERS, extract_text, source_lines
from praxis.adapters.web.limits import WebLimits, load_web_limits
from praxis.adapters.web.search import (
    DEFAULT_MAX_RESULTS,
    MAX_RESULTS_LIMIT,
    MIN_MAX_RESULTS,
    create_web_search_tools,
)
from praxis.adapters.web.tools import create_web_fetch_tools
from praxis.adapters.web.urls import is_public_address, validate_public_url
from praxis.core.events import EventBus
from praxis.core.permission import decide
from praxis.core.pipeline import ExecCtx, execute
from praxis.protocol.floor import floor_reason, host_private_paths
from praxis.protocol.hooks import (
    MAX_FEEDBACK_BYTES,
    MAX_HOOKS,
    MAX_HOOK_STDOUT_BYTES,
    HookOutcome,
    HookSetup,
    HookSpec,
    fold_hook_items,
    hook_invocation_id,
)
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import thaw_json
from praxis.protocol.permission import HookDecision, PermissionContext
from praxis.protocol.policy import PermissionPolicy, ToolRule
from praxis.protocol.sandbox import SandboxSpec, ToolExecContext
from praxis.protocol.skills import (
    MAX_INVOKE_ARGS_CHARS,
    MAX_SKILL_INSTRUCTIONS_CHARS,
    MAX_SKILL_NAME_CHARS,
    SkillErrorCode,
    SkillSource,
)
from praxis.protocol.tools import (
    ResolvedUse,
    Tool,
    ToolCall,
    ToolContentBlock,
    ToolEffect,
    ToolResult,
    ToolSource,
    ToolValidationError,
    args_digest,
    json_schema_input,
)
from praxis.store.log import Store
from praxis.store.turns import create_turn
from praxis.tools.builtins.child_env import child_env
from praxis.tools.builtins.process import signal_group
from praxis.tools.builtins.skills import (
    create_invoke_skill_tool,
    create_materialize_skill_asset_tool,
    load_skill_catalog,
    skill_fingerprint,
)
from tests.praxis.conftest import make_minimal_tool
from tests.praxis.test_web_urls_dns import (
    PUBLIC_IP,
    FakeBackend,
    FakeResolver,
    FakeStream,
    response_chunks,
)

# ---------------------------------------------------------------------------
# 共用脚手架
# ---------------------------------------------------------------------------


def _schema_properties(tool: Tool) -> Mapping[str, Any]:
    schema = thaw_json(tool.definition.input_schema)
    assert isinstance(schema, Mapping)
    properties = schema.get("properties")
    assert isinstance(properties, Mapping)
    return properties


def _code_string_constants(module: Any) -> set[str]:
    """模块内**代码**字符串常量(排除 docstring)——"错误码缺席"断言用。

    与 scripts/check_anti_patterns.py 的 docstring 排除同口径:docstring 里
    提及一个已删除的码(历史注记)不等于该码仍可达。
    """
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
    }


def _assert_rejects(tool: Tool, arguments: Mapping[str, Any]) -> ToolValidationError:
    with pytest.raises(ToolValidationError) as excinfo:
        tool.validate_input(dict(arguments))
    return excinfo.value


def _run_tool(tool: Tool, arguments: Mapping[str, Any], exec_ctx: ToolExecContext | None = None) -> ToolResult:
    ctx = exec_ctx if exec_ctx is not None else ToolExecContext()
    if inspect.iscoroutinefunction(tool.run):
        return asyncio.run(tool.run(dict(arguments), ctx))
    result = tool.run(dict(arguments), ctx)  # 同步工具(管道经 to_thread 派发)
    assert isinstance(result, ToolResult)
    return result


def _hook_spec(script: Path, host: Path, **overrides: Any) -> HookSpec:
    defaults: dict[str, Any] = dict(
        name="policy",
        revision="1",
        event="PreToolUse",
        argv=(str(script),),
        directory=str(host),
    )
    defaults.update(overrides)
    return HookSpec(**defaults)


def _write_hook_script(host: Path, body: str, name: str = "hook.sh") -> Path:
    host.mkdir(parents=True, exist_ok=True)
    script = host / name
    script.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    script.chmod(0o755)
    return script


async def _real_spawn(spec: HookSpec, request: Mapping[str, Any]) -> HookOutcome:
    return await spawn_hook(spec, request, kill_group=signal_group)


def _write_hooks_config(host: Path, entries: list[dict[str, Any]]) -> Path:
    host.mkdir(parents=True, exist_ok=True)
    host.chmod(0o700)
    path = host / "hooks.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    path.chmod(0o600)
    return path


def _write_skill(
    workspace: Path,
    name: str,
    *,
    body: str,
    extra_frontmatter: str = "",
) -> Path:
    (workspace / ".git").mkdir(exist_ok=True)  # 锚定 repo root,PROJECT walk 只扫本 workspace
    skill_dir = workspace / ".agents" / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        f"---\nname: {name}\ndescription: test skill\n{extra_frontmatter}---\n{body}",
        encoding="utf-8",
    )
    return skill_file


def _offline_client(
    script: Sequence[FakeStream | Exception],
    *,
    authorize: Any = None,
    limits: WebLimits | None = None,
    resolver_answers: Sequence[str] = (PUBLIC_IP,),
) -> PublicHttpClient:
    calls: list[str] = []

    def recording(url: str) -> None:
        calls.append(url)

    return PublicHttpClient(
        authorize=authorize if authorize is not None else recording,
        limits=limits if limits is not None else WebLimits(),
        resolver=FakeResolver(list(resolver_answers)),
        backend=FakeBackend(script),
        environ={},
    )


def _fetch_tool(client: PublicHttpClient, **kwargs: Any) -> Tool:
    (tool,) = create_web_fetch_tools(client, limits=WebLimits(), **kwargs)
    return tool


def _stdio_config(**overrides: Any) -> StdioServerConfig:
    defaults: dict[str, Any] = dict(
        name="probe",
        command="/bin/echo",
        args=("hello",),
        env={"SECRET_MCP_KEY": "sk-parity-value"},
        cwd=None,
        tools_allowlist=frozenset(),
        allow_all_tools=True,
        startup_timeout_seconds=20.0,
    )
    defaults.update(overrides)
    return StdioServerConfig(**defaults)


def _http_config(**overrides: Any) -> HttpServerConfig:
    defaults: dict[str, Any] = dict(
        name="remote",
        url="https://mcp.example.com/rpc",
        headers={},
        bearer_token_env_var=None,
        keychain=False,
        read_only=True,
        tools_allowlist=frozenset(),
        allow_all_tools=True,
        startup_timeout_seconds=20.0,
    )
    defaults.update(overrides)
    return HttpServerConfig(**defaults)


def _mcp_tool(name: str = "echo", **annotations: Any) -> McpTool:
    return McpTool(
        name=name,
        description=f"{name} tool",
        inputSchema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
        annotations=annotations or None,
    )


class _FakeSession:
    """duck-typed ClientSession:call_tool 脚本化;记录调用供守卫断言。"""

    def __init__(self, result: CallToolResult | None = None) -> None:
        self.result = result if result is not None else CallToolResult(
            content=[TextContent(type="text", text="echo: hi")], isError=False
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        self.calls.append((name, dict(arguments or {})))
        return self.result


def _fake_connection(tools: tuple[McpTool, ...], session: _FakeSession | None = None) -> McpConnection:
    return McpConnection(
        server_name="probe",
        session=session if session is not None else _FakeSession(),  # type: ignore[arg-type]
        tools=tools,
    )


def _mcp_tools(
    config: Any = None,
    *,
    tools: tuple[McpTool, ...] | None = None,
    sha: list[str] | None = None,
    session: _FakeSession | None = None,
) -> tuple[Tool, ...]:
    baseline = sha if sha is not None else ["baseline-sha"]
    return mcp_tools_for_connection(
        _fake_connection(tools if tools is not None else (_mcp_tool(),), session),
        config if config is not None else _stdio_config(),
        config_sha_provider=lambda: baseline[0],
    )


# ===========================================================================
# hooks:16 / 2048 / 16_385 / 64KiB / timeout(0,30] 默认 5 / PATH=os.defpath /
#        7 状态全集 / 阻断文案形状
# ===========================================================================


def test_hooks_constants_match_legacy() -> None:
    # 旧 hooks.py:21-22 MAX_HOOKS=16 / MAX_FEEDBACK_BYTES=2048。
    assert MAX_HOOKS == 16
    assert MAX_FEEDBACK_BYTES == 2048
    # 旧 hooks.py:188 readexactly(16_385):>16KiB(16_384)即拒。
    assert MAX_HOOK_STDOUT_BYTES == 16_385


def test_hook_spec_timeout_bounds_and_default_match_legacy() -> None:
    # 旧 hooks.py:64 默认 5.0;:77 校验 0 < x <= 30(int/float、有限)。
    script = Path("/bin/sh")
    spec = _hook_spec(script, Path("/tmp"), argv=(str(script),))
    assert spec.timeout_seconds == 5.0
    assert _hook_spec(script, Path("/tmp"), argv=(str(script),), timeout_seconds=30).timeout_seconds == 30.0
    assert _hook_spec(script, Path("/tmp"), argv=(str(script),), timeout_seconds=0.5).timeout_seconds == 0.5
    for bad in (0, -1, 30.1, float("inf"), float("nan"), True, "5"):
        with pytest.raises(ValueError):
            _hook_spec(script, Path("/tmp"), argv=(str(script),), timeout_seconds=bad)
    # 旧 hooks.py:70-72 name/revision 1–128。
    with pytest.raises(ValueError):
        _hook_spec(script, Path("/tmp"), argv=(str(script),), name="x" * 129)
    with pytest.raises(ValueError):
        _hook_spec(script, Path("/tmp"), argv=(str(script),), revision="")


def test_hook_status_full_set_matches_legacy() -> None:
    # 旧状态全集:harness/lifecycle.py:159-173(allowed/blocked/observed/timeout/error)
    # + :133/:139 unknown + :148 unavailable。praxis 协议层逐字保留七值。
    expected = {"allowed", "blocked", "observed", "timeout", "error", "unavailable", "unknown"}
    assert set(typing.get_args(typing.get_type_hints(HookOutcome)["status"])) == expected
    for status in sorted(expected):
        assert HookOutcome(status).status == status  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        HookOutcome("cancelled")  # type: ignore[arg-type]


def test_hook_feedback_truncates_instead_of_raising(tmp_path: Path) -> None:
    # 有意差异:旧 HookDecision.__post_init__(hooks.py:54-55)对 >2048 字节
    # **抛错**;praxis HookOutcome 构造期按 UTF-8 截断(运行器不把超长宿主
    # 输出变成异常)。2048 字节上限本身逐字保留。
    outcome = HookOutcome("blocked", "x" * 3000)
    assert len(outcome.feedback.encode("utf-8")) == 2048
    # 多字节码点不切半:截断后仍是合法 UTF-8 且 ≤2048 字节。
    multibyte = HookOutcome("blocked", "中" * 2000)
    encoded = multibyte.feedback.encode("utf-8")
    assert len(encoded) <= 2048
    assert encoded.decode("utf-8")  # 无半个码点
    assert len(HookOutcome("blocked", "y" * 2048).feedback) == 2048


def test_hook_stdout_exact_16384_16385_boundary(tmp_path: Path) -> None:
    # 旧 hooks.py:186-191:readexactly(16_385) 哨兵——16_384 字节可读(合法
    # 裁决),16_385 字节即超 16KiB 上限 ⇒ error。T4 留档①的精确边界例。
    host = tmp_path / "host"
    host.mkdir()

    def payload(size: int) -> bytes:
        prefix = b'{"block": false, "feedback": "'
        suffix = b'"}'
        return prefix + b"a" * (size - len(prefix) - len(suffix)) + suffix

    for size, expected in ((16_384, "allowed"), (16_385, "error")):
        data_file = host / f"out-{size}.json"
        data_file.write_bytes(payload(size))
        script = _write_hook_script(host, f"cat {data_file}", name=f"hook-{size}.sh")
        spec = _hook_spec(script, host)
        outcome = asyncio.run(spawn_hook(spec, {"event": "PreToolUse"}, kill_group=signal_group))
        assert outcome.status == expected, size


def test_hook_child_env_is_exactly_path_defpath(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 旧 hooks.py:166 env={"PATH": os.defpath};:170 start_new_session=True。
    # T4 留档③:断"除 PATH 外零键"**全等**(不是子集)。观测者自身运行时
    # 自添的变量如实枚举并排除:/bin/sh 自添 PWD/SHLVL/_(execve 传入的
    # 只有 PATH);宿主哨兵绝不出现(B33 安全性质,全强度断言)。
    monkeypatch.setenv("PRAXIS_HOOK_CANARY", "SECRET-HOOK-XYZ")
    host = tmp_path / "host"
    record = host / "env.txt"
    script = _write_hook_script(host, f"env > {record}")
    spec = _hook_spec(script, host)
    outcome = asyncio.run(spawn_hook(spec, {}, kill_group=signal_group))
    assert outcome.status == "allowed"  # 空 stdout ⇒ allowed(旧 hooks.py:196-197)
    lines = record.read_text(encoding="utf-8").splitlines()
    env = dict(line.split("=", 1) for line in lines if "=" in line)
    shell_self_added = {"PWD", "SHLVL", "_"}
    assert set(env) - shell_self_added == {"PATH"}
    assert env["PATH"] == os.defpath
    assert "PRAXIS_HOOK_CANARY" not in env
    assert "SECRET-HOOK-XYZ" not in record.read_text(encoding="utf-8")


def test_hook_success_path_sends_no_signal(tmp_path: Path) -> None:
    # T4 留档④:成功路径零信号(旧 hooks.py:203-208 只在 returncode None
    # 或 stdout 未 EOF 时 killpg)。记录型 signaler 证明成功路径不发射。
    host = tmp_path / "host"
    script = _write_hook_script(host, "printf '%s' '{\"block\": false}'")
    spec = _hook_spec(script, host)
    sent: list[tuple[int, int]] = []

    def recorder(pid: int, sig: int) -> None:
        sent.append((pid, sig))
        signal_group(pid, sig)

    outcome = asyncio.run(spawn_hook(spec, {}, kill_group=recorder))
    assert outcome.status == "allowed"
    assert sent == []


def test_hook_timeout_kills_process_group(tmp_path: Path) -> None:
    # 旧 hooks.py:203-216:超时/未 EOF ⇒ os.killpg(SIGKILL) 整组 + 0.5s 有界排空。
    host = tmp_path / "host"
    script = _write_hook_script(host, "sleep 30")
    spec = _hook_spec(script, host, timeout_seconds=0.5)
    sent: list[tuple[int, int]] = []

    def recorder(pid: int, sig: int) -> None:
        sent.append((pid, sig))
        signal_group(pid, sig)

    outcome = asyncio.run(spawn_hook(spec, {}, kill_group=recorder))
    assert outcome.status == "timeout"
    assert len(sent) == 1 and sent[0][1] == signal.SIGKILL
    # 整组已死:组长 pid 不再存在(start_new_session ⇒ pid==pgid)。
    with pytest.raises(ProcessLookupError):
        os.kill(sent[0][0], 0)


def test_hook_decision_shape_and_unknown_keys(tmp_path: Path) -> None:
    # 旧 hooks.py:196-201:空 stdout ⇒ allowed(None);JSON object ⇒
    # HookDecision(**result),未知键抛错 ⇒ error;非 0 退出 ⇒ error。
    host = tmp_path / "host"
    cases = (
        ('printf \'%s\' \'{"block": true, "feedback": "no"}\'', "blocked", "no"),
        ('printf \'%s\' \'{"block": false}\'', "allowed", ""),
        ("printf '%s' '{\"block\": true, \"extra\": 1}'", "error", ""),
        ("printf '%s' 'not-json'", "error", ""),
        ("printf '%s' '[1]'", "error", ""),
        ("printf '%s' '{\"block\": \"yes\"}'", "error", ""),
        ("exit 3", "error", ""),
        ("", "allowed", ""),
    )
    for index, (body, status, feedback) in enumerate(cases):
        script = _write_hook_script(host, body, name=f"hook-{index}.sh")
        outcome = asyncio.run(spawn_hook(_hook_spec(script, host), {}, kill_group=signal_group))
        assert outcome.status == status, (index, body)
        assert outcome.feedback == feedback


def test_hooks_config_64kib_and_16_entries_boundary(tmp_path: Path) -> None:
    # 旧 hooks.py:119-124:read(65_537) 判 >65_536;数组 ≤16 项。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    host = tmp_path / "host"
    entry = {"event": "PreToolUse", "name": "h", "argv": ["/bin/sh"]}
    ok = _write_hooks_config(host, [entry] * 16)
    # 16 项同名同事件 ⇒ 唯一性拒绝;先钉大小门,再换唯一名钉 16 项通过。
    with pytest.raises(ValueError, match="unique"):
        load_host_hooks(workspace, ok)
    named = [{"event": "PreToolUse", "name": f"h{i}", "argv": ["/bin/sh"]} for i in range(16)]
    ok16 = _write_hooks_config(host, named)
    assert len(load_host_hooks(workspace, ok16)) == 16
    over = _write_hooks_config(host, named + [{"event": "PreToolUse", "name": "h16", "argv": ["/bin/sh"]}])
    with pytest.raises(ValueError, match="at most 16"):
        load_host_hooks(workspace, over)
    # 65_536 字节恰好通过(JSON 尾部空白填充,条目本身合法);65_537 拒绝。
    def sized(total: int) -> str:
        base = json.dumps([{"event": "PreToolUse", "name": "pad", "argv": ["/bin/sh"]}])
        return base + " " * (total - len(base))

    ok_bytes = host / "hooks.json"
    ok_bytes.write_text(sized(65_536), encoding="utf-8")
    ok_bytes.chmod(0o600)
    assert ok_bytes.stat().st_size == 65_536
    assert len(load_host_hooks(workspace, ok_bytes)) == 1
    ok_bytes.write_text(sized(65_537), encoding="utf-8")
    assert ok_bytes.stat().st_size == 65_537
    with pytest.raises(ValueError, match="64 KiB"):
        load_host_hooks(workspace, ok_bytes)


def test_hooks_config_stop_rejected_at_load_time(tmp_path: Path) -> None:
    # 有意差异(phase 6):旧 EVENTS 含 "Stop"(hooks.py:20)且 Stop hook 在
    # harness/stop.py:370-383 执行;praxis 3B 只有 Pre/PostToolUse 两执行点,
    # Stop 在**加载期明确拒绝**(不静默忽略)。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    host = tmp_path / "host"
    path = _write_hooks_config(host, [{"event": "Stop", "name": "s", "argv": ["/bin/sh"]}])
    with pytest.raises(ValueError, match="stop hooks land in phase 6"):
        load_host_hooks(workspace, path)
    with pytest.raises(ValueError, match="invalid hook event"):
        _hook_spec(Path("/bin/sh"), host, event="Stop", argv=("/bin/sh",))


def test_hooks_config_entry_shape_and_hygiene(tmp_path: Path) -> None:
    # 旧 hooks.py:125-141:必填 event/name/argv,未知键直接报错;argv[0]
    # 绝对可执行且不在 workspace 内;revision 1–128。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    host = tmp_path / "host"
    good = {"event": "PreToolUse", "name": "h", "argv": ["/bin/sh"]}
    path = _write_hooks_config(host, [good])
    assert len(load_host_hooks(workspace, path)) == 1
    for bad_entry, match in (
        ({**good, "extra_key": 1}, "event, name and argv"),
        ({"event": "PreToolUse", "name": "h"}, "event, name and argv"),
        ({**good, "argv": ["relative/sh"]}, "absolute executable"),
        ({**good, "argv": [str(workspace / "inside.sh")]}, "absolute executable"),
        ({**good, "revision": "x" * 129}, "1–128"),
        ({**good, "event": "Bogus"}, "invalid hook event"),
    ):
        path = _write_hooks_config(host, [bad_entry])
        with pytest.raises(ValueError, match=match):
            load_host_hooks(workspace, path)
    # 卫生门(旧 decide_mcp_config_trust 个人配置标准,3B 笔记 §4):
    # group/world 可写 ⇒ 拒。
    path = _write_hooks_config(host, [good])
    path.chmod(0o666)
    with pytest.raises(ValueError, match="owned by the current user"):
        load_host_hooks(workspace, path)
    # workspace 内的配置文件 ⇒ 拒(旧 hooks.py:114-116)。
    inside = workspace / "hooks.json"
    inside.write_text(json.dumps([good]), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the workspace"):
        load_host_hooks(workspace, inside)


def test_hook_no_handler_callable_dual_entry() -> None:
    # 有意差异:旧 LifecycleHook.handler 允许任意 async callable(hooks.py:62,
    # :74-75),与 load_host_hooks 的 command-only 构成两个入口两种信任模型。
    # praxis 只留 command hook:HookSpec 无 handler 字段,spawn 是唯一机制注入。
    field_names = {field.name for field in dataclasses.fields(HookSpec)}
    assert "handler" not in field_names
    assert field_names == {
        "name", "revision", "event", "argv", "directory",
        "tool_names", "effects", "timeout_seconds",
    }
    setup = HookSetup(specs=(), spawn=_real_spawn)
    assert setup.specs == () and callable(setup.spawn)


def test_hook_block_string_shape_praxis_vs_legacy() -> None:
    # 旧阻断文案(harness/lifecycle.py:78-79):
    #   f"{event} hook {name!r} ({status}): {feedback}"
    # praxis 有意差异:hook 结论作为 HookDecision 进唯一 decide,deny reason
    # 形如 "hook:<name>:<status>: <feedback>"(R-PF-1;core/permission._hook_reason),
    # 不再是独立的 ExecutionStartRejectedError 入口(旧 tool_orchestrator.py:869-872)。
    tool = make_minimal_tool()
    use = tool.resolve_use({})
    ctx = PermissionContext(
        workspace="/tmp",
        policy=PermissionPolicy(),
        mode="on-request",
        hook_decisions=(HookDecision(source="policy:blocked", verdict="deny", feedback="no"),),
    )
    decision = decide(tool, use, ctx)
    assert decision.is_deny
    assert decision.reason == "hook:policy:blocked: no"
    # 无 feedback 时只有前缀段。
    ctx2 = PermissionContext(
        workspace="/tmp",
        policy=PermissionPolicy(),
        mode="on-request",
        hook_decisions=(HookDecision(source="policy:timeout", verdict="deny"),),
    )
    assert decide(tool, use, ctx2).reason == "hook:policy:timeout"
    # allow 结构上无裁决:hook allow 不进 hook_decisions 的 deny/ask 分支,
    # 规则 deny 依旧生效(旧实现不存在 allow 短路,3B 笔记 §4 grep 结论)。
    ctx3 = PermissionContext(
        workspace="/tmp",
        policy=PermissionPolicy(tools=(ToolRule(pattern="minimal", decision="deny"),)),
        mode="bypass",
        hook_decisions=(HookDecision(source="policy:allowed", verdict="allow"),),
    )
    assert decide(tool, use, ctx3).reason == "rule: tool 'minimal' denied"


def _pipeline_ctx(
    store: Store,
    workspace: Path,
    tools: Mapping[str, Tool],
    *,
    hooks: HookSetup | None = None,
    policy: PermissionPolicy | None = None,
) -> ExecCtx:
    effective = policy if policy is not None else PermissionPolicy()
    create_turn(store, turn_id="turn-1", thread_id="thread-1", binding={})
    return ExecCtx(
        store=store,
        bus=EventBus(),
        workspace=workspace,
        tools=dict(tools),
        permissions=PermissionContext(
            workspace=str(workspace), policy=effective, mode="on-request"
        ),
        turn_id="turn-1",
        thread_id="thread-1",
        policy_revision=effective.revision,
        hooks=hooks,
    )


def test_post_hook_digest_is_args_not_result(tmp_path: Path) -> None:
    # 有意差异(R-T5-2):旧 PostToolUse 幂等键含结果侧 target_id/证据修订
    # (harness/lifecycle.py:71-75);praxis Pre/Post **都用 args_digest**——
    # 结果摘要会在 cancel→resume 重执行时改变,导致 Post hook 再 spawn 一次,
    # 重新引入副作用重复。同参数不同结果 ⇒ 同一 invocation_id ⇒ fold 复用。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    host = tmp_path / "host"
    counter = host / "spawns.txt"
    record = host / "request.json"
    host.mkdir(parents=True, exist_ok=True)
    script = _write_hook_script(
        host,
        f"echo run >> {counter}\ncat > {record}\nprintf '%s' '{{\"block\": false}}'",
    )
    spec = _hook_spec(script, host, event="PostToolUse", tool_names=("vary",))
    calls = {"n": 0}

    async def vary_run(arguments: Mapping[str, Any], exec_ctx: ToolExecContext) -> ToolResult:
        calls["n"] += 1
        return ToolResult(
            tool_call_id="pending",
            tool_name="vary",
            content=(ToolContentBlock(type="text", data={"text": f"result-{calls['n']}"}),),
        )

    tool = dataclasses.replace(
        make_minimal_tool(),
        definition=dataclasses.replace(make_minimal_tool().definition, name="vary"),
        validate_input=json_schema_input({"type": "object"}),
        run=vary_run,
        static_effects=frozenset(),
        resolve_use=lambda _args: ResolvedUse(effects=frozenset(), targets=()),
    )
    store = Store(workspace / "store.db")
    try:
        ctx = _pipeline_ctx(store, workspace, {"vary": tool}, hooks=HookSetup((spec,), _real_spawn))
        arguments = {"x": 1}
        first = asyncio.run(execute(ToolCall("c1", "vary", arguments), ctx))
        second = asyncio.run(execute(ToolCall("c2", "vary", arguments), ctx))
        assert first.status.value == "ok" and second.status.value == "ok"
        assert first.content[0].data["text"] != second.content[0].data["text"]
        # 结果不同但参数相同 ⇒ Post hook 只 spawn 一次(fold 复用)。
        assert counter.read_text().splitlines() == ["run"]
        request = json.loads(record.read_text(encoding="utf-8"))
        digest = args_digest(tool.validate_input(arguments))
        assert request["invocation_id"] == hook_invocation_id("turn-1", "PostToolUse", "policy", digest)
        # Pre 与 Post 身份只差 event 段。
        assert hook_invocation_id("turn-1", "PreToolUse", "policy", digest) != request["invocation_id"]
        # 两次执行都带观察摘要(第二次来自 fold,不再 spawn)。
        assert first.metadata["post_tool_hook"]["status"] == "allowed"
        assert second.metadata["post_tool_hook"]["status"] == "allowed"
    finally:
        store.close()


def test_post_hook_request_omits_source_item_id(tmp_path: Path) -> None:
    # 有意差异(R-T5-1):旧 HookRequest 含 source_item_id/result/answer/
    # completion 等字段(hooks.py:25-43),PostToolUse 填 result+source_item_id。
    # praxis 结算序里结果 item 尚未提交 ⇒ Post 请求**不含** source_item_id,
    # 也不含 result;请求键集合在此钉死。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    host = tmp_path / "host"
    record = host / "request.json"
    host.mkdir(parents=True, exist_ok=True)
    script = _write_hook_script(host, f"cat > {record}\nprintf '%s' '{{\"block\": false}}'")
    spec = _hook_spec(script, host, event="PostToolUse")
    tool = make_minimal_tool()
    store = Store(workspace / "store.db")
    try:
        ctx = _pipeline_ctx(store, workspace, {"minimal": tool}, hooks=HookSetup((spec,), _real_spawn))
        asyncio.run(execute(ToolCall("c1", "minimal", {}), ctx))
        request = json.loads(record.read_text(encoding="utf-8"))
        assert set(request) == {
            "event", "invocation_id", "thread_id", "turn_id", "workspace",
            "tool_call_id", "tool_name", "arguments", "effects", "input_revision",
        }
        assert request["event"] == "PostToolUse"
        assert request["workspace"] == str(workspace.resolve())
    finally:
        store.close()


def _hook_item(invocation_id: str, phase: str, **extra: Any) -> Item:
    payload: dict[str, Any] = {"invocation_id": invocation_id, "phase": phase, **extra}
    return Item(
        item_id=f"i-{phase}",
        thread_id="t",
        turn_id="turn-1",
        seq=1,
        kind=ItemKind.HOOK_INVOCATION,
        payload=payload,
        created_at=0.0,
    )


def test_hook_settlement_folds_pipeline_items_not_dict_lookup() -> None:
    # 有意差异:旧实现用运行时 dict 反查 + binding() 比对(harness/lifecycle.py
    # 的 _run_registered/注册漂移 unavailable);praxis 的结算事实是 append-only
    # **pipeline items** 的纯函数投影(protocol.hooks.fold_hook_items):
    # settled 优先、started-only ⇒ unknown(fail-closed)、无记录 ⇒ None。
    invocation = "hook_abc"
    assert fold_hook_items([], invocation) is None
    started = _hook_item(invocation, "started", event="PreToolUse", hook="policy")
    assert fold_hook_items([started], invocation) == HookOutcome("unknown", "crash before settlement")
    settled = _hook_item(invocation, "settled", status="blocked", feedback="no")
    # settled 优先,与顺序无关。
    assert fold_hook_items([started, settled], invocation) == HookOutcome("blocked", "no")
    assert fold_hook_items([settled, started], invocation) == HookOutcome("blocked", "no")
    # 畸形 payload 一律跳过,绝不抛(损坏宿主数据不是当前调用的崩溃点)。
    malformed = (
        _hook_item(invocation, "settled", status="bogus"),
        _hook_item(invocation, "settled", status="allowed", feedback=3),
        _hook_item(invocation, "unknown-phase"),
        Item("i-x", "t", "turn-1", 9, ItemKind.HOOK_INVOCATION, ["not-a-mapping"], 0.0),  # type: ignore[arg-type]
    )
    for item in malformed:
        assert fold_hook_items([item], invocation) is None
    # 其它 invocation 的记录不串用。
    assert fold_hook_items([_hook_item("other", "settled", status="allowed")], invocation) is None


# ===========================================================================
# skills:40_000 / 500 / 4000 / 4096 / scripts|references / 300_000 / 50_000 /
#         timeout 5.0 / 错误码含 drift / fingerprint 形状
# ===========================================================================


def test_invoke_skill_schema_bounds_match_legacy(tmp_path: Path) -> None:
    # 旧 integrations/skills.py:39-51 InvokeSkillInput:name 1..500(:44)、
    # args ≤4000(:49)、extra=forbid。
    tool = create_invoke_skill_tool(load_skill_catalog(tmp_path))
    properties = _schema_properties(tool)
    assert properties["name"]["maxLength"] == MAX_SKILL_NAME_CHARS == 500
    assert properties["name"]["minLength"] == 1
    assert properties["args"]["maxLength"] == MAX_INVOKE_ARGS_CHARS == 4000
    assert thaw_json(tool.definition.input_schema)["additionalProperties"] is False
    assert tool.validate_input({"name": "x" * 500})["name"] == "x" * 500
    assert tool.validate_input({"name": "s", "args": "a" * 4000})["args"] == "a" * 4000
    _assert_rejects(tool, {"name": "x" * 501})
    _assert_rejects(tool, {"name": "s", "args": "a" * 4001})
    _assert_rejects(tool, {"name": ""})
    _assert_rejects(tool, {"name": "s", "extra": 1})
    # 有意差异:旧 available_skill_ids 注入 name enum(integrations/skills.py:
    # 122-129);praxis 目录经 context 纯函数步骤注入(T7),schema 无 enum。
    assert "enum" not in properties["name"]


def test_invoke_skill_tool_attributes_match_legacy(tmp_path: Path) -> None:
    # 旧 integrations/skills.py:144 static_effects=frozenset()、:154 timeout 5.0、
    # :155 max_model_output_bytes=300_000、:146-153 idempotent/COOPERATIVE/CANCEL。
    tool = create_invoke_skill_tool(load_skill_catalog(tmp_path))
    assert tool.static_effects == frozenset()
    assert tool.timeout_seconds == 5.0
    assert tool.max_model_output_bytes == 300_000
    assert tool.idempotent is True
    assert tool.concurrency_safe is False
    assert tool.interrupt_behavior.value == "cancel"
    assert tool.source is ToolSource.BUILTIN
    use = tool.resolve_use({"name": "demo"})
    assert use.effects == frozenset()
    assert [(t.kind, t.value) for t in use.targets] == [("skill", "demo")]


def test_invoke_skill_instructions_cap_40000(tmp_path: Path) -> None:
    # 旧 integrations/skills.py:30 MAX_SKILL_INSTRUCTIONS_CHARS=40_000、
    # :242 截断。40_000 恰好不截;40_001 截断且 truncated=True。
    for size, truncated in ((40_000, False), (40_001, True)):
        workspace = tmp_path / f"ws-{size}"
        workspace.mkdir()
        _write_skill(workspace, "big", body="A" * size)
        tool = create_invoke_skill_tool(load_skill_catalog(workspace))
        result = _run_tool(tool, {"name": "big"})
        assert result.status.value == "ok"
        assert result.truncated is truncated
        text = result.content[0].data["text"]
        instructions = text.split("\n", 1)[1]
        assert len(instructions) == min(size, MAX_SKILL_INSTRUCTIONS_CHARS)
        assert result.metadata["truncated"] is truncated


def test_invoke_skill_substitution_is_literal_not_regex_template(tmp_path: Path) -> None:
    # 旧 skills/runtime.py:90-93:字面 ${SKILL_DIR} 先、word-boundary $SKILL_DIR
    # 正则 lambda、$ARGUMENTS 仅 args 非 None 时替换;替换文本是字面数据,
    # 绝不被当正则模板(\1 反引用不解释)。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    skill_file = _write_skill(
        workspace, "sub", body="a=${SKILL_DIR} b=$SKILL_DIR c=$ARGUMENTS d=$SKILL_DIRECTORY"
    )
    tool = create_invoke_skill_tool(load_skill_catalog(workspace))
    root = str(skill_file.parent.resolve())
    result = _run_tool(tool, {"name": "sub", "args": r"\1$&"})
    text = result.content[0].data["text"]
    assert f"a={root}" in text
    assert f"b={root}" in text
    assert r"c=\1$&" in text  # 反引用/组引用原样保留
    assert "d=$SKILL_DIRECTORY" in text  # word-boundary:不匹配更长标识符
    # args=None ⇒ $ARGUMENTS 不替换(旧 runtime.py:92-93)。
    result2 = _run_tool(tool, {"name": "sub"})
    assert "c=$ARGUMENTS" in result2.content[0].data["text"]


def test_skill_error_codes_praxis_set_vs_legacy(tmp_path: Path) -> None:
    # 旧 SkillErrorCode(skills/models.py:144-151)六码:skill_not_found /
    # skill_disabled / invalid_skill_manifest / skill_source_untrusted /
    # skill_content_changed_on_resume / unsupported_skill_constraints。
    # praxis 有意差异:①trust-model 删除 ⇒ skill_source_untrusted 缺席;
    # ②每次激活都查漂移(不止 resume)⇒ 更名 skill_content_changed。
    codes = set(typing.get_args(SkillErrorCode))
    assert codes == {
        "skill_not_found",
        "skill_disabled",
        "invalid_skill_manifest",
        "skill_content_changed",
        "unsupported_skill_constraints",
    }
    assert "skill_source_untrusted" not in codes
    assert "skill_content_changed_on_resume" not in codes
    # 错误码可达性:未知名 ⇒ skill_not_found;disable ⇒ skill_disabled。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # 注意:skill 名避开 YAML 1.1 布尔字面量(off/on/yes/no 会被 safe_load
    # 解析成 bool ⇒ name 非字符串 ⇒ loader skip;旧 loader 同款行为)。
    _write_skill(workspace, "disabled", body="b", extra_frontmatter="disable_model_invocation: true\n")
    tool = create_invoke_skill_tool(load_skill_catalog(workspace))
    assert _run_tool(tool, {"name": "ghost"}).error_code == "skill_not_found"
    assert _run_tool(tool, {"name": "disabled"}).error_code == "skill_disabled"


def test_skill_content_drift_rejected_on_every_activation(tmp_path: Path) -> None:
    # 旧实现只在 resume 时查 content_changed_on_resume;praxis 每次激活重读
    # 重算(R-T6-1 procedure)。文件在装载后被改 ⇒ skill_content_changed。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    skill_file = _write_skill(workspace, "drift", body="original\n")
    tool = create_invoke_skill_tool(load_skill_catalog(workspace))
    assert _run_tool(tool, {"name": "drift"}).status.value == "ok"
    skill_file.write_text(
        "---\nname: drift\ndescription: test skill\n---\nrewritten\n", encoding="utf-8"
    )
    result = _run_tool(tool, {"name": "drift"})
    assert result.error_code == "skill_content_changed"
    # 文件消失 ⇒ skill_not_found(旧同语义)。
    skill_file.unlink()
    assert _run_tool(tool, {"name": "drift"}).error_code == "skill_not_found"


def test_skill_constraints_excluded_at_loader_and_drift_reachable(tmp_path: Path) -> None:
    # 有意差异(R-T8-2):旧 loader 对 unsupported_constraints 硬拒
    # (skills/loader.py:183-186 抛 SkillLoadError;:390-391 scan 路径降级
    # warning),且 invoke 期再查一次(三处检查)。praxis 收敛为**装载期一处**
    # skip+warning(字段不落清单);unsupported_skill_constraints 只经
    # "装载后文件被改成带约束"的漂移路径可达。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    skill_file = _write_skill(
        workspace, "guarded", body="b", extra_frontmatter="allowed_tools:\n  - bash\n"
    )
    catalog = load_skill_catalog(workspace)
    assert catalog.get("guarded") is None  # 装载期排除
    # 先装一个干净版本,再漂移成带约束。
    skill_file.write_text(
        "---\nname: guarded\ndescription: test skill\n---\nclean\n", encoding="utf-8"
    )
    tool = create_invoke_skill_tool(load_skill_catalog(workspace))
    assert _run_tool(tool, {"name": "guarded"}).status.value == "ok"
    skill_file.write_text(
        "---\nname: guarded\ndescription: test skill\npaths:\n  - src/**\n---\nclean\n",
        encoding="utf-8",
    )
    assert _run_tool(tool, {"name": "guarded"}).error_code == "unsupported_skill_constraints"


def test_skill_sources_trimmed_to_project_external() -> None:
    # 有意差异:旧 SkillSource 七成员(skills/models.py:19-29:BUNDLED/PROJECT/
    # USER/EXTERNAL/MANAGED/MCP/PLUGIN),其中只有 PROJECT/EXTERNAL 有生产者
    # (3B 笔记 §5 grep 结论)。praxis 剪到实际支持的两个根。
    assert set(SkillSource) == {SkillSource.PROJECT, SkillSource.EXTERNAL}
    assert SkillSource.PROJECT.value == "project"
    assert SkillSource.EXTERNAL.value == "external"


def test_skill_fingerprint_recipe_matches_legacy(tmp_path: Path) -> None:
    # 旧 loader 指纹:sha256(yaml.dump(排序 frontmatter) + "\n" + body)
    # (3B 笔记 §5;skills/loader.py)。praxis skill_fingerprint 是唯一配方,
    # 产出 64 位小写 hex(旧 SkillActivationEvent.fingerprint ≤128,
    # integrations/skills.py:62)。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_skill(workspace, "fp", body="body text\n", extra_frontmatter="version: '2'\n")
    catalog = load_skill_catalog(workspace)
    manifest = catalog.get("fp")
    assert manifest is not None
    assert len(manifest.fingerprint) == 64
    assert manifest.fingerprint == manifest.fingerprint.lower()
    frontmatter = {"name": "fp", "description": "test skill", "version": "2"}
    canonical = yaml.dump(frontmatter, sort_keys=True, default_flow_style=False)
    expected = hashlib.sha256(f"{canonical}\nbody text\n".encode()).hexdigest()
    assert manifest.fingerprint == expected
    assert skill_fingerprint(frontmatter, "body text\n") == expected


def test_materialize_schema_and_path_rules_match_legacy(tmp_path: Path) -> None:
    # 旧 integrations/skills.py:70-83 MaterializeSkillAssetInput:skill_id 1..500
    # (:75)、relative_path 1..4096(:80);:328-339 路径校验逐字(相对、无 ..、
    # 首段 scripts|references、≥2 段)。
    tool = create_materialize_skill_asset_tool(tmp_path)
    properties = _schema_properties(tool)
    assert properties["skill_id"]["maxLength"] == 500
    assert properties["relative_path"]["maxLength"] == 4096
    assert tool.validate_input({"skill_id": "s", "relative_path": "scripts/x.py"})
    _assert_rejects(tool, {"skill_id": "s" * 501, "relative_path": "scripts/x.py"})
    _assert_rejects(tool, {"skill_id": "s", "relative_path": "r" * 4097})
    for bad, message in (
        ("/abs/scripts/x.py", "relative without parent traversal"),
        ("scripts/../x.py", "relative without parent traversal"),
        ("../scripts/x.py", "relative without parent traversal"),
        ("assets/x.py", "under scripts/ or references/"),
        ("scripts", "include a file name"),
        ("references", "include a file name"),
    ):
        error = _assert_rejects(tool, {"skill_id": "s", "relative_path": bad})
        assert message in error.message
    # 旧错误文案逐字(integrations/skills.py:330-345)。
    assert (
        _assert_rejects(tool, {"skill_id": "s", "relative_path": "x/y"}).message
        == "skill asset must be under scripts/ or references/"
    )


def test_materialize_tool_attributes_and_divergences(tmp_path: Path) -> None:
    # 旧 integrations/skills.py:209 timeout 5.0、:210 max_model_output_bytes=50_000。
    tool = create_materialize_skill_asset_tool(tmp_path)
    assert tool.timeout_seconds == 5.0
    assert tool.max_model_output_bytes == 50_000
    assert tool.idempotent is True
    assert tool.concurrency_safe is False
    # 有意差异①:旧 static_effects={WRITE_WORKSPACE}(:184);praxis 加
    # READ_WORKSPACE(读 skill 源文件是真实副作用面)。
    assert tool.static_effects == frozenset({ToolEffect.READ_WORKSPACE, ToolEffect.WRITE_WORKSPACE})
    # 有意差异②:旧 interrupt_behavior=FINISH_CURRENT(:208);praxis CANCEL
    # (同步 run 经 to_thread 派发,O_NONBLOCK/S_ISREG 使阻塞有界)。
    assert tool.interrupt_behavior.value == "cancel"
    # 有意差异③:旧 target kinds active_skill/workspace_path(:186-201);
    # praxis skill/path(与唯一 policy 代数同词汇)。
    use = tool.resolve_use({"skill_id": "demo", "relative_path": "scripts/x.py"})
    assert [t.kind for t in use.targets] == ["skill", "path"]
    assert use.targets[0].value == "demo"


def test_materialize_destination_is_scratch_not_floor_praxis(tmp_path: Path) -> None:
    # 有意差异(Q8):旧落点 workspace.scratch/skills/<safe_id>/<rel>
    # (integrations/skills.py:325),scratch=.praxis/runtime/scratch
    # (workspace.py:99 runtime_root/scratch)。.praxis 是 FLOOR_ENTRIES deny
    # (protocol/floor.py:51)⇒ 沙箱内 bash/run_python 读不到资产。praxis 落
    # .scratch(工作区内普通目录,执行类工具可直接消费;需自行 gitignore)。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    skill_file = _write_skill(workspace, "demo", body="b")
    scripts = skill_file.parent / "scripts"
    scripts.mkdir()
    (scripts / "run.py").write_text("print('demo')\n", encoding="utf-8")
    tool = create_materialize_skill_asset_tool(workspace)
    result = _run_tool(
        tool,
        {"skill_id": "demo", "relative_path": "scripts/run.py"},
        ToolExecContext(active_skill_ids=frozenset({"demo"})),
    )
    assert result.status.value == "ok"
    rel = result.metadata["workspace_path"]
    assert rel == ".scratch/skills/demo/scripts/run.py"
    assert (workspace / rel).read_text(encoding="utf-8") == "print('demo')\n"
    assert result.metadata["size_bytes"] == len(b"print('demo')\n")
    assert result.metadata["source_fingerprint"] == hashlib.sha256(b"print('demo')\n").hexdigest()
    # 地板对照:.praxis 落点被地板拒,.scratch 落点放行(理由可诊断)。
    private = host_private_paths()
    old_dest = workspace / ".praxis" / "runtime" / "scratch" / "skills" / "demo" / "run.py"
    new_dest = workspace / rel
    assert floor_reason(old_dest, write=True, private_paths=private) != ""
    assert floor_reason(new_dest, write=True, private_paths=private) == ""
    # 描述文案明确告知 .scratch 不自动 gitignore(Q8 代价)。
    assert "NOT git-ignored automatically" in tool.definition.description


def test_materialize_preserves_source_mode_bits(tmp_path: Path) -> None:
    # 旧 shutil.copy2(integrations/skills.py:311)保留权限位;praxis 经
    # atomic_write_bytes(mode=...) 在新落点保真(M5:exec bit 不丢)。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    skill_file = _write_skill(workspace, "exec", body="b")
    scripts = skill_file.parent / "scripts"
    scripts.mkdir()
    source = scripts / "tool.sh"
    source.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    source.chmod(0o755)
    tool = create_materialize_skill_asset_tool(workspace)
    result = _run_tool(
        tool,
        {"skill_id": "exec", "relative_path": "scripts/tool.sh"},
        ToolExecContext(active_skill_ids=frozenset({"exec"})),
    )
    assert result.status.value == "ok"
    dest = workspace / result.metadata["workspace_path"]
    assert stat.S_IMODE(dest.stat().st_mode) == 0o755


def test_materialize_asset_size_cap_is_praxis_only(tmp_path: Path) -> None:
    # 有意差异(praxis-only 防御,T9 M2):旧实现无资产大小上限;praxis 整块
    # 进内存复制必须有界 ⇒ 4MiB(4_194_304)上限 + 新码 asset_too_large。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    skill_file = _write_skill(workspace, "big", body="b")
    scripts = skill_file.parent / "scripts"
    scripts.mkdir()
    with (scripts / "big.bin").open("wb") as stream:
        stream.truncate(4_194_304 + 1)  # 稀疏文件,不真实分配
    tool = create_materialize_skill_asset_tool(workspace)
    result = _run_tool(
        tool,
        {"skill_id": "big", "relative_path": "scripts/big.bin"},
        ToolExecContext(active_skill_ids=frozenset({"big"})),
    )
    assert result.error_code == "asset_too_large"
    assert "4194304" in result.error_message
    assert not (workspace / ".scratch" / "skills" / "big" / "scripts" / "big.bin").exists()


def test_materialize_requires_activation_fact(tmp_path: Path) -> None:
    # 旧 skills×权限唯一耦合:active_skill_ids 从已提交事实重算
    # (permissions.py:59/:429-436,tool_orchestrator:1476-1499)。praxis 同构:
    # 执行侧事实查询 exec_ctx.active_skill_ids,未激活 ⇒ skill_not_active。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    skill_file = _write_skill(workspace, "demo", body="b")
    (skill_file.parent / "scripts").mkdir()
    (skill_file.parent / "scripts" / "x.py").write_text("x\n", encoding="utf-8")
    tool = create_materialize_skill_asset_tool(workspace)
    result = _run_tool(tool, {"skill_id": "demo", "relative_path": "scripts/x.py"})
    assert result.error_code == "skill_not_active"
    ok = _run_tool(
        tool,
        {"skill_id": "demo", "relative_path": "scripts/x.py"},
        ToolExecContext(active_skill_ids=frozenset({"demo"})),
    )
    assert ok.status.value == "ok"


# ===========================================================================
# web:validate 全拒形 / 8 候选 / 20s deadline / min(3.0,rem/n) / 剥凭证四头 /
#      Location 4096 / 5 跳 / gzip 解码后限 / 2_000_000 / 65_536 / 12000 窗 /
#      行窗 / 45s+75s / 8MB 单响应含主文档 / 64 req / 8MB net / DOM 2MB /
#      render 30s / 无账本
# ===========================================================================


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/",
        "file:///etc/passwd",
        "https://user:pass@example.com/",
        "https://example.com:8443/",
        "https://example.com:9999/",
        "https://exa%mple.com/",
        "https://localhost/",
        "https://foo.localhost/",
        "https://127.0.0.1/",
        "https://10.0.0.1/",
        "https://192.168.1.1/",
        "https://[::1]/",
        "https://[::ffff:127.0.0.1]/",
        "https://[::ffff:7f00:1]/",
        "",
        "https://example.com/\x00",
        "https://example.com/\n",
        "https://💩.example/",
    ],
)
def test_validate_public_url_rejection_forms_match_legacy(url: str) -> None:
    # 旧 web_http.py:82-109 全拒形:scheme∉{http,https}、userinfo、端口
    # ∉{None,80,443}、host 含 %、localhost/.localhost、非公网字面 IP
    # (:72-79 IPv4-mapped 先还原)、空白/控制字符。praxis 另加 IDNA 三层
    # (R-T10-1,💩 类在预检层拒——旧 httpx.URL 预检同样拒,语义一致)。
    with pytest.raises(PublicWebError) as excinfo:
        validate_public_url(url)
    assert excinfo.value.code == "invalid_url"
    assert str(excinfo.value) == "A public HTTP(S) URL on port 80 or 443 is required."


def test_validate_public_url_normalization_matches_legacy() -> None:
    # 旧 web_http.py:88-89 host 小写去尾点;:109 fragment 丢弃。
    assert validate_public_url("HTTPS://EXAMPLE.COM./Path#frag") == ("example.com", 443, "/Path")
    assert validate_public_url("http://example.com") == ("example.com", 80, "/")
    assert validate_public_url("http://example.com:443/x?q=1") == ("example.com", 443, "/x?q=1")
    # 字面 IP 公网判定:IPv4-mapped 还原后判(旧 :72-79)。
    assert is_public_address("::ffff:93.184.216.34") is True
    assert is_public_address("::ffff:127.0.0.1") is False
    assert is_public_address("example.com") is False  # 主机名不在预检裁决


def test_web_fetch_invalid_url_maps_to_legacy_message(tmp_path: Path) -> None:
    # 旧 builtins/web.py:116 的文案逐字;praxis 映射 invalid_input($.url)。
    tool = _fetch_tool(_offline_client([]))
    error = _assert_rejects(tool, {"url": "http://127.0.0.1/"})
    assert error.path == "$.url"
    assert error.message == "a public HTTP(S) URL without credentials on port 80 or 443 is required"


def test_connection_candidates_eight_cap_and_shared_deadline() -> None:
    # 旧 web_http.py:153 去重后 [:8];:155 共享 deadline 默认 20s;
    # :172 每候选预算 min(3.0, remaining/剩余数)。
    class _StubBackend:
        async def connect_tcp(self, *args: Any, **kwargs: Any) -> Any:
            raise OSError("stub")

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        addresses = [f"93.184.216.{i}" for i in range(1, 13)] + ["93.184.216.1"]
        candidates = _ConnectionCandidates(_StubBackend(), addresses, 443, None, None, None)  # type: ignore[arg-type]
        assert len(candidates.addresses) == 8  # 去重 + 上限 8
        assert candidates.addresses[0] == "93.184.216.1"
        assert candidates.deadline - loop.time() == pytest.approx(20.0, abs=1.0)
        # 默认 deadline 20s、8 候选 ⇒ 首候选预算 min(3.0, 20/8)=2.5。
        budget = min(3.0, candidates.remaining() / len(candidates.addresses))
        assert budget == pytest.approx(2.5, abs=0.2)
        # 小预算:timeout=2、4 候选 ⇒ 每候选 0.5(为其他候选保留时间)。
        small = _ConnectionCandidates(_StubBackend(), addresses[:4], 443, 2.0, None, None)  # type: ignore[arg-type]
        assert min(3.0, small.remaining() / 4) == pytest.approx(0.5, abs=0.1)
        # 大预算:timeout=60、4 候选 ⇒ 每候选帽 3.0。
        big = _ConnectionCandidates(_StubBackend(), addresses[:4], 443, 60.0, None, None)  # type: ignore[arg-type]
        assert min(3.0, big.remaining() / 4) == 3.0

    asyncio.run(scenario())


def test_cross_hop_strips_credentials_to_four_header_whitelist() -> None:
    # 旧 web_http.py:538-544:跨跳只留 accept/accept-language/accept-encoding/
    # user-agent。Authorization/Cookie 绝不随重定向外泄。
    redirect = response_chunks(
        "302 Found", headers=[("Location", "https://example.com/landing")], body=b""
    )
    landing = response_chunks(body=b"<p>ok</p>", headers=[("Content-Type", "text/html")])
    backend = FakeBackend([FakeStream(redirect), FakeStream(landing)])
    client = PublicHttpClient(
        authorize=lambda _url: None,
        limits=WebLimits(),
        resolver=FakeResolver([PUBLIC_IP]),
        backend=backend,
        environ={},
    )
    response = asyncio.run(
        client.fetch(
            "https://example.com/start",
            headers={"Authorization": "Bearer secret", "Cookie": "sid=1", "User-Agent": "praxis"},
        )
    )
    assert response.status == 200
    assert len(backend.issued) == 2
    second_request = backend.issued[1].writes[0].lower()
    assert b"authorization" not in second_request
    assert b"cookie" not in second_request
    assert b"user-agent: praxis" in second_request
    assert b"accept-encoding: gzip, identity" in second_request


def test_location_4096_boundary_and_five_hop_limit() -> None:
    # 旧 web_http.py:517 Location join ≤4096;:370/:527 max_redirects=5,
    # 第 6 跳 too_many_redirects;无 Location ⇒ invalid_redirect。
    def hop(location: str) -> list[bytes]:
        return response_chunks("302 Found", headers=[("Location", location)], body=b"")

    # 4096 字符恰好放行(构造长度恰好的 path)。
    path = "/" + "a" * (4096 - len("https://example.com/"))
    assert len("https://example.com" + path) == 4096
    ok_backend = FakeBackend(
        [FakeStream(hop(path)), FakeStream(response_chunks(body=b"ok", headers=[("Content-Type", "text/plain")]))]
    )
    client = PublicHttpClient(
        authorize=lambda _u: None, limits=WebLimits(),
        resolver=FakeResolver([PUBLIC_IP]), backend=ok_backend, environ={},
    )
    assert asyncio.run(client.fetch("https://example.com/")).status == 200
    # 4097 ⇒ invalid_redirect。
    long_path = path + "a"
    over_backend = FakeBackend([FakeStream(hop(long_path))])
    client2 = PublicHttpClient(
        authorize=lambda _u: None, limits=WebLimits(),
        resolver=FakeResolver([PUBLIC_IP]), backend=over_backend, environ={},
    )
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client2.fetch("https://example.com/"))
    assert excinfo.value.code == "invalid_redirect"
    # 5 跳放行、第 6 跳拒绝(hop 0..4 各消费一次重定向,hop 5 取终态)。
    streams = [FakeStream(hop(f"https://example.com/{i}")) for i in range(1, 6)]
    streams.append(FakeStream(response_chunks(body=b"end", headers=[("Content-Type", "text/plain")])))
    client3 = PublicHttpClient(
        authorize=lambda _u: None, limits=WebLimits(),
        resolver=FakeResolver([PUBLIC_IP]), backend=FakeBackend(streams), environ={},
    )
    assert asyncio.run(client3.fetch("https://example.com/0")).status == 200
    client4 = PublicHttpClient(
        authorize=lambda _u: None, limits=WebLimits(),
        resolver=FakeResolver([PUBLIC_IP]),
        backend=FakeBackend([FakeStream(hop(f"https://example.com/{i}")) for i in range(7)]),
        environ={},
    )
    with pytest.raises(PublicWebError) as excinfo4:
        asyncio.run(client4.fetch("https://example.com/0"))
    assert excinfo4.value.code == "too_many_redirects"
    # 无 Location ⇒ invalid_redirect(旧 :531-537 同语义)。
    client5 = PublicHttpClient(
        authorize=lambda _u: None, limits=WebLimits(),
        resolver=FakeResolver([PUBLIC_IP]),
        backend=FakeBackend([FakeStream(response_chunks("302 Found", body=b""))]),
        environ={},
    )
    with pytest.raises(PublicWebError) as excinfo5:
        asyncio.run(client5.fetch("https://example.com/"))
    assert excinfo5.value.code == "invalid_redirect"


def test_gzip_decoded_byte_limit_and_encoding_whitelist() -> None:
    # 旧 web_http.py:575-611:gzip 增量解压、**解码后**同限;非 gzip/identity
    # 拒绝(旧 unsupported_encoding/invalid_encoding,praxis 统一
    # encoding_unsupported——有意差异,码集见 CLIENT_ERROR_CODES 断言)。
    bomb = gzip.compress(b"A" * 100_000)
    assert len(bomb) < 4096  # 压缩体很小
    client = _offline_client(
        [FakeStream(response_chunks(body=bomb, headers=[("Content-Encoding", "gzip"), ("Content-Type", "text/plain")]))],
        limits=WebLimits(max_bytes=4096),
    )
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com/"))
    assert excinfo.value.code == "max_bytes_exceeded"
    # 解码后恰好等于上限 ⇒ 放行。
    exact = gzip.compress(b"B" * 4096)
    client2 = _offline_client(
        [FakeStream(response_chunks(body=exact, headers=[("Content-Encoding", "gzip"), ("Content-Type", "text/plain")]))],
        limits=WebLimits(max_bytes=4096),
    )
    assert len(asyncio.run(client2.fetch("https://example.com/")).body) == 4096
    # br 编码拒绝。
    client3 = _offline_client(
        [FakeStream(response_chunks(body=b"x", headers=[("Content-Encoding", "br"), ("Content-Type", "text/plain")]))]
    )
    with pytest.raises(PublicWebError) as excinfo3:
        asyncio.run(client3.fetch("https://example.com/"))
    assert excinfo3.value.code == "encoding_unsupported"


def test_web_defaults_and_no_turn_budget_ledger() -> None:
    # 旧 web_http.py:368-370 默认 max_bytes=2_000_000 / timeout=20 /
    # max_redirects=5;web_limits.py:9-11 browser_requests=64(1..128)、
    # browser_network_bytes=8_000_000(1e6..32e6)。
    limits = WebLimits()
    assert (limits.max_bytes, limits.timeout_seconds, limits.max_redirects) == (2_000_000, 20.0, 5)
    assert (limits.browser_requests, limits.browser_network_bytes) == (64, 8_000_000)
    # T15(Q8)兑现:旧 turn_requests=32(1..256,web_limits.py:9)回归字段集,
    # 作为 composition 折进 root Limits 的默认来源。有意差异(T17 对照表钉):
    # 旧 reserve_resource_unit SQLite 原子预扣 → core.budget 事实折叠+进程内
    # gate(跨进程原子归阶段 5);web_request_budget_exceeded 错误码住在
    # core.pipeline 的 dispatch gate,**不在** web client/工具层(3B 逐请求
    # 硬上限面不含 turn 账本机制)。
    field_names = {field.name for field in dataclasses.fields(WebLimits)}
    assert field_names == {
        "max_bytes", "timeout_seconds", "max_redirects", "turn_requests",
        "browser_requests", "browser_network_bytes",
    }
    assert WebLimits(turn_requests=32).turn_requests == 32
    assert "web_request_budget_exceeded" not in CLIENT_ERROR_CODES
    import praxis.adapters.web.tools as web_tools_module

    assert "web_request_budget_exceeded" not in _code_string_constants(web_tools_module)
    # fetch 路径自产码集逐字(T10 有意差异:http_error 不抛、network_error
    # 拆 connect_failed/tls_failed、request/response_too_large 统一
    # max_bytes_exceeded、unsupported/invalid_encoding 统一 encoding_unsupported)。
    assert CLIENT_ERROR_CODES == frozenset(
        {
            "nonpublic_address", "invalid_redirect", "too_many_redirects",
            "max_bytes_exceeded", "encoding_unsupported", "connect_failed",
            "tls_failed", "timeout", "denied",
        }
    )


def test_web_limits_file_is_path_not_inline_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 旧 cli.py:97-99/_load_web_limits:794-803:PRAXIS_WEB_LIMITS 值是
    # **JSON 文件路径**(≤4096 字节),不是内联 JSON。
    limits_file = tmp_path / "limits.json"
    limits_file.write_text(json.dumps({"max_bytes": 8192}), encoding="utf-8")
    monkeypatch.setenv("PRAXIS_WEB_LIMITS", str(limits_file))
    assert load_web_limits().max_bytes == 8192
    monkeypatch.setenv("PRAXIS_WEB_LIMITS", '{"max_bytes": 8192}')
    with pytest.raises(ValueError):  # 内联 JSON 不是合法路径
        load_web_limits()
    monkeypatch.setenv("PRAXIS_WEB_LIMITS", str(limits_file))
    # T15:turn_requests 是合法扁平键(旧 WebLimits 形状 parity);未知键仍拒。
    limits_file.write_text(json.dumps({"max_bytes": 8192, "turn_requests": 8}), encoding="utf-8")
    assert load_web_limits() == WebLimits(max_bytes=8192, turn_requests=8)
    limits_file.write_text(json.dumps({"max_bytes": 8192, "nope": 8}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_web_limits()
    # 文件 >4096 字节拒绝(旧 read(4097) 判 >4096)。
    limits_file.write_text(" " * 4097, encoding="utf-8")
    with pytest.raises(ValueError, match="4096"):
        load_web_limits()


def test_web_fetch_schema_absent_keys_and_bounds() -> None:
    # 旧 WebFetchInput(builtins/web.py:71-101):url XOR source_id、
    # compare_to_source_id、view、section_id、start_link、start_section、
    # find、max_tokens、refresh 都是模型可见面。praxis 有意差异
    # (R-PF-2'/S8 工具化):这些键**全部缺席**——超限回读由管道外部化 +
    # read_artifact 覆盖,不建第二套 source_id/截断体系。render 自 T14 入
    # schema(默认 false,与旧默认 auto 有意差异:渲染起进程,显式选择才走)。
    tool = _fetch_tool(_offline_client([]))
    properties = _schema_properties(tool)
    assert set(properties) == {"url", "max_bytes", "timeout_seconds", "start_line", "max_lines", "render"}
    for absent in (
        "source_id", "compare_to_source_id", "view", "section_id",
        "start_link", "start_section", "find", "max_tokens", "refresh",
    ):
        assert absent not in properties
        _assert_rejects(tool, {"url": "https://example.com/", absent: "x"})
    # 保留键的边界逐字对齐旧 :73/:82-85。
    assert properties["url"]["maxLength"] == 4096
    assert (properties["max_bytes"]["minimum"], properties["max_bytes"]["maximum"], properties["max_bytes"]["default"]) == (4096, 16000, 12000)
    assert (properties["max_lines"]["minimum"], properties["max_lines"]["maximum"], properties["max_lines"]["default"]) == (1, 500, 100)
    assert properties["start_line"]["minimum"] == 1
    assert properties["render"]["default"] is False  # 旧 :99-100 默认 'auto'
    assert tool.validate_input({"url": "https://example.com/", "max_bytes": 16000})["max_bytes"] == 16000
    _assert_rejects(tool, {"url": "https://example.com/", "max_bytes": 16001})
    _assert_rejects(tool, {"url": "https://example.com/", "max_bytes": 4095})
    _assert_rejects(tool, {"url": "https://example.com/", "max_lines": 501})
    _assert_rejects(tool, {"url": "https://example.com/", "start_line": 0})


def test_web_tool_timeouts_and_output_caps_match_legacy() -> None:
    # 旧 builtins/web.py:751 web_search timeout 45.0 / max_model_output_bytes
    # 65_536;:853 web_fetch 75.0 / 65_536。
    fetch = _fetch_tool(_offline_client([]))
    assert fetch.timeout_seconds == 75.0
    assert fetch.max_model_output_bytes == 65_536
    (search,) = create_web_search_tools(
        _offline_client([]), config_provider=lambda: None, credential_provider=lambda _cfg: None
    )
    assert search.timeout_seconds == 45.0
    assert search.max_model_output_bytes == 65_536
    # 模型参数不能抬高宿主硬上限:timeout 钳制 min(requested, limits, 75)。
    properties = _schema_properties(fetch)
    assert (properties["timeout_seconds"]["minimum"], properties["timeout_seconds"]["maximum"], properties["timeout_seconds"]["default"]) == (5.0, 120.0, 45.0)


def test_web_fetch_excerpt_budget_is_raw_utf8_not_json_escaped() -> None:
    # 有意差异:旧行窗预算按 json.dumps(line) 转义后字节计(builtins/web.py:
    # 980:每个 " 转义成 \" 计 2 字节);praxis 按渲染行的**原始 UTF-8** 字节
    # (tools.py _window)。20 行 × 400 个双引号:raw 每行 ≈404 字节 ⇒ 4096
    # 预算放得下 10 行;旧口径每行 ≈805 字节 ⇒ 只放得下 5 行。断言恰 10 行
    # ⇒ 钉死 raw-byte 口径(若按 JSON 转义计,此断言必红)。
    body = "".join('"' * 400 + "<br>" for _ in range(20))
    client = _offline_client(
        [FakeStream(response_chunks(body=body.encode(), headers=[("Content-Type", "text/html")]))]
    )
    tool = _fetch_tool(client)
    result = _run_tool(tool, {"url": "https://example.com/", "max_bytes": 4096, "max_lines": 500})
    assert result.status.value == "ok"
    text = result.content[0].data["text"]
    assert result.metadata["end_line"] == 10
    assert text.count("\n") == 9
    assert '"' * 400 in text  # 引号未被转义口径提前截断
    assert len(text.encode("utf-8")) <= 4096
    # 多字节字符按 UTF-8 字节计(不是字符数):中文字符 3 字节。
    client2 = _offline_client(
        [FakeStream(response_chunks(body=("<p>" + "中" * 2000 + "</p>").encode(), headers=[("Content-Type", "text/html")]))]
    )
    result2 = _run_tool(_fetch_tool(client2), {"url": "https://example.com/", "max_bytes": 4096})
    assert result2.status.value == "ok"
    assert len(result2.content[0].data["text"].encode("utf-8")) <= 4096


def test_web_fetch_line_window_512_wrap_and_cursor() -> None:
    # 旧 builtins/web.py:859-862 _source_lines:真实行按 512 字符切虚拟行;
    # :83 max_lines 1..500 默认 100;行窗游标 start_line/end_line。
    assert source_lines("a" * 1025) == ["a" * 512, "a" * 512, "a"]
    assert source_lines("") == []  # splitlines 语义;空文本先被 web_content_empty 拒
    long_line = "x" * 1200  # 3 虚拟行
    client = _offline_client(
        [FakeStream(response_chunks(body=f"<p>{long_line}</p>".encode(), headers=[("Content-Type", "text/html")]))]
    )
    tool = _fetch_tool(client)
    result = _run_tool(tool, {"url": "https://example.com/", "max_lines": 2, "max_bytes": 16000})
    assert result.metadata["line_count"] == 3
    assert result.metadata["start_line"] == 1
    assert result.metadata["end_line"] == 2
    assert result.metadata["truncated"] is True  # 行窗可续读
    assert result.metadata["source_truncated"] is False
    # 续读:start_line=end_line+1。
    client2 = _offline_client(
        [FakeStream(response_chunks(body=f"<p>{long_line}</p>".encode(), headers=[("Content-Type", "text/html")]))]
    )
    result2 = _run_tool(_fetch_tool(client2), {"url": "https://example.com/", "start_line": 3, "max_bytes": 16000})
    assert result2.metadata["end_line"] == 3
    assert result2.metadata["truncated"] is False
    # start_line 越界 ⇒ web_line_out_of_range(旧 web_line_out_of_range 同码)。
    client3 = _offline_client(
        [FakeStream(response_chunks(body=b"<p>one</p>", headers=[("Content-Type", "text/html")]))]
    )
    result3 = _run_tool(_fetch_tool(client3), {"url": "https://example.com/", "start_line": 99})
    assert result3.error_code == "web_fetch_failed:web_line_out_of_range"


def test_web_extract_source_cap_200000_and_no_links_surface() -> None:
    # 旧 web_content.py:24 MAX_SOURCE_CHARACTERS=200_000;:25 MAX_SOURCE_LINKS=200。
    # praxis 保留 200_000 抽取上限;**有意差异**:links 清单/view=links 面
    # 不移植(超限回读走 read_artifact),解析在进程内线程(旧 worker 子进程
    # 10s+RLIMIT_CPU 8s/RLIMIT_AS 768MB,web_content.py:57/:93-95)——
    # ExtractedText 只有 text/truncated 两字段。
    assert MAX_SOURCE_CHARACTERS == 200_000
    extracted = extract_text(("<p>" + "word " * 60_000 + "</p>").encode(), "text/html")
    assert extracted.truncated is True
    assert len(extracted.text) <= 200_000
    assert {field.name for field in dataclasses.fields(extracted)} == {"text", "truncated"}
    # 二进制/未知类型拒收码(旧 web_content_unsupported 同码)。
    with pytest.raises(PublicWebError) as excinfo:
        extract_text(b"\x00\x01", "application/octet-stream")
    assert excinfo.value.code == "web_content_unsupported"
    with pytest.raises(PublicWebError) as excinfo2:
        extract_text(b"<html><body>   </body></html>", "text/html")
    assert excinfo2.value.code == "web_content_empty"


def test_web_search_schema_divergences_from_legacy() -> None:
    # 旧 builtins/web.py:56-57:query 1..2000、max_results 1..10 默认 5。
    # praxis 有意差异(T13):query 1..512、max_results 1..20 默认 8
    # (更窄的模型面;provider 计数上限由工具侧钳制)。
    (search,) = create_web_search_tools(
        _offline_client([]), config_provider=lambda: None, credential_provider=lambda _cfg: None
    )
    properties = _schema_properties(search)
    assert properties["query"]["maxLength"] == 512
    assert (MIN_MAX_RESULTS, MAX_RESULTS_LIMIT, DEFAULT_MAX_RESULTS) == (1, 20, 8)
    assert properties["max_results"]["default"] == 8
    _assert_rejects(search, {"query": "q" * 513})
    _assert_rejects(search, {"query": "q", "max_results": 21})
    # 未配置 ⇒ web_search_unconfigured(旧 web_search_not_configured,
    # builtins/web.py:313——有意更名,零网络)。
    result = _run_tool(search, {"query": "praxis"})
    assert result.error_code == "web_search_unconfigured"
    # 有意差异:旧 key_file 配置面(builtins/web.py web.json {provider,key_file})
    # 不采用——praxis 只存 api_key_env(变量名),值走 credentials 纪律;
    # PRAXIS_WEB_SEARCH_KEY_FILE 完全惰性。resolve_use 未配置时无 url target。
    use = search.resolve_use({"query": "q"})
    assert use.effects == frozenset({ToolEffect.NETWORK})
    assert use.targets == ()


def test_browser_constants_match_legacy_and_single_response_includes_main() -> None:
    # 旧 web_browser.py:27-30 MAX_REQUESTS=64 / MAX_NETWORK_BYTES=8_000_000 /
    # MAX_DOM_BYTES=2_000_000 / RENDER_TIMEOUT_SECONDS=30。
    from praxis.adapters.web import browser

    assert browser.MAX_REQUESTS == 64
    assert browser.MAX_NETWORK_BYTES == 8_000_000
    assert browser.MAX_DOM_BYTES == 2_000_000
    assert browser.RENDER_TIMEOUT_SECONDS == 30
    # 有意差异(T14):旧单响应 8MB 只施加于 Script 子资源(:152),主文档走
    # 客户端普通上限(默认 2MB);praxis 主文档与子资源**同一口径**
    # min(8MB, 剩余 browser 预算)——>2MB 页面默认可读,宿主抬 max_bytes 到
    # 32MB 也不会一次读穿 8MB 单响应帽。RenderedPage 字段面钉死计数口径。
    assert {field.name for field in dataclasses.fields(browser.RenderedPage)} == {
        "text", "final_url", "requests_used", "bytes_used", "dom_bytes", "truncated",
        "blocked_ask", "blocked_deny", "blocked_other", "budget_aborted",
        "budget_stopped", "failed_resources", "redirected_subresources", "load_timeouts",
    }
    # 有意差异:旧浏览器 env 白名单六键(HOME/TMPDIR/TMP/TEMP/PATH/LANG,
    # web_browser.py:231-235);praxis 收窄到三键(HOME/TMPDIR/PATH,
    # 控制器裁决;常量只允许出现在 browser.py,CI 规则⑳)。
    assert browser._BROWSER_PATH == "/usr/bin:/bin"
    # 出口硬禁(praxis-only,review R1):WebRTC 非代理 UDP + WebTransport。
    assert "--force-webrtc-ip-handling-policy=disable_non_proxied_udp" in browser._HARDENING_ARGS
    assert "--disable-features=WebTransport" in browser._HARDENING_ARGS
    # 死代理纵深防御逐字(旧 :300-303)。
    assert browser._DEAD_PROXY == {"server": "http://127.0.0.1:9", "bypass": "<-loopback>"}
    # browser 预算范围逐字(旧 web_limits.py:10-11)。
    with pytest.raises(ValueError):
        WebLimits(browser_requests=129)
    with pytest.raises(ValueError):
        WebLimits(browser_requests=0)
    with pytest.raises(ValueError):
        WebLimits(browser_network_bytes=999_999)
    with pytest.raises(ValueError):
        WebLimits(browser_network_bytes=32_000_001)


def test_web_fetch_render_face_effects_split() -> None:
    # R-T14-2:render-capable 面 static_effects={NETWORK,EXECUTE_PROCESS}
    # (浏览器 helper 起进程);resolve_use 按参数分流——render=false 只
    # {NETWORK}(不起进程),true/auto 加 EXECUTE_PROCESS。纯文本面
    # (renderer_factory=None)static_effects 只 {NETWORK}。
    plain = _fetch_tool(_offline_client([]))
    assert plain.static_effects == frozenset({ToolEffect.NETWORK})
    assert plain.resolve_use({"url": "https://example.com/"}).effects == frozenset({ToolEffect.NETWORK})

    def _factory() -> Any:
        raise AssertionError("parity must not render")

    capable = _fetch_tool(_offline_client([]), renderer_factory=_factory)
    assert capable.static_effects == frozenset({ToolEffect.NETWORK, ToolEffect.EXECUTE_PROCESS})
    assert capable.resolve_use({"url": "https://example.com/"}).effects == frozenset({ToolEffect.NETWORK})
    for render in (True, "auto"):
        use = capable.resolve_use({"url": "https://example.com/", "render": render})
        assert use.effects == frozenset({ToolEffect.NETWORK, ToolEffect.EXECUTE_PROCESS})
    # target 是模型给的原始 URL 串(归一化归 policy/authorizer,T11 纪律)。
    assert capable.resolve_use({"url": "HTTPS://Example.COM./x"}).targets[0].value == "HTTPS://Example.COM./x"
    # 纯文本面 render=true ⇒ render_unavailable(可诊断、零网络)。
    result = _run_tool(plain, {"url": "https://example.com/", "render": True})
    assert result.error_code == "render_unavailable"


# ===========================================================================
# MCP:4096 / 256 / 4MiB / 20.0 启动 / 30.0 调用 / 1_000_000 / read_only=True /
#      enabled=False / envp-S7 / config sha 漂移唯一守卫
# ===========================================================================


def test_mcp_catalog_resource_limits_match_legacy() -> None:
    # 旧 runtime/mcp.py:42-44 _MAX_CATALOG_TOOLS=4096 / _MAX_CATALOG_PAGES=256 /
    # _MAX_CATALOG_BYTES=4MiB;:522-549 全或无分页。
    assert (_MAX_CATALOG_TOOLS, _MAX_CATALOG_PAGES, _MAX_CATALOG_BYTES) == (4096, 256, 4 * 1024 * 1024)

    class _Pages:
        def __init__(self, pages: dict[str | None, ListToolsResult]) -> None:
            self.pages = pages
            self.cursors: list[str | None] = []

        async def list_tools(self, params: Any = None) -> ListToolsResult:
            cursor = getattr(params, "cursor", None)
            self.cursors.append(cursor)
            return self.pages[cursor]

    def tool_page(names: Sequence[str], next_cursor: str | None = None) -> ListToolsResult:
        return ListToolsResult(tools=[_mcp_tool(name) for name in names], nextCursor=next_cursor)

    # 4096 恰好放行;4097 拒绝(全或无,绝不发布半页)。
    ok = _Pages({None: tool_page([f"t{i}" for i in range(4096)])})
    assert len(asyncio.run(list_all_tools(ok))) == 4096  # type: ignore[arg-type]
    over = _Pages({None: tool_page([f"t{i}" for i in range(4097)])})
    with pytest.raises(ValueError, match="resource limit"):
        asyncio.run(list_all_tools(over))  # type: ignore[arg-type]
    # 字节口径:tool.model_dump_json() UTF-8 长度 >4MiB 拒绝。
    huge = McpTool(name="huge", description="x" * (4 * 1024 * 1024), inputSchema={"type": "object"})
    bytes_page = _Pages({None: ListToolsResult(tools=[huge])})
    with pytest.raises(ValueError, match="resource limit"):
        asyncio.run(list_all_tools(bytes_page))  # type: ignore[arg-type]
    # 重复工具名拒绝(跨页)。
    dup = _Pages({None: tool_page(["a"], "c1"), "c1": tool_page(["a"])})
    with pytest.raises(ValueError, match="duplicate tool name"):
        asyncio.run(list_all_tools(dup))  # type: ignore[arg-type]
    # 重复 cursor 拒绝(server 打转)。
    loop = _Pages({None: tool_page(["a"], "c1"), "c1": tool_page(["b"], "c1")})
    with pytest.raises(ValueError, match="repeated or empty cursor"):
        asyncio.run(list_all_tools(loop))  # type: ignore[arg-type]
    # 空 cursor 拒绝。
    empty = _Pages({None: tool_page(["a"], "")})
    with pytest.raises(ValueError, match="repeated or empty cursor"):
        asyncio.run(list_all_tools(empty))  # type: ignore[arg-type]
    # 256 页恰好放行;257 页拒绝。
    pages: dict[str | None, ListToolsResult] = {}
    cursor: str | None = None
    for index in range(256):
        next_cursor = f"c{index}" if index < 255 else None
        pages[cursor] = tool_page([f"p{index}"], next_cursor)
        cursor = next_cursor
    assert len(asyncio.run(list_all_tools(_Pages(pages)))) == 256  # type: ignore[arg-type]
    pages2: dict[str | None, ListToolsResult] = {}
    cursor = None
    for index in range(257):
        next_cursor = f"c{index}"
        pages2[cursor] = tool_page([f"p{index}"], next_cursor)
        cursor = next_cursor
    with pytest.raises(ValueError, match="page limit"):
        asyncio.run(list_all_tools(_Pages(pages2)))  # type: ignore[arg-type]


def test_mcp_config_defaults_match_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 旧 runtime/mcp.py:718 enabled 默认 False;:743 startup_timeout 默认 20.0;
    # :795 HTTP read_only 默认 True;transport 默认 stdio;allowlist/allow_all
    # 至少其一(:751-756)。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    host = tmp_path / "host"
    host.mkdir()
    host.chmod(0o700)
    config = host / "mcp_servers.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "servers": [
                    {"name": "off", "command": "/bin/echo", "allow_all_tools": True},
                    {
                        "name": "on",
                        "enabled": True,
                        "command": "/bin/echo",
                        "allow_all_tools": True,
                    },
                    {
                        "name": "remote",
                        "enabled": True,
                        "transport": "http",
                        "url": "https://mcp.example.com/rpc",
                        "allow_all_tools": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)
    configs = load_mcp_configs(workspace, config)
    assert [c.name for c in configs] == ["on", "remote"]  # enabled=False 默认跳过
    stdio = configs[0]
    assert isinstance(stdio, StdioServerConfig)
    assert stdio.startup_timeout_seconds == 20.0
    remote = configs[1]
    assert isinstance(remote, HttpServerConfig)
    assert remote.read_only is True
    assert remote.startup_timeout_seconds == 20.0
    # include_disabled=True 时完整校验并保留状态(CLI list/show)。
    all_configs = load_mcp_configs(workspace, config, include_disabled=True)
    assert [c.name for c in all_configs] == ["off", "on", "remote"]
    assert all_configs[0].enabled is False
    # allowlist/allow_all 都缺 ⇒ 拒。
    config.write_text(
        yaml.safe_dump({"servers": [{"name": "x", "enabled": True, "command": "/bin/echo"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tools_allowlist or allow_all_tools"):
        load_mcp_configs(workspace, config)
    # startup_timeout 非正/非数拒绝(旧 :744-748)。
    for bad in (0, -1, "fast", True):
        config.write_text(
            yaml.safe_dump(
                {"servers": [{"name": "x", "enabled": True, "command": "/bin/echo", "allow_all_tools": True, "startup_timeout_seconds": bad}]}
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="startup_timeout_seconds"):
            load_mcp_configs(workspace, config)


def test_mcp_env_reference_expansion_names_var_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 旧 runtime/mcp.py:41 _ENV_REFERENCE + :845-859 _environment():${VAR}
    # 加载期展开,缺失即抛。praxis 同语义且**只点名变量**(值不进消息)。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    host = tmp_path / "host"
    host.mkdir()
    host.chmod(0o700)
    config = host / "mcp_servers.yaml"
    config.write_text(
        yaml.safe_dump(
            {"servers": [{"name": "s", "enabled": True, "command": "/bin/echo", "allow_all_tools": True, "env": {"KEY": "${PARITY_SECRET}"}}]}
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)
    monkeypatch.setenv("PARITY_SECRET", "sk-expanded-value")
    (stdio,) = load_mcp_configs(workspace, config)
    assert isinstance(stdio, StdioServerConfig)
    assert stdio.env["KEY"] == "sk-expanded-value"
    monkeypatch.delenv("PARITY_SECRET")
    with pytest.raises(ValueError, match="PARITY_SECRET") as excinfo:
        load_mcp_configs(workspace, config)
    assert "sk-expanded-value" not in str(excinfo.value)
    # 有意差异(R-PF-4):stdio cwd 必须在 workspace 之内(旧 cwd 相对配置
    # 文件目录,runtime/mcp.py:702-814;新地板下沙箱只授权 workspace+scratch)。
    config.write_text(
        yaml.safe_dump(
            {"servers": [{"name": "s", "enabled": True, "command": "/bin/echo", "allow_all_tools": True, "cwd": str(tmp_path)}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cwd must be inside the workspace"):
        load_mcp_configs(workspace, config)


def test_mcp_oauth_rejected_and_normalized_collision_rejected(tmp_path: Path) -> None:
    # 有意差异①(Q6):旧 _HTTPServerConfig 支持 oauth/oauth_client_id/
    # oauth_scopes/oauth_allowed_domains(runtime/mcp.py:59-73,:766-769);
    # praxis 3B 明确拒绝 OAuth 族键(阶段 5 宿主),不静默忽略。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    host = tmp_path / "host"
    host.mkdir()
    host.chmod(0o700)
    config = host / "mcp_servers.yaml"
    config.write_text(
        yaml.safe_dump(
            {"servers": [{"name": "s", "enabled": True, "transport": "http", "url": "https://mcp.example.com/rpc", "allow_all_tools": True, "oauth": True}]}
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)
    with pytest.raises(ValueError, match="OAuth lands in phase 5"):
        load_mcp_configs(workspace, config)
    # 有意差异②(M3/T18):归一化后同名(a-b vs a_b)加载期拒绝并点名两个
    # 原始名——旧实现到运行期才拼出碰撞的 canonical 名/共享凭据 service。
    config.write_text(
        yaml.safe_dump(
            {
                "servers": [
                    {"name": "a-b", "enabled": True, "command": "/bin/echo", "allow_all_tools": True},
                    {"name": "a_b", "enabled": True, "command": "/bin/echo", "allow_all_tools": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="collide after normalization") as excinfo:
        load_mcp_configs(workspace, config)
    assert "'a-b'" in str(excinfo.value) and "'a_b'" in str(excinfo.value)
    # 全标点名字规范化为空 ⇒ 加载期拒(旧运行期 mcp____* 垃圾的边界收紧)。
    config.write_text(
        yaml.safe_dump({"servers": [{"name": "---", "enabled": True, "command": "/bin/echo", "allow_all_tools": True}]})
    )
    with pytest.raises(ValueError, match="normalize to a non-empty value"):
        load_mcp_configs(workspace, config)


def test_mcp_names_normalize_parity() -> None:
    # 旧 tools/integrations/mcp.py:70-75 normalize:strip→lower→-/空格→_→
    # 非法→_→折叠任意连续下划线→去首尾 _;:78-83 canonical 拼装。
    assert normalize_mcp_name("  My-Server Name  ") == "my_server_name"
    assert normalize_mcp_name("a--b") == "a_b"
    assert normalize_mcp_name("a__b") == "a_b"
    assert normalize_mcp_name("__x__") == "x"
    assert normalize_mcp_name("日本語") == ""  # 全非法 → 空(加载期拒)
    assert normalize_mcp_name("a.b/c") == "a_b_c"
    assert canonical_tool_name("My Server", "Do-Thing") == "mcp__my_server__do_thing"
    assert server_wildcard("My Server") == "mcp__my_server__*"
    assert global_wildcard() == "mcp__*"
    with pytest.raises(ValueError):
        canonical_tool_name("---", "tool")
    with pytest.raises(TypeError):
        normalize_mcp_name(3)  # type: ignore[arg-type]


def test_mcp_descriptor_attributes_match_legacy() -> None:
    # 旧 tools/integrations/mcp.py:46 timeout_seconds=30.0;:168
    # max_model_output_bytes=1_000_000;:141 static_effects;:155-159 targets。
    (tool,) = _mcp_tools()
    assert tool.timeout_seconds == 30.0
    assert tool.max_model_output_bytes == 1_000_000
    assert tool.source is ToolSource.MCP
    assert tool.check_permission is None  # Q3:MCP 不自带权限回调
    assert tool.cancellation_mode.value == "remote_best_effort"
    assert tool.interrupt_behavior.value == "cancel"
    assert tool.definition.name == "mcp__probe__echo"
    # 有意差异①(T17,controller 钉死):旧对所有 transport 声明 {NETWORK}
    # (integrations/mcp.py:116);praxis stdio 工具 effects=**空集**(server
    # 进程已在沙箱内,调用只是 client RPC;网络能力由沙箱 profile 决定,
    # 与 bash 同一 D9 面)。HTTP 工具仍 {NETWORK}。
    assert tool.static_effects == frozenset()
    assert [(t.kind, t.value) for t in tool.resolve_use({}).targets] == [("mcp_tool", "mcp__probe__echo")]
    (http_tool,) = _mcp_tools(_http_config(read_only=False), tools=(_mcp_tool(),))
    assert http_tool.static_effects == frozenset({ToolEffect.NETWORK})
    assert [(t.kind, t.value) for t in http_tool.resolve_use({}).targets] == [
        ("mcp_tool", "mcp__remote__echo"),
        ("url", "https://mcp.example.com/rpc"),  # 旧 kind=public_web(:159),praxis 统一 url 词汇
    ]
    # destructiveHint ⇒ 两条路径都叠加 DESTRUCTIVE(旧 :117-118)。
    (destructive,) = _mcp_tools(tools=(_mcp_tool("boom", destructiveHint=True),))
    assert destructive.static_effects == frozenset({ToolEffect.DESTRUCTIVE})
    # 有意差异②:旧 idempotent=not destructive and (read_only or idempotent_hint)
    # (:163)、concurrency_safe=read_only and not destructive(:164);
    # praxis idempotent=idempotentHint、concurrency_safe=not destructiveHint
    # (hint 直译,不再从 read_only 推重放安全性)。
    (readonly,) = _mcp_tools(tools=(_mcp_tool("ro", readOnlyHint=True),))
    assert readonly.idempotent is False  # 旧口径会是 True
    assert readonly.concurrency_safe is True
    (idem,) = _mcp_tools(tools=(_mcp_tool("idem", idempotentHint=True),))
    assert idem.idempotent is True
    assert destructive.concurrency_safe is False
    # 有意差异③:旧 approval_profile=MCP_ACTION(:142-148)与 execution_revision
    # 前缀 "integration-mcp-v1:"(:162,result.py:235-237 前缀分类)删除——
    # 分类一律 tool.source 枚举,批准由唯一 decide + 审批绑定承担。
    assert not hasattr(tool, "approval_profile")
    assert not hasattr(tool, "execution_revision")


def test_mcp_read_only_filter_and_allowlist_guard() -> None:
    # 旧 runtime/mcp.py:257-262:HTTP read_only=True 只暴露
    # readOnlyHint and not destructiveHint 的工具;:242-246 allowlist 缺失
    # 工具即启动错误。
    tools = (
        _mcp_tool("ro", readOnlyHint=True),
        _mcp_tool("rw"),
        _mcp_tool("danger", readOnlyHint=True, destructiveHint=True),
    )
    filtered = _mcp_tools(_http_config(read_only=True), tools=tools)
    assert [t.definition.name for t in filtered] == ["mcp__remote__ro"]
    unfiltered = _mcp_tools(_http_config(read_only=False), tools=tools)
    assert len(unfiltered) == 3
    # allowlist 点名缺失工具 ⇒ ValueError(启动错误,构造期)。
    with pytest.raises(ValueError, match="missing allowlisted tools"):
        _mcp_tools(_stdio_config(tools_allowlist=frozenset({"ghost"}), allow_all_tools=False))
    # allowlist 过滤生效。
    (only_echo,) = _mcp_tools(
        _stdio_config(tools_allowlist=frozenset({"echo"}), allow_all_tools=False),
        tools=(_mcp_tool("echo"), _mcp_tool("other")),
    )
    assert only_echo.definition.name == "mcp__probe__echo"
    # canonical 名重复 ⇒ 构造期拒(旧同边界)。
    with pytest.raises(ValueError, match="duplicate canonical MCP tool name"):
        _mcp_tools(tools=(_mcp_tool("a-b"), _mcp_tool("a_b")))


def test_mcp_stdio_parameters_envp_only_s7_shape(tmp_path: Path) -> None:
    # S7 主案:旧病灶 runtime/mcp.py:625-667(:654 server_env={**environment,
    # **config.env},:659-661 args=[...,"/usr/bin/env","-i",*(f"{key}={value}"
    # ...),command,*args])把密钥明文写进 ps 可见的 argv。praxis 全部 env 值
    # 只经 envp:_stdio_parameters 产物 argv 零 KEY=VALUE、零 /usr/bin/env、
    # 零 -i(构造期护栏,T16 证据分工)。
    workspace = tmp_path / "ws"
    workspace.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    spec = SandboxSpec(backend="test-passthrough", argv_prefix=("/usr/bin/env",), scratch_dir=str(scratch))
    config = _stdio_config(env={"SECRET_MCP_KEY": "sk-s7-parity", "LC_ALL": "C.UTF-8"})
    params = _stdio_parameters(config, workspace=workspace, spec=spec, env_builder=child_env)
    # argv = 沙箱前缀余部 + 真实命令 + 命令参数;command 取前缀首元素。
    assert params.command == "/usr/bin/env"
    assert params.args == ["/bin/echo", "hello"]
    assert not any("=" in arg for arg in params.args)
    assert "/usr/bin/env" not in params.args
    assert "-i" not in params.args
    assert "sk-s7-parity" not in " ".join(params.args)
    # env = {**child_env(scratch, workspace), **config.env}(拼装顺序钉死)。
    expected_env = {**child_env(scratch, workspace), **config.env}
    assert params.env == expected_env
    assert params.env["SECRET_MCP_KEY"] == "sk-s7-parity"
    assert params.env["HOME"] == os.path.realpath(scratch)
    assert params.cwd is None  # cwd 逐字取 config.cwd(None ⇒ 装配侧给 workspace 默认)
    # 负向护栏:旧病灶形态必须被同一判据报警(护栏不是空转)。
    legacy_args = ["/usr/bin/env", "-i", "SECRET_MCP_KEY=sk-s7-parity", "/bin/echo", "hello"]
    assert any("=" in arg for arg in legacy_args)
    assert "/usr/bin/env" in legacy_args and "-i" in legacy_args


def test_mcp_config_sha_drift_is_the_only_call_guard(tmp_path: Path) -> None:
    # 旧 call_tool 三层守卫(runtime/mcp.py:350-367):(a)配置字节 sha256 漂移
    # :355-356;(b)permission revision 变化 :362-363;(c)当前+启动两套策略
    # 各查 deny :365-366。praxis(Q3/R-T17-2)只保留 (a)——重连问题;
    # (b)/(c)由唯一 decide + 审批 args_digest/policy_revision 承担,工具层
    # 不再自带第二/第三决策路径。mcp_tool_unknown 码已删除(目录=构造期快照)。
    sha = ["baseline-sha"]
    session = _FakeSession()
    (tool,) = _mcp_tools(sha=sha, session=session)
    # 基线一致 ⇒ 正常调用。
    ok = _run_tool(tool, {"message": "hi"})
    assert ok.status.value == "ok"
    assert session.calls == [("echo", {"message": "hi"})]
    # 漂移 ⇒ mcp_config_changed,固定文案 "config changed; reconnect",零 RPC。
    sha[0] = "drifted-sha"
    drifted = _run_tool(tool, {"message": "hi"})
    assert drifted.error_code == "mcp_config_changed"
    assert drifted.error_message == "config changed; reconnect"
    assert drifted.metadata["mcp_server"] == "probe"
    assert drifted.metadata["mcp_tool"] == "echo"
    assert len(session.calls) == 1  # 漂移后绝不触达 server
    # 读失败同样 fail-closed 到重连(不裸泄 IO 异常)。构造期基线必须可读,
    # 之后 provider 转坏 ⇒ run 期读失败归一 mcp_config_changed。
    provider_state = ["baseline-sha"]

    def _provider() -> str:
        if provider_state[0] == "raise":
            raise OSError("disk gone")
        return provider_state[0]

    (tool2,) = mcp_tools_for_connection(
        _fake_connection((_mcp_tool(),)), _stdio_config(), config_sha_provider=_provider
    )
    provider_state[0] = "raise"
    assert _run_tool(tool2, {"message": "hi"}).error_code == "mcp_config_changed"
    # 策略变化不在工具层拦:policy deny 由 decide 承担(管道级),工具 run
    # 本身不读策略——同一 tool 在"策略已 deny"的上下文外仍可 RPC(证明没有
    # 第二守卫);管道级 deny 在 fault-injection/三入口测试钉。
    sha[0] = "baseline-sha"
    assert _run_tool(tool, {"message": "again"}).status.value == "ok"
    assert len(session.calls) == 2
    # 缺席码面:mcp_tool_unknown(R-T17-2 删除)/ mcp_connection_required
    # (lazy 占位,阶段 5)在**代码字符串常量**里不存在(docstring 历史注记不算)。
    import praxis.adapters.mcp.descriptors as descriptors_module
    import praxis.adapters.mcp.manager as manager_module

    assert "mcp_tool_unknown" not in _code_string_constants(descriptors_module)
    assert "mcp_connection_required" not in _code_string_constants(manager_module)


def test_mcp_no_lazy_catalog_tools_are_immediately_callable() -> None:
    # 有意差异(Q4):旧 HTTP lazy 目录(runtime/mcp_catalog.py:171-301)从缓存
    # 出工具、用时才连,未缓存暴露占位 mcp__<server>__connect,漂移置
    # mcp_connection_required。praxis 3B eager连接:factory 产物立即可调用,
    # 无 connect 占位工具、无缓存 binding 比对层。
    tools = _mcp_tools()
    assert [t.definition.name for t in tools] == ["mcp__probe__echo"]
    assert all(t.definition.name != "mcp__probe__connect" for t in tools)
    assert _run_tool(tools[0], {"message": "x"}).status.value == "ok"


def test_mcp_output_normalization_and_structured_content_landing() -> None:
    # 旧 integrations/mcp.py:172-241 _normalize_mcp_output:text/image/resource
    # 块 + structuredContent;isError ⇒ mcp_tool_error,metadata 带
    # mcp_server/mcp_tool。有意差异:praxis ToolResult 无 structured_content
    # 字段(protocol 删除),落 metadata["structured_content"]。
    raw = CallToolResult(
        content=[TextContent(type="text", text="hello")],
        structuredContent={"answer": 42},
        isError=False,
    )
    result = normalize_mcp_output(raw, server_name="My Server", tool_name="Do-Thing")
    assert result.tool_name == "mcp__my_server__do_thing"
    assert result.status.value == "ok"
    assert result.content[0].data == {"text": "hello"}
    assert result.metadata["mcp_server"] == "My Server"
    assert result.metadata["mcp_tool"] == "Do-Thing"
    assert thaw_json(result.metadata["structured_content"]) == {"answer": 42}
    assert not hasattr(result, "structured_content")
    # isError ⇒ mcp_tool_error + 首条非空 text 前 512 字符。
    error_raw = CallToolResult(content=[TextContent(type="text", text="e" * 600)], isError=True)
    error = normalize_mcp_output(error_raw, server_name="probe", tool_name="echo")
    assert error.error_code == "mcp_tool_error"
    assert error.status.value == "error"
    assert error.error_message == "e" * 512
    # 无文本 isError ⇒ 固定文案。
    bare = normalize_mcp_output(CallToolResult(content=[], isError=True), server_name="probe", tool_name="echo")
    assert bare.error_message == "MCP tool reported an error"


def test_mcp_http_transport_guards_and_no_destination_check() -> None:
    # 旧 _MCPHTTPTransport(runtime/mcp.py:400-445):URL 逐字等于 endpoint、
    # Host 与 netloc 一致、3xx 一律拒、非 oauth 401/403 ⇒ PermissionError。
    # 有意差异(Q3/R-T18-1):旧 OAuth 腿 destination_check(:483-491)每请求
    # 复查策略交集——praxis 不再移植(策略漂移只有 decide 一个面;HTTP 工具
    # 带 url target,调用期 live 裁决)。传输只留工具层无法表达的完整性守卫。
    import httpx

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    transport = GuardedMcpTransport("https://mcp.example.com/rpc", httpx.MockTransport(handler))

    async def scenario() -> None:
        ok = await transport.handle_async_request(
            httpx.Request("POST", "https://mcp.example.com/rpc", headers={"Host": "mcp.example.com"})
        )
        assert ok.status_code == 200
        # endpoint 逐字相等:path 变化即拒(凭证不可能带到同主机另一路径)。
        with pytest.raises(McpTransportError) as excinfo:
            await transport.handle_async_request(
                httpx.Request("POST", "https://mcp.example.com/other", headers={"Host": "mcp.example.com"})
            )
        assert excinfo.value.code == "mcp_http_endpoint_changed"
        # Host 混淆拒。
        with pytest.raises(McpTransportError) as host_err:
            await transport.handle_async_request(
                httpx.Request("POST", "https://mcp.example.com/rpc", headers={"Host": "evil.example.com"})
            )
        assert host_err.value.code == "mcp_http_host_mismatch"
        assert len(seen) == 1  # 守卫拒绝的请求绝不触达内层传输

    asyncio.run(scenario())
    # 3xx 与 401/403 映射。
    for status, code in ((302, "mcp_http_redirect"), (401, "mcp_http_unauthorized"), (403, "mcp_http_unauthorized")):
        redirect_transport = GuardedMcpTransport(
            "https://mcp.example.com/rpc",
            httpx.MockTransport(lambda _req, status=status: httpx.Response(status)),
        )

        async def denied() -> None:
            with pytest.raises(McpTransportError) as excinfo:
                await redirect_transport.handle_async_request(
                    httpx.Request("POST", "https://mcp.example.com/rpc", headers={"Host": "mcp.example.com"})
                )
            assert excinfo.value.code == code
            assert "secret" not in str(excinfo.value)

        asyncio.run(denied())
    # 构造签名只有 endpoint+inner:无策略回调/destination_check 注入位。
    import inspect

    assert list(inspect.signature(GuardedMcpTransport.__init__).parameters) == ["self", "endpoint", "inner"]


def test_mcp_credential_revision_is_presence_only() -> None:
    # 有意差异:旧 credential_revision=sha256(headers+oauth storage.read())
    # (runtime/mcp.py:58,含 OAuth 存储字节);praxis 只含 headers 展开值 +
    # bearer **是否存在**(controller 裁决:OAuth 存储不读、秘密不进修订)。
    headers = {"X-Api-Key": "expanded-value"}
    present = credential_revision(headers, bearer_present=True)
    absent = credential_revision(headers, bearer_present=False)
    assert present != absent
    assert len(present) == 64
    # 函数签名根本不接收 bearer 值 ⇒ 不同 bearer 值不可能改变修订。
    import inspect

    assert list(inspect.signature(credential_revision).parameters) == ["headers", "bearer_present"]
    assert credential_revision(headers, bearer_present=True) == present
    # headers 变化 ⇒ 修订变化(形状漂移可检测)。
    assert credential_revision({"X-Api-Key": "other"}, bearer_present=True) != present


def test_mcp_config_sha256_is_hexdigest_str(tmp_path: Path) -> None:
    # 旧 runtime/mcp.py:355 hashlib.sha256(config_path.read_bytes()).hexdigest()
    # ——str 形(T15 裁决:brief 笔误,legacy 即 hexdigest str)。
    config = tmp_path / "mcp_servers.yaml"
    config.write_bytes(b"servers: []\n")
    digest = config_sha256(config)
    assert isinstance(digest, str)
    assert digest == hashlib.sha256(b"servers: []\n").hexdigest()
    config.write_bytes(b"servers: []\n# drift\n")
    assert config_sha256(config) != digest
    with pytest.raises(ValueError, match="could not be read"):
        config_sha256(tmp_path / "missing.yaml")


def test_mcp_call_failure_is_bounded_and_flattened() -> None:
    # SDK/传输异常一律归一 mcp_call_failed(有界 ≤512,ExceptionGroup 摊平,
    # 绝不带 TaskGroup 包装文案;T17/T19)。
    class _BrokenSession:
        async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
            raise ExceptionGroup("taskgroup", [ExceptionGroup("inner", [ConnectionError("pipe broken")])])

    (tool,) = _mcp_tools(session=_BrokenSession())  # type: ignore[arg-type]
    result = _run_tool(tool, {"message": "hi"})
    assert result.error_code == "mcp_call_failed"
    assert "pipe broken" in result.error_message
    assert "TaskGroup" not in result.error_message and "taskgroup" not in result.error_message
    assert len(result.error_message) <= 512
    assert result.metadata == {"mcp_server": "probe", "mcp_tool": "echo"}


def test_mcp_http_read_timeout_and_startup_budget_shape() -> None:
    # 旧 runtime/mcp.py:509 client timeout=Timeout(startup, read=300)。
    # praxis http_transport._READ_TIMEOUT_SECONDS=300.0 逐字;startup 取
    # config.startup_timeout_seconds(默认 20.0,T15 已校验正有限)。
    from praxis.adapters.mcp import http_transport

    assert http_transport._READ_TIMEOUT_SECONDS == 300.0
    assert _stdio_config().startup_timeout_seconds == 20.0
    assert _http_config().startup_timeout_seconds == 20.0


def test_mcp_bundle_close_timeout_default_matches_legacy() -> None:
    # 旧 runtime/mcp.py:133/:165 close_timeout_seconds 默认 5.0;praxis
    # manager._CLOSE_TIMEOUT_SECONDS 与 lifecycle 默认同值。
    from praxis.adapters.mcp import manager

    assert manager._CLOSE_TIMEOUT_SECONDS == 5.0
    import inspect

    from praxis.adapters.mcp.lifecycle import owned_stdio_connection

    assert inspect.signature(owned_stdio_connection).parameters["close_timeout_seconds"].default == 5.0


# ---------------------------------------------------------------------------
# 跨切:child_env 唯一最小环境(B33)在 3B 新 spawn 面同样生效
# ---------------------------------------------------------------------------


def test_child_env_still_the_only_minimal_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 3A parity 已钉 child_env 键集合;3B 复用同一实现(MCP stdio/hook 各自
    # 姿态:hook 更窄——只有 PATH=os.defpath,见上文全等断言)。此处钉 MCP
    # 侧宿主哨兵不继承(B33 在 3B 的延续)。
    monkeypatch.setenv("PRAXIS_CANARY_3B", "SECRET-XYZ")
    scratch = tmp_path / "scratch"
    workspace = tmp_path / "ws"
    scratch.mkdir()
    workspace.mkdir()
    env = child_env(scratch, workspace)
    assert "PRAXIS_CANARY_3B" not in env
    assert set(env) == {
        "PATH", "HOME", "TMPDIR", "TEMP", "TMP",
        "GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM", "LC_ALL",
    }


def test_ps_orphan_census_helper_smoke() -> None:
    # 验收装置自检:ps 探针可用(fault-injection 与 kill9 文件共用同一口径)。
    completed = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=20, check=False
    )
    assert completed.returncode == 0
    assert completed.stdout.strip()