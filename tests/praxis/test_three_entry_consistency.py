"""Task 9:三入口决策一致性属性测试(设计 §7.2/§8 验收;硬规则 3)。

设计 §7.2 要求"同一策略在 CLI/模型/子代理三入口决策一致(属性测试)"。子代理入口
阶段 4 才落地,当前可达面 = **模型 / CLI / resume** 三入口——它们一律经唯一管道
``praxis.core.pipeline.execute`` 执行工具,唯一决策入口 ``permission.decide``。
``ENTRIES`` 注册表是扩展位:阶段 4 子代理入口复用同一 execute,注册即纳入本属性。

观测口径(brief 钉死):比较**首次 execute 的决策类别 + reason**——
- executed = tool_call item(管道已 dispatch;工具自身业务错误如
  manage_command 的未知 session 不改变决策类别);
- denied   = tool_result status=denied,或 error_code ∈ {permission_denied,
  sandbox_refused},或 error_message 以 "floor" 开头;
- asked    = approval_request 且无 matching approval_decision。
reason 口径:denied 取 tool_result.error_message(即 decide 的真实 reason);
asked 取状态机 reason "approval_required";executed 的 allow reason 不落 item 日志
(管道只记结果),以规范化标记 "executed" 参与比较。唯一 decide 保证三类跨入口一致。

resume 入口构造:submit 以同 policy 触发首次决策——asked 则再
resume(approval={call: allow}) 重入,断言重入后执行成功且 args_digest/policy_revision
绑定生效;executed/denied 则直接观测 submit 快照,无需 resume。跨入口比较用的是
**submit 的首次决策**(重入是第二次决策,单独断言)。

真实面:模型/resume 经 sdk.open_session(唯一组合根),CLI 经 typer CliRunner 进程内
真实分派 ``run`` 命令(--scripted + --policy 临时文件)。ScriptedAdapter 只是确定性
模型替身,循环/管道/工具/store 全是真的。

沙箱面分层(brief):
- 广度属性(全平台):fake availability(monkeypatch composition.probe_backend →
  available,compile/finalize → None 单测路径)——decide 层一致性,避免真实沙箱主导
  时长;bash 以无沙箱直跑 trivial 命令。强制面不在此证明。
- 真实子集(darwin seatbelt,skipif 无后端):不 monkeypatch,经真实组合根 + 真实
  sandbox-exec,证明真实强制下三入口仍一致。
- 每例独立 workspace/store(tmp_path_factory);reason 比较把各入口 workspace 根
  规范化为 "<ws>"——这是逐入口隔离的产物,非决策差异;规则/地板/裁决内容逐字比较。
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import subprocess
import sys
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as McpTool
from typer.testing import CliRunner

from praxis import composition
from praxis.adapters.mcp.config import StdioServerConfig
from praxis.adapters.mcp.descriptors import mcp_tools_for_connection
from praxis.adapters.mcp.lifecycle import McpConnection
from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.adapters.sandbox.base import probe_backend
from praxis.adapters.web.client import PublicHttpClient
from praxis.adapters.web.limits import WebLimits
from praxis.adapters.web.search import create_web_search_tools, load_web_search_config
from praxis.adapters.web.tools import create_web_fetch_tools
from praxis.core import permission
from praxis.interfaces.cli import app
from praxis.interfaces.sdk import open_session
from praxis.protocol.floor import host_config_dir
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.models import ModelResponse, ModelToolCall, Usage
from praxis.protocol.permission import PermissionContext, PermissionMode, SandboxAvailability
from praxis.protocol.policy import NetworkPolicy, PathRule, PermissionPolicy, ToolRule
from praxis.protocol.tools import Tool, ToolSource
from praxis.store.log import Store
from praxis.tools.builtins.command import create_bash_tool
from praxis.tools.builtins.command_sessions import CommandSessions, create_manage_command_tool
from praxis.tools.builtins.filesystem import create_read_tool, create_write_tool
from praxis.tools.builtins.patch import create_apply_patch_tool
from praxis.tools.builtins.planning import create_update_plan_tool
from praxis.tools.builtins.python_tool import create_run_python_tool
from praxis.tools.builtins.search import create_search_text_tool
from praxis.tools.builtins.skills import (
    create_invoke_skill_tool,
    create_materialize_skill_asset_tool,
    load_skill_catalog,
)
from tests.praxis.test_web_urls_dns import (
    PUBLIC_IP,
    FakeBackend,
    FakeResolver,
    FakeStream,
    response_chunks,
)

CALL_ID = "c1"
_TOOL_NAMES = (
    "read",
    "write",
    "bash",
    "search_text",
    "apply_patch",
    "update_plan",
    "manage_command",
    "run_python",
    # T21 扩展:3B 四个新 builtin 纳入同一一致性属性(默认表 T20 已装配)。
    "web_fetch",
    "web_search",
    "invoke_skill",
    "materialize_skill_asset",
)
_FAKE_SANDBOX = SandboxAvailability(backend="seatbelt", available=True)
_AVAILABLE = probe_backend()
_CLEAN_ENV = {"PRAXIS_MODEL": None, "PRAXIS_BASE_URL": None, "PRAXIS_API_KEY": None, "OPENAI_API_KEY": None}
_runner = CliRunner()

# 预置 read 目标(保证 read 执行即 ok,不落入 not_found);write/bash 目标见策略。
# T21 扩展:demo skill(含 scripts/ 资产)供 invoke_skill/materialize 用例;
# 内容不含 "seed"/"x" 模式,既有 search_text 用例的匹配面不变。
_SEED_FILES = {
    "note.txt": "seed note\n",
    ".git/config": "orig\n",
    ".praxis/state.json": "{}\n",
    ".agent_memory/mem.txt": "memory\n",
    "sub/deep.txt": "deep\n",
    ".agents/skills/demo/SKILL.md": "---\nname: demo\ndescription: demo skill\n---\ndemo instructions\n",
    ".agents/skills/demo/scripts/run.py": "print('demo')\n",
}


# ---------------------------------------------------------------------------
# 案例数据与策略
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Call:
    tool: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class Scenario:
    policy: PermissionPolicy
    mode: PermissionMode | None  # None = 用 policy.approval(C5 正交)


@dataclass(frozen=True)
class Observation:
    decision_class: str  # executed | denied | asked
    reason: str


_READ_PATHS = ("note.txt", ".git/config", ".praxis/state.json", ".agent_memory/mem.txt", "sub/deep.txt")
_WRITE_PATHS = (
    "out.txt", ".git/config", ".praxis/state.json", ".agent_memory/mem.txt",
    "locked/file.txt", "granted/file.txt", "sub/new.txt",
)
_BASH_COMMANDS = ("echo praxis-ok", "true", "printf hello")
_SEARCH_PATHS = (".", "note.txt", ".praxis", "sub")
_CMD_ID = "cmd_" + "0" * 32
_PATCH_NOTE = "*** Begin Patch\n*** Update File: note.txt\n@@\n-seed note\n+patched\n*** End Patch\n"
_PATCH_GIT = "*** Begin Patch\n*** Update File: .git/config\n@@\n-orig\n+pwned\n*** End Patch\n"


@st.composite
def call_strategy(draw: st.DrawFn) -> Call:
    tool = draw(st.sampled_from(_TOOL_NAMES))
    if tool == "read":
        return Call(tool, {"path": draw(st.sampled_from(_READ_PATHS))})
    if tool == "write":
        return Call(tool, {"path": draw(st.sampled_from(_WRITE_PATHS)), "content": "praxis property\n"})
    if tool == "bash":
        return Call(tool, {"command": draw(st.sampled_from(_BASH_COMMANDS))})
    if tool == "search_text":
        return Call(tool, {"pattern": "seed", "path": draw(st.sampled_from(_SEARCH_PATHS))})
    if tool == "apply_patch":
        return Call(tool, {"patch": draw(st.sampled_from((_PATCH_NOTE, _PATCH_GIT)))})
    if tool == "update_plan":
        return Call(tool, {"plan": [{"step": "check state", "status": "in_progress"}]})
    if tool == "manage_command":
        return Call(tool, {"session_id": _CMD_ID, "action": "status"})
    if tool == "web_fetch":
        return Call(tool, {"url": "https://example.com/"})
    if tool == "web_search":
        return Call(tool, {"query": draw(st.sampled_from(("praxis", "public docs")))})
    if tool == "invoke_skill":
        return Call(tool, {"name": draw(st.sampled_from(("demo", "absent")))})
    if tool == "materialize_skill_asset":
        return Call(tool, {"skill_id": "demo", "relative_path": "scripts/run.py"})
    return Call(tool, {"code": "print('praxis property')"})


@st.composite
def policy_strategy(draw: st.DrawFn) -> Scenario:
    profile = draw(st.sampled_from(("workspace", "read-only", "trusted-local")))
    approval = draw(st.sampled_from(("on-request", "never")))
    filesystem: list[PathRule] = []
    if draw(st.booleans()):
        filesystem.append(PathRule(path="locked", access="deny"))
    if draw(st.booleans()) and profile != "read-only":  # read-only 禁 write 授权(readonly_cannot_widen)
        filesystem.append(PathRule(path="granted", access="write"))
    if draw(st.booleans()):
        filesystem.append(PathRule(path="sub", access="read"))
    tools: list[ToolRule] = []
    for name in _TOOL_NAMES:
        verdict = draw(st.sampled_from((None, "allow", "ask", "deny")))
        if verdict is not None:
            tools.append(ToolRule(pattern=name, decision=verdict))
    network = NetworkPolicy(decision=draw(st.sampled_from(("allow", "ask", "deny"))))
    policy = PermissionPolicy(
        profile=profile, approval=approval, filesystem=tuple(filesystem), tools=tuple(tools), network=network
    )
    mode = draw(st.sampled_from((None, "on-request", "never", "bypass")))
    return Scenario(policy, mode)


# ---------------------------------------------------------------------------
# 观测:从 item 日志读首次决策类别 + reason
# ---------------------------------------------------------------------------


def _normalize_reason(reason: str, workspace: Path) -> str:
    """把逐入口 workspace 根规范化为 <ws>(隔离产物,非决策差异);其余内容逐字保留。"""
    for form in dict.fromkeys((str(workspace), str(workspace.resolve()))):
        reason = reason.replace(form, "<ws>")
    return reason


def _payloads(items: list[Item], kind: str, call_id: str) -> list[Mapping[str, Any]]:
    return [i.payload for i in items if i.kind == kind and i.payload.get("call_id") == call_id]


def observe_first_decision(items: list[Item], call_id: str, workspace: Path) -> Observation:
    results = _payloads(items, ItemKind.TOOL_RESULT, call_id)
    calls = _payloads(items, ItemKind.TOOL_CALL, call_id)
    requests = _payloads(items, ItemKind.APPROVAL_REQUEST, call_id)
    decisions = _payloads(items, ItemKind.APPROVAL_DECISION, call_id)
    if results:
        result = results[0]
        status = result.get("status")
        error_code = result.get("error_code")
        message = result.get("error_message") or ""
        if status == "denied" or error_code in {"permission_denied", "sandbox_refused"} or message.startswith("floor"):
            return Observation("denied", _normalize_reason(str(message), workspace))
        if calls:
            # executed = decide 放行且管道已 dispatch(存在 tool_call item)。
            # 工具自身的业务错误(manage_command 的未知 session、run_python 的
            # 非零退出)不改变**决策类别**——目标正是三入口同一 decide。
            return Observation("executed", "executed")
        raise AssertionError(f"unexpected first result shape: {result!r}")
    if requests and not decisions:
        return Observation("asked", "approval_required")
    raise AssertionError(f"no first decision observed for {call_id}: kinds={[i.kind for i in items]}")


def _assert_reentry_outcome(items: list[Item], call_id: str, policy: PermissionPolicy) -> None:
    """resume 重入断言:审批 args_digest/policy_revision 绑定 + 重入结局正确。

    重入两途皆合法:
    - 一般 ask 批准 → 执行(tool_call + 非 denied 结果,digest 绑定到执行参数);
      工具自身业务错误(如 manage_command 的未知 session)仍算已执行;
    - 地板目标经 ask 规则(decide 第 2 步先于第 4 步地板)被批准 → 执行时地板守卫
      (R5c)deny——地板在任何入口都不可批准,无 tool_call、error_message 以 floor 开头。
    """
    decisions = _payloads(items, ItemKind.APPROVAL_DECISION, call_id)
    requests = _payloads(items, ItemKind.APPROVAL_REQUEST, call_id)
    calls = _payloads(items, ItemKind.TOOL_CALL, call_id)
    results = _payloads(items, ItemKind.TOOL_RESULT, call_id)
    assert decisions and decisions[-1].get("verdict") == "allow"
    assert decisions[-1].get("policy_revision") == policy.revision, "approval must bind current policy revision"
    assert requests and decisions[-1].get("args_digest") == requests[-1].get("args_digest"), "digest binding"
    assert results, "reentry must produce a result"
    result = results[-1]
    if calls:
        # 批准后的非地板调用必须被 dispatch(存在 tool_call item)。工具自身业务
        # 失败(如 manage_command 的未知 session)不算决策层面的 deny。
        assert result.get("status") != "denied", "approved non-floor call must execute"
        assert decisions[-1].get("args_digest") == calls[-1].get("args_digest"), "approval digest binds executed args"
    else:
        assert result.get("status") == "denied", "floor target must be denied at execution guard"
        assert str(result.get("error_message") or "").startswith("floor"), "R5c execution floor guard"


# ---------------------------------------------------------------------------
# T21 扩展装置:离线 web client / MCP 形状样本 / MCP 真 harness 接线槽
# ---------------------------------------------------------------------------


class _UnlimitedBackend(FakeBackend):
    """每次 connect 都返回一条新的 200 HTML 假流(属性测试绝不触网)。"""

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: object = None,
    ) -> FakeStream:
        self.connects.append((host, port, timeout))
        stream = FakeStream(
            response_chunks(body=b"<p>praxis offline</p>", headers=[("Content-Type", "text/html")])
        )
        self.issued.append(stream)
        return stream


def _offline_http_client(*, authorize: Callable[[str], None], limits: WebLimits) -> PublicHttpClient:
    """composition.PublicHttpClient 的离线替身(签名兼容组合根装配点)。

    web_fetch 加入工具集后,executed 类用例会真实 dispatch 抓取——必须经
    T10 的 resolver/backend 接缝注入假答案,属性测试保持零网络。authorize
    仍由组合根传入**生产闭包**(make_web_authorizer 读 live 权限),裁决面
    不被替身绕过。
    """
    return PublicHttpClient(
        authorize=authorize,
        limits=limits,
        resolver=FakeResolver([PUBLIC_IP]),
        backend=_UnlimitedBackend(),
        environ={},
    )


def _mcp_shape_tool() -> Tool:
    """ONE source=MCP 形状样本:经 ``mcp_tools_for_connection`` 在 fake
    McpConnection 上构造(T21 派单口径)。echo schema 与真 fixture server
    (tests/praxis/fixtures/mcp_probe_server.py)逐字一致,validate/resolve
    行为与真连接投影不可区分;decide 不得因 source 特殊对待(见下方等价测试)。
    """

    class _Session:
        async def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> CallToolResult:
            message = (arguments or {}).get("message", "")
            return CallToolResult(content=[TextContent(type="text", text=f"echo: {message}")], isError=False)

    config = StdioServerConfig(
        name="probe",
        command="/bin/echo",
        args=(),
        env={},
        cwd=None,
        tools_allowlist=frozenset(),
        allow_all_tools=True,
        startup_timeout_seconds=20.0,
    )
    catalog_tool = McpTool(
        name="echo",
        description="Echo the supplied message.",
        inputSchema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
    )
    connection = McpConnection(server_name="probe", session=_Session(), tools=(catalog_tool,))  # type: ignore[arg-type]
    (tool,) = mcp_tools_for_connection(connection, config, config_sha_provider=lambda: "shape-baseline")
    return tool


# T17 carry(T21 落实):MCP 决策场景的**真 harness** 接线槽。None = 既有路径
# 逐字不变;设置后 model/resume 入口在**同一事件循环**内 connect → open →
# submit → aclose(CLI 入口经 _mcp_configured 自动连接,不需要槽)。
MCP_WIRING: Callable[[Path, Scenario], Awaitable[tuple[tuple[Tool, ...], tuple[Any, ...]]]] | None = None


# ---------------------------------------------------------------------------
# 入口运行器(统一签名;ENTRIES 注册表 = 阶段 4 子代理扩展位)
# ---------------------------------------------------------------------------


def _seed_workspace(workspace: Path) -> None:
    for rel, content in _SEED_FILES.items():
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _read_items(workspace: Path, thread_id: str) -> list[Item]:
    store = Store(workspace / ".praxis" / "praxis.sqlite")
    try:
        return store.read_items(thread_id)
    finally:
        store.close()


def _resp_call(call: Call) -> ModelResponse:
    return ModelResponse(
        content="",
        tool_calls=(ModelToolCall(call_id=CALL_ID, name=call.tool, arguments=dict(call.arguments)),),
        usage=Usage(input_tokens=1, output_tokens=1),
        finish_reason="tool_calls",
    )


def _resp_done() -> ModelResponse:
    return ModelResponse(content="done", tool_calls=(), usage=Usage(input_tokens=1, output_tokens=1), finish_reason="stop")


def _effective_mode(scenario: Scenario) -> PermissionMode:
    return scenario.mode if scenario.mode is not None else scenario.policy.approval


def run_via_model(workspace: Path, call: Call, scenario: Scenario, example_root: Path) -> Observation:
    adapter = ScriptedAdapter([_resp_call(call), _resp_done()])
    if MCP_WIRING is None:
        session = open_session(workspace, adapter, policy=scenario.policy, mode=scenario.mode, thread_id="model")
        try:
            asyncio.run(session.submit("three-entry property"))
        finally:
            session.close()
    else:
        wiring = MCP_WIRING

        async def _go() -> None:
            extra_tools, integrations = await wiring(workspace, scenario)
            session = open_session(
                workspace, adapter, policy=scenario.policy, mode=scenario.mode, thread_id="model",
                extra_tools=extra_tools, integrations=integrations,
            )
            try:
                await session.submit("three-entry property")
            finally:
                await session.aclose()

        asyncio.run(_go())
    return observe_first_decision(_read_items(workspace, "model"), CALL_ID, workspace)


def run_via_cli(workspace: Path, call: Call, scenario: Scenario, example_root: Path) -> Observation:
    policy_file = example_root / "policy.json"
    policy_file.write_text(scenario.policy.model_dump_json(), encoding="utf-8")
    script_file = example_root / "scripted.json"
    script_file.write_text(
        json.dumps(
            [
                {"content": "", "tool_calls": [{"call_id": CALL_ID, "name": call.tool, "arguments": dict(call.arguments)}]},
                {"content": "done"},
            ]
        ),
        encoding="utf-8",
    )
    args = [
        "run", "three-entry property", "--workspace", str(workspace), "--thread", "cli",
        "--scripted", str(script_file), "--policy", str(policy_file),
    ]
    if scenario.mode is not None:
        args += ["--mode", scenario.mode]
    result = _runner.invoke(app, args, env=_CLEAN_ENV)
    assert result.exit_code in (0, 2), (result.exit_code, result.output, repr(result.exception))
    return observe_first_decision(_read_items(workspace, "cli"), CALL_ID, workspace)


def run_via_resume(workspace: Path, call: Call, scenario: Scenario, example_root: Path) -> Observation:
    adapter = ScriptedAdapter([_resp_call(call), _resp_done()])
    if MCP_WIRING is None:
        session = open_session(workspace, adapter, policy=scenario.policy, mode=scenario.mode, thread_id="resume")
        try:
            result = asyncio.run(session.submit("three-entry property"))
            first = observe_first_decision(session.items(), CALL_ID, workspace)
            if first.decision_class == "asked":
                asyncio.run(session.resume(result.turn_id, approval={CALL_ID: "allow"}))
                _assert_reentry_outcome(session.items(), CALL_ID, scenario.policy)
        finally:
            session.close()
        return first
    wiring = MCP_WIRING

    async def _go() -> Observation:
        extra_tools, integrations = await wiring(workspace, scenario)
        session = open_session(
            workspace, adapter, policy=scenario.policy, mode=scenario.mode, thread_id="resume",
            extra_tools=extra_tools, integrations=integrations,
        )
        try:
            result = await session.submit("three-entry property")
            first = observe_first_decision(session.items(), CALL_ID, workspace)
            if first.decision_class == "asked":
                await session.resume(result.turn_id, approval={CALL_ID: "allow"})
                _assert_reentry_outcome(session.items(), CALL_ID, scenario.policy)
            return first
        finally:
            await session.aclose()

    return asyncio.run(_go())


# 阶段 4 子代理入口复用同一 execute,届时在此注册即纳入一致性属性。
ENTRIES: dict[str, Callable[[Path, Call, Scenario, Path], Observation]] = {
    "model": run_via_model,
    "cli": run_via_cli,
    "resume": run_via_resume,
}


def _reference_decision(workspace: Path, call: Call, scenario: Scenario, sandbox: SandboxAvailability) -> permission.Decision:
    sessions = CommandSessions(workspace)
    web_client = _offline_http_client(authorize=lambda _url: None, limits=WebLimits())
    tools: dict[str, Tool] = {
        "read": create_read_tool(workspace),
        "write": create_write_tool(workspace),
        "bash": create_bash_tool(workspace),
        "search_text": create_search_text_tool(workspace),
        "apply_patch": create_apply_patch_tool(workspace),
        "update_plan": create_update_plan_tool(),
        "manage_command": create_manage_command_tool(sessions),
        "run_python": create_run_python_tool(workspace),
        # T21:与组合根默认表同工厂/同配置源(web.json 缺失 ⇒ unconfigured,
        # 与三入口实际装配一致);render=false 的 resolve_use 与 renderer
        # 注入与否无关(只 NETWORK),decide 输入逐字相同。
        "web_fetch": create_web_fetch_tools(web_client, limits=WebLimits())[0],
        "web_search": create_web_search_tools(
            web_client, config_provider=load_web_search_config, credential_provider=lambda _cfg: None
        )[0],
        "invoke_skill": create_invoke_skill_tool(load_skill_catalog(workspace)),
        "materialize_skill_asset": create_materialize_skill_asset_tool(workspace),
        "mcp__probe__echo": _mcp_shape_tool(),
    }
    tool = tools[call.tool]
    args = tool.validate_input(dict(call.arguments))
    use = tool.resolve_use(args)
    ctx = PermissionContext(
        workspace=str(workspace), policy=scenario.policy, mode=_effective_mode(scenario), sandbox=sandbox
    )
    return permission.decide(tool, use, ctx)


def _run_all_entries(example_root: Path, call: Call, scenario: Scenario) -> dict[str, Observation]:
    observations: dict[str, Observation] = {}
    for name, runner_fn in ENTRIES.items():
        workspace = example_root / name
        workspace.mkdir()
        _seed_workspace(workspace)
        observations[name] = runner_fn(workspace, call, scenario, example_root)
    return observations


def _assert_consistent(
    observations: dict[str, Observation], call: Call, scenario: Scenario, reference: permission.Decision, workspace: Path
) -> str:
    classes = {o.decision_class for o in observations.values()}
    reasons = {o.reason for o in observations.values()}
    assert len(classes) == 1, f"decision class diverged: {observations} (call={call}, scenario={scenario})"
    assert len(reasons) == 1, f"reason diverged: {observations} (call={call}, scenario={scenario})"
    observed = next(iter(classes))
    # 唯一 decide 强化:三入口不仅彼此一致,且与唯一决策入口一致(防三入口同偏)。
    expected = "executed" if reference.is_allow else "denied" if reference.is_deny else "asked"
    assert observed == expected, f"entries={observed} but decide={reference.verdict} ({reference.reason})"
    if reference.is_deny:
        assert next(iter(reasons)) == _normalize_reason(reference.reason, workspace), (
            f"denied reason mismatch: {observations} vs decide {reference.reason!r}"
        )
    return observed


# ---------------------------------------------------------------------------
# 广度属性(fake availability,全平台)
# ---------------------------------------------------------------------------


def test_three_entries_same_decision(tmp_path_factory: pytest.TempPathFactory) -> None:
    distribution: Counter[str] = Counter()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(composition, "probe_backend", lambda: _FAKE_SANDBOX)
        mp.setattr(composition, "compile_sandbox", None)
        mp.setattr(composition, "finalize_sandbox", None)
        # T21:web_fetch 入表后 executed 类会真实 dispatch 抓取——离线替身
        # 保持属性测试零网络(authorize 仍是组合根的生产闭包)。
        mp.setattr(composition, "PublicHttpClient", _offline_http_client)

        @example(Call("write", {"path": "out.txt", "content": "x"}), Scenario(PermissionPolicy(), None))  # executed
        @example(Call("write", {"path": ".git/config", "content": "x"}), Scenario(PermissionPolicy(), None))  # denied(floor)
        @example(Call("bash", {"command": "echo hi"}), Scenario(PermissionPolicy(), None))  # asked(network ask)
        @given(call=call_strategy(), scenario=policy_strategy())
        @settings(max_examples=60, deadline=None, derandomize=True, suppress_health_check=[HealthCheck.too_slow])
        def prop(call: Call, scenario: Scenario) -> None:
            example_root = tmp_path_factory.mktemp("case")
            observations = _run_all_entries(example_root, call, scenario)
            reference = _reference_decision(example_root / "model", call, scenario, _FAKE_SANDBOX)
            observed = _assert_consistent(observations, call, scenario, reference, example_root / "model")
            distribution[observed] += 1

        prop()

    # 非空泛:三个 @example 保证三类必现;打印分布供执行日志存证。
    print(f"decision-class distribution: {dict(distribution)}")
    assert set(distribution) == {"executed", "denied", "asked"}, distribution


# ---------------------------------------------------------------------------
# 确定性矩阵:每类代表案例 × 三入口(fake availability;非空泛显式枚举)
# ---------------------------------------------------------------------------


_NET_ALLOW = PermissionPolicy(network=NetworkPolicy(decision="allow"))
_NET_DENY = PermissionPolicy(network=NetworkPolicy(decision="deny"))


@pytest.mark.parametrize(
    ("label", "call", "scenario", "expected"),
    [
        ("allow-write-workspace", Call("write", {"path": "out.txt", "content": "x"}), Scenario(PermissionPolicy(), None), "executed"),
        ("allow-read-workspace", Call("read", {"path": "note.txt"}), Scenario(PermissionPolicy(), None), "executed"),
        ("allow-bash-net-allow", Call("bash", {"command": "echo ok"}), Scenario(_NET_ALLOW, None), "executed"),
        ("deny-floor-git-write", Call("write", {"path": ".git/config", "content": "x"}), Scenario(PermissionPolicy(), None), "denied"),
        ("deny-floor-praxis-read", Call("read", {"path": ".praxis/state.json"}), Scenario(PermissionPolicy(), None), "denied"),
        ("deny-floor-memory-write", Call("write", {"path": ".agent_memory/mem.txt", "content": "x"}), Scenario(PermissionPolicy(), None), "denied"),
        ("deny-readonly-write", Call("write", {"path": "out.txt", "content": "x"}), Scenario(PermissionPolicy(profile="read-only"), None), "denied"),
        ("deny-tool-rule-bash", Call("bash", {"command": "echo ok"}), Scenario(PermissionPolicy(tools=(ToolRule(pattern="bash", decision="deny"),)), None), "denied"),
        ("deny-filesystem-rule", Call("write", {"path": "locked/file.txt", "content": "x"}), Scenario(PermissionPolicy(filesystem=(PathRule(path="locked", access="deny"),)), None), "denied"),
        ("deny-network-bash", Call("bash", {"command": "echo ok"}), Scenario(_NET_DENY, None), "denied"),
        ("ask-bash-default-network", Call("bash", {"command": "echo ok"}), Scenario(PermissionPolicy(), None), "asked"),
        ("ask-tool-rule-write", Call("write", {"path": "out.txt", "content": "x"}), Scenario(PermissionPolicy(tools=(ToolRule(pattern="write", decision="ask"),)), None), "asked"),
        ("ask-tool-rule-read", Call("read", {"path": "note.txt"}), Scenario(PermissionPolicy(tools=(ToolRule(pattern="read", decision="ask"),)), None), "asked"),
        ("floor-bypass-immune", Call("write", {"path": ".git/config", "content": "x"}), Scenario(PermissionPolicy(), "bypass"), "denied"),
        ("never-ask-to-deny", Call("bash", {"command": "echo ok"}), Scenario(PermissionPolicy(), "never"), "denied"),
        # ask 规则(decide 第2步)先于地板(第4步)→ 首次 asked;resume 批准后执行时地板守卫 deny(R5c)。
        ("ask-rule-before-floor", Call("read", {"path": ".praxis/state.json"}), Scenario(PermissionPolicy(tools=(ToolRule(pattern="read", decision="ask"),)), None), "asked"),
        # Task 18 Step 5:新增五个工具也走同一 decide(每类至少一例)。
        # S-1:search_text 现声明 EXECUTE_PROCESS(worker 在沙箱内运行),因此与
        # bash/run_python 同一姿态受 command-network 闸支配:默认 network ask
        # (decide 第 2 步)先于 floor/allow(第 4/6 步)。下面两例用 _NET_ALLOW
        # 保留 executed/denied 覆盖(网络闸清零后,沙箱 allow / 地板 deny 才是
        # 决策),并新增一例钉住"默认 network ask → asked"的新语义。
        ("allow-search-text", Call("search_text", {"pattern": "seed"}), Scenario(_NET_ALLOW, None), "executed"),
        ("deny-floor-search-praxis", Call("search_text", {"pattern": "x", "path": ".praxis"}), Scenario(_NET_ALLOW, None), "denied"),
        ("ask-search-text-default-network", Call("search_text", {"pattern": "seed"}), Scenario(PermissionPolicy(), None), "asked"),
        ("allow-apply-patch", Call("apply_patch", {"patch": _PATCH_NOTE}), Scenario(PermissionPolicy(), None), "executed"),
        ("deny-floor-patch-git", Call("apply_patch", {"patch": _PATCH_GIT}), Scenario(PermissionPolicy(), None), "denied"),
        ("ask-tool-rule-apply-patch", Call("apply_patch", {"patch": _PATCH_NOTE}), Scenario(PermissionPolicy(tools=(ToolRule(pattern="apply_patch", decision="ask"),)), None), "asked"),
        ("allow-update-plan", Call("update_plan", {"plan": [{"step": "check state", "status": "in_progress"}]}), Scenario(PermissionPolicy(), None), "executed"),
        ("allow-manage-command", Call("manage_command", {"session_id": _CMD_ID, "action": "status"}), Scenario(PermissionPolicy(), None), "executed"),
        ("ask-manage-command", Call("manage_command", {"session_id": _CMD_ID, "action": "status"}), Scenario(PermissionPolicy(tools=(ToolRule(pattern="manage_command", decision="ask"),)), None), "asked"),
        ("allow-run-python", Call("run_python", {"code": "print('ok')"}), Scenario(_NET_ALLOW, None), "executed"),
        ("ask-run-python-default-network", Call("run_python", {"code": "print('ok')"}), Scenario(PermissionPolicy(), None), "asked"),
        ("deny-tool-rule-run-python", Call("run_python", {"code": "print('ok')"}), Scenario(PermissionPolicy(tools=(ToolRule(pattern="run_python", decision="deny"),)), None), "denied"),
        # T21 扩展:3B 四个新 builtin 每类至少一例(同一 decide、同一观测口径)。
        # web_fetch:URL target 走 network 面——allow ⇒ 离线抓取执行;默认 ask ⇒
        # 第 2 步暂停;deny ⇒ 第 1 步拒。web_search 未配置(隔离宿主无 web.json):
        # NETWORK **无 url target** ⇒ 第 6 步完整授权覆盖不成立 ⇒ fallback ask
        # ——即使 network allow 也 ask(决策类别与业务错误 unconfigured 无关)。
        ("allow-web-fetch-net-allow", Call("web_fetch", {"url": "https://example.com/"}), Scenario(_NET_ALLOW, None), "executed"),
        ("ask-web-fetch-default-network", Call("web_fetch", {"url": "https://example.com/"}), Scenario(PermissionPolicy(), None), "asked"),
        ("deny-web-fetch-net-deny", Call("web_fetch", {"url": "https://example.com/"}), Scenario(_NET_DENY, None), "denied"),
        ("ask-web-search-unconfigured-net-allow", Call("web_search", {"query": "praxis"}), Scenario(_NET_ALLOW, None), "asked"),
        ("ask-web-search-default", Call("web_search", {"query": "praxis"}), Scenario(PermissionPolicy(), None), "asked"),
        ("deny-tool-rule-web-search", Call("web_search", {"query": "praxis"}), Scenario(PermissionPolicy(tools=(ToolRule(pattern="web_search", decision="deny"),)), None), "denied"),
        ("allow-invoke-skill", Call("invoke_skill", {"name": "demo"}), Scenario(PermissionPolicy(), None), "executed"),
        ("ask-invoke-skill", Call("invoke_skill", {"name": "demo"}), Scenario(PermissionPolicy(tools=(ToolRule(pattern="invoke_skill", decision="ask"),)), None), "asked"),
        ("deny-invoke-skill", Call("invoke_skill", {"name": "demo"}), Scenario(PermissionPolicy(tools=(ToolRule(pattern="invoke_skill", decision="deny"),)), None), "denied"),
        # materialize:单调用 turn 未激活 ⇒ 业务错误 skill_not_active(已 dispatch,
        # 决策类别 executed);read-only profile 写拒 ⇒ denied;tool ask ⇒ asked。
        ("allow-materialize-skill-asset", Call("materialize_skill_asset", {"skill_id": "demo", "relative_path": "scripts/run.py"}), Scenario(PermissionPolicy(), None), "executed"),
        ("deny-materialize-readonly", Call("materialize_skill_asset", {"skill_id": "demo", "relative_path": "scripts/run.py"}), Scenario(PermissionPolicy(profile="read-only"), None), "denied"),
        ("ask-materialize-tool-rule", Call("materialize_skill_asset", {"skill_id": "demo", "relative_path": "scripts/run.py"}), Scenario(PermissionPolicy(tools=(ToolRule(pattern="materialize_skill_asset", decision="ask"),)), None), "asked"),
    ],
)
def test_decision_matrix_all_entries(
    label: str, call: Call, scenario: Scenario, expected: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(composition, "probe_backend", lambda: _FAKE_SANDBOX)
        mp.setattr(composition, "compile_sandbox", None)
        mp.setattr(composition, "finalize_sandbox", None)
        mp.setattr(composition, "PublicHttpClient", _offline_http_client)  # T21:零网络
        example_root = tmp_path_factory.mktemp(label)
        observations = _run_all_entries(example_root, call, scenario)
        reference = _reference_decision(example_root / "model", call, scenario, _FAKE_SANDBOX)
        observed = _assert_consistent(observations, call, scenario, reference, example_root / "model")
    assert observed == expected, (label, observations)


# ---------------------------------------------------------------------------
# 真实沙箱子集(darwin seatbelt;真实组合根 + 真实 sandbox-exec,不 monkeypatch)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _AVAILABLE.available, reason="bwrap deferred / no sandbox backend")
@pytest.mark.parametrize(
    ("label", "call", "scenario", "expected"),
    [
        ("real-bash-allow", Call("bash", {"command": "echo praxis-real"}), Scenario(_NET_ALLOW, None), "executed"),
        ("real-bash-ask", Call("bash", {"command": "echo praxis-real"}), Scenario(PermissionPolicy(), None), "asked"),
        ("real-write-allow", Call("write", {"path": "out.txt", "content": "real\n"}), Scenario(PermissionPolicy(), None), "executed"),
        ("real-read-allow", Call("read", {"path": "note.txt"}), Scenario(PermissionPolicy(), None), "executed"),
        ("real-floor-git-write", Call("write", {"path": ".git/config", "content": "x"}), Scenario(PermissionPolicy(), None), "denied"),
        # N-3:矩阵与真沙箱子集必须有一条**真正跑成功**的 search(前面广度矩阵的
        # executed 只是"已 dispatch 但业务失败"。network allow 清掉 command-network
        # 闸,让沙箱 allow / 地板 deny 成为决策。
        ("real-search-text-allow", Call("search_text", {"pattern": "seed"}), Scenario(_NET_ALLOW, None), "executed"),
        ("real-search-text-floor-deny", Call("search_text", {"pattern": "x", "path": ".praxis"}), Scenario(_NET_ALLOW, None), "denied"),
    ],
)
def test_real_sandbox_three_entries(
    label: str, call: Call, scenario: Scenario, expected: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    example_root = tmp_path_factory.mktemp(label)
    observations = _run_all_entries(example_root, call, scenario)
    reference = _reference_decision(example_root / "model", call, scenario, _AVAILABLE)
    observed = _assert_consistent(observations, call, scenario, reference, example_root / "model")
    assert observed == expected, (label, observations)


# ---------------------------------------------------------------------------
# T21 扩展:MCP 形状样本(source=MCP)与真 harness 三入口 MCP 场景(T17 carry)
# ---------------------------------------------------------------------------


def test_mcp_shape_sample_decides_without_special_treatment() -> None:
    """fake McpConnection 经 mcp_tools_for_connection 构造的 source=MCP 样本:
    decide 不得因 source 特殊对待——把 source 换成 BUILTIN 的克隆在全部策略
    形态下 verdict+reason 逐字相同(分类只消费 tool.source 的展示/审计面,
    决策面没有 MCP 分支;Q3 三层自带守卫消灭的终态)。
    """
    tool = _mcp_shape_tool()
    assert tool.source is ToolSource.MCP
    assert tool.definition.name == "mcp__probe__echo"
    assert tool.static_effects == frozenset()  # stdio {} effects(T17 钉死值)
    clone = dataclasses.replace(tool, source=ToolSource.BUILTIN)
    use = tool.resolve_use({"message": "hi"})
    policies = (
        PermissionPolicy(),
        PermissionPolicy(tools=(ToolRule(pattern="mcp__probe__echo", decision="allow"),)),
        PermissionPolicy(tools=(ToolRule(pattern="mcp__probe__echo", decision="ask"),)),
        PermissionPolicy(tools=(ToolRule(pattern="mcp__probe__echo", decision="deny"),)),
        PermissionPolicy(tools=(ToolRule(pattern="mcp__probe__*", decision="deny"),)),
        PermissionPolicy(network=NetworkPolicy(decision="deny")),
    )
    for policy in policies:
        for mode in ("on-request", "never", "bypass"):
            ctx = PermissionContext(
                workspace="/tmp", policy=policy, mode=mode, sandbox=_FAKE_SANDBOX  # type: ignore[arg-type]
            )
            mcp_decision = permission.decide(tool, use, ctx)
            builtin_decision = permission.decide(clone, use, ctx)
            assert (mcp_decision.verdict, mcp_decision.reason) == (
                builtin_decision.verdict, builtin_decision.reason
            ), (policy, mode)


_MCP_CALL = Call("mcp__probe__echo", {"message": "hi"})
_MCP_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mcp_probe_server.py"


@pytest.mark.skipif(not _AVAILABLE.available, reason="bwrap deferred / no sandbox backend")
@pytest.mark.parametrize(
    ("label", "scenario", "expected"),
    [
        # stdio MCP 工具 effects={} ⇒ 默认策略 policy-grant 直接执行;
        # tool 规则 ask/deny 与 builtin 同一 decide 口径。allow_all_tools=True
        # 使连接预检只在 wildcard deny 时跳过(Q4),exact-name 规则留给 decide。
        ("mcp-executed-default", Scenario(PermissionPolicy(), None), "executed"),
        ("mcp-ask-tool-rule", Scenario(PermissionPolicy(tools=(ToolRule(pattern="mcp__probe__echo", decision="ask"),)), None), "asked"),
        ("mcp-deny-tool-rule", Scenario(PermissionPolicy(tools=(ToolRule(pattern="mcp__probe__echo", decision="deny"),)), None), "denied"),
    ],
)
def test_mcp_three_real_harness_entries(
    label: str, scenario: Scenario, expected: str, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T17 carry:一个 MCP 决策场景经**真 harness 三入口**(run_via_model /
    run_via_cli / run_via_resume),而不是三次相同的 execute——model/resume
    经 MCP_WIRING 在同一 loop 内 connect→open→submit→aclose;CLI 经
    ``_mcp_configured`` 自动连接(生产 compile_sandbox,真 seatbelt)。
    真 fixture stdio server(mcp SDK),参考决策用同形状 fake-connection 样本。
    """
    token = "t21-entry-" + uuid.uuid4().hex
    directory = host_config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    config = directory / "mcp_servers.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "servers": [
                    {
                        "name": "probe",
                        "enabled": True,
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": ["-I", str(_MCP_FIXTURE), "--token", token],
                        "allow_all_tools": True,
                        "startup_timeout_seconds": 10.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)

    async def wiring(workspace: Path, scen: Scenario) -> tuple[tuple[Tool, ...], tuple[Any, ...]]:
        bundle = await composition.connect_mcp_servers(workspace, policy=scen.policy)
        assert [s.connected for s in bundle.statuses] == [True], bundle.statuses
        return bundle.tools, (bundle,)

    monkeypatch.setattr(sys.modules[__name__], "MCP_WIRING", wiring)
    example_root = tmp_path_factory.mktemp(label)
    observations = _run_all_entries(example_root, _MCP_CALL, scenario)
    reference = _reference_decision(example_root / "model", _MCP_CALL, scenario, _AVAILABLE)
    observed = _assert_consistent(observations, _MCP_CALL, scenario, reference, example_root / "model")
    assert observed == expected, (label, observations)
    # 三入口各自的 server 进程组都必须已回收(session.aclose / CLI finally)。
    completed = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=20, check=False
    )
    assert [row for row in completed.stdout.splitlines() if token in row] == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
