"""Task 9:findings 回归矩阵 —— S1/S2/S8/S9/B10 的真实强制验收(执行文档 §5.8)。

不是 happy-path:本文件以真实组合根(sdk.open_session)、真实 typer CLI 分派、
真实 store item、真实 sandbox-exec 子进程(darwin seatbelt)证明阶段 2 的安全
findings 已结构闭合。ScriptedAdapter 只是确定性模型替身,循环/管道/工具/沙箱全真。

覆盖:
- S1 唯一编译器:profile 构造字面量只许出现在 praxis/adapters/sandbox/(AST 扫描
  代码字符串,排除注释/docstring——它们不是构造器);Task 10 把此钉进 CI 反模式。
- S2 地板矩阵:地板目录 × {read,write} × 三入口 decide 层 deny;真实 seatbelt bash
  强制(.git 写拒读放 / .praxis 读拒 / workspace 写放,阳性对照);bypass 免疫。
- S8 网络 ask 不降级:合成 NETWORK(url target)工具 + ToolRule allow 共存 → ask;
  network deny → denied。model/resume 经合成工具;CLI 的网络可达面是 bash 命令网络
  (url 级 web 工具阶段 3 落地,CLI 内置 read/write/bash 不dispatch 合成工具)。
- S9 别名攻击:preflight 跨地板硬链接 → SandboxRefusal(攻击 + 无误报);运行时
  沙箱内 ln 探针 → 阻断。SBPL 限制 xfail 留档于 test_sandbox_seatbelt.py(不重复)。
- B10 断链 symlink:read/write/bash × model/resume → 干净 ToolResult,无异常炸穿;
  decide 对断链/loop/越界目标全函数(totality)。

平台条件:依赖真实 bash 的强制测试 skipif(not probe_backend().available),原因
"bwrap deferred / no sandbox backend";decide/preflight 层全平台照跑(fake/纯代数)。
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.adapters.sandbox.base import probe_backend
from praxis.adapters.sandbox.compile import compile_sandbox, finalize_sandbox, validate_aliases
from praxis.core import permission
from praxis.interfaces.cli import app
from praxis.interfaces.sdk import open_session
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.models import ModelResponse, ModelToolCall, Usage
from praxis.protocol.permission import Decision, PermissionContext, PermissionMode
from praxis.protocol.policy import NetworkPolicy, PermissionPolicy, ToolRule, resolve_policy
from praxis.protocol.sandbox import SandboxRefusal
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolResult,
    ToolTarget,
    json_schema_input,
)
from praxis.store.log import Store
from praxis.tools.builtins.filesystem import create_read_tool

_AVAILABLE = probe_backend()
_NET_ALLOW = PermissionPolicy(network=NetworkPolicy(decision="allow"))
_CLEAN_ENV = {"PRAXIS_MODEL": None, "PRAXIS_BASE_URL": None, "PRAXIS_API_KEY": None, "OPENAI_API_KEY": None}
_runner = CliRunner()
_CALL = "c1"
_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}

_SEED = {
    "note.txt": "seed\n",
    ".git/config": "orig\n",
    ".praxis/state.json": "{}\n",
    ".agent_memory/mem.txt": "mem\n",
}


# ---------------------------------------------------------------------------
# 通用脚手架:真实入口 + item 观测(口径同三入口属性测试)
# ---------------------------------------------------------------------------


def _seed(workspace: Path) -> None:
    for rel, content in _SEED.items():
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _read_items(workspace: Path, thread_id: str) -> list[Item]:
    store = Store(workspace / ".praxis" / "praxis.sqlite")
    try:
        return store.read_items(thread_id)
    finally:
        store.close()


def _resp(content: str, calls: tuple[tuple[str, str, Mapping[str, Any]], ...] = ()) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tuple(ModelToolCall(call_id=c, name=n, arguments=dict(a)) for c, n, a in calls),
        usage=Usage(input_tokens=1, output_tokens=1),
        finish_reason="tool_calls" if calls else "stop",
    )


def _observe(items: list[Item], call_id: str) -> tuple[str, str]:
    """首次决策类别 + reason(executed/denied/asked;denied reason=error_message)。"""
    results = [i.payload for i in items if i.kind == ItemKind.TOOL_RESULT and i.payload.get("call_id") == call_id]
    calls = [i.payload for i in items if i.kind == ItemKind.TOOL_CALL and i.payload.get("call_id") == call_id]
    requests = [i.payload for i in items if i.kind == ItemKind.APPROVAL_REQUEST and i.payload.get("call_id") == call_id]
    decisions = [i.payload for i in items if i.kind == ItemKind.APPROVAL_DECISION and i.payload.get("call_id") == call_id]
    if results:
        r = results[0]
        msg = str(r.get("error_message") or "")
        if r.get("status") == "ok" and calls:
            return "executed", "executed"
        if r.get("status") == "denied" or r.get("error_code") in {"permission_denied", "sandbox_refused"} or msg.startswith("floor"):
            return "denied", msg
        raise AssertionError(f"unexpected result: {r!r}")
    if requests and not decisions:
        return "asked", "approval_required"
    raise AssertionError(f"no decision for {call_id}: {[i.kind for i in items]}")


def _via_model(
    workspace: Path, tool: str, args: Mapping[str, Any], policy: PermissionPolicy, mode: PermissionMode | None,
    tools: Sequence[Tool] | None = None,
) -> list[Item]:
    adapter = ScriptedAdapter([_resp("", ((_CALL, tool, args),)), _resp("done")])
    session = open_session(workspace, adapter, policy=policy, mode=mode, thread_id="model", tools=tools)
    try:
        asyncio.run(session.submit("enforcement acceptance"))
    finally:
        session.close()
    return _read_items(workspace, "model")


def _via_resume(
    workspace: Path, tool: str, args: Mapping[str, Any], policy: PermissionPolicy, mode: PermissionMode | None,
    tools: Sequence[Tool] | None = None, *, auto_resume: bool = True,
) -> list[Item]:
    adapter = ScriptedAdapter([_resp("", ((_CALL, tool, args),)), _resp("done")])
    session = open_session(workspace, adapter, policy=policy, mode=mode, thread_id="resume", tools=tools)
    try:
        result = asyncio.run(session.submit("enforcement acceptance"))
        items = session.items()
        # 待决审批 = 有 approval_request 且无 matching decision(不依赖结果分类,
        # 兼容 B10 的 not_found 等干净工具错误)。
        requested = any(i.kind == ItemKind.APPROVAL_REQUEST and i.payload.get("call_id") == _CALL for i in items)
        decided = any(i.kind == ItemKind.APPROVAL_DECISION and i.payload.get("call_id") == _CALL for i in items)
        if auto_resume and requested and not decided:
            asyncio.run(session.resume(result.turn_id, approval={_CALL: "allow"}))
            items = session.items()
    finally:
        session.close()
    return items


def _via_cli(
    workspace: Path, tool: str, args: Mapping[str, Any], policy: PermissionPolicy, mode: PermissionMode | None,
    files: Path,
) -> list[Item]:
    policy_file = files / "policy.json"
    policy_file.write_text(policy.model_dump_json(), encoding="utf-8")
    script_file = files / "scripted.json"
    script_file.write_text(
        json.dumps([{"content": "", "tool_calls": [{"call_id": _CALL, "name": tool, "arguments": dict(args)}]}, {"content": "done"}]),
        encoding="utf-8",
    )
    argv = ["run", "acceptance", "--workspace", str(workspace), "--thread", "cli", "--scripted", str(script_file), "--policy", str(policy_file)]
    if mode is not None:
        argv += ["--mode", mode]
    result = _runner.invoke(app, argv, env=_CLEAN_ENV)
    assert result.exit_code in (0, 2), (result.exit_code, result.output, repr(result.exception))
    return _read_items(workspace, "cli")


def _three_entries(
    root: Path, tool: str, args: Mapping[str, Any], policy: PermissionPolicy, mode: PermissionMode | None = None,
) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for name in ("model", "cli", "resume"):
        ws = root / name
        ws.mkdir()
        _seed(ws)
        if name == "model":
            items = _via_model(ws, tool, args, policy, mode)
        elif name == "cli":
            items = _via_cli(ws, tool, args, policy, mode, root)
        else:
            items = _via_resume(ws, tool, args, policy, mode)
        out[name] = _observe(items, _CALL)
    return out


def _assert_all(obs: dict[str, tuple[str, str]], expected_class: str) -> None:
    assert {c for c, _ in obs.values()} == {expected_class}, obs
    assert len({r for _, r in obs.values()}) == 1, obs


# ---------------------------------------------------------------------------
# S1:唯一 profile 编译器(代码字符串扫描,排除注释/docstring)
# ---------------------------------------------------------------------------

_PROFILE_LITERALS = ("(deny default)", "(version 1)", "sandbox-exec", "--unshare-all", "bwrap")


def _code_string_literals(source: str) -> list[str]:
    """AST 取代码字符串常量,排除 docstring(注释本就不进 AST)——只钉构造器,不误伤文档。"""
    tree = ast.parse(source)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
    ]


def test_s1_no_second_profile_builder() -> None:
    root = Path(__file__).resolve().parents[3] / "praxis"
    offenders: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("adapters/sandbox/"):
            continue
        for literal in _code_string_literals(path.read_text(encoding="utf-8")):
            for marker in _PROFILE_LITERALS:
                if marker in literal:
                    offenders.append((rel, marker))
    assert offenders == [], f"profile construction literals outside adapters/sandbox: {offenders}"
    # 非空泛:唯一编译器确实持有这些字面量。
    seatbelt = (root / "adapters" / "sandbox" / "seatbelt.py").read_text(encoding="utf-8")
    assert "(deny default)" in seatbelt and "(version 1)" in seatbelt
    base = (root / "adapters" / "sandbox" / "base.py").read_text(encoding="utf-8")
    assert "sandbox-exec" in base


# ---------------------------------------------------------------------------
# S2:地板矩阵 —— decide 层 × 三入口 + 真实 seatbelt bash 强制 + bypass 免疫
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "args", "expected"),
    [
        ("read", {"path": ".git/config"}, "executed"),  # .git 只读地板:read 放(阳性对照)
        ("write", {"path": ".git/config", "content": "x"}, "denied"),
        ("read", {"path": ".praxis/state.json"}, "denied"),
        ("write", {"path": ".praxis/state.json", "content": "x"}, "denied"),
        ("read", {"path": ".agent_memory/mem.txt"}, "denied"),
        ("write", {"path": ".agent_memory/mem.txt", "content": "x"}, "denied"),
    ],
)
def test_s2_floor_matrix_all_entries(tool: str, args: Mapping[str, Any], expected: str, tmp_path: Path) -> None:
    obs = _three_entries(tmp_path, tool, args, PermissionPolicy())
    _assert_all(obs, expected)
    if expected == "denied":
        assert all(reason.startswith("floor:") for _, reason in obs.values()), obs


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("write", {"path": ".git/config", "content": "x"}),
        ("read", {"path": ".praxis/state.json"}),
        ("write", {"path": ".agent_memory/mem.txt", "content": "x"}),
    ],
)
def test_s2_floor_bypass_immune_all_entries(tool: str, args: Mapping[str, Any], tmp_path: Path) -> None:
    obs = _three_entries(tmp_path, tool, args, PermissionPolicy(), mode="bypass")
    _assert_all(obs, "denied")
    assert all(reason.startswith("floor:") for _, reason in obs.values()), obs


@pytest.mark.skipif(not _AVAILABLE.available, reason="bwrap deferred / no sandbox backend")
def test_s2_real_seatbelt_floor_enforcement(tmp_path: Path) -> None:
    """真实 bash 经组合根 → sandbox-exec:.git 写拒(内容不变)/ .praxis 读拒 / workspace 写 + .git 读放。"""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _seed(workspace)
    adapter = ScriptedAdapter(
        [
            _resp("", (("c1", "bash", {"command": "echo hacked > .git/config"}),)),
            _resp("", (("c2", "bash", {"command": "cat .praxis/state.json"}),)),
            _resp("", (("c3", "bash", {"command": "echo ok > note.txt && cat .git/config"}),)),
            _resp("done"),
        ]
    )
    session = open_session(workspace, adapter, policy=_NET_ALLOW)
    try:
        result = asyncio.run(session.submit("real floor enforcement"))
        items = session.items()
    finally:
        session.close()
    assert result.status.value == "completed"
    results = {i.payload["call_id"]: i.payload for i in items if i.kind == ItemKind.TOOL_RESULT}
    # .git 写:syscall 层失败,内容不变
    assert results["c1"]["metadata"]["exit_code"] != 0  # type: ignore[index]
    assert (workspace / ".git" / "config").read_text() == "orig\n"
    # .praxis 读:失败
    assert results["c2"]["metadata"]["exit_code"] != 0  # type: ignore[index]
    # 阳性对照:workspace 写成功 + .git 读放行
    assert results["c3"]["metadata"]["exit_code"] == 0, results["c3"]  # type: ignore[index]
    assert (workspace / "note.txt").read_text() == "ok\n"
    assert "orig" in results["c3"]["content"][0]["data"]["text"]  # type: ignore[index]


@pytest.mark.skipif(not _AVAILABLE.available, reason="bwrap deferred / no sandbox backend")
def test_s2_real_bash_floor_bypass_immune(tmp_path: Path) -> None:
    """mode=bypass 也不能让 bash 写地板:seatbelt 地板段与 mode 无关。"""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _seed(workspace)
    adapter = ScriptedAdapter([_resp("", (("c1", "bash", {"command": "echo hacked > .git/config"}),)), _resp("done")])
    session = open_session(workspace, adapter, policy=_NET_ALLOW, mode="bypass")
    try:
        asyncio.run(session.submit("bypass floor"))
        items = session.items()
    finally:
        session.close()
    results = {i.payload["call_id"]: i.payload for i in items if i.kind == ItemKind.TOOL_RESULT}
    assert results["c1"]["metadata"]["exit_code"] != 0  # type: ignore[index]
    assert (workspace / ".git" / "config").read_text() == "orig\n"


# ---------------------------------------------------------------------------
# S8:网络 ask 全入口不降级(合成 NETWORK 工具 + ToolRule allow 共存)
# ---------------------------------------------------------------------------


def _network_tool(workspace: Path) -> Tool:
    """合成 URL 级 NETWORK 工具:effects={NETWORK},target kind=url。"""

    async def run(arguments: Mapping[str, Any], exec_ctx: Any) -> ToolResult:
        return ToolResult(
            tool_call_id="pending", tool_name="netfetch",
            content=(ToolContentBlock(type="text", data={"text": "fetched"}),), status=ResultStatus.OK,
        )

    return Tool(
        definition=ToolDefinition(name="netfetch", description="synthetic url tool", input_schema=_EMPTY_SCHEMA),
        validate_input=json_schema_input(_EMPTY_SCHEMA),
        run=run,
        static_effects=frozenset({ToolEffect.NETWORK}),
        resolve_use=lambda arguments: ResolvedUse(
            effects=frozenset({ToolEffect.NETWORK}),
            targets=(ToolTarget(kind="url", value="https://example.com/api"),),
        ),
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30.0,
        max_model_output_bytes=4096,
    )


@pytest.mark.parametrize("network_decision", ["ask", "deny"])
def test_s8_network_not_widened_by_tool_allow_model_resume(network_decision: str, tmp_path: Path) -> None:
    """合成 NETWORK 工具 + ToolRule allow 共存:network ask→asked,deny→denied(model/resume)。"""
    policy = PermissionPolicy(
        network=NetworkPolicy(decision=network_decision),  # type: ignore[arg-type]
        tools=(ToolRule(pattern="netfetch", decision="allow"),),
    )
    expected = "asked" if network_decision == "ask" else "denied"
    for name in ("model", "resume"):
        ws = tmp_path / name
        ws.mkdir()
        _seed(ws)
        tools = [_network_tool(ws)]
        # 观测首次决策:resume 入口不自动重入(asked 即 submit 的暂停)。
        items = (
            _via_model(ws, "netfetch", {}, policy, None, tools=tools)
            if name == "model"
            else _via_resume(ws, "netfetch", {}, policy, None, tools=tools, auto_resume=False)
        )
        assert _observe(items, _CALL)[0] == expected, (name, _observe(items, _CALL))


def test_s8_network_ask_cli_bash_face(tmp_path: Path) -> None:
    """CLI 网络可达面(bash 命令网络):network ask + ToolRule bash allow → asked;deny → denied。

    url 级 web 工具阶段 3 落地;CLI 内置 read/write/bash,命令网络是 CLI 当前网络面。
    """
    for decision, expected in (("ask", "asked"), ("deny", "denied")):
        policy = PermissionPolicy(
            network=NetworkPolicy(decision=decision),  # type: ignore[arg-type]
            tools=(ToolRule(pattern="bash", decision="allow"),),
        )
        ws = tmp_path / decision
        ws.mkdir()
        _seed(ws)
        items = _via_cli(ws, "bash", {"command": "echo ok"}, policy, None, tmp_path)
        assert _observe(items, _CALL)[0] == expected, (decision, _observe(items, _CALL))


def test_s8_network_ask_decide_never_widened(tmp_path: Path) -> None:
    """decide 代数:network ask 先于 bypass/allow/回调 allow,任何来源不得放宽(S8)。"""
    tool = _network_tool(tmp_path)
    use = tool.resolve_use({})
    url = "https://example.com/api"
    ask_policy = PermissionPolicy(network=NetworkPolicy(decision="ask"), tools=(ToolRule(pattern="netfetch", decision="allow"),))
    for mode in ("on-request", "bypass"):
        ctx = PermissionContext(workspace=str(tmp_path), policy=ask_policy, mode=mode, sandbox=_AVAILABLE)  # type: ignore[arg-type]
        decision = permission.decide(tool, use, ctx)
        assert decision.is_ask, (mode, decision)
        assert url in decision.reason
    # 回调 allow 也不放宽 network ask(D5)。
    allow_callback = replace(tool, check_permission=lambda use, ctx: Decision.allow("callback"))
    ctx = PermissionContext(workspace=str(tmp_path), policy=ask_policy, mode="on-request", sandbox=_AVAILABLE)
    assert permission.decide(allow_callback, use, ctx).is_ask
    # network deny 恒 deny。
    deny_policy = PermissionPolicy(network=NetworkPolicy(decision="deny"), tools=(ToolRule(pattern="netfetch", decision="allow"),))
    ctx = PermissionContext(workspace=str(tmp_path), policy=deny_policy, mode="bypass", sandbox=_AVAILABLE)
    assert permission.decide(tool, use, ctx).is_deny


# ---------------------------------------------------------------------------
# S9:别名攻击 —— preflight 跨地板硬链接拒绝(攻击 + 无误报)+ 运行时 ln 探针
# ---------------------------------------------------------------------------


def _resolved(workspace: Path, policy: PermissionPolicy | None = None):
    return resolve_policy(policy if policy is not None else PermissionPolicy(), workspace)


def test_s9_cross_floor_hardlink_preflight_refused(tmp_path: Path) -> None:
    """攻击:工作区硬链接别名指向 .git/config(跨地板)→ preflight SandboxRefusal。"""
    workspace = tmp_path / "ws"
    (workspace / ".git").mkdir(parents=True)
    (workspace / ".git" / "config").write_text("orig")
    os.link(workspace / ".git" / "config", workspace / "alias")
    with pytest.raises(SandboxRefusal, match="hardlink alias crosses the safety floor"):
        validate_aliases(_resolved(workspace), workspace, allow_write=True)


def test_s9_cross_floor_hardlink_no_false_positive(tmp_path: Path) -> None:
    """无误报:常规文件 / 工作区内硬链接 / 地板内硬链接 均不触发跨地板拒绝。"""
    workspace = tmp_path / "ws"
    (workspace / ".git").mkdir(parents=True)
    (workspace / ".git" / "config").write_text("orig")
    (workspace / "a.txt").write_text("a")
    os.link(workspace / "a.txt", workspace / "b.txt")  # 工作区内(非地板)硬链接
    os.link(workspace / ".git" / "config", workspace / ".git" / "config2")  # 地板内硬链接
    validate_aliases(_resolved(workspace), workspace, allow_write=True)  # 不抛


@pytest.mark.skipif(not _AVAILABLE.available, reason="bwrap deferred / no sandbox backend")
def test_s9_runtime_ln_probe_blocked(tmp_path: Path) -> None:
    """真实沙箱内 ln .git/config 硬链接再写 → 阻断(link 需源写权限),内容不变。"""
    workspace = tmp_path / "ws"
    (workspace / ".git").mkdir(parents=True)
    (workspace / ".git" / "config").write_text("orig")
    ctx = PermissionContext(workspace=str(workspace), policy=PermissionPolicy(), mode="on-request")
    use = ResolvedUse(effects=frozenset({ToolEffect.EXECUTE_PROCESS}), targets=(ToolTarget(kind="path", value=str(workspace)),))
    spec = compile_sandbox(ctx, use, workspace)
    try:
        result = subprocess.run(
            [*spec.argv_prefix, "/bin/bash", "-lc", "ln .git/config alias && echo hacked > alias"],
            cwd=workspace, capture_output=True, text=True, timeout=30,
        )
        assert (workspace / ".git" / "config").read_text() == "orig"
        assert result.returncode != 0 or "Operation not permitted" in (result.stdout + result.stderr)
    finally:
        finalize_sandbox(spec, workspace)


# ---------------------------------------------------------------------------
# B10:断链 symlink × read/write/bash × model/resume → 干净 ToolResult,无炸穿
# ---------------------------------------------------------------------------


# bash 结局平台相关:有沙箱→命令执行(ok,非零退出不是 is_error);无沙箱→fail-closed denied。
_BASH_CLEAN_STATUS = "ok" if _AVAILABLE.available else "denied"


@pytest.mark.parametrize(
    ("tool", "args", "expected_status"),
    [
        ("read", {"path": "dangling.txt"}, "error"),  # not_found(干净)
        ("write", {"path": "dangling.txt", "content": "x"}, "ok"),  # 创建目标(干净)
        ("bash", {"command": "cat dangling.txt"}, _BASH_CLEAN_STATUS),
    ],
)
def test_b10_broken_symlink_clean_results(tool: str, args: Mapping[str, Any], expected_status: str, tmp_path: Path) -> None:
    for name, runner in (("model", _via_model), ("resume", _via_resume)):
        ws = tmp_path / name
        ws.mkdir()
        _seed(ws)
        (ws / "dangling.txt").symlink_to(ws / "missing_target.txt")  # 断链:目标不存在
        items = runner(ws, tool, args, _NET_ALLOW, None)
        results = [i.payload for i in items if i.kind == ItemKind.TOOL_RESULT and i.payload.get("call_id") == _CALL]
        assert results, (name, "no tool_result — exception escaped the turn")
        assert results[0]["status"] == expected_status, (name, results[0])
        # turn 必须自然收尾(COMPLETED),而非异常 FAILED(turn_status payload = {from,to,reason})。
        statuses = [i.payload for i in items if i.kind == ItemKind.TURN_STATUS]
        assert statuses[-1]["to"] == "completed", (name, statuses[-1])
        if tool == "read":
            assert results[0]["error_code"] == "not_found"


def test_b10_decide_totality_on_broken_targets(tmp_path: Path) -> None:
    """decide 全函数:断链/loop/越界 symlink 目标都不抛,返回 Decision(B10)。"""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "dangling").symlink_to(workspace / "missing")
    (workspace / "loop").symlink_to(workspace / "loop")
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "escape").symlink_to(outside / "target")
    tool = create_read_tool(workspace)
    ctx = PermissionContext(workspace=str(workspace), policy=PermissionPolicy(), mode="on-request", sandbox=_AVAILABLE)
    for target in (workspace / "dangling", workspace / "loop", workspace / "escape", workspace / "missing"):
        use = ResolvedUse(effects=frozenset({ToolEffect.READ_WORKSPACE}), targets=(ToolTarget(kind="path", value=str(target)),))
        decision = permission.decide(tool, use, ctx)  # 不抛
        assert decision.verdict in {"allow", "ask", "deny"}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))