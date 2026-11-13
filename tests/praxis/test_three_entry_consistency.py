"""Task 9:三入口决策一致性属性测试(设计 §7.2/§8 验收;硬规则 3)。

设计 §7.2 要求"同一策略在 CLI/模型/子代理三入口决策一致(属性测试)"。子代理入口
阶段 4 才落地,当前可达面 = **模型 / CLI / resume** 三入口——它们一律经唯一管道
``praxis.core.pipeline.execute`` 执行工具,唯一决策入口 ``permission.decide``。
``ENTRIES`` 注册表是扩展位:阶段 4 子代理入口复用同一 execute,注册即纳入本属性。

观测口径(brief 钉死):比较**首次 execute 的决策类别 + reason**——
- executed = tool_call item + status=ok 的 tool_result;
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
import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from praxis import composition
from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.adapters.sandbox.base import probe_backend
from praxis.core import permission
from praxis.interfaces.cli import app
from praxis.interfaces.sdk import open_session
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.models import ModelResponse, ModelToolCall, Usage
from praxis.protocol.permission import PermissionContext, PermissionMode, SandboxAvailability
from praxis.protocol.policy import NetworkPolicy, PathRule, PermissionPolicy, ToolRule
from praxis.protocol.tools import Tool
from praxis.store.log import Store
from praxis.tools.builtins.command import create_bash_tool
from praxis.tools.builtins.filesystem import create_read_tool, create_write_tool

CALL_ID = "c1"
_FAKE_SANDBOX = SandboxAvailability(backend="seatbelt", available=True)
_AVAILABLE = probe_backend()
_CLEAN_ENV = {"PRAXIS_MODEL": None, "PRAXIS_BASE_URL": None, "PRAXIS_API_KEY": None, "OPENAI_API_KEY": None}
_runner = CliRunner()

# 预置 read 目标(保证 read 执行即 ok,不落入 not_found);write/bash 目标见策略。
_SEED_FILES = {
    "note.txt": "seed note\n",
    ".git/config": "orig\n",
    ".praxis/state.json": "{}\n",
    ".agent_memory/mem.txt": "memory\n",
    "sub/deep.txt": "deep\n",
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


@st.composite
def call_strategy(draw: st.DrawFn) -> Call:
    tool = draw(st.sampled_from(("read", "write", "bash")))
    if tool == "read":
        return Call(tool, {"path": draw(st.sampled_from(_READ_PATHS))})
    if tool == "write":
        return Call(tool, {"path": draw(st.sampled_from(_WRITE_PATHS)), "content": "praxis property\n"})
    return Call(tool, {"command": draw(st.sampled_from(_BASH_COMMANDS))})


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
    for name in ("read", "write", "bash"):
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
        if status == "ok" and calls:
            return Observation("executed", "executed")
        if status == "denied" or error_code in {"permission_denied", "sandbox_refused"} or message.startswith("floor"):
            return Observation("denied", _normalize_reason(str(message), workspace))
        raise AssertionError(f"unexpected first result shape: {result!r}")
    if requests and not decisions:
        return Observation("asked", "approval_required")
    raise AssertionError(f"no first decision observed for {call_id}: kinds={[i.kind for i in items]}")


def _assert_reentry_outcome(items: list[Item], call_id: str, policy: PermissionPolicy) -> None:
    """resume 重入断言:审批 args_digest/policy_revision 绑定 + 重入结局正确。

    重入两途皆合法:
    - 一般 ask 批准 → 执行(tool_call + ok,digest 绑定到执行参数);
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
        assert result.get("status") == "ok", "approved non-floor call must execute"
        assert decisions[-1].get("args_digest") == calls[-1].get("args_digest"), "approval digest binds executed args"
    else:
        assert result.get("status") == "denied", "floor target must be denied at execution guard"
        assert str(result.get("error_message") or "").startswith("floor"), "R5c execution floor guard"


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
    session = open_session(workspace, adapter, policy=scenario.policy, mode=scenario.mode, thread_id="model")
    try:
        asyncio.run(session.submit("three-entry property"))
    finally:
        session.close()
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


# 阶段 4 子代理入口复用同一 execute,届时在此注册即纳入一致性属性。
ENTRIES: dict[str, Callable[[Path, Call, Scenario, Path], Observation]] = {
    "model": run_via_model,
    "cli": run_via_cli,
    "resume": run_via_resume,
}


def _reference_decision(workspace: Path, call: Call, scenario: Scenario, sandbox: SandboxAvailability) -> permission.Decision:
    tools: dict[str, Tool] = {
        "read": create_read_tool(workspace),
        "write": create_write_tool(workspace),
        "bash": create_bash_tool(workspace),
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
    ],
)
def test_decision_matrix_all_entries(
    label: str, call: Call, scenario: Scenario, expected: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(composition, "probe_backend", lambda: _FAKE_SANDBOX)
        mp.setattr(composition, "compile_sandbox", None)
        mp.setattr(composition, "finalize_sandbox", None)
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))