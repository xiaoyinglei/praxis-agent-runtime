"""Task 5:hooks 管道前后两执行点 +「hook allow 不豁免规则」(设计 §5.14)。

钉住七件事(全部经**真管道** ``pipeline.execute`` + **真 command hook 子进程**,
不 mock ``spawn_hook``):

1. hook allow 结构上无裁决:规则 deny(第 1 步)、bypass(第 5 步)都不会因
   hook 放行而豁免;hook deny 与规则 deny 同时在场时 reason 取规则(第 1 步内序);
2. hook deny 走既有 ``permission_denied`` 单一口径,reason 形如
   ``hook:<name>:<status>: <feedback>``;
3. EXECUTE_PROCESS 的沙箱裁决(第 7 步)不因 hook allow 旁路;
4. 审批重入幂等:``started``/``settled`` item 折叠复用,子进程 spawn 恰 1 次;
5. PostToolUse 只观察:结果 status/content 逐字不变,metadata 记
   ``post_tool_hook``,settled item 照写,绝不阻断;
6. 伪造「仅 started 无 settled」⇒ decide 之前即 fail-closed deny(unknown),
   0 次 spawn;
7. 三入口(model/CLI/resume)同 call+policy+hooks ⇒ 同 decision+reason。

仓库无 pytest-asyncio:与 test_pipeline.py 同风格,asyncio.run 包装。
hook 脚本落在 workspace 之外的 host 目录(与真实 ``HookSpec.directory`` 语义一致)。
"""

from __future__ import annotations

import asyncio
import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from praxis.adapters.hooks.runner import spawn_hook
from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, execute
from praxis.core.statemachine import TurnStatus, transition
from praxis.interfaces.cli import app
from praxis.interfaces.sdk import open_session
from praxis.protocol.hooks import (
    HookOutcome,
    HookSetup,
    HookSpec,
    hook_invocation_id,
)
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import ModelResponse, ModelToolCall, Usage
from praxis.protocol.permission import PermissionContext, SandboxAvailability
from praxis.protocol.policy import NetworkPolicy, PathRule, PermissionPolicy, ToolRule
from praxis.protocol.tools import ResultStatus, Tool, ToolCall, args_digest
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import create_turn
from praxis.tools.builtins.command import create_bash_tool
from praxis.tools.builtins.filesystem import create_read_tool
from praxis.tools.builtins.process import signal_group

_CALL_ID = "c1"
_FEEDBACK = "policy says no"
_BLOCKED = json.dumps({"block": True, "feedback": _FEEDBACK})
_ALLOWED = json.dumps({"block": False})
_runner = CliRunner()
_CLEAN_ENV = {
    "PRAXIS_MODEL": None,
    "PRAXIS_BASE_URL": None,
    "PRAXIS_API_KEY": None,
    "OPENAI_API_KEY": None,
}


# ---------------------------------------------------------------------------
# 脚手架:真实 command hook / ExecCtx / item 读取
# ---------------------------------------------------------------------------


async def _real_spawn(spec: HookSpec, request: Mapping[str, JsonValue]) -> HookOutcome:
    """与 composition 相同的真实绑定(整组 SIGKILL 语义)。"""
    return await spawn_hook(spec, request, kill_group=signal_group)


def _hook_setup(
    host: Path,
    *,
    event: str = "PreToolUse",
    stdout_json: str = _ALLOWED,
    counter: Path | None = None,
    name: str = "policy",
) -> HookSetup:
    """落一个真实 /bin/sh hook 并绑定真实 spawn;counter 记录每次进程启动。"""
    host.mkdir(parents=True, exist_ok=True)
    body = ""
    if counter is not None:
        body += f"echo run >> {shlex.quote(str(counter))}\n"
    body += "cat >/dev/null\n"
    body += f"printf '%s' {shlex.quote(stdout_json)}"
    script = host / f"{name}-{event}.sh"
    script.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    script.chmod(0o755)
    spec = HookSpec(
        name=name,
        revision="1",
        event=cast("Any", event),
        argv=(str(script),),
        directory=str(host),
    )
    return HookSetup(specs=(spec,), spawn=_real_spawn)


def _ctx(
    store: Store,
    workspace: Path,
    tools: Mapping[str, Tool],
    *,
    policy: PermissionPolicy | None = None,
    mode: str = "on-request",
    hooks: HookSetup | None = None,
    sandbox: SandboxAvailability | None = None,
    turn_id: str = "turn-1",
) -> ExecCtx:
    effective = policy if policy is not None else PermissionPolicy()
    create_turn(store, turn_id=turn_id, thread_id="thread-1", binding={})
    return ExecCtx(
        store=store,
        bus=EventBus(),
        workspace=workspace,
        tools=dict(tools),
        permissions=PermissionContext(
            workspace=str(workspace),
            policy=effective,
            mode=cast("Any", mode),
            sandbox=sandbox
            if sandbox is not None
            else SandboxAvailability(backend="fake", available=True),
        ),
        turn_id=turn_id,
        thread_id="thread-1",
        policy_revision=effective.revision,
        hooks=hooks,
    )


def _hook_items(ctx: ExecCtx) -> list[Item]:
    return [
        item
        for item in ctx.store.read_turn_items(ctx.turn_id)
        if item.kind == ItemKind.HOOK_INVOCATION
    ]


def _phases(items: list[Item]) -> list[str]:
    return [str(item.payload.get("phase")) for item in items]


def _spawn_count(counter: Path) -> int:
    return len(counter.read_text(encoding="utf-8").splitlines()) if counter.exists() else 0


def _digest(tool: Tool, arguments: Mapping[str, Any]) -> str:
    return args_digest(tool.validate_input(arguments))


def _run(ctx: ExecCtx, tool_name: str, arguments: Mapping[str, Any]) -> Any:
    return asyncio.run(execute(ToolCall(_CALL_ID, tool_name, dict(arguments)), ctx))


# ---------------------------------------------------------------------------
# ① 规则 deny + hook allow ⇒ 仍 deny(reason 仍是 rule)
# ---------------------------------------------------------------------------


def test_rule_deny_not_exempted_by_hook_allow(store: Store, workspace: Path, tmp_path: Path) -> None:
    (workspace / "secret.txt").write_text("secret\n", encoding="utf-8")
    counter = tmp_path / "spawns.txt"
    ctx = _ctx(
        store,
        workspace,
        {"read": create_read_tool(workspace)},
        policy=PermissionPolicy(filesystem=(PathRule(path="secret.txt", access="deny"),)),
        hooks=_hook_setup(tmp_path / "host", counter=counter, stdout_json=_ALLOWED),
    )
    result = _run(ctx, "read", {"path": "secret.txt"})
    assert result.status is ResultStatus.DENIED
    assert result.error_code == "permission_denied"
    assert result.error_message.startswith("rule: ")
    assert "hook:" not in result.error_message
    assert _spawn_count(counter) == 1  # hook 确实执行了;allow 只是结构上不产生裁决


# ---------------------------------------------------------------------------
# ② bypass + hook deny ⇒ deny(hook deny 在合并器第 1 步)
# ---------------------------------------------------------------------------


def test_bypass_mode_does_not_exempt_hook_deny(store: Store, workspace: Path, tmp_path: Path) -> None:
    (workspace / "note.txt").write_text("seed\n", encoding="utf-8")
    counter = tmp_path / "spawns.txt"
    ctx = _ctx(
        store,
        workspace,
        {"read": create_read_tool(workspace)},
        mode="bypass",
        hooks=_hook_setup(tmp_path / "host", stdout_json=_BLOCKED, counter=counter),
    )
    result = _run(ctx, "read", {"path": "note.txt"})
    assert result.status is ResultStatus.DENIED
    assert result.error_code == "permission_denied"
    assert result.error_message == f"hook:policy:blocked: {_FEEDBACK}"
    assert _spawn_count(counter) == 1


# ---------------------------------------------------------------------------
# ③ hook deny 无规则意见 ⇒ permission_denied,message 含 hook: 与 feedback
# ---------------------------------------------------------------------------


def test_hook_deny_reason_carries_feedback(store: Store, workspace: Path, tmp_path: Path) -> None:
    (workspace / "note.txt").write_text("seed\n", encoding="utf-8")
    counter = tmp_path / "spawns.txt"
    ctx = _ctx(
        store,
        workspace,
        {"read": create_read_tool(workspace)},
        hooks=_hook_setup(tmp_path / "host", stdout_json=_BLOCKED, counter=counter),
    )
    result = _run(ctx, "read", {"path": "note.txt"})
    assert result.status is ResultStatus.DENIED
    assert result.error_code == "permission_denied"
    assert result.error_message == f"hook:policy:blocked: {_FEEDBACK}"
    assert _spawn_count(counter) == 1


# ---------------------------------------------------------------------------
# ④ hook deny 与规则 deny 同时存在 ⇒ reason 取规则(第 1 步内序)
# ---------------------------------------------------------------------------


def test_rule_deny_wins_over_hook_deny(store: Store, workspace: Path, tmp_path: Path) -> None:
    (workspace / "secret.txt").write_text("secret\n", encoding="utf-8")
    ctx = _ctx(
        store,
        workspace,
        {"read": create_read_tool(workspace)},
        policy=PermissionPolicy(filesystem=(PathRule(path="secret.txt", access="deny"),)),
        hooks=_hook_setup(tmp_path / "host", stdout_json=_BLOCKED),
    )
    result = _run(ctx, "read", {"path": "secret.txt"})
    assert result.status is ResultStatus.DENIED
    assert result.error_message.startswith("rule: ")
    assert "hook:" not in result.error_message


# ---------------------------------------------------------------------------
# ⑤ EXECUTE_PROCESS 沙箱裁决照旧(hook allow 不旁路第 7 步)
# ---------------------------------------------------------------------------


def test_execute_process_sandbox_verdict_not_bypassed_by_hook_allow(
    store: Store, workspace: Path, tmp_path: Path
) -> None:
    counter = tmp_path / "spawns.txt"
    ctx = _ctx(
        store,
        workspace,
        {"bash": create_bash_tool(workspace)},
        policy=PermissionPolicy(network=NetworkPolicy(decision="allow")),
        hooks=_hook_setup(tmp_path / "host", counter=counter),
        sandbox=SandboxAvailability(backend=None, available=False),
    )
    result = _run(ctx, "bash", {"command": "echo hi"})
    assert result.status is ResultStatus.DENIED
    assert result.error_code == "permission_denied"
    assert result.error_message == "sandbox_unavailable"
    assert _spawn_count(counter) == 1


# ---------------------------------------------------------------------------
# ⑥ 审批重入:fold 复用 ⇒ 子进程 spawn 恰 1 次、settled 恰 1 组
# ---------------------------------------------------------------------------


def test_approval_reentry_reuses_settled_hook_without_respawn(
    store: Store, workspace: Path, tmp_path: Path
) -> None:
    (workspace / "note.txt").write_text("hello hook\n", encoding="utf-8")
    counter = tmp_path / "spawns.txt"
    tool = create_read_tool(workspace)
    ctx = _ctx(
        store,
        workspace,
        {"read": tool},
        policy=PermissionPolicy(tools=(ToolRule(pattern="read", decision="ask"),)),
        hooks=_hook_setup(tmp_path / "host", counter=counter),
    )
    first = _run(ctx, "read", {"path": "note.txt"})
    assert first.status is ResultStatus.PAUSED
    assert _spawn_count(counter) == 1
    assert _phases(_hook_items(ctx)) == ["started", "settled"]

    ctx.store.append_items(
        ctx.thread_id,
        ctx.turn_id,
        [
            ItemEntry(
                kind=ItemKind.APPROVAL_DECISION,
                payload={
                    "call_id": _CALL_ID,
                    "verdict": "allow",
                    "args_digest": _digest(tool, {"path": "note.txt"}),
                    "destination": "once",
                    "policy_revision": ctx.policy_revision,
                },
            )
        ],
    )
    transition(store, ctx.bus, ctx.turn_id, TurnStatus.RUNNING, reason="approval_decided")
    second = _run(ctx, "read", {"path": "note.txt"})
    assert second.status is ResultStatus.OK
    assert second.text == "hello hook\n"
    assert _spawn_count(counter) == 1  # 审批重入 0 次新 spawn
    hook_items = _hook_items(ctx)
    assert _phases(hook_items) == ["started", "settled"]  # 只有原一组,不重复写
    assert hook_items[-1].payload["status"] == "allowed"


# ---------------------------------------------------------------------------
# ⑦ PostToolUse 只观察:status/content 逐字不变,metadata 记摘要,settled 照写
# ---------------------------------------------------------------------------


def test_post_hook_observes_without_blocking(store: Store, workspace: Path, tmp_path: Path) -> None:
    (workspace / "note.txt").write_text("unchanged\n", encoding="utf-8")
    tool = create_read_tool(workspace)
    control_ctx = _ctx(store, workspace, {"read": tool}, turn_id="turn-control")
    control = _run(control_ctx, "read", {"path": "note.txt"})

    counter = tmp_path / "posts.txt"
    observed_ctx = _ctx(
        store,
        workspace,
        {"read": tool},
        hooks=_hook_setup(
            tmp_path / "host", event="PostToolUse", stdout_json=_BLOCKED, counter=counter
        ),
        turn_id="turn-observed",
    )
    observed = _run(observed_ctx, "read", {"path": "note.txt"})

    assert observed.status is control.status is ResultStatus.OK
    assert observed.text == control.text == "unchanged\n"
    assert observed.is_error is control.is_error is False
    assert observed.truncated is control.truncated is False
    assert observed.error_code == control.error_code is None
    assert "post_tool_hook" not in control.metadata
    assert observed.metadata["post_tool_hook"] == {
        "status": "blocked",
        "feedback": _FEEDBACK,
    }

    assert _spawn_count(counter) == 1
    kinds = [
        item.kind
        for item in observed_ctx.store.read_turn_items(observed_ctx.turn_id)
        if item.kind in (ItemKind.TOOL_CALL, ItemKind.HOOK_INVOCATION, ItemKind.TOOL_RESULT)
    ]
    # Post 序列在工具返回后、result item 之前:观察记录绝不改也绝不阻断结果。
    assert kinds == [
        ItemKind.TOOL_CALL,
        ItemKind.HOOK_INVOCATION,
        ItemKind.HOOK_INVOCATION,
        ItemKind.TOOL_RESULT,
    ]
    hook_items = _hook_items(observed_ctx)
    assert _phases(hook_items) == ["started", "settled"]
    assert all(item.payload["event"] == "PostToolUse" for item in hook_items)
    assert hook_items[-1].payload["status"] == "blocked"
    result_item = observed_ctx.store.read_turn_items(observed_ctx.turn_id)[-1]
    assert result_item.kind == ItemKind.TOOL_RESULT
    assert result_item.payload["status"] == "ok"
    assert result_item.payload["content"][0]["data"]["text"] == "unchanged\n"
    assert result_item.payload["metadata"]["post_tool_hook"] == {
        "status": "blocked",
        "feedback": _FEEDBACK,
    }


# ---------------------------------------------------------------------------
# ⑧ 伪造「仅 started 无 settled」⇒ decide 前即 deny(unknown),0 次 spawn
# ---------------------------------------------------------------------------


def test_started_without_settled_fails_closed_before_decide(
    store: Store, workspace: Path, tmp_path: Path
) -> None:
    (workspace / "note.txt").write_text("seed\n", encoding="utf-8")
    counter = tmp_path / "spawns.txt"
    tool = create_read_tool(workspace)
    ctx = _ctx(
        store,
        workspace,
        {"read": tool},
        hooks=_hook_setup(tmp_path / "host", counter=counter),
    )
    invocation_id = hook_invocation_id(
        ctx.turn_id, "PreToolUse", "policy", _digest(tool, {"path": "note.txt"})
    )
    store.append_items(
        ctx.thread_id,
        ctx.turn_id,
        [
            ItemEntry(
                kind=ItemKind.HOOK_INVOCATION,
                payload={
                    "invocation_id": invocation_id,
                    "event": "PreToolUse",
                    "hook": "policy",
                    "phase": "started",
                },
            )
        ],
    )
    result = _run(ctx, "read", {"path": "note.txt"})
    assert result.status is ResultStatus.DENIED
    assert result.error_code == "permission_denied"
    assert result.error_message == "hook:policy:unknown: crash before settlement"
    assert _spawn_count(counter) == 0  # fail-closed 投影面:不再 spawn


# ---------------------------------------------------------------------------
# ⑨ 三入口一致性:同 call + policy + hooks ⇒ 同 decision + reason
# ---------------------------------------------------------------------------


def _seed_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "note.txt").write_text("seed\n", encoding="utf-8")


def _scripted() -> ScriptedAdapter:
    return ScriptedAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=(
                    ModelToolCall(call_id=_CALL_ID, name="read", arguments={"path": "note.txt"}),
                ),
                usage=Usage(0, 0),
                finish_reason="tool_calls",
            ),
            ModelResponse(content="done", tool_calls=(), usage=Usage(0, 0), finish_reason="stop"),
        ]
    )


def _observe_denial(items: list[Item]) -> tuple[str, str]:
    results = [
        item.payload
        for item in items
        if item.kind == ItemKind.TOOL_RESULT and item.payload.get("call_id") == _CALL_ID
    ]
    assert len(results) == 1, [item.kind for item in items]
    return str(results[0]["status"]), str(results[0]["error_message"])


def _write_hooks_config(host: Path, script: Path) -> Path:
    host.mkdir(parents=True, exist_ok=True)
    host.chmod(0o700)
    config = host / "hooks.json"
    config.write_text(
        json.dumps([{"event": "PreToolUse", "name": "policy", "argv": [str(script)]}]),
        encoding="utf-8",
    )
    config.chmod(0o600)
    return config


def test_three_entries_same_hook_decision(tmp_path: Path) -> None:
    host = tmp_path / "host"
    hooks = _hook_setup(host, stdout_json=_BLOCKED)
    script = Path(cast("HookSpec", hooks.specs[0]).argv[0])
    hooks_config = _write_hooks_config(host, script)
    observations: dict[str, tuple[str, str]] = {}

    for entry in ("model", "resume"):
        workspace = tmp_path / entry / "workspace"
        _seed_workspace(workspace)
        session = open_session(workspace, _scripted(), thread_id=entry, hooks=hooks)
        try:
            asyncio.run(session.submit("hook parity"))
            observations[entry] = _observe_denial(session.items())
        finally:
            session.close()

    workspace = tmp_path / "cli" / "workspace"
    _seed_workspace(workspace)
    scripted_file = tmp_path / "cli" / "scripted.json"
    scripted_file.write_text(
        json.dumps(
            [
                {
                    "content": "",
                    "tool_calls": [
                        {"call_id": _CALL_ID, "name": "read", "arguments": {"path": "note.txt"}}
                    ],
                },
                {"content": "done"},
            ]
        ),
        encoding="utf-8",
    )
    cli_result = _runner.invoke(
        app,
        [
            "run",
            "hook parity",
            "--workspace",
            str(workspace),
            "--thread",
            "cli",
            "--scripted",
            str(scripted_file),
        ],
        env={**_CLEAN_ENV, "PRAXIS_HOOKS_CONFIG": str(hooks_config)},
    )
    assert cli_result.exit_code in (0, 2), (
        cli_result.exit_code,
        cli_result.output,
        repr(cli_result.exception),
    )
    cli_store = Store(workspace / ".praxis" / "praxis.sqlite")
    try:
        observations["cli"] = _observe_denial(cli_store.read_items("cli"))
    finally:
        cli_store.close()

    assert set(observations) == {"model", "resume", "cli"}
    assert len(set(observations.values())) == 1, observations
    assert observations["model"] == ("denied", f"hook:policy:blocked: {_FEEDBACK}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))