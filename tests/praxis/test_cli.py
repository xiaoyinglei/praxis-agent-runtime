"""Task 15:最小 CLI(设计 §5.12;findings C1/C7/C8 形态预防)+ R14 会话只读面。

CliRunner 直调 typer app;RULING R14 先测(Session.items/turns/verify + sdk re-export)。
契约上 cli 包只许 import sdk——测试文件不受此限制,可直接 import core/protocol/store
构造夹具(如 monkeypatch permission.decide 制造审批暂停)。
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from praxis.core import permission
from praxis.core.locks import thread_execution_lock
from praxis.interfaces import cli as cli_pkg
from praxis.interfaces import sdk as sdk_pkg
from praxis.interfaces.cli import app
from praxis.interfaces.cli import main as cli_main
from praxis.interfaces.cli.render import render_item
from praxis.interfaces.cli.slash import REGISTRY
from praxis.interfaces.sdk import (
    ConversationBusy,
    ModelResponse,
    ModelToolCall,
    ScriptedAdapter,
    TurnResult,
    TurnStatus,
    Usage,
    open_session,
)
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.permission import Decision

runner = CliRunner()

# 运行态环境变量隔离:None = click 测试隔离里"删除该变量"
_CLEAN_ENV = {
    "PRAXIS_MODEL_ID": None,
    "PRAXIS_MODEL": None,
    "PRAXIS_BASE_URL": None,
    "PRAXIS_API_KEY": None,
    "OPENAI_API_KEY": None,
}


@pytest.fixture(autouse=True)
def _never_touch_real_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    """所有 CLI 用例走 POSIX 文件分支:不读/写开发者真钥匙串。"""
    import praxis.adapters.config.credentials as credentials_module

    monkeypatch.setattr(credentials_module, "_PLATFORM", "linux")


def _env(config_dir: Path, **extra: str | None) -> dict[str, str | None]:
    return {**_CLEAN_ENV, "PRAXIS_CONFIG_DIR": str(config_dir), **extra}


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


def _write_script(tmp_path: Path, name: str, entries: list[dict[str, Any]]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _write_call_entry(content: str) -> dict[str, Any]:
    """剧本条目:一个 c1/write 工具调用(审批暂停与 allow 路径夹具)。"""
    return {
        "content": "",
        "tool_calls": [{"call_id": "c1", "name": "write", "arguments": {"path": "f.txt", "content": content}}],
    }


def _resp(content: str, calls: tuple[tuple[str, str, dict[str, Any]], ...] = ()) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tuple(ModelToolCall(call_id=c, name=n, arguments=a) for c, n, a in calls),
        usage=Usage(input_tokens=1, output_tokens=1),
        finish_reason="tool_calls" if calls else "stop",
    )


def _item(kind: str, payload: dict[str, Any], seq: int = 7) -> Item:
    return Item(
        item_id="item-x",
        thread_id="t",
        turn_id=None,
        seq=seq,
        kind=kind,
        payload=payload,
        created_at=0.0,
    )


# ---------------------------------------------------------------------------
# RULING R14:Session 只读面 + sdk re-export(先写、先红)
# ---------------------------------------------------------------------------


def test_r14_sdk_reexports_names() -> None:
    for name in ("Item", "ItemKind", "Turn", "VerifyReport"):
        assert name in sdk_pkg.__all__, f"sdk.__all__ missing {name}"
        assert hasattr(sdk_pkg, name), f"sdk missing attribute {name}"


def test_sdk_reexports_model_registry_names() -> None:
    """Task 6:CLI 的 models 子命令只经 sdk 触达注册表与凭据面。"""
    for name in (
        "ModelConfigStore",
        "ModelEntry",
        "ConnectionEntry",
        "CostMicros",
        "ConfigVersionConflict",
        "ModelSpec",
        "resolve_model_credential",
        "save_model_credential",
        "credential_source",
        "model_adapter_for",
        "validate_model_spec",
        "probe_model",
        "ProbeOutcome",
    ):
        assert name in sdk_pkg.__all__, f"sdk.__all__ missing {name}"
        assert hasattr(sdk_pkg, name), f"sdk missing attribute {name}"


def test_sdk_reexports_artifact_plan_and_event_surface() -> None:
    """Task 17:artifact/计划/流式事件面只经 sdk 对外暴露。"""
    for name in (
        "ArtifactStore",
        "PlanSnapshot",
        "latest_plan",
        "ModelEvent",
        "ModelTextDelta",
        "ModelReasoningDelta",
        "ModelToolCallDelta",
        "assemble_model_events",
    ):
        assert name in sdk_pkg.__all__, f"sdk.__all__ missing {name}"
        assert hasattr(sdk_pkg, name), f"sdk missing attribute {name}"


def test_default_tool_table_is_stable(workspace: Path) -> None:
    """Task 17:默认工具表的成员与顺序是稳定契约(tools 参数仍可整体替换)。"""
    session = open_session(workspace, ScriptedAdapter([_resp("ok")]))
    try:
        assert list(session._tools) == [  # noqa: SLF001 — 装配契约:顺序可断言
            "read",
            "write",
            "bash",
            "manage_command",
            "list_files",
            "search_text",
            "inspect_data_file",
            "apply_patch",
            "update_plan",
            "run_python",
            "read_artifact",
        ]
    finally:
        session.close()


def test_r14_session_items_and_turns(workspace: Path) -> None:
    session = open_session(workspace, ScriptedAdapter([_resp("hi there")]))
    try:
        result = asyncio.run(session.submit("hello"))
        assert result.status is TurnStatus.COMPLETED

        items = session.items()
        assert [i.kind for i in items][0] == ItemKind.USER_MESSAGE
        assert all(isinstance(i, Item) for i in items)
        if len(items) > 1:
            assert session.items(after_seq=items[0].seq)[0].seq == items[1].seq

        session.steer("mid-flight")
        assert session.items()[-1].kind == ItemKind.STEERING

        turns = session.turns()
        assert len(turns) == 1 and turns[0].turn_id == result.turn_id and turns[0].status == "completed"
    finally:
        session.close()


def test_r14_session_verify_ok_on_fresh_store(workspace: Path) -> None:
    session = open_session(workspace, ScriptedAdapter([]))
    try:
        report = session.verify()
        assert report.ok is True and report.problems == () and report.checked_items == 0
    finally:
        session.close()


# ---------------------------------------------------------------------------
# render.py:fail-open,永不抛(B1 读端最后一环)
# ---------------------------------------------------------------------------


def test_render_unknown_kind_never_raises() -> None:
    out = render_item(_item("alien", {"a": 1, "b": "z"}))
    assert "[alien]" in out
    assert "a" in out and "b" in out


@pytest.mark.parametrize("kind", sorted(ItemKind))
def test_render_known_kinds_empty_payload_never_raises(kind: str) -> None:
    assert isinstance(render_item(_item(kind, {})), str)


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        (ItemKind.USER_MESSAGE, {"text": 42}),
        (ItemKind.MODEL_RESPONSE, {"content": None, "tool_calls": "not-a-list"}),
        (ItemKind.TOOL_RESULT, {"content": None, "status": 9}),
        (ItemKind.TOOL_RESULT, {"content": [{"type": "text", "data": {"text": "hello result"}}], "status": "ok"}),
        (ItemKind.TURN_STATUS, {"to": 5}),
        (ItemKind.APPROVAL_REQUEST, {}),
        (ItemKind.COMPACTION_BOUNDARY, {"algorithm_version": "v2"}),
        (ItemKind.COMPACTION_BOUNDARY, {"algorithm_version": "v1", "covered_from": 1, "covered_to": 4}),
    ],
)
def test_render_malformed_payloads_never_raise(kind: str, payload: dict[str, Any]) -> None:
    assert isinstance(render_item(_item(kind, payload)), str)


def test_render_known_kinds_friendly_shapes() -> None:
    assert "user: hi" in render_item(_item(ItemKind.USER_MESSAGE, {"text": "hi"}))
    assert "steer: note" in render_item(_item(ItemKind.STEERING, {"text": "note"}))
    model = render_item(
        _item(
            ItemKind.MODEL_RESPONSE,
            {
                "content": "thinking",
                "tool_calls": ({"call_id": "c1", "name": "bash", "arguments": {"command": "ls"}},),
            },
        )
    )
    assert "thinking" in model and "bash" in model and "c1" in model
    call = render_item(
        _item(ItemKind.TOOL_CALL, {"call_id": "c1", "tool_name": "bash", "args_digest": "abc123def456789"})
    )
    assert "bash" in call and "abc123def456" in call
    result = render_item(
        _item(
            ItemKind.TOOL_RESULT,
            {
                "call_id": "c1",
                "tool_name": "bash",
                "status": "ok",
                "content": ({"type": "text", "data": {"text": "hello result"}},),
            },
        )
    )
    assert "hello result" in result and "ok" in result
    status = render_item(
        _item(ItemKind.TURN_STATUS, {"from": "running", "to": "completed", "reason": "model_finished"})
    )
    assert "running" in status and "completed" in status and "model_finished" in status
    request = render_item(
        _item(ItemKind.APPROVAL_REQUEST, {"call_id": "c1", "tool_name": "write", "args_digest": "d" * 64})
    )
    assert "approval" in request.lower() and "write" in request
    decision = render_item(_item(ItemKind.APPROVAL_DECISION, {"call_id": "c1", "verdict": "allow"}))
    assert "allow" in decision
    assert "digest text" in render_item(_item(ItemKind.SUMMARY, {"text": "digest text"}))


# ---------------------------------------------------------------------------
# slash.py:注册表驱动(C1 无 facade loop)
# ---------------------------------------------------------------------------


def test_slash_registry_shape() -> None:
    assert set(REGISTRY) == {"help", "status", "exit"}
    for name, command in REGISTRY.items():
        assert command.name == name
        assert command.help
        assert callable(command.run)


# ---------------------------------------------------------------------------
# run 命令:退出码 0/2/3 与脚本装配
# ---------------------------------------------------------------------------


def test_run_scripted_exit_0(workspace: Path, tmp_path: Path) -> None:
    script = _write_script(tmp_path, "done.json", [{"content": "hello world"}])
    result = runner.invoke(
        app,
        ["run", "hi", "--workspace", str(workspace), "--scripted", str(script)],
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 0, result.output
    assert "hello world" in result.output


def test_run_scripted_tool_allow_path(workspace: Path, tmp_path: Path) -> None:
    script = _write_script(tmp_path, "tool.json", [_write_call_entry("x"), {"content": "file written"}])
    result = runner.invoke(
        app,
        ["run", "write it", "--workspace", str(workspace), "--thread", "toolrun", "--scripted", str(script)],
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 0, result.output
    assert "file written" in result.output
    assert (workspace / "f.txt").read_text(encoding="utf-8") == "x"


def test_run_paused_exit_2(workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    entry = _write_call_entry("y")
    entry["content"] = "about to write"
    script = _write_script(tmp_path, "ask.json", [entry])
    result = runner.invoke(
        app,
        ["run", "write it", "--workspace", str(workspace), "--scripted", str(script)],
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 2, result.output
    assert "approval" in result.output.lower()
    assert "c1" in result.output
    assert "resume" in result.output.lower()  # 提示如何续跑
    assert not (workspace / "f.txt").exists()


def test_run_missing_model_exit_3(workspace: Path, tmp_path: Path) -> None:
    """Task 6:无注册模型且无逃生门 → 指向 praxis models add,不再猜 OPENAI_API_KEY。"""
    result = runner.invoke(
        app, ["run", "hi", "--workspace", str(workspace)], env=_env(tmp_path)
    )
    assert result.exit_code == 3, result.output
    assert "praxis models add" in result.output


def test_run_unregistered_model_exit_3(workspace: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["run", "hi", "--workspace", str(workspace), "--model", "gpt-sim"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 3, result.output
    assert "not registered" in result.output
    assert "Traceback" not in result.output


def test_run_model_id_escape_hatch_reports_missing_key(workspace: Path, tmp_path: Path) -> None:
    """PRAXIS_MODEL_ID 逃生门只构造 ModelSpec;缺 PRAXIS_API_KEY 时干净失败(退出码 1)。"""
    result = runner.invoke(
        app,
        ["run", "hi", "--workspace", str(workspace)],
        env={
            **_env(tmp_path),
            "PRAXIS_MODEL_ID": "gpt-from-env",
            "PRAXIS_BASE_URL": "https://escape.test/v1",
        },
    )
    assert result.exit_code == 1, result.output
    # M-1:缺凭据文案点名模型/provider 与下一步(逃生门没有 api_key_env)。
    assert "no credential for model 'gpt-from-env'" in result.output
    assert "praxis models add gpt-from-env --api-key-env" in result.output
    assert "Traceback" not in result.output


def test_run_bad_script_json_exit_3(workspace: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    result = runner.invoke(
        app, ["run", "hi", "--workspace", str(workspace), "--scripted", str(bad)], env=_CLEAN_ENV
    )
    assert result.exit_code == 3, result.output


def test_run_scripted_missing_entry_exit_3(workspace: Path, tmp_path: Path) -> None:
    bad = _write_script(tmp_path, "bad-entry.json", [{"tool_calls": []}])
    result = runner.invoke(
        app, ["run", "hi", "--workspace", str(workspace), "--scripted", str(bad)], env=_CLEAN_ENV
    )
    assert result.exit_code == 3, result.output


def test_scripted_entry_accepts_optional_reasoning() -> None:
    # R-T2-6:reasoning 可选,缺省空串;既有剧本 JSON 零改动。
    with_reasoning = cli_main._scripted_entry(0, {"content": "hi", "reasoning": "why"})
    assert with_reasoning.reasoning == "why"
    assert cli_main._scripted_entry(0, {"content": "hi"}).reasoning == ""


def test_usage_errors_exit_3(workspace: Path) -> None:
    """C8:click 原生用法错误(缺参数/未知命令/未知选项)统一退出码 3,非默认 2。"""
    missing_arg = runner.invoke(app, ["run", "--workspace", str(workspace)], env=_CLEAN_ENV)
    assert missing_arg.exit_code == 3, missing_arg.output
    unknown_cmd = runner.invoke(app, ["bogus-cmd"], env=_CLEAN_ENV)
    assert unknown_cmd.exit_code == 3, unknown_cmd.output
    unknown_opt = runner.invoke(app, ["run", "hi", "--nope"], env=_CLEAN_ENV)
    assert unknown_opt.exit_code == 3, unknown_opt.output


# ---------------------------------------------------------------------------
# resume 命令:审批续跑、确认守卫、参数校验
# ---------------------------------------------------------------------------


def _turn_id_from_output(output: str) -> str:
    match = re.search(r"turn ([0-9a-f]{32})", output)
    assert match is not None, output
    return match.group(1)


def test_resume_allow_then_complete(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    script1 = _write_script(tmp_path, "ask.json", [_write_call_entry("approved")])
    paused = runner.invoke(
        app,
        ["run", "write it", "--workspace", str(workspace), "--thread", "ar1", "--scripted", str(script1)],
        env=_CLEAN_ENV,
    )
    assert paused.exit_code == 2, paused.output
    turn_id = _turn_id_from_output(paused.output)

    script2 = _write_script(tmp_path, "final.json", [{"content": "all done"}])
    done = runner.invoke(
        app,
        [
            "resume",
            turn_id,
            "--workspace",
            str(workspace),
            "--thread",
            "ar1",
            "--allow",
            "c1",
            "--scripted",
            str(script2),
        ],
        env=_CLEAN_ENV,
    )
    assert done.exit_code == 0, done.output
    assert "all done" in done.output
    assert (workspace / "f.txt").read_text(encoding="utf-8") == "approved"

    # 终态 turn 不可再 resume → 参数/状态错误 = 3
    again = runner.invoke(
        app,
        [
            "resume",
            turn_id,
            "--workspace",
            str(workspace),
            "--thread",
            "ar1",
            "--allow",
            "c1",
            "--scripted",
            str(script2),
        ],
        env=_CLEAN_ENV,
    )
    assert again.exit_code == 3, again.output


def test_resume_paused_without_input_exit_3(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    script = _write_script(tmp_path, "ask.json", [_write_call_entry("z")])
    paused = runner.invoke(
        app, ["run", "go", "--workspace", str(workspace), "--thread", "ar3", "--scripted", str(script)], env=_CLEAN_ENV
    )
    assert paused.exit_code == 2, paused.output
    turn_id = _turn_id_from_output(paused.output)

    bare = runner.invoke(
        app,
        ["resume", turn_id, "--workspace", str(workspace), "--thread", "ar3", "--scripted", str(script)],
        env=_CLEAN_ENV,
    )
    assert bare.exit_code == 3, bare.output
    assert "approval or reconcile" in bare.output


def test_resume_bad_reconcile_format_exit_3(workspace: Path, tmp_path: Path) -> None:
    script = _write_script(tmp_path, "final.json", [{"content": "x"}])
    result = runner.invoke(
        app,
        ["resume", "abc", "--workspace", str(workspace), "--reconcile", "c1=maybe", "--scripted", str(script)],
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 3, result.output
    assert "reconcile" in result.output.lower()


def test_resume_unknown_turn_exit_3(workspace: Path, tmp_path: Path) -> None:
    script = _write_script(tmp_path, "final.json", [{"content": "x"}])
    result = runner.invoke(
        app, ["resume", "deadbeef", "--workspace", str(workspace), "--confirm", "--scripted", str(script)],
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 3, result.output


def test_resume_interrupted_confirm_flow(workspace: Path, tmp_path: Path) -> None:
    """锁被占 → run 报忙(退出码 1,留 running 孤儿);释放后 status sweep → interrupted。

    resume 无 --confirm → ResumeNotConfirmed → 3;加 --confirm → 续跑完成 → 0。
    """
    script1 = _write_script(tmp_path, "orphan.json", [{"content": "never runs"}])
    with thread_execution_lock(workspace, "orph"):
        busy = runner.invoke(
            app,
            ["run", "hi", "--workspace", str(workspace), "--thread", "orph", "--scripted", str(script1)],
            env=_CLEAN_ENV,
        )
    assert busy.exit_code == 1, busy.output
    assert "busy" in busy.output.lower()

    shown = runner.invoke(app, ["status", "--workspace", str(workspace), "--thread", "orph"], env=_CLEAN_ENV)
    assert shown.exit_code == 0, shown.output
    assert "interrupted" in shown.output
    match = re.search(r"([0-9a-f]{32})\s+interrupted", shown.output)
    assert match is not None, shown.output
    turn_id = match.group(1)

    script2 = _write_script(tmp_path, "resume.json", [{"content": "resumed ok"}])
    denied = runner.invoke(
        app,
        ["resume", turn_id, "--workspace", str(workspace), "--thread", "orph", "--scripted", str(script2)],
        env=_CLEAN_ENV,
    )
    assert denied.exit_code == 3, denied.output
    assert "confirm" in denied.output.lower()

    confirmed = runner.invoke(
        app,
        [
            "resume",
            turn_id,
            "--workspace",
            str(workspace),
            "--thread",
            "orph",
            "--confirm",
            "--scripted",
            str(script2),
        ],
        env=_CLEAN_ENV,
    )
    assert confirmed.exit_code == 0, confirmed.output
    assert "resumed ok" in confirmed.output


# ---------------------------------------------------------------------------
# chat 命令:C1 错误边界 / Ctrl+C / 审批 y/n / slash
# ---------------------------------------------------------------------------


class _FlakySession:
    """chat 错误边界夹具:submit 按剧本抛异常(不消耗 output)/返回结果,close 记录。"""

    def __init__(self, failures: list[BaseException | None], outputs: list[str]) -> None:
        self.thread_id = "flaky"
        self.failures = failures
        self.outputs = outputs
        self.closed = False
        self._attempt = 0
        self._succeeded = 0

    async def submit(self, text: str) -> TurnResult:
        failure = self.failures[self._attempt] if self._attempt < len(self.failures) else None
        self._attempt += 1
        if failure is not None:
            raise failure
        output = self.outputs[self._succeeded] if self._succeeded < len(self.outputs) else "done"
        self._succeeded += 1
        return TurnResult(
            turn_id=f"turn{self._succeeded}", status=TurnStatus.COMPLETED, output=output, reason="model_finished"
        )

    def close(self) -> None:
        self.closed = True


def _stub_chat_open(monkeypatch: pytest.MonkeyPatch, session: Any) -> None:
    monkeypatch.setattr(cli_main, "open_session", lambda *args, **kwargs: session)


def test_chat_error_boundary_survives(workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _write_script(tmp_path, "unused.json", [{"content": "x"}])
    session = _FlakySession([RuntimeError("boom"), None], ["back again"])  # 首个 submit 抛,次个成功
    _stub_chat_open(monkeypatch, session)
    result = runner.invoke(
        app,
        ["chat", "--workspace", str(workspace), "--scripted", str(script)],
        input="break it\ncarry on\n/exit\n",
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert "boom" in result.output
    assert "back again" in result.output
    assert session.closed


def test_chat_conversation_busy_continues(workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _write_script(tmp_path, "unused.json", [{"content": "x"}])
    session = _FlakySession([ConversationBusy("held"), None], ["later"])
    _stub_chat_open(monkeypatch, session)
    result = runner.invoke(
        app, ["chat", "--workspace", str(workspace), "--scripted", str(script)], input="a\nb\n/exit\n", env=_CLEAN_ENV
    )
    assert result.exit_code == 0, result.output
    assert "busy" in result.output.lower()
    assert "later" in result.output


def test_chat_keyboard_interrupt_during_turn(workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _write_script(tmp_path, "unused.json", [{"content": "x"}])
    session = _FlakySession([KeyboardInterrupt(), None], ["second fine"])
    _stub_chat_open(monkeypatch, session)
    result = runner.invoke(
        app,
        ["chat", "--workspace", str(workspace), "--scripted", str(script)],
        input="run\nagain\n/exit\n",
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 0, result.output
    assert "cancelled" in result.output.lower()
    assert "second fine" in result.output


def test_chat_ctrl_c_at_prompt_exits_0(workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _write_script(tmp_path, "done.json", [{"content": "fine"}])
    calls = {"n": 0}

    def fake_input(prompt: str = "") -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "hello"
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", fake_input)
    result = runner.invoke(app, ["chat", "--workspace", str(workspace), "--scripted", str(script)], env=_CLEAN_ENV)
    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert "fine" in result.output


def test_chat_slash_commands(workspace: Path, tmp_path: Path) -> None:
    script = _write_script(tmp_path, "unused.json", [{"content": "x"}])
    result = runner.invoke(
        app,
        ["chat", "--workspace", str(workspace), "--scripted", str(script)],
        input="/help\n/status\n/notathing\n/exit\n",
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 0, result.output
    assert "/help" in result.output
    assert "no turns" in result.output.lower()
    assert "unknown command" in result.output.lower()


def test_chat_bare_slash_never_kills_session(workspace: Path, tmp_path: Path) -> None:
    """C1 回归(Fix Round 2):裸 "/" 与 "/ " 必须走 unknown-command 提示并继续循环。

    审查发现:text[1:].split(maxsplit=1)[0] 对恰好 "/" 的输入抛 IndexError,
    逸出所有错误边界 → traceback → 会话终止。此测试钉死修复。
    """
    script = _write_script(tmp_path, "slash.json", [{"content": "normal reply"}])
    result = runner.invoke(
        app,
        ["chat", "--workspace", str(workspace), "--scripted", str(script)],
        input="/\n/ \n/nosuch\nhello\n/exit\n",
        env=_CLEAN_ENV,
    )
    assert result.exception is None, repr(result.exception)
    assert result.exit_code == 0, result.output
    assert "unknown command" in result.output.lower()
    assert "normal reply" in result.output  # 裸斜杠后普通 turn 照常执行
    assert "Traceback" not in result.output


def test_chat_paused_approve_y_n(workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    script = _write_script(tmp_path, "chat-ask.json", [_write_call_entry("chat-approved"), {"content": "chat done"}])
    result = runner.invoke(
        app,
        ["chat", "--workspace", str(workspace), "--scripted", str(script)],
        input="write it\ny\n/exit\n",
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 0, result.output
    assert "approval" in result.output.lower()
    assert "chat done" in result.output
    assert (workspace / "f.txt").read_text(encoding="utf-8") == "chat-approved"


# ---------------------------------------------------------------------------
# status / log / steer / verify 只读与旁路命令
# ---------------------------------------------------------------------------


def _run_scripted(workspace: Path, script: Path, thread: str) -> None:
    result = runner.invoke(
        app,
        ["run", "go", "--workspace", str(workspace), "--thread", thread, "--scripted", str(script)],
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 0, result.output


def test_status_table_and_detail(workspace: Path, tmp_path: Path) -> None:
    script = _write_script(tmp_path, "done.json", [{"content": "hello world"}])
    _run_scripted(workspace, script, "st1")

    table = runner.invoke(app, ["status", "--workspace", str(workspace), "--thread", "st1"], env=_CLEAN_ENV)
    assert table.exit_code == 0, table.output
    assert "completed" in table.output
    match = re.search(r"([0-9a-f]{32})\s+completed", table.output)
    assert match is not None, table.output
    turn_id = match.group(1)

    detail = runner.invoke(app, ["status", turn_id, "--workspace", str(workspace), "--thread", "st1"], env=_CLEAN_ENV)
    assert detail.exit_code == 0, detail.output
    assert "hello world" in detail.output

    missing = runner.invoke(
        app, ["status", "ffffffff", "--workspace", str(workspace), "--thread", "st1"], env=_CLEAN_ENV
    )
    assert missing.exit_code == 3, missing.output


def test_log_renders_items_and_after_seq(workspace: Path, tmp_path: Path) -> None:
    script = _write_script(tmp_path, "done.json", [{"content": "hello world"}])
    _run_scripted(workspace, script, "lg1")

    full = runner.invoke(app, ["log", "--workspace", str(workspace), "--thread", "lg1"], env=_CLEAN_ENV)
    assert full.exit_code == 0, full.output
    assert "user: go" in full.output
    assert "hello world" in full.output
    lines = [line for line in full.output.splitlines() if line.strip()]

    session = open_session(workspace, ScriptedAdapter([]), thread_id="lg1")
    try:
        last = session.items()[-1].seq
    finally:
        session.close()
    tail = runner.invoke(
        app, ["log", "--workspace", str(workspace), "--thread", "lg1", "--after-seq", str(last - 1)], env=_CLEAN_ENV
    )
    assert tail.exit_code == 0, tail.output
    assert len([line for line in tail.output.splitlines() if line.strip()]) < len(lines)


def test_steer_appends_and_log_shows(workspace: Path, tmp_path: Path) -> None:
    script = _write_script(tmp_path, "done.json", [{"content": "x"}])
    _run_scripted(workspace, script, "st2")

    steered = runner.invoke(
        app, ["steer", "note this", "--workspace", str(workspace), "--thread", "st2"], env=_CLEAN_ENV
    )
    assert steered.exit_code == 0, steered.output
    assert "ok" in steered.output.lower()

    logged = runner.invoke(app, ["log", "--workspace", str(workspace), "--thread", "st2"], env=_CLEAN_ENV)
    assert "note this" in logged.output


def test_verify_command_reports(workspace: Path, tmp_path: Path) -> None:
    script = _write_script(tmp_path, "done.json", [{"content": "x"}])
    _run_scripted(workspace, script, "vz1")

    clean = runner.invoke(app, ["verify", "--workspace", str(workspace)], env=_CLEAN_ENV)
    assert clean.exit_code == 0, clean.output
    assert "ok" in clean.output.lower()

    db = workspace / ".praxis" / "praxis.sqlite"
    conn = sqlite3.connect(db)
    try:
        max_seq = conn.execute("SELECT COALESCE(MAX(seq), 0) FROM items WHERE thread_id = 'vz1'").fetchone()[0]
        conn.execute(
            "INSERT INTO items(item_id, thread_id, turn_id, seq, kind, payload, created_at) "
            "VALUES ('gaprow', 'vz1', NULL, ?, 'user_message', '{\"text\": \"gap\"}', 0.0)",
            (max_seq + 2,),
        )
        conn.commit()
    finally:
        conn.close()

    flagged = runner.invoke(app, ["verify", "--workspace", str(workspace)], env=_CLEAN_ENV)
    assert flagged.exit_code == 1, flagged.output
    assert "seq gap" in flagged.output


# ---------------------------------------------------------------------------
# R15:稳定默认线程 "default"(跨命令可发现)+ run/resume 打印 turn id
# ---------------------------------------------------------------------------


def test_default_thread_persists_across_invocations(workspace: Path, tmp_path: Path) -> None:
    """run 不传 --thread → 会话落 "default";log/status 不传 --thread 必须找到它。"""
    script = _write_script(tmp_path, "done.json", [{"content": "hello default"}])
    run_res = runner.invoke(
        app, ["run", "hi", "--workspace", str(workspace), "--scripted", str(script)], env=_CLEAN_ENV
    )
    assert run_res.exit_code == 0, run_res.output
    assert "hello default" in run_res.output

    logged = runner.invoke(app, ["log", "--workspace", str(workspace)], env=_CLEAN_ENV)
    assert logged.exit_code == 0, logged.output
    assert "user: hi" in logged.output
    assert "hello default" in logged.output

    table = runner.invoke(app, ["status", "--workspace", str(workspace)], env=_CLEAN_ENV)
    assert table.exit_code == 0, table.output
    assert "completed" in table.output


def test_run_and_resume_print_copyable_turn_id(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run/resume 输出 "turn: <32hex> (thread: <id>)" 行,供复制到 praxis resume。"""
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    script1 = _write_script(tmp_path, "ask.json", [_write_call_entry("r15")])
    paused = runner.invoke(
        app,
        ["run", "write it", "--workspace", str(workspace), "--thread", "r15t", "--scripted", str(script1)],
        env=_CLEAN_ENV,
    )
    assert paused.exit_code == 2, paused.output
    id_match = re.search(r"turn: ([0-9a-f]{32}) \(thread: r15t\)", paused.output)
    assert id_match is not None, paused.output
    turn_id = id_match.group(1)

    script2 = _write_script(tmp_path, "final.json", [{"content": "r15 done"}])
    done = runner.invoke(
        app,
        [
            "resume",
            turn_id,
            "--workspace",
            str(workspace),
            "--thread",
            "r15t",
            "--allow",
            "c1",
            "--scripted",
            str(script2),
        ],
        env=_CLEAN_ENV,
    )
    assert done.exit_code == 0, done.output
    assert re.search(rf"turn: {turn_id} \(thread: r15t\)", done.output)
    assert "r15 done" in done.output


def test_thread_override_still_isolates(workspace: Path, tmp_path: Path) -> None:
    """--thread 覆盖仍生效:run --thread alpha 后,log --thread beta 为空、alpha 有内容。"""
    script = _write_script(tmp_path, "done.json", [{"content": "isolated"}])
    run_res = runner.invoke(
        app,
        ["run", "hi", "--workspace", str(workspace), "--thread", "alpha", "--scripted", str(script)],
        env=_CLEAN_ENV,
    )
    assert run_res.exit_code == 0, run_res.output

    other = runner.invoke(app, ["log", "--workspace", str(workspace), "--thread", "beta"], env=_CLEAN_ENV)
    assert other.exit_code == 0, other.output
    assert "(no items)" in other.output

    same = runner.invoke(app, ["log", "--workspace", str(workspace), "--thread", "alpha"], env=_CLEAN_ENV)
    assert "isolated" in same.output


# ---------------------------------------------------------------------------
# 入口点自检:pyproject 的 console script 目标存在且可调用
# ---------------------------------------------------------------------------


def test_console_entry_object() -> None:
    """pyproject 入口 praxis = praxis.interfaces.cli:app:包必须直接暴露 app。"""
    assert cli_pkg.app is app
    assert callable(app)


# ---------------------------------------------------------------------------
# Task 8: --policy / --mode(策略入口,C5 正交;坏输入 → 退出码 3)
# ---------------------------------------------------------------------------


def _write_policy(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _bash_call_entry() -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [{"call_id": "c1", "name": "bash", "arguments": {"command": "echo hi"}}],
    }


_POLICY_BASH_DENY = json.dumps({"tools": [{"pattern": "bash", "decision": "deny"}]})


def test_cli_policy_denies_bash_visible_in_log(workspace: Path, tmp_path: Path) -> None:
    policy = _write_policy(tmp_path, "pol.json", _POLICY_BASH_DENY)
    script = _write_script(tmp_path, "bash.json", [_bash_call_entry(), {"content": "done"}])
    run_res = runner.invoke(
        app,
        [
            "run", "run bash", "--workspace", str(workspace), "--thread", "pol1",
            "--scripted", str(script), "--policy", str(policy),
        ],
        env=_CLEAN_ENV,
    )
    assert run_res.exit_code == 0, run_res.output
    logged = runner.invoke(app, ["log", "--workspace", str(workspace), "--thread", "pol1"], env=_CLEAN_ENV)
    assert "[denied]" in logged.output  # 规则 deny 落到 tool_result,经日志可查


def test_cli_mode_never_denies_bash_default_ask(workspace: Path, tmp_path: Path) -> None:
    """默认策略 bash 是 ask(命令网络);--mode never 把 ask 统一转 deny(R7 代数)。"""
    script = _write_script(tmp_path, "bash.json", [_bash_call_entry(), {"content": "done"}])
    run_res = runner.invoke(
        app,
        ["run", "run bash", "--workspace", str(workspace), "--thread", "mode1", "--scripted", str(script),
         "--mode", "never"],
        env=_CLEAN_ENV,
    )
    assert run_res.exit_code == 0, run_res.output
    logged = runner.invoke(app, ["log", "--workspace", str(workspace), "--thread", "mode1"], env=_CLEAN_ENV)
    assert "[denied]" in logged.output
    assert not (workspace / ".praxis" / "permissions.json").exists()  # mode 永不落盘(C5)


def test_cli_policy_bad_json_exit_3(workspace: Path, tmp_path: Path) -> None:
    bad = _write_policy(tmp_path, "bad.json", "{ not json")
    script = _write_script(tmp_path, "done.json", [{"content": "x"}])
    result = runner.invoke(
        app,
        ["run", "hi", "--workspace", str(workspace), "--scripted", str(script), "--policy", str(bad)],
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 3, result.output
    assert "policy" in result.output.lower()


@pytest.mark.parametrize("bad_text", ['{"nope": 1}', '{"profile": "turbo"}', "[]"])
def test_cli_policy_invalid_shape_exit_3(workspace: Path, tmp_path: Path, bad_text: str) -> None:
    bad = _write_policy(tmp_path, "bad.json", bad_text)
    script = _write_script(tmp_path, "done.json", [{"content": "x"}])
    result = runner.invoke(
        app,
        ["run", "hi", "--workspace", str(workspace), "--scripted", str(script), "--policy", str(bad)],
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 3, result.output


def test_cli_policy_missing_file_exit_3(workspace: Path, tmp_path: Path) -> None:
    script = _write_script(tmp_path, "done.json", [{"content": "x"}])
    result = runner.invoke(
        app,
        [
            "run", "hi", "--workspace", str(workspace), "--scripted", str(script),
            "--policy", str(tmp_path / "absent.json"),
        ],
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 3, result.output


def test_cli_mode_bad_choice_exit_3(workspace: Path, tmp_path: Path) -> None:
    script = _write_script(tmp_path, "done.json", [{"content": "x"}])
    result = runner.invoke(
        app,
        ["run", "hi", "--workspace", str(workspace), "--scripted", str(script), "--mode", "turbo"],
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 3, result.output


def test_cli_policy_and_mode_forwarded_orthogonally(workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C5 透传契约:--policy 解析为 PermissionPolicy、--mode 原样传 open_session,互不折叠。"""
    from praxis.interfaces.sdk import PermissionPolicy

    captured: dict[str, Any] = {}

    def fake_open(root: Any, adapter: Any, **kwargs: Any) -> Any:
        captured["workspace"] = root
        captured.update(kwargs)
        return _FlakySession([], ["ok"])

    monkeypatch.setattr(cli_main, "open_session", fake_open)
    policy = _write_policy(
        tmp_path, "pol.json", json.dumps({"approval": "never", "tools": [{"pattern": "bash", "decision": "deny"}]})
    )
    script = _write_script(tmp_path, "done.json", [{"content": "x"}])
    result = runner.invoke(
        app,
        [
            "run", "hi", "--workspace", str(workspace), "--scripted", str(script),
            "--policy", str(policy), "--mode", "bypass",
        ],
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 0, result.output
    assert isinstance(captured["policy"], PermissionPolicy)
    assert captured["policy"].approval == "never"  # mode 不进 policy(正交)
    assert captured["mode"] == "bypass"
    assert captured["thread_id"] == "default"


# ---------------------------------------------------------------------------
# Task 9 RULING R12:损坏的既有 .praxis/permissions.json → 干净错误 + 退出码 1
# (状态损坏≈运行时失败,非用法错误;--policy 坏参仍 exit 3;C4 形态无裸 traceback)
# ---------------------------------------------------------------------------


def _corrupt_existing_policy(workspace: Path) -> None:
    praxis_dir = workspace / ".praxis"
    praxis_dir.mkdir(parents=True, exist_ok=True)
    (praxis_dir / "permissions.json").write_text("{ this is not json", encoding="utf-8")


def test_cli_corrupt_existing_policy_run_exit_1_clean(workspace: Path, tmp_path: Path) -> None:
    """R12(a):既有策略文件损坏 → run 退出 1、干净消息、无 ValueError 裸 traceback。"""
    _corrupt_existing_policy(workspace)
    script = _write_script(tmp_path, "done.json", [{"content": "x"}])
    result = runner.invoke(
        app,
        ["run", "hi", "--workspace", str(workspace), "--scripted", str(script)],
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 1, result.output
    assert not isinstance(result.exception, ValueError), repr(result.exception)
    assert "Traceback" not in result.output
    assert "error" in result.output.lower()
    assert "permissions.json" in result.output  # 消息指向损坏文件


def test_cli_corrupt_existing_policy_readonly_commands_exit_1_clean(workspace: Path) -> None:
    """R12:只读命令(log/status)同样经 open_session → 损坏状态干净退出 1,非 traceback。"""
    _corrupt_existing_policy(workspace)
    for command in (["log"], ["status"]):
        result = runner.invoke(app, [*command, "--workspace", str(workspace)], env=_CLEAN_ENV)
        assert result.exit_code == 1, (command, result.output)
        assert not isinstance(result.exception, ValueError), repr(result.exception)
        assert "Traceback" not in result.output
        assert "error" in result.output.lower()


def test_cli_corrupt_existing_policy_overridden_by_valid_policy_flag(workspace: Path, tmp_path: Path) -> None:
    """R12:显式 --policy 覆盖损坏的既有文件(来源序:显式 > 文件)→ 正常退出 0。"""
    _corrupt_existing_policy(workspace)
    policy = _write_policy(tmp_path, "good.json", json.dumps({"profile": "workspace"}))
    script = _write_script(tmp_path, "done.json", [{"content": "x"}])
    result = runner.invoke(
        app,
        ["run", "hi", "--workspace", str(workspace), "--scripted", str(script), "--policy", str(policy)],
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 0, result.output


def test_cli_valid_existing_policy_file_runs_normal(workspace: Path, tmp_path: Path) -> None:
    """R12(c):既有策略文件合法 → 正常加载并执行(退出 0),无误伤。"""
    praxis_dir = workspace / ".praxis"
    praxis_dir.mkdir(parents=True, exist_ok=True)
    (praxis_dir / "permissions.json").write_text(json.dumps({"profile": "workspace"}), encoding="utf-8")
    script = _write_script(tmp_path, "done.json", [{"content": "all good"}])
    result = runner.invoke(
        app,
        ["run", "hi", "--workspace", str(workspace), "--scripted", str(script)],
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 0, result.output
    assert "all good" in result.output


def test_cli_policy_bad_json_still_exit_3_not_1(workspace: Path, tmp_path: Path) -> None:
    """R12(b):--policy 坏 JSON 是用户用法错误 → 仍 exit 3(与既有状态损坏的 exit 1 区分)。"""
    bad = _write_policy(tmp_path, "bad.json", "{ not json")
    script = _write_script(tmp_path, "done.json", [{"content": "x"}])
    result = runner.invoke(
        app,
        ["run", "hi", "--workspace", str(workspace), "--scripted", str(script), "--policy", str(bad)],
        env=_CLEAN_ENV,
    )
    assert result.exit_code == 3, result.output


# ---------------------------------------------------------------------------
# Task 6:praxis models(全局模型库 + 凭据卫生)+ composition 单一构造路径
# ---------------------------------------------------------------------------


def _invoke_models(args: list[str], env: dict[str, str | None]) -> Any:
    return runner.invoke(app, ["models", *args], env=env)


def _add_openai(tmp_path: Path, **extra: str | None) -> Any:
    env = _env(tmp_path, REGISTRY_KEY="sk-test")
    env.update(extra)
    return _invoke_models(
        [
            "add",
            "ds",
            "--provider",
            "openai-compatible",
            "--base-url",
            "https://api.deepseek.com/v1",
            "--api-key-env",
            "REGISTRY_KEY",
            "--remote-model-id",
            "deepseek-chat",
            "--price-in-per-1m",
            "0.27",
            "--price-out-per-1m",
            "1.1",
        ],
        env,
    )


def test_models_price_conversion_is_dollars_per_1m() -> None:
    from praxis.interfaces.cli.models_cli import dollars_per_1m_to_micros

    assert dollars_per_1m_to_micros("0.27") == 270_000
    assert dollars_per_1m_to_micros(1.1) == 1_100_000
    assert dollars_per_1m_to_micros("0") == 0


def test_models_add_list_show_roundtrip(tmp_path: Path) -> None:
    added = _add_openai(tmp_path)
    assert added.exit_code == 0, added.output
    assert "ok: registered ds" in added.output

    listed = _invoke_models(["list"], _env(tmp_path))
    assert listed.exit_code == 0, listed.output
    assert "ds\topenai-compatible" in listed.output

    shown = _invoke_models(["show", "ds"], _env(tmp_path))
    assert shown.exit_code == 0, shown.output
    assert "remote_model_id: deepseek-chat" in shown.output
    assert "api_key_env: REGISTRY_KEY" in shown.output
    assert "input=0.27" in shown.output
    assert "sk-test" not in shown.output  # 凭据值绝不显示

    defaulted = _invoke_models(["set-default", "ds"], _env(tmp_path))
    assert defaulted.exit_code == 0, defaulted.output
    marked = _invoke_models(["list"], _env(tmp_path))
    assert "* ds" in marked.output


def _sse(text: str) -> bytes:
    def chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
        return (
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "m",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
            )
            + "\n\n"
        ).encode()

    return b"".join(
        [
            chunk({"role": "assistant", "content": text}),
            chunk({}, finish="stop"),
            b"data: [DONE]\n\n",
        ]
    )


def test_models_add_login_check_end_to_end_without_preexisting_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-T6-6:add(无凭据) → login(0600 文件) → check(stub transport) 三步全 0。"""
    env = _env(tmp_path)
    added = _invoke_models(
        [
            "add",
            "ds",
            "--provider",
            "openai-compatible",
            "--base-url",
            "https://api.deepseek.com/v1",
            "--api-key-env",
            "REGISTRY_KEY",
        ],
        env,
    )
    assert added.exit_code == 0, added.output

    secret = "sk-e2e-secret"
    logged = runner.invoke(
        app,
        ["models", "login", "ds"],
        input=secret + "\n",
        env=_env(tmp_path, REGISTRY_KEY=None),
    )
    assert logged.exit_code == 0, logged.output

    import httpx

    import praxis.interfaces.cli.models_cli as models_cli
    from praxis.interfaces.sdk import OpenAICompatAdapter

    captured: dict[str, Any] = {}

    def fake_adapter(spec: Any, api_key: Any = None, **kwargs: Any) -> Any:
        captured["api_key"] = api_key
        return OpenAICompatAdapter(
            spec,
            api_key="stub",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        200, content=_sse("ok"), headers={"content-type": "text/event-stream"}
                    )
                )
            ),
        )

    monkeypatch.setattr(models_cli, "model_adapter_for", fake_adapter)
    checked = _invoke_models(["check", "ds"], _env(tmp_path, REGISTRY_KEY=None))
    assert checked.exit_code == 0, checked.output
    assert "ok: ds" in checked.output
    # login 写入的 0600 文件确实被 check 解析到(而非 env)。
    assert captured["api_key"] == secret


def test_models_add_ollama_without_credential_is_normal(tmp_path: Path) -> None:
    result = _invoke_models(
        [
            "add",
            "local",
            "--provider",
            "ollama",
            "--base-url",
            "http://127.0.0.1:11434",
            "--remote-model-id",
            "qwen2.5:7b",
        ],
        _env(tmp_path),
    )
    assert result.exit_code == 0, result.output
    assert "ok: registered local" in result.output


def test_models_add_rejects_public_ollama_base_url(tmp_path: Path) -> None:
    result = _invoke_models(
        [
            "add",
            "remote-ollama",
            "--provider",
            "ollama",
            "--base-url",
            "https://ollama.example.com",
        ],
        _env(tmp_path),
    )
    assert result.exit_code == 3, result.output
    assert "loopback or private" in result.output


def test_models_add_rejects_unknown_ollama_provider_option(tmp_path: Path) -> None:
    result = _invoke_models(
        [
            "add",
            "local",
            "--provider",
            "ollama",
            "--base-url",
            "http://127.0.0.1:11434",
            "--option",
            "mystery=1",
        ],
        _env(tmp_path),
    )
    assert result.exit_code == 3, result.output
    assert "unsupported ollama provider option: mystery" in result.output


def test_models_add_accepts_extended_ollama_options(tmp_path: Path) -> None:
    result = _invoke_models(
        [
            "add",
            "local",
            "--provider",
            "ollama",
            "--base-url",
            "http://127.0.0.1:11434",
            "--option",
            "num_ctx=8192",
            "--option",
            "repeat_penalty=1.1",
            "--option",
            "format=json",
        ],
        _env(tmp_path),
    )
    assert result.exit_code == 0, result.output


def test_models_add_rejects_invalid_context_window(tmp_path: Path) -> None:
    result = _invoke_models(
        [
            "add",
            "local",
            "--provider",
            "ollama",
            "--base-url",
            "http://127.0.0.1:11434",
            "--context-window",
            "1024",
            "--max-output",
            "4096",
        ],
        _env(tmp_path),
    )
    assert result.exit_code == 3, result.output


def _add_model(
    tmp_path: Path,
    model_id: str,
    base_url: str,
    *,
    extra: list[str] | None = None,
    env: dict[str, str | None] | None = None,
) -> Any:
    return _invoke_models(
        [
            "add",
            model_id,
            "--provider",
            "openai-compatible",
            "--base-url",
            base_url,
            "--api-key-env",
            "REGISTRY_KEY",
            *(extra or []),
        ],
        env if env is not None else _env(tmp_path),
    )


def test_models_add_same_provider_two_endpoints_are_isolated(tmp_path: Path) -> None:
    """R-T6-7:默认连接名是 model id,同 provider 两个端点互不接管。"""
    first = _add_model(tmp_path, "vendor-a", "https://a.test/v1")
    second = _add_model(tmp_path, "vendor-b", "https://b.test/v1")
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output

    shown_a = _invoke_models(["show", "vendor-a"], _env(tmp_path))
    shown_b = _invoke_models(["show", "vendor-b"], _env(tmp_path))
    assert "base_url: https://a.test/v1" in shown_a.output
    assert "base_url: https://b.test/v1" in shown_b.output


def test_models_add_rejects_mismatched_connection_reuse(tmp_path: Path) -> None:
    """R-T6-7:--connection 指向既有连接时,不一致必须拒绝且连接行字节不变。"""
    created = _invoke_models(
        [
            "connection",
            "add",
            "shared",
            "--provider",
            "openai-compatible",
            "--base-url",
            "https://a.test/v1",
            "--api-key-env",
            "REGISTRY_KEY",
        ],
        _env(tmp_path),
    )
    assert created.exit_code == 0, created.output

    from praxis.adapters.config.store import ModelConfigStore

    before = ModelConfigStore(tmp_path / "config.sqlite")
    try:
        original = before.get_connection("shared")
    finally:
        before.close()
    assert original is not None

    rejected = _add_model(
        tmp_path,
        "shared-model",
        "https://b.test/v1",
        extra=["--connection", "shared"],
    )
    assert rejected.exit_code == 3, rejected.output
    assert "already exists with different" in rejected.output

    after = ModelConfigStore(tmp_path / "config.sqlite")
    try:
        unchanged = after.get_connection("shared")
    finally:
        after.close()
    assert unchanged == original

    # 完全一致的复用被接受,且不要求重写连接。
    reused = _add_model(
        tmp_path,
        "shared-model",
        "https://a.test/v1",
        extra=["--connection", "shared"],
    )
    assert reused.exit_code == 0, reused.output


def test_models_add_replace_connection_requires_connection(tmp_path: Path) -> None:
    result = _invoke_models(
        [
            "add",
            "m",
            "--provider",
            "ollama",
            "--base-url",
            "http://127.0.0.1:11434",
            "--replace-connection",
        ],
        _env(tmp_path),
    )
    assert result.exit_code == 3, result.output
    assert "requires --connection" in result.output


def test_models_list_sqlite_error_is_clean_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-T6-9:sqlite3.Error 不得裸 traceback。"""
    from praxis.adapters.config.store import ModelConfigStore

    def broken(self: Any) -> Any:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(ModelConfigStore, "list_models", broken)
    result = _invoke_models(["list"], _env(tmp_path))
    assert result.exit_code == 1, result.output
    assert "disk I/O error" in result.output
    assert "Traceback" not in result.output


def test_models_login_oserror_is_clean_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-T6-9:凭据写入的 OSError 归成干净失败。"""
    assert _add_openai(tmp_path).exit_code == 0
    import praxis.interfaces.cli.models_cli as models_cli

    def denied(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("credential store is not writable")

    monkeypatch.setattr(models_cli, "save_model_credential", denied)
    result = runner.invoke(
        app, ["models", "login", "ds"], input="sk-test\n", env=_env(tmp_path, REGISTRY_KEY=None)
    )
    assert result.exit_code == 1, result.output
    assert "not writable" in result.output
    assert "Traceback" not in result.output


def test_models_remove_unknown_is_clean_exit_1(tmp_path: Path) -> None:
    result = _invoke_models(["remove", "nope"], _env(tmp_path))
    assert result.exit_code == 1, result.output
    assert "not registered" in result.output
    assert "Traceback" not in result.output


def test_models_remove_cas_conflict_is_clean_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _add_openai(tmp_path).exit_code == 0
    from praxis.adapters.config.store import ConfigVersionConflict, ModelConfigStore

    def conflict(self: Any, model_id: str, *, expected_revision: int) -> None:
        raise ConfigVersionConflict(observed_revision=7)

    monkeypatch.setattr(ModelConfigStore, "remove_model", conflict)
    result = _invoke_models(["remove", "ds"], _env(tmp_path))
    assert result.exit_code == 1, result.output
    assert "revision conflict" in result.output
    assert "Traceback" not in result.output


def test_models_connection_add_and_list(tmp_path: Path) -> None:
    added = _invoke_models(
        [
            "connection",
            "add",
            "local-ollama",
            "--provider",
            "ollama",
            "--base-url",
            "http://127.0.0.1:11434",
            "--location",
            "local",
        ],
        _env(tmp_path),
    )
    assert added.exit_code == 0, added.output
    listed = _invoke_models(["connection", "list"], _env(tmp_path))
    assert listed.exit_code == 0, listed.output
    assert "local-ollama\tollama\thttp://127.0.0.1:11434" in listed.output


def test_models_login_keeps_secret_out_of_sqlite_and_output(tmp_path: Path) -> None:
    """硬测试①:login 之后全表扫描 config.sqlite,任何字段都不得含密钥值。"""
    secret = "sk-super-secret-value-123"
    assert _add_openai(tmp_path, REGISTRY_KEY=secret).exit_code == 0

    # login 时环境变量不存在 → 必须由隐藏 prompt 提供并写入 0600 文件。
    logged = runner.invoke(
        app,
        ["models", "login", "ds"],
        input=secret + "\n",
        env=_env(tmp_path, REGISTRY_KEY=None),
    )
    assert logged.exit_code == 0, logged.output
    assert secret not in logged.output

    credential_file = tmp_path / "credentials" / "models.json"
    assert credential_file.exists()
    assert credential_file.stat().st_mode & 0o777 == 0o600
    assert secret in credential_file.read_text(encoding="utf-8")  # 确实保存了

    connection = sqlite3.connect(tmp_path / "config.sqlite")
    try:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        ]
        assert tables
        scanned = 0
        for table in tables:
            for row in connection.execute(f'SELECT * FROM "{table}"').fetchall():
                scanned += 1
                for value in row:
                    assert secret not in str(value)
        assert scanned > 0
    finally:
        connection.close()


def test_models_login_requires_api_key_env(tmp_path: Path) -> None:
    assert (
        _invoke_models(
            ["add", "local", "--provider", "ollama", "--base-url", "http://127.0.0.1:11434"],
            _env(tmp_path),
        ).exit_code
        == 0
    )
    result = _invoke_models(["login", "local"], _env(tmp_path))
    assert result.exit_code == 1, result.output
    assert "api_key_env" in result.output


def test_models_check_missing_credential_names_env_and_next_step(tmp_path: Path) -> None:
    """M-1:缺凭据报错必须点名模型/provider、api_key_env 与两条可执行出路。"""
    assert _add_openai(tmp_path).exit_code == 0
    result = _invoke_models(["check", "ds"], _env(tmp_path, REGISTRY_KEY=None))
    assert result.exit_code == 1, result.output
    output = result.output
    assert "ds" in output
    assert "openai-compatible" in output
    assert "REGISTRY_KEY" in output
    assert "export REGISTRY_KEY=" in output
    assert "praxis models login ds" in output
    assert "sk-test" not in output  # 凭据值绝不出现


def test_models_check_missing_api_key_env_points_at_models_add(tmp_path: Path) -> None:
    """M-1:api_key_env is None 时如实说明,并指向 models add --api-key-env。"""
    added = _invoke_models(
        [
            "add",
            "glm-5.3",
            "--provider",
            "openai-compatible",
            "--base-url",
            "https://unit.test/v1",
        ],
        _env(tmp_path),
    )
    assert added.exit_code == 0, added.output
    result = _invoke_models(["check", "glm-5.3"], _env(tmp_path))
    assert result.exit_code == 1, result.output
    output = result.output
    assert "glm-5.3" in output
    assert "openai-compatible" in output
    assert "no api_key_env configured" in output
    assert "praxis models add glm-5.3 --api-key-env" in output


def test_models_check_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _add_openai(tmp_path).exit_code == 0
    import praxis.interfaces.cli.models_cli as models_cli
    from praxis.interfaces.sdk import ProbeOutcome

    async def fake_ok(
        adapter: Any,
        *,
        tool_name: str = "__probe_noop",
        credential_source: Any = None,
    ) -> ProbeOutcome:
        return ProbeOutcome(ok=True, steps=("text",), detail=None, usage=None)

    monkeypatch.setattr(models_cli, "probe_model", fake_ok)
    passed = _invoke_models(["check", "ds"], _env(tmp_path, REGISTRY_KEY="sk-test"))
    assert passed.exit_code == 0, passed.output
    assert "ok: ds" in passed.output

    async def fake_fail(
        adapter: Any,
        *,
        tool_name: str = "__probe_noop",
        credential_source: Any = None,
    ) -> ProbeOutcome:
        return ProbeOutcome(
            ok=False,
            steps=("text",),
            detail="text step failed: boom (authentication)",
            usage=None,
        )

    monkeypatch.setattr(models_cli, "probe_model", fake_fail)
    failed = _invoke_models(["check", "ds"], _env(tmp_path, REGISTRY_KEY="sk-test"))
    assert failed.exit_code == 1, failed.output
    assert "text step failed: boom" in failed.output


def test_cli_run_forwards_model_and_no_adapter(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-T6-2:main.py 不再构造 adapter;模型 id 原样交给 composition。"""
    captured: dict[str, Any] = {}

    def fake_open(root: Any, adapter: Any, **kwargs: Any) -> Any:
        captured["adapter"] = adapter
        captured.update(kwargs)
        return _FlakySession([], ["forwarded"])

    monkeypatch.setattr(cli_main, "open_session", fake_open)
    result = runner.invoke(
        app,
        ["run", "hi", "--workspace", str(workspace), "--model", "ds"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0, result.output
    assert captured["adapter"] is None
    assert captured["model"] == "ds"
    assert "forwarded" in result.output


def test_build_session_resolves_registered_model_spec(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import praxis.composition as composition
    from praxis.adapters.config.store import ConnectionEntry, ModelConfigStore, ModelEntry
    from praxis.protocol.models import ModelSpec

    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path))
    store = ModelConfigStore()
    try:
        connection = ConnectionEntry(
            "c", "openai-compatible", "https://unit.test/v1", "UNIT_KEY"
        )
        store.upsert_connection(connection)
        store.upsert_model(ModelEntry(ModelSpec(model_id="m"), connection, 0))
        monkeypatch.setenv("UNIT_KEY", "unit-secret")

        captured: dict[str, Any] = {}

        def fake_adapter(spec: Any, api_key: Any = None, *, http_client: Any = None) -> Any:
            captured["spec"] = spec
            captured["api_key"] = api_key
            return ScriptedAdapter([])

        monkeypatch.setattr(composition, "_adapter_for", fake_adapter)
        session = composition.build_session(workspace, model="m", config_store=store)
        session.close()
    finally:
        store.close()

    assert captured["spec"].model_id == "m"
    assert captured["spec"].base_url == "https://unit.test/v1"
    assert captured["api_key"] == "unit-secret"


def test_build_session_escape_hatch_only_builds_a_spec(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import praxis.composition as composition
    from praxis.adapters.config.store import ModelConfigStore

    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("PRAXIS_MODEL_ID", "temporary-model")
    monkeypatch.setenv("PRAXIS_BASE_URL", "https://temporary.test/v1")
    monkeypatch.setenv("PRAXIS_API_KEY", "temporary-secret")
    store = ModelConfigStore()
    try:
        captured: dict[str, Any] = {}

        def fake_adapter(spec: Any, api_key: Any = None, *, http_client: Any = None) -> Any:
            captured["spec"] = spec
            captured["api_key"] = api_key
            return ScriptedAdapter([])

        monkeypatch.setattr(composition, "_adapter_for", fake_adapter)
        session = composition.build_session(workspace, config_store=store)
        session.close()
    finally:
        store.close()

    assert captured["spec"].model_id == "temporary-model"
    assert captured["spec"].base_url == "https://temporary.test/v1"
    assert captured["api_key"] == "temporary-secret"


def test_build_session_without_registry_is_actionable(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import praxis.composition as composition
    from praxis.adapters.config.store import ModelConfigStore

    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path))
    store = ModelConfigStore()
    try:
        with pytest.raises(KeyError, match="praxis models add"):
            composition.build_session(workspace, config_store=store)
    finally:
        store.close()
