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
_CLEAN_ENV = {"PRAXIS_MODEL": None, "PRAXIS_BASE_URL": None, "PRAXIS_API_KEY": None, "OPENAI_API_KEY": None}


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


def test_run_missing_model_exit_3(workspace: Path) -> None:
    result = runner.invoke(app, ["run", "hi", "--workspace", str(workspace)], env=_CLEAN_ENV)
    assert result.exit_code == 3, result.output
    assert "PRAXIS_MODEL" in result.output


def test_run_missing_api_key_exit_3(workspace: Path) -> None:
    result = runner.invoke(
        app, ["run", "hi", "--workspace", str(workspace), "--model", "gpt-sim"], env=_CLEAN_ENV
    )
    assert result.exit_code == 3, result.output
    assert "PRAXIS_API_KEY" in result.output or "OPENAI_API_KEY" in result.output


def test_run_model_from_env(workspace: Path, tmp_path: Path) -> None:
    """PRAXIS_MODEL 有值但缺 key:错误文案必须指向凭据而非模型(证明 env 生效)。"""
    result = runner.invoke(
        app,
        ["run", "hi", "--workspace", str(workspace)],
        env={**_CLEAN_ENV, "PRAXIS_MODEL": "gpt-from-env"},
    )
    assert result.exit_code == 3, result.output
    assert "PRAXIS_MODEL" not in result.output
    assert "API_KEY" in result.output


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
