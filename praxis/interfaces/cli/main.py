"""Praxis 最小 CLI(设计 §5.12;findings C1/C7/C8 形态预防)。

硬契约(import-linter praxis_cli_via_sdk):本包只 import praxis.interfaces.sdk,
core/store 一律经 sdk re-export(R14)触达,装配一律 sdk.open_session。

退出码统一(C7/C8):``0 完成 / 1 失败或取消 / 2 暂停待输入 / 3 用法错误``。
click 原生 UsageError 默认退出码 2,由 _UsageExit3Group 在参数解析与子命令分派
两个入口拦截重映射为 3——console 入口与 typer.testing.CliRunner 共用
``get_command(app)`` 构造的同一个 Group 类,两条路径行为一致。

chat(C1):每 turn 一个错误边界,任何 Exception 显示后回到输入态,永不退出会话。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

import click
import typer
from typer.core import TyperGroup

from praxis.interfaces.cli.render import render_item
from praxis.interfaces.cli.slash import REGISTRY, ExitChat
from praxis.interfaces.sdk import (
    ConversationBusy,
    ModelResponse,
    ModelSpec,
    ModelToolCall,
    OpenAICompatAdapter,
    PermissionMode,
    PermissionPolicy,
    ResumeNotConfirmed,
    ScriptedAdapter,
    Session,
    TurnResult,
    TurnStatus,
    Usage,
    open_session,
)

EXIT_COMPLETED = 0
EXIT_FAILED = 1
EXIT_PAUSED = 2
EXIT_USAGE = 3

_DEFAULT_THREAD = "default"

_WORKSPACE_OPT = typer.Option(None, "--workspace", help="Workspace directory (default: cwd).")
_MODEL_OPT = typer.Option(None, "--model", help="Model id (else env PRAXIS_MODEL).")
_BASE_URL_OPT = typer.Option(None, "--base-url", help="OpenAI-compatible base URL (else env PRAXIS_BASE_URL).")
_SCRIPTED_OPT = typer.Option(None, "--scripted", help="Scripted responses JSON path (offline/acceptance).")
_THREAD_OPT = typer.Option(None, "--thread", help='Thread id (default: "default"; R15 stable across commands).')
_CONFIRM_OPT = typer.Option(False, "--confirm", help="Resume an interrupted turn (human guard, §5.2).")
_ALLOW_OPT = typer.Option([], "--allow", help="Approve this pending approval call id (repeatable).")
_DENY_OPT = typer.Option([], "--deny", help="Deny this pending approval call id (repeatable).")
_RECONCILE_OPT = typer.Option([], "--reconcile", help="Reconcile outcome-unknown call: CALL_ID=ok|failed.")
_AFTER_SEQ_OPT = typer.Option(0, "--after-seq", help="Only render items with seq greater than this.")
_TURN_ARG = typer.Argument(None, help="Turn id (omit: list the thread's turns).")
# Task 8 策略入口(R6-6):--policy 覆盖 .praxis/permissions.json 声明;
# --mode 正交于策略(C5:bypass 永不进策略、永不落盘)。
_POLICY_OPT = typer.Option(
    None, "--policy", help="PermissionPolicy JSON file (overrides .praxis/permissions.json)."
)
_MODE_OPT = typer.Option(
    None,
    "--mode",
    click_type=click.Choice(["on-request", "never", "bypass"]),
    help="Approval mode override; orthogonal to policy rules, never persisted.",
)


# ---------------------------------------------------------------------------
# typer app:C8 用法错误 → 退出码 3
# ---------------------------------------------------------------------------


def _fail_usage(exc: click.UsageError) -> NoReturn:
    exc.show()
    raise SystemExit(EXIT_USAGE)


class _UsageExit3Group(TyperGroup):
    """UsageError(含 BadParameter/缺参/未知命令)统一重映射为退出码 3。"""

    def make_context(
        self,
        info_name: str | None,
        args: list[str],
        parent: click.Context | None = None,
        **extra: Any,
    ) -> click.Context:
        try:
            return super().make_context(info_name, args, parent, **extra)
        except click.UsageError as exc:
            _fail_usage(exc)

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except click.UsageError as exc:
            _fail_usage(exc)


app = typer.Typer(
    cls=_UsageExit3Group,
    name="praxis",
    help="Praxis workspace agent — minimal console (run/chat/resume/status/log/steer/verify).",
    add_completion=False,
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# 装配与会话打开(全部经 sdk;每命令 <60 行)
# ---------------------------------------------------------------------------


def _resolve_workspace(workspace: Path | None) -> Path:
    return Path(workspace) if workspace is not None else Path.cwd()


def _scripted_entry(index: int, entry: object) -> ModelResponse:
    """一条剧本 JSON 记录 → ModelResponse;畸形输入 → BadParameter(退出码 3)。"""
    if not isinstance(entry, Mapping):
        raise typer.BadParameter(f"scripted entry [{index}] must be an object, got {type(entry).__name__}")
    content = entry.get("content")
    if not isinstance(content, str):
        raise typer.BadParameter(f"scripted entry [{index}] needs a string 'content' field")
    calls: list[ModelToolCall] = []
    raw_calls = entry.get("tool_calls") or []
    if not isinstance(raw_calls, (list, tuple)):
        raise typer.BadParameter(f"scripted entry [{index}] 'tool_calls' must be a list")
    for call_index, call in enumerate(raw_calls):
        if not isinstance(call, Mapping):
            raise typer.BadParameter(f"scripted entry [{index}] tool_calls[{call_index}] must be an object")
        call_id, name = call.get("call_id"), call.get("name")
        arguments = call.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(arguments, dict):
            raise typer.BadParameter(
                f"scripted entry [{index}] tool_calls[{call_index}] needs string call_id/name + object arguments"
            )
        calls.append(ModelToolCall(call_id=call_id, name=name, arguments=cast("Mapping[str, Any]", arguments)))
    finish_reason = entry.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        finish_reason = "tool_calls" if calls else "stop"
    return ModelResponse(content=content, tool_calls=tuple(calls), usage=Usage(0, 0), finish_reason=finish_reason)


def _load_scripted(path: Path) -> list[ModelResponse]:
    """剧本格式:``[{content, tool_calls?: [{call_id, name, arguments?}], finish_reason?}]``。"""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise typer.BadParameter(f"--scripted file not found: {path}") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"cannot read scripted JSON {path}: {exc}") from None
    if not isinstance(raw, list):
        raise typer.BadParameter("scripted JSON must be a list of response objects")
    return [_scripted_entry(index, entry) for index, entry in enumerate(raw)]


def _resolve_thread(thread: str | None) -> str:
    """R15:省略 --thread 一律落稳定线程 "default"——跨调用可发现(log/status 找得到),
    也供阶段16 真实模型 smoke 检查;--thread 显式覆盖保持线程隔离语义。"""
    return thread if thread is not None else _DEFAULT_THREAD


def _load_policy_file(path: Path) -> PermissionPolicy:
    """--policy JSON → PermissionPolicy;坏文件/坏 JSON → BadParameter(退出码 3,边界拒绝)。"""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise typer.BadParameter(f"--policy file not found: {path}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise typer.BadParameter(f"cannot read policy file {path}: {exc}") from None
    try:
        return PermissionPolicy.model_validate_json(raw)
    except ValueError as exc:  # pydantic strict 冻结模型:ValidationError ⊂ ValueError
        raise typer.BadParameter(f"invalid PermissionPolicy JSON {path}: {exc}") from None


def _open_session_guarded(
    workspace: Path,
    adapter: Any,
    *,
    policy: PermissionPolicy | None,
    mode: PermissionMode | None,
    thread_id: str,
) -> Session:
    """R12:open_session 的 ValueError(损坏的既有状态,如 .praxis/permissions.json
    无法解析)是运行时失败 → 干净错误消息 + EXIT_FAILED(1),绝不裸 traceback(C4)。
    用法错误(--policy/--scripted 坏参)在此之前已以 BadParameter 抛出,保持退出码 3。"""
    try:
        return open_session(workspace, adapter, policy=policy, mode=mode, thread_id=thread_id)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_FAILED) from None


def _open_from_options(
    workspace: Path | None,
    model: str | None,
    base_url: str | None,
    scripted: Path | None,
    thread: str | None,
    policy: Path | None = None,
    mode: str | None = None,
) -> Session:
    # Task 8:policy/mode 经 sdk 透传给 composition(来源序在 composition 执行;
    # mode 与 policy 正交,C5——这里不折叠、不改写)。
    resolved_policy = _load_policy_file(Path(policy)) if policy is not None else None
    resolved_mode = cast("PermissionMode | None", mode)
    if scripted is not None:
        adapter: Any = ScriptedAdapter(_load_scripted(Path(scripted)))
        return _open_session_guarded(
            _resolve_workspace(workspace),
            adapter,
            policy=resolved_policy,
            mode=resolved_mode,
            thread_id=_resolve_thread(thread),
        )
    resolved_model = model or os.environ.get("PRAXIS_MODEL", "")
    if not resolved_model:
        raise typer.BadParameter("no model configured: pass --model, set PRAXIS_MODEL, or use --scripted")
    api_key = os.environ.get("PRAXIS_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise typer.BadParameter("missing API key: set PRAXIS_API_KEY (or OPENAI_API_KEY), or use --scripted")
    resolved_base_url = base_url or os.environ.get("PRAXIS_BASE_URL") or None
    adapter = OpenAICompatAdapter(ModelSpec(model_id=resolved_model), api_key=api_key, base_url=resolved_base_url)
    return _open_session_guarded(
        _resolve_workspace(workspace),
        adapter,
        policy=resolved_policy,
        mode=resolved_mode,
        thread_id=_resolve_thread(thread),
    )


def _open_read_only(workspace: Path | None, thread: str | None) -> Session:
    """log/status/steer/verify 不需要模型:空剧本 adapter 只是 open_session 的必填参数。"""
    return _open_session_guarded(
        _resolve_workspace(workspace),
        ScriptedAdapter([]),
        policy=None,
        mode=None,
        thread_id=_resolve_thread(thread),
    )


# ---------------------------------------------------------------------------
# 结果打印与退出码(C7)
# ---------------------------------------------------------------------------


def _report_turn(result: TurnResult) -> int:
    """turn 终态 → 打印 + 统一退出码:0 完成 / 1 失败或取消 / 2 暂停待输入。"""
    status = result.status
    if status is TurnStatus.COMPLETED:
        if result.output:
            typer.echo(result.output)
        return EXIT_COMPLETED
    if status is TurnStatus.CANCELLED:
        typer.echo("cancelled")
        return EXIT_FAILED
    if status is TurnStatus.FAILED:
        if result.output:
            typer.echo(result.output)
        typer.echo(f"turn failed: {result.reason or 'unknown reason'}", err=True)
        return EXIT_FAILED
    if status is TurnStatus.PAUSED:
        typer.echo(f"turn {result.turn_id} paused ({result.reason or 'needs input'})")
        for approval in result.pending_approvals:
            effects = ",".join(approval.effects) or "-"
            typer.echo(
                f"  approval request: call {approval.call_id} tool={approval.tool_name} "
                f"digest={approval.args_digest[:12]} effects={effects}"
            )
            typer.echo(f"  resume: praxis resume {result.turn_id} --allow {approval.call_id}  (or --deny)")
        if result.unknown_calls:
            typer.echo(f"  outcome unknown calls: {', '.join(result.unknown_calls)}")
            typer.echo("  resume with --reconcile CALL_ID=ok|failed (side effects are never auto-replayed)")
        return EXIT_PAUSED
    typer.echo(f"unexpected turn status: {status.value}", err=True)
    return EXIT_FAILED


def _fmt_time(timestamp: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def _print_turn_ref(session: Session, result: TurnResult) -> None:
    """R15:run/resume 打印可复制的 turn/thread 参照行(`praxis resume <turn_id>` 的入据)。"""
    typer.echo(f"turn: {result.turn_id} (thread: {session.thread_id})")


# ---------------------------------------------------------------------------
# run / resume
# ---------------------------------------------------------------------------


@app.command()
def run(
    prompt: str,
    workspace: Path | None = _WORKSPACE_OPT,
    model: str | None = _MODEL_OPT,
    base_url: str | None = _BASE_URL_OPT,
    scripted: Path | None = _SCRIPTED_OPT,
    thread: str | None = _THREAD_OPT,
    policy: Path | None = _POLICY_OPT,
    mode: str | None = _MODE_OPT,
) -> None:
    """一次性 turn:提交 prompt,打印结果,按 turn 终态决定退出码。"""
    session = _open_from_options(workspace, model, base_url, scripted, thread, policy, mode)
    try:
        result = asyncio.run(session.submit(prompt))
    except ConversationBusy as exc:
        typer.echo(f"busy: {exc}", err=True)
        raise typer.Exit(EXIT_FAILED) from None
    finally:
        session.close()
    _print_turn_ref(session, result)
    raise typer.Exit(_report_turn(result))


def _parse_approval(allow: list[str], deny: list[str]) -> dict[str, Literal["allow", "deny"]]:
    verdicts: dict[str, Literal["allow", "deny"]] = {}
    for call_id in allow:
        verdicts[call_id] = "allow"
    for call_id in deny:
        if call_id in verdicts:
            raise typer.BadParameter(f"call {call_id!r} given both --allow and --deny")
        verdicts[call_id] = "deny"
    return verdicts


def _parse_reconcile(specs: list[str]) -> dict[str, Literal["ok", "failed"]]:
    outcomes: dict[str, Literal["ok", "failed"]] = {}
    for spec in specs:
        call_id, sep, outcome = spec.partition("=")
        if not sep or not call_id or outcome not in ("ok", "failed"):
            raise typer.BadParameter(f"--reconcile expects CALL_ID=ok|failed, got {spec!r}")
        if call_id in outcomes:
            raise typer.BadParameter(f"duplicate --reconcile for call {call_id!r}")
        outcomes[call_id] = cast("Literal['ok', 'failed']", outcome)
    return outcomes


@app.command()
def resume(
    turn_id: str,
    workspace: Path | None = _WORKSPACE_OPT,
    model: str | None = _MODEL_OPT,
    base_url: str | None = _BASE_URL_OPT,
    scripted: Path | None = _SCRIPTED_OPT,
    thread: str | None = _THREAD_OPT,
    policy: Path | None = _POLICY_OPT,
    mode: str | None = _MODE_OPT,
    confirm: bool = _CONFIRM_OPT,
    allow: list[str] = _ALLOW_OPT,
    deny: list[str] = _DENY_OPT,
    reconcile: list[str] = _RECONCILE_OPT,
) -> None:
    """续跑 paused/interrupted turn:--confirm 人工确认,--allow/--deny 审批,--reconcile 对账。"""
    approval_map = _parse_approval(allow, deny)
    reconcile_map = _parse_reconcile(reconcile)
    session = _open_from_options(workspace, model, base_url, scripted, thread, policy, mode)
    try:
        result = asyncio.run(
            session.resume(
                turn_id,
                confirm=confirm,
                approval=approval_map or None,
                reconcile=reconcile_map or None,
            )
        )
    except (ResumeNotConfirmed, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_USAGE) from None
    except ConversationBusy as exc:
        typer.echo(f"busy: {exc}", err=True)
        raise typer.Exit(EXIT_FAILED) from None
    finally:
        session.close()
    _print_turn_ref(session, result)
    raise typer.Exit(_report_turn(result))


# ---------------------------------------------------------------------------
# chat(C1 错误边界 + slash 注册表分派)
# ---------------------------------------------------------------------------


def _read_answer(prompt_text: str) -> str | None:
    """input() 一行;EOF/Ctrl+C → None(调用方决定退出路径)。"""
    try:
        return input(prompt_text)
    except (EOFError, KeyboardInterrupt):
        return None


def _ask_verdict(question: str, choices: Mapping[str, str]) -> str | None:
    while True:
        answer = _read_answer(question)
        if answer is None:
            return None
        verdict = choices.get(answer.strip().lower())
        if verdict is not None:
            return verdict
        typer.echo(f"please answer one of: {', '.join(choices)}")


_VERDICTS_YES_NO: Mapping[str, str] = {"y": "allow", "yes": "allow", "n": "deny", "no": "deny"}
_VERDICTS_OUTCOME: Mapping[str, str] = {"ok": "ok", "failed": "failed"}


def _chat_result(session: Session, result: TurnResult) -> None:
    """打印结果;PAUSED 则逐条询问 → resume,直到非暂停(错误边界保证永不逸出)。"""
    while True:
        code = _report_turn(result)
        if code != EXIT_PAUSED:
            return
        approval: dict[str, Literal["allow", "deny"]] = {}
        for summary in result.pending_approvals:
            verdict = _ask_verdict(
                f"allow {summary.tool_name} (call {summary.call_id}, digest {summary.args_digest[:12]})? [y/n] ",
                _VERDICTS_YES_NO,
            )
            if verdict is None:
                typer.echo("cancelled")
                return
            approval[summary.call_id] = cast("Literal['allow', 'deny']", verdict)
        reconcile: dict[str, Literal["ok", "failed"]] = {}
        for call_id in result.unknown_calls:
            outcome = _ask_verdict(f"call {call_id} side effect already happened? [ok/failed] ", _VERDICTS_OUTCOME)
            if outcome is None:
                typer.echo("cancelled")
                return
            reconcile[call_id] = cast("Literal['ok', 'failed']", outcome)
        if not approval and not reconcile:
            return
        try:
            result = asyncio.run(
                session.resume(result.turn_id, approval=approval or None, reconcile=reconcile or None)
            )
        except KeyboardInterrupt:
            typer.echo("cancelled")
            return
        except Exception as exc:  # noqa: BLE001 — C1 错误边界
            typer.echo(f"error: {type(exc).__name__}: {exc}")
            return


def _chat_turn(session: Session, text: str) -> None:
    """一次输入 = 一个错误边界:Exception → 显示 → 回到输入态,永不退出会话(C1)。"""
    try:
        result = asyncio.run(session.submit(text))
    except KeyboardInterrupt:
        typer.echo("cancelled")
        return
    except ConversationBusy as exc:
        typer.echo(f"busy: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 — C1 错误边界
        typer.echo(f"error: {type(exc).__name__}: {exc}")
        return
    _chat_result(session, result)


def _chat_loop(session: Session) -> None:
    while True:
        line = _read_answer("> ")
        if line is None:
            return
        text = line.strip()
        if not text:
            continue
        if text.startswith("/"):
            # Fix Round 2:名字提取必须对裸 "/"(及 "/ " 经 strip 后)安全——
            # split 结果为空时 name="",与未知命令同路:提示后继续循环,绝不抛。
            parts = text[1:].split(maxsplit=1)
            name = parts[0].lower() if parts else ""
            command = REGISTRY.get(name)
            if command is None:
                typer.echo(f"unknown command: /{name} (try /help)")
                continue
            try:
                output = command.run(session)
            except ExitChat:
                return
            except Exception as exc:  # noqa: BLE001 — slash 也走错误边界
                typer.echo(f"error: {type(exc).__name__}: {exc}")
                continue
            if output:
                typer.echo(output)
            continue
        _chat_turn(session, text)


@app.command()
def chat(
    workspace: Path | None = _WORKSPACE_OPT,
    model: str | None = _MODEL_OPT,
    base_url: str | None = _BASE_URL_OPT,
    scripted: Path | None = _SCRIPTED_OPT,
    thread: str | None = _THREAD_OPT,
    policy: Path | None = _POLICY_OPT,
    mode: str | None = _MODE_OPT,
) -> None:
    """交互式 REPL:每 turn 错误边界(C1);/help /status /exit 查表分派。"""
    session = _open_from_options(workspace, model, base_url, scripted, thread, policy, mode)
    typer.echo(f"praxis chat — thread {session.thread_id} — /help for commands, /exit to leave")
    try:
        _chat_loop(session)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# 只读与旁路命令
# ---------------------------------------------------------------------------


@app.command()
def status(
    turn_id: str | None = _TURN_ARG,
    workspace: Path | None = _WORKSPACE_OPT,
    thread: str | None = _THREAD_OPT,
) -> None:
    """带 turn_id 打印 turn 详情;省略则打印本 thread 的 turn 表(id/status/updated)。"""
    session = _open_read_only(workspace, thread)
    try:
        if turn_id is None:
            turns = session.turns()
            if not turns:
                typer.echo("no turns")
                return
            typer.echo(f"thread {session.thread_id}:")
            for turn in turns:
                typer.echo(f"  {turn.turn_id}  {turn.status:<12} {_fmt_time(turn.updated_at)}")
            return
        try:
            turn = session.status(turn_id)
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(EXIT_USAGE) from None
        result = session.result(turn_id)
        typer.echo(f"turn {turn.turn_id}")
        typer.echo(f"  thread:  {turn.thread_id}")
        typer.echo(f"  status:  {turn.status}")
        typer.echo(f"  updated: {_fmt_time(turn.updated_at)}")
        if result.reason:
            typer.echo(f"  reason:  {result.reason}")
        if result.output:
            typer.echo(f"  output:  {result.output}")
        for approval in result.pending_approvals:
            typer.echo(f"  pending approval: call {approval.call_id} tool={approval.tool_name}")
        if result.unknown_calls:
            typer.echo(f"  outcome unknown calls: {', '.join(result.unknown_calls)}")
    finally:
        session.close()


@app.command()
def log(
    workspace: Path | None = _WORKSPACE_OPT,
    thread: str | None = _THREAD_OPT,
    after_seq: int = _AFTER_SEQ_OPT,
) -> None:
    """打印本 thread 的 item 日志(seq > after_seq 升序);fail-open 渲染(B1)。"""
    session = _open_read_only(workspace, thread)
    try:
        items = session.items(after_seq=after_seq)
        if not items:
            typer.echo("(no items)")
            return
        for item in items:
            typer.echo(render_item(item))
    finally:
        session.close()


@app.command()
def steer(
    text: str,
    workspace: Path | None = _WORKSPACE_OPT,
    thread: str | None = _THREAD_OPT,
) -> None:
    """追加一条 steering(append-only,不取执行锁,§5.8);执行者在步边界读到。"""
    session = _open_read_only(workspace, thread)
    try:
        session.steer(text)
        typer.echo("ok: steering appended")
    finally:
        session.close()


@app.command()
def verify(
    workspace: Path | None = _WORKSPACE_OPT,
) -> None:
    """离线只读校验 store(seq 连续/payload JSON/status 合法/tool_call 配对),永不修复。"""
    session = _open_read_only(workspace, None)
    try:
        report = session.verify()
    finally:
        session.close()
    typer.echo(f"checked {report.checked_items} item(s)")
    if report.ok:
        typer.echo("ok")
        raise typer.Exit(EXIT_COMPLETED)
    for problem in report.problems:
        typer.echo(f"problem: {problem}")
    raise typer.Exit(EXIT_FAILED)
