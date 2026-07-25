from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sys
from collections.abc import Coroutine, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import click
import typer
import yaml

from agent_runtime.knowledge import RAGKnowledgeConfig
from agent_runtime.models import (
    ModelNotAvailableError,
    ModelPolicyError,
    ModelSpec,
    format_model_rows,
)
from agent_runtime.result import AgentDiagnostic, AgentResult
from agent_runtime.runtime.builder import build_model_control_plane
from rag.agent.core.llm_registry import UnknownModelAliasError
from rag.agent.streaming.events import EventType, StreamEvent
from rag.agent.turns import (
    TurnNotFoundError,
    TurnRecord,
    TurnStateError,
    TurnStore,
)

if TYPE_CHECKING:
    from agent_runtime.agent import Agent
    from agent_runtime.result import AgentPause

agent_app = typer.Typer(add_completion=False, no_args_is_help=True)
model_app = typer.Typer(add_completion=False, no_args_is_help=True)
agent_app.add_typer(model_app, name="model", help="查看和切换当前模型会话。")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_SESSION_PATH = Path(".rag/agent_model_session.json")
DEFAULT_CHECKPOINT_PATH = Path(".rag/agent_checkpoints.sqlite")


class _CLIToolEventDisplay:
    """Project canonical stream events into bounded terminal output."""

    def __init__(self) -> None:
        self._displayed_tool_ids: set[str] = set()
        self._displayed_tool_events: set[tuple[EventType, str]] = set()
        self._displayed_plan_revisions: set[int | str] = set()
        self._tool_names: dict[str, str] = {}
        self._line_open = False
        self.answer_streamed = False

    async def emit(self, event: StreamEvent) -> None:
        if event.type is EventType.TEXT_DELTA:
            text = event.data.get("text")
            if not isinstance(text, str) or not text:
                return
            print(text, end="", flush=True)
            self.answer_streamed = True
            self._line_open = not text.endswith("\n")
            return

        if event.type is EventType.TOOL_USE_START:
            self._render_tool_start(event)
            return
        if event.type is EventType.TOOL_USE_PROGRESS:
            self._render_tool_progress(event)
            return
        if event.type is EventType.TOOL_USE_RESULT:
            self._render_tool_result(event)
            return
        if event.type is EventType.TOOL_USE_ERROR:
            self._render_tool_error(event)
            return
        if event.type is EventType.PLAN_UPDATED:
            self._render_plan(event)
            return
        if event.type is EventType.RECOVERY:
            strategy = event.data.get("strategy")
            if not isinstance(strategy, str) or not strategy:
                return
            detail = event.data.get("detail")
            suffix = f" — {detail}" if isinstance(detail, str) and detail else ""
            self._write_line(f"↻ 恢复: {strategy}{suffix}")

    def _render_tool_start(self, event: StreamEvent) -> None:
        tool_id = event.data.get("tool_id")
        if not isinstance(tool_id, str) or not tool_id:
            return
        if tool_id in self._displayed_tool_ids:
            return
        self._displayed_tool_ids.add(tool_id)
        tool_name = event.data.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            return
        self._tool_names[tool_id] = tool_name
        preview = event.data.get("input_preview")
        suffix = f": {preview}" if isinstance(preview, str) and preview else ""
        self._write_line(f"→ {tool_name}{suffix}")

    def _render_tool_progress(self, event: StreamEvent) -> None:
        tool_id = event.data.get("tool_id")
        progress = event.data.get("progress")
        if not isinstance(tool_id, str) or not isinstance(progress, str):
            return
        tool_name = self._tool_names.get(tool_id, "tool")
        percent = event.data.get("percent")
        percent_text = f" ({percent:g}%)" if isinstance(percent, (int, float)) else ""
        self._write_line(f"… {tool_name}: {progress}{percent_text}")

    def _render_tool_result(self, event: StreamEvent) -> None:
        tool_id = event.data.get("tool_id")
        if not isinstance(tool_id, str) or not tool_id:
            return
        marker = (EventType.TOOL_USE_RESULT, tool_id)
        if marker in self._displayed_tool_events:
            return
        self._displayed_tool_events.add(marker)
        event_name = event.data.get("tool_name")
        tool_name = event_name if isinstance(event_name, str) and event_name else self._tool_names.get(tool_id, "tool")
        result = event.data.get("result")
        suffix = f": {_bounded_cli_text(str(result))}" if result is not None else ""
        self._write_line(f"✓ {tool_name}{suffix}")
        details = event.data.get("details")
        if not isinstance(details, Mapping):
            return
        diff = details.get("diff")
        if isinstance(diff, str) and diff:
            self._write_block(diff)

    def _render_tool_error(self, event: StreamEvent) -> None:
        tool_id = event.data.get("tool_id")
        if not isinstance(tool_id, str) or not tool_id:
            return
        marker = (EventType.TOOL_USE_ERROR, tool_id)
        if marker in self._displayed_tool_events:
            return
        self._displayed_tool_events.add(marker)
        error = event.data.get("error")
        suffix = f": {_bounded_cli_text(error)}" if isinstance(error, str) and error else ""
        self._write_line(f"✗ {self._tool_names.get(tool_id, 'tool')}{suffix}")

    def _render_plan(self, event: StreamEvent) -> None:
        plan = event.data.get("plan")
        if not isinstance(plan, Mapping):
            return
        revision = plan.get("revision")
        if not isinstance(revision, (int, str)):
            return
        if revision in self._displayed_plan_revisions:
            return
        self._displayed_plan_revisions.add(revision)
        self._write_line(f"计划 (revision {revision})")
        steps = plan.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            return
        symbols = {
            "completed": "✓",
            "in_progress": "→",
            "failed": "✗",
        }
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            title = step.get("title")
            if not isinstance(title, str) or not title:
                continue
            status = step.get("status")
            symbol = symbols.get(status, "○") if isinstance(status, str) else "○"
            self._write_line(f"  {symbol} {title}")

    def _write_line(self, value: str) -> None:
        if self._line_open:
            print()
        print(value, flush=True)
        self._line_open = False

    def _write_block(self, value: str) -> None:
        if self._line_open:
            print()
        print(value.rstrip("\n"), flush=True)
        self._line_open = False

    def begin_turn(self) -> None:
        self.finish()
        self.answer_streamed = False

    def finish(self) -> None:
        if self._line_open:
            print(flush=True)
            self._line_open = False


def _bounded_cli_text(value: str, *, limit: int = 180) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 1]}…"


def _load_knowledge_config(path: Path | None) -> RAGKnowledgeConfig | None:
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise typer.BadParameter(f"无法读取 knowledge config {path}: {exc}") from exc
    try:
        payload: object = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
        if not isinstance(payload, Mapping):
            raise TypeError("顶层必须是对象")
        return RAGKnowledgeConfig.model_validate(dict(payload))
    except Exception as exc:
        raise typer.BadParameter(f"无效的 knowledge config {path}: {exc}") from exc


def _format_tool_summary(result: AgentResult) -> str:
    if not result.tool_calls:
        return ""
    lines = ["", "─" * 40, "工具执行:"]
    for tool_call in result.tool_calls:
        status_icon = "✗" if tool_call.is_error else "✓"
        tool_info = f"  {status_icon} {tool_call.tool_name}"
        if tool_call.is_error:
            tool_info += (
                f" ({tool_call.error_code or 'tool_error'}: {(tool_call.error_message or 'unknown tool error')[:60]})"
            )
        lines.append(tool_info)
    return "\n".join(lines)


def _format_plan_summary(result: AgentResult) -> str:
    plan = result.plan
    if plan is None or not any(event.event_type == "llm_update" for event in result.plan_events):
        return ""
    icons = {
        "pending": "○",
        "in_progress": "→",
        "completed": "✓",
        "blocked": "!",
        "skipped": "-",
    }
    lines = ["", "─" * 40, f"计划 (revision {plan.revision}):"]
    if plan.summary:
        lines.append(f"  {plan.summary}")
    lines.extend(f"  {icons.get(step.status, '?')} {step.title}" for step in plan.steps)
    return "\n".join(lines)


def _failure_title(stop_reason: str | None) -> str:
    if stop_reason == "model_provider_failed":
        return "错误: 模型调用失败 (model_provider_failed)"
    if stop_reason:
        return f"错误: Agent 运行失败 ({stop_reason})"
    return "错误: Agent 运行失败"


def _failure_diagnostics(
    diagnostics: Sequence[AgentDiagnostic],
    *,
    stop_reason: str | None,
) -> list[AgentDiagnostic]:
    all_diagnostics = list(diagnostics)
    if not all_diagnostics:
        return []
    if stop_reason:
        exact = [diagnostic for diagnostic in all_diagnostics if diagnostic.code == stop_reason]
        if exact:
            return exact
    errors = [diagnostic for diagnostic in all_diagnostics if diagnostic.severity == "error"]
    return errors or all_diagnostics[:1]


def _print_diagnostic(diagnostic: AgentDiagnostic) -> None:
    suffix = f", {diagnostic.error_type}" if diagnostic.error_type is not None else ""
    print(f"  [{diagnostic.component}] {diagnostic.code}: {diagnostic.message}{suffix}")


def _display_failure(
    *,
    stop_reason: str | None,
    diagnostics: Sequence[AgentDiagnostic],
    verbose: bool,
) -> None:
    print(f"\n{_failure_title(stop_reason)}")
    selected = _failure_diagnostics(diagnostics, stop_reason=stop_reason)
    if selected:
        for diagnostic in selected:
            _print_diagnostic(diagnostic)
    elif verbose and stop_reason:
        print(f"  stop_reason: {stop_reason}")
    if verbose:
        selected_ids = {id(diagnostic) for diagnostic in selected}
        for diagnostic in diagnostics:
            if id(diagnostic) not in selected_ids:
                _print_diagnostic(diagnostic)


def _display_agent_result(
    result: AgentResult,
    *,
    verbose: bool,
    answer_streamed: bool = False,
) -> None:
    if result.status == "failed":
        _display_failure(
            stop_reason=result.stop_reason,
            diagnostics=result.diagnostics,
            verbose=verbose,
        )
    elif result.diagnostics:
        degraded = sum(1 for diagnostic in result.diagnostics if diagnostic.degraded)
        if degraded:
            print(f"\n警告: Agent 以降级模式运行（{degraded} 项诊断）")
        if verbose:
            for diagnostic in result.diagnostics:
                _print_diagnostic(diagnostic)

    if result.answer and not answer_streamed:
        print(f"\n{result.answer}")

    plan_summary = _format_plan_summary(result)
    if plan_summary:
        print(plan_summary)
    if result.tool_calls:
        print(_format_tool_summary(result))

    if verbose:
        usage = result.usage
        print(
            "\n调用统计: "
            f"native={usage.native_calls}"
            f"({usage.native_errors}err/"
            f"{usage.native_latency_ms_total:.0f}ms) "
            f"deferred={usage.deferred_calls} "
            f"mcp={usage.mcp_calls}"
            f"({usage.mcp_errors}err/"
            f"{usage.mcp_latency_ms_total:.0f}ms)"
        )
        print(
            "耗时: "
            f"total={usage.latency_ms:.0f}ms "
            f"startup={usage.startup_ms:.0f}ms "
            f"build={usage.build_service_ms:.0f}ms "
            f"model_ready={usage.model_ready_ms:.0f}ms "
            f"model={usage.model_latency_ms:.0f}ms "
            f"tool={usage.tool_latency_ms:.0f}ms "
            f"finalize={usage.finalize_latency_ms:.0f}ms "
            f"prompt_bytes={usage.prompt_bytes} "
            f"tool_schema_bytes={usage.tool_schema_bytes}"
        )
        if result.evidence:
            print(f"证据: {len(result.evidence)} 条")
        if result.stop_reason:
            print(f"停止原因: {result.stop_reason}")

    print(f"Turn: {result.turn_id}")

    if verbose:
        print(f"状态: {result.status}")


def _handle_pause(
    result: AgentResult,
) -> str | None:
    """展示暂停信息，获取用户决策。返回 None 表示退出。"""
    req = result.pause
    if req is None:
        return None

    print(f"\n⏸  需要确认: {req.question}")

    if req.tool_calls:
        for tc in req.tool_calls:
            risk_mark = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(tc.risk_level, "")
            print(f"  {risk_mark} {tc.tool_name}: {tc.args_preview}")
            if tc.reason:
                print(f"     原因: {tc.reason}")

    options = req.options or ("allow_once", "deny", "continue", "abort")
    print(f"  选项: {', '.join(options)}")

    while True:
        choice = input("> ").strip()
        if choice in options:
            break
        if choice in {"a", "y", "yes"}:
            choice = "allow_once"
            break
        if choice in {"n", "no", "d"}:
            choice = "deny"
            break
        if choice in {"c"}:
            choice = "continue"
            break
        if choice in {"q", "exit", "/exit"}:
            return None
        print(f"  请输入: {', '.join(options)} (或 a=允许, n=拒绝, q=退出)")

    return choice


def _display_pending_recovery(
    request: AgentPause | None,
    *,
    turn_id: str,
    checkpoint_db: Path,
) -> None:
    print(f"\n⏸  待恢复 Turn: {turn_id}")
    if request is None:
        print("   未检测到待处理的人机请求；可继续中断执行。")
        options = ["continue", "abort"]
    else:
        print(f"   请求: {request.question}")
        for tool_call in request.tool_calls:
            risk_mark = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢",
            }.get(tool_call.risk_level, "")
            print(f"   {risk_mark} {tool_call.tool_name}: {tool_call.args_preview}")
            if tool_call.reason:
                print(f"      原因: {tool_call.reason}")
        options = list(request.options)
        if not options:
            options = (
                ["allow_once", "deny", "abort"]
                if request.kind in {"tool_approval", "tool_reconciliation"}
                else ["continue", "abort"]
            )
    print("   可用恢复命令:")
    for option in options:
        command = [
            "agent",
            "resume",
            turn_id,
            "--action",
            option,
            "--checkpoint-db",
            str(checkpoint_db),
        ]
        print(f"     {shlex.join(command)}")
    if request is not None and request.kind in {"choice", "clarification"}:
        print("   需要文本时，在 continue 命令后添加 --input '<text>'。")


def _create_agent_facade(
    *,
    model: str | None = None,
    checkpoint_db: Path | None = None,
    workspace_path: Path | str | None = None,
    model_session_path: Path | None = None,
    knowledge: RAGKnowledgeConfig | None = None,
) -> Agent:
    from agent_runtime.agent import Agent

    return Agent(
        model=model,
        checkpoint_db=checkpoint_db,
        workspace_path=workspace_path,
        model_session_path=model_session_path,
        knowledge=knowledge,
    )


def _is_interactive_terminal() -> bool:
    if any(_environment_flag(name) for name in ("CI", "GITHUB_ACTIONS")):
        return False
    return _is_tty(sys.stdin) and _is_tty(sys.stdout)


def _environment_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _is_tty(stream: object) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(isatty()) if callable(isatty) else False


def _run_cli_async[T](awaitable: Coroutine[Any, Any, T]) -> T:
    """Run one public async operation and render known model setup failures."""

    try:
        return asyncio.run(awaitable)
    except (
        FileNotFoundError,
        KeyError,
        ModelNotAvailableError,
        ModelPolicyError,
        TurnNotFoundError,
        TurnStateError,
        UnknownModelAliasError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc


def _resolve_cli_input_files(values: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    for value in values:
        path = Path(value).expanduser()
        if not path.exists():
            raise typer.BadParameter(
                f"输入文件不存在: {value}",
                param_hint="--file",
            )
        if not path.is_file():
            raise typer.BadParameter(
                f"输入路径不是文件: {value}",
                param_hint="--file",
            )
        resolved.append(str(path.resolve()))
    return resolved


async def _run_facade_command(
    facade: Agent,
    *,
    task: str,
    previous_turn_id: str | None = None,
    files: Sequence[str],
    max_tokens_total: int | None,
    interactive_approval: bool,
    max_turns: int | None = None,
    require_workspace_change: bool = False,
    allow_write_tools: bool = False,
    allow_execute_tools: bool = False,
    event_display: _CLIToolEventDisplay | None = None,
) -> AgentResult:
    """Run one Turn and optionally drive its approval lifecycle."""
    display = event_display or _CLIToolEventDisplay()
    try:
        display.begin_turn()
        result = await facade.arun(
            task,
            previous_turn_id=previous_turn_id,
            files=files,
            max_turns=max_turns,
            max_tokens_total=max_tokens_total,
            require_workspace_change=require_workspace_change,
            allow_write_tools=allow_write_tools,
            allow_execute_tools=allow_execute_tools,
            event_sink=display,
        )
        while result.status == "paused" and interactive_approval:
            display.finish()
            action = _handle_pause(result)
            if action is None:
                break
            display.begin_turn()
            result = await facade.aresume(
                result.turn_id,
                action,
                event_sink=display,
            )
        return result
    finally:
        display.finish()


async def _resume_facade_command(
    facade: Agent,
    *,
    turn_id: str,
    action: str,
    user_input: str | None = None,
    event_display: _CLIToolEventDisplay | None = None,
) -> AgentResult:
    display = event_display or _CLIToolEventDisplay()
    display.begin_turn()
    try:
        return await facade.aresume(
            turn_id,
            action,
            user_input=user_input,
            event_sink=display,
        )
    finally:
        display.finish()


async def _pending_resume_request(
    facade: Agent,
    *,
    turn_id: str,
    event_display: _CLIToolEventDisplay | None = None,
) -> AgentPause | None:
    display = event_display or _CLIToolEventDisplay()
    display.begin_turn()
    try:
        return await facade.apending_input(turn_id)
    finally:
        display.finish()


def _print_startup_banner(model_alias: str) -> None:
    print(f"Agent 就绪 (模型: {model_alias})")
    print("输入查询，或输入 /help 查看交互命令。")
    print()


def _print_chat_help() -> None:
    print("交互命令:")
    print("  /help              显示本帮助")
    print("  /status            显示当前 Turn、模型和工作区")
    print("  /new, /clear       下一条消息不继承当前上下文")
    print("  /model [current|list|switch <id>]")
    print("  /verbose           切换详细输出")
    print("  /exit              退出")


async def _chat_facade_loop(
    facade: Agent,
    *,
    max_tokens_total: int | None,
    max_turns: int | None = None,
    previous_turn_id: str | None = None,
    allow_write_tools: bool = False,
    allow_execute_tools: bool = False,
) -> None:
    event_display = _CLIToolEventDisplay()
    current_turn_id = previous_turn_id
    verbose = False
    default_chat_workspace = facade.workspace_path or Path.cwd()
    chat_workspace = default_chat_workspace
    model_alias = facade.current_model().id
    _print_startup_banner(model_alias)
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return
        if not query:
            continue
        if query == "/exit":
            return
        if query == "/help":
            _print_chat_help()
            continue
        if query == "/status":
            print(f"Previous Turn: {current_turn_id or '(none)'}")
            print(f"模型: {model_alias}")
            print(f"工作区: {chat_workspace}")
            print(f"详细输出: {'开' if verbose else '关'}")
            continue
        if query in {"/new", "/clear"}:
            current_turn_id = None
            chat_workspace = default_chat_workspace
            model_alias = facade.current_model().id
            print("下一条消息将使用空历史。")
            continue
        if query == "/verbose":
            verbose = not verbose
            print(f"详细输出: {'开' if verbose else '关'}")
            continue
        if query == "/model" or query.startswith("/model "):
            _handle_model_slash_command(
                query,
                agent=facade,
                allow_switch=current_turn_id is None,
            )
            model_alias = facade.current_model().id
            continue
        if query.startswith("/"):
            print(f"未知命令: {query.split()[0]}；输入 /help 查看可用命令。")
            continue

        event_display.begin_turn()
        result = await facade.arun(
            query,
            previous_turn_id=current_turn_id,
            max_turns=max_turns,
            max_tokens_total=max_tokens_total,
            allow_write_tools=allow_write_tools,
            allow_execute_tools=allow_execute_tools,
            event_sink=event_display,
        )
        current_turn_id = result.turn_id
        while result.status == "paused":
            event_display.finish()
            action = _handle_pause(result)
            if action is None:
                print("已取消。")
                break
            event_display.begin_turn()
            result = await facade.aresume(
                result.turn_id,
                action,
                event_sink=event_display,
            )
        event_display.finish()
        _display_agent_result(
            result,
            verbose=verbose,
            answer_streamed=event_display.answer_streamed,
        )


def _print_current_model(spec: ModelSpec) -> None:
    print(f"{spec.id}")
    print(f"provider: {spec.provider}")
    print(f"provider_model: {spec.provider_model}")
    print(f"context_window: {spec.context_window}")
    print(f"location: {spec.location}")


def _handle_model_slash_command(
    query: str,
    *,
    agent: Agent | None,
    allow_switch: bool = True,
) -> None:
    if agent is None:
        print("模型控制平面不可用。")
        return
    parts = query.split()
    action = parts[1] if len(parts) > 1 else "current"
    switch_requested = action in {"switch", "use"} or (len(parts) == 2 and action not in {"current", "list"})
    if switch_requested and not allow_switch:
        print("当前上下文的模型绑定已冻结；请先使用 /new，再切换模型。")
        return
    try:
        if action == "list":
            for line in format_model_rows(
                agent.models(),
                current_model_id=agent.current_model().id,
            ):
                print(line)
            return
        if action == "current":
            _print_current_model(agent.current_model())
            return
        if action in {"switch", "use"}:
            if len(parts) < 3:
                print("用法: /model switch <model_id>")
                return
            spec = agent.switch_model(parts[2])
            print(f"已切换模型: {spec.id}")
            return
        if len(parts) == 2:
            spec = agent.switch_model(action)
            print(f"已切换模型: {spec.id}")
            return
    except (ModelPolicyError, UnknownModelAliasError) as exc:
        print(f"模型切换失败: {exc}")
        return
    print("用法: /model [current|list|switch <model_id>]")


# ── CLI Commands ──


@model_app.command(name="list")
def model_list(
    session_path: Annotated[
        Path,
        typer.Option("--session-path", help="模型 session state 文件"),
    ] = DEFAULT_MODEL_SESSION_PATH,
) -> None:
    """列出可用模型，并标记当前会话模型。"""
    control_plane = build_model_control_plane(session_path=session_path)
    for line in format_model_rows(
        control_plane.list_models(),
        current_model_id=control_plane.current_model().id,
    ):
        print(line)


@model_app.command(name="current")
def model_current(
    session_path: Annotated[
        Path,
        typer.Option("--session-path", help="模型 session state 文件"),
    ] = DEFAULT_MODEL_SESSION_PATH,
) -> None:
    """显示当前会话模型。"""
    _print_current_model(build_model_control_plane(session_path=session_path).current_model())


@model_app.command(name="switch")
def model_switch(
    model_id: Annotated[str, typer.Argument(help="要切换到的模型 id")],
    session_path: Annotated[
        Path,
        typer.Option("--session-path", help="模型 session state 文件"),
    ] = DEFAULT_MODEL_SESSION_PATH,
) -> None:
    """切换当前模型 session state，不修改 models.yaml。"""
    control_plane = build_model_control_plane(session_path=session_path)
    try:
        spec = control_plane.switch_model(model_id, requested_by="user")
    except (ModelPolicyError, UnknownModelAliasError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    print(f"已切换模型: {spec.id}")


def _latest_cli_turn(
    checkpoint_db: Path,
    *,
    workspace_path: Path | None,
) -> TurnRecord | None:
    store = TurnStore(checkpoint_db)
    try:
        return store.latest_resumable_turn(workspace_path=workspace_path)
    finally:
        store.close()


def _latest_completed_cli_turn(
    checkpoint_db: Path,
    *,
    workspace_path: Path | None,
) -> TurnRecord | None:
    store = TurnStore(checkpoint_db)
    try:
        return store.latest_turn(workspace_path=workspace_path)
    finally:
        store.close()


@agent_app.command(name="chat")
def agent_chat(
    previous_turn_id: Annotated[
        str | None,
        typer.Option(
            "--previous-turn-id",
            help="从指定 Turn 继续；省略则从空上下文开始。",
        ),
    ] = None,
    last: Annotated[
        bool,
        typer.Option("--last", help="从当前工作区最近的已完成 Turn 继续"),
    ] = False,
    checkpoint_db: Annotated[
        Path,
        typer.Option("--checkpoint-db", help="SQLite checkpoint 文件"),
    ] = DEFAULT_CHECKPOINT_PATH,
    knowledge_config: Annotated[
        Path | None,
        typer.Option(
            "--knowledge-config",
            help="新上下文使用的 RAGKnowledgeConfig JSON/YAML。",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="主生成模型别名，对应 configs/models.yaml 中 capability=chat 的条目"),
    ] = None,
    max_tokens_total: Annotated[
        int | None,
        typer.Option(
            "--max-tokens-total",
            min=1,
            help="每条消息的 LLM token 总预算。",
        ),
    ] = None,
    max_turns: Annotated[
        int | None,
        typer.Option("--max-turns", min=1, help="每条消息允许的最大模型回合数"),
    ] = None,
    allow_write_tools: Annotated[
        bool,
        typer.Option("--allow-write-tools", help="预授权 workspace 写入类调用"),
    ] = False,
    allow_execute_tools: Annotated[
        bool,
        typer.Option("--allow-execute-tools", help="预授权进程执行类调用"),
    ] = False,
) -> None:
    """交互式 Agent 对话。暂停时支持工具审批。"""
    if previous_turn_id is not None and last:
        raise typer.BadParameter("--previous-turn-id 与 --last 不能同时使用")
    effective_previous_turn_id = previous_turn_id
    if last:
        latest = _latest_completed_cli_turn(
            checkpoint_db,
            workspace_path=Path.cwd(),
        )
        if latest is None:
            raise typer.BadParameter("当前工作区没有可继续的 Turn")
        effective_previous_turn_id = latest.turn_id
    if effective_previous_turn_id is not None and knowledge_config is not None:
        raise typer.BadParameter("继续已有 Turn 时不能传 --knowledge-config；Turn 的 RuntimeBinding 是唯一配置来源")
    if effective_previous_turn_id is not None and model is not None:
        raise typer.BadParameter(
            "继续已有 Turn 时不能传 --model；模型已由前一个 Turn 绑定"
        )
    facade = _create_agent_facade(
        model=model,
        checkpoint_db=checkpoint_db,
        workspace_path=Path.cwd(),
        model_session_path=DEFAULT_MODEL_SESSION_PATH,
        knowledge=_load_knowledge_config(knowledge_config),
    )
    _run_cli_async(
        _chat_facade_loop(
            facade,
            max_tokens_total=max_tokens_total,
            max_turns=max_turns,
            previous_turn_id=effective_previous_turn_id,
            allow_write_tools=allow_write_tools,
            allow_execute_tools=allow_execute_tools,
        )
    )


@agent_app.command(name="run")
def agent_run(
    task: Annotated[str, typer.Argument(help="查询任务")],
    previous_turn_id: Annotated[
        str | None,
        typer.Option(
            "--previous-turn-id",
            help="从指定 Turn 继续；省略则使用空历史。",
        ),
    ] = None,
    last: Annotated[
        bool,
        typer.Option("--last", help="从当前工作区最近的已完成 Turn 继续"),
    ] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive", help="非交互模式")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="详细输出")] = False,
    checkpoint_db: Annotated[
        Path,
        typer.Option("--checkpoint-db", help="SQLite checkpoint 文件；启用后可跨进程 resume"),
    ] = DEFAULT_CHECKPOINT_PATH,
    model_session_path: Annotated[
        Path,
        typer.Option(
            "--model-session-path",
            help="模型会话状态文件；可放在 workspace 外部。",
        ),
    ] = DEFAULT_MODEL_SESSION_PATH,
    max_tokens_total: Annotated[
        int | None,
        typer.Option(
            "--max-tokens-total",
            min=1,
            help="本次 Agent 运行的 LLM token 总预算。",
        ),
    ] = None,
    max_turns: Annotated[
        int | None,
        typer.Option("--max-turns", min=1, help="本次运行允许的最大模型回合数"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="主生成模型别名，对应 configs/models.yaml 中 capability=chat 的条目"),
    ] = None,
    knowledge_config: Annotated[
        Path | None,
        typer.Option(
            "--knowledge-config",
            help="RAGKnowledgeConfig JSON/YAML 文件。",
        ),
    ] = None,
    input_files: Annotated[
        list[str] | None,
        typer.Option(
            "--file",
            "-f",
            help="附加本次 Turn 的输入文件，可多次指定",
        ),
    ] = None,
    allow_write_tools: Annotated[
        bool,
        typer.Option("--allow-write-tools", help="预授权本次 workspace 写入类调用"),
    ] = False,
    require_workspace_change: Annotated[
        bool,
        typer.Option(
            "--require-workspace-change",
            help=(
                "仅在产生真实 workspace 变更后允许完成 Turn；若同时允许"
                "进程执行，还要求在最后一次真实变更后成功运行测试、lint、"
                "类型检查或构建。"
            ),
        ),
    ] = False,
    allow_execute_tools: Annotated[
        bool,
        typer.Option("--allow-execute-tools", help="预授权本次进程执行类调用"),
    ] = False,
) -> None:
    """执行一个新 Turn；可从已完成 Turn 继续上下文。"""
    if previous_turn_id is not None and last:
        raise typer.BadParameter("--previous-turn-id 与 --last 不能同时使用")
    effective_previous_turn_id = previous_turn_id
    if last:
        latest = _latest_completed_cli_turn(
            checkpoint_db,
            workspace_path=Path.cwd(),
        )
        if latest is None:
            raise typer.BadParameter("当前工作区没有可继续的 Turn")
        effective_previous_turn_id = latest.turn_id
    if effective_previous_turn_id is not None and knowledge_config is not None:
        raise typer.BadParameter("继续已有 Turn 时不能传 --knowledge-config；Turn 的 RuntimeBinding 是唯一配置来源")
    if effective_previous_turn_id is not None and model is not None:
        raise typer.BadParameter(
            "继续已有 Turn 时不能传 --model；模型已由前一个 Turn 绑定"
        )
    facade = _create_agent_facade(
        model=model,
        checkpoint_db=checkpoint_db,
        workspace_path=Path.cwd(),
        model_session_path=model_session_path,
        knowledge=_load_knowledge_config(knowledge_config),
    )
    interactive_approval = not non_interactive and _is_interactive_terminal()
    event_display = _CLIToolEventDisplay()
    result = _run_cli_async(
        _run_facade_command(
            facade,
            task=task,
            previous_turn_id=effective_previous_turn_id,
            files=_resolve_cli_input_files(input_files or ()),
            max_tokens_total=max_tokens_total,
            interactive_approval=interactive_approval,
            max_turns=max_turns,
            require_workspace_change=require_workspace_change,
            allow_write_tools=allow_write_tools,
            allow_execute_tools=allow_execute_tools,
            event_display=event_display,
        )
    )
    _display_agent_result(
        result,
        verbose=verbose,
        answer_streamed=event_display.answer_streamed,
    )

    if result.status == "paused":
        print()
        print("⏸  已保存 checkpoint，可跨进程恢复:")
        resume_args = [
            "agent",
            "resume",
            result.turn_id,
            "--checkpoint-db",
            str(checkpoint_db),
        ]
        print(f"   {shlex.join(resume_args)}")

        if result.pause is not None:
            print(f"\n   待处理: {result.pause.question}")

        raise typer.Exit(code=2)

    if result.status == "failed":
        raise typer.Exit(code=1)


@agent_app.command(name="resume")
def agent_resume(
    turn_id: Annotated[
        str | None,
        typer.Argument(help="要恢复的 UUID Turn ID"),
    ] = None,
    last: Annotated[
        bool,
        typer.Option("--last", help="恢复最近的可恢复 Turn"),
    ] = False,
    all_workspaces: Annotated[
        bool,
        typer.Option("--all", help="--last 搜索所有工作区"),
    ] = False,
    checkpoint_db: Annotated[
        Path,
        typer.Option("--checkpoint-db", help="SQLite checkpoint 文件"),
    ] = DEFAULT_CHECKPOINT_PATH,
    action: Annotated[
        str | None,
        typer.Option(
            "--action",
            help=("allow_once | deny | continue | mark_completed | mark_failed | abort"),
        ),
    ] = None,
    user_input: Annotated[
        str | None,
        typer.Option("--input", help="clarification/choice 恢复时的用户输入"),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="详细输出")] = False,
) -> None:
    """先读取持久化 Turn 元数据，再恢复未完成的 Turn。"""
    if turn_id is not None and last:
        raise typer.BadParameter("Turn ID 与 --last 不能同时使用")
    if all_workspaces and not last:
        raise typer.BadParameter("--all 只能与 --last 一起使用")
    effective_turn_id = turn_id
    if last:
        latest = _latest_cli_turn(
            checkpoint_db,
            workspace_path=None if all_workspaces else Path.cwd(),
        )
        if latest is None:
            scope = "所有工作区" if all_workspaces else "当前工作区"
            raise typer.BadParameter(f"{scope}没有可恢复 Turn")
        effective_turn_id = latest.turn_id
    if effective_turn_id is None:
        print("请提供 Turn ID，或使用 --last 恢复最近 Turn。")
        raise typer.Exit(code=2)
    if action is None and user_input is not None:
        raise typer.BadParameter("--input 需要同时指定 --action")
    facade = _create_agent_facade(
        checkpoint_db=checkpoint_db,
    )
    event_display = _CLIToolEventDisplay()
    if action is None:
        pending = _run_cli_async(
            _pending_resume_request(
                facade,
                turn_id=effective_turn_id,
                event_display=event_display,
            )
        )
        _display_pending_recovery(
            pending,
            turn_id=effective_turn_id,
            checkpoint_db=checkpoint_db,
        )
        raise typer.Exit(code=2)
    result = _run_cli_async(
        _resume_facade_command(
            facade,
            turn_id=effective_turn_id,
            action=action,
            user_input=user_input,
            event_display=event_display,
        )
    )
    _display_agent_result(
        result,
        verbose=verbose,
        answer_streamed=event_display.answer_streamed,
    )
    if result.status == "paused":
        raise typer.Exit(code=2)
    if result.status == "failed":
        raise typer.Exit(code=1)
