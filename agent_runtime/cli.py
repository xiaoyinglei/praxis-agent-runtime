from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sqlite3
import sys
import time
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict, cast

import click
import typer
import yaml
from pydantic import ValidationError

from agent_runtime.approval import APPROVAL_CHOICES, APPROVAL_MODES, ApprovalMode, approval_mode_hint
from agent_runtime.command_cli import commands
from agent_runtime.connection_manager import connections_command
from agent_runtime.core.llm_config import ModelProvider
from agent_runtime.core.llm_registry import UnknownModelIdError
from agent_runtime.harness import RuntimeState, TurnSnapshot
from agent_runtime.harness.ownership import ConversationBusyError
from agent_runtime.history import ChatSessionSummary, list_chat_sessions, read_chat_messages
from agent_runtime.knowledge import RAGKnowledgeConfig
from agent_runtime.local_files import parse_file_input, resolve_local_file
from agent_runtime.local_runtime import EndpointConflictError, LocalRuntimeError
from agent_runtime.mcp_cli import mcp_app
from agent_runtime.model_admin import (
    ModelAdminService,
    ModelDefinitionArguments,
    ModelMutationOutcome,
)
from agent_runtime.model_probe import ModelProbeEvidence, ProbeLevel
from agent_runtime.models import (
    ModelNotAvailableError,
    ModelPolicyError,
    ModelSpec,
    ModelSwitchRequester,
    format_model_rows,
)
from agent_runtime.result import AgentDiagnostic, AgentResult
from agent_runtime.runtime.builder import build_model_admin_service
from agent_runtime.runtime_clock import capture_runtime_clock
from agent_runtime.setup_cli import setup
from agent_runtime.terminal_input import CHAT_COMMANDS, TerminalComposer, resolve_pause_choice
from agent_runtime.terminal_render import (
    TerminalToolEventDisplay,
    format_usage_summary,
    safe_terminal_text,
    web_error_message,
)
from agent_runtime.tools.web_limits import WebLimits
from agent_runtime.web_cli import web_app
from agent_runtime.workspace import DEFAULT_CHECKPOINT_PATH, DEFAULT_MODEL_SESSION_PATH

if TYPE_CHECKING:
    from agent_runtime.agent import Agent
    from agent_runtime.harness.session import Session
    from agent_runtime.result import AgentPause
    from agent_runtime.terminal_app import TerminalChatApp
    from agent_runtime.worktrees import CreateReport

agent_app = typer.Typer(add_completion=False, no_args_is_help=True)
model_app = typer.Typer(add_completion=False, no_args_is_help=True)
model_trust_app = typer.Typer(add_completion=False, no_args_is_help=True)
worktree_app = typer.Typer(add_completion=False, no_args_is_help=True)
agent_app.add_typer(model_app, name="model", help="查看和切换当前模型会话。")
agent_app.add_typer(mcp_app, name="mcp", help="注册、认证和检查 MCP 连接。")
agent_app.add_typer(worktree_app, name="worktree", help="管理并行会话的 Git worktree。")
agent_app.add_typer(web_app, name="web", help="配置和检查搜索与网页读取。")
agent_app.command("setup")(setup)
agent_app.command("connections")(connections_command)
agent_app.command("commands")(commands)


def _open_cli_worktree(name: str | None) -> CreateReport:
    """Create the managed worktree that a new ``--worktree`` session runs in."""
    from agent_runtime.worktrees import WorktreeError, WorktreeManager

    try:
        report = WorktreeManager(Path.cwd()).create(name)
    except WorktreeError as error:
        raise typer.BadParameter(str(error)) from None
    worktree = report.worktree
    typer.echo(
        f"已在独立 worktree 中开始：分支 {worktree.branch}（基准 {worktree.base_commit[:12]}）\n"
        f"目录：{worktree.root}\n"
        f"之后继续该会话：cd {worktree.root} && agent chat --continue"
    )
    if report.uncommitted_in_source:
        typer.echo(
            f"注意：源工作区有 {report.uncommitted_in_source} 个未提交变更，它们不在 worktree 中；"
            "需要 .env 等被忽略文件时，在仓库根添加 .worktreeinclude。"
        )
    return report


@worktree_app.command("list")
def worktree_list(
    root: Annotated[
        Path | None, typer.Option("--root", envvar="PRAXIS_WORKTREE_ROOT", help="worktree 根目录"),
    ] = None,
) -> None:
    """列出托管 worktree 与状态；不调用模型或执行工具。"""
    from agent_runtime.worktrees import WorktreeError, WorktreeManager

    try:
        manager = WorktreeManager(Path.cwd(), root=root)
        rows = []
        for entry in manager.list():
            inspection = manager.inspect(entry)
            rows.append({
                "name": entry.name,
                "root": str(entry.root),
                "branch": entry.branch,
                "base_commit": entry.base_commit,
                "source_root": str(entry.source_root),
                "created_at": entry.created_at,
                "exists": inspection.exists,
                "dirty_count": inspection.dirty_count,
                "ahead_count": inspection.ahead_count,
            })
    except WorktreeError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from None
    typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))


@worktree_app.command("remove")
def worktree_remove(
    name: Annotated[str, typer.Argument(help="worktree 名称或检出路径")],
    force: Annotated[
        bool, typer.Option("--force", help="放弃未提交变更与未合并提交后删除"),
    ] = False,
    root: Annotated[
        Path | None, typer.Option("--root", envvar="PRAXIS_WORKTREE_ROOT", help="worktree 根目录"),
    ] = None,
) -> None:
    """删除托管 worktree；有未提交变更或未合并提交时需要 --force。"""
    from agent_runtime.worktrees import WorktreeError, WorktreeManager

    try:
        report = WorktreeManager(Path.cwd(), root=root).remove(name, force=force)
    except WorktreeError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from None
    typer.echo(json.dumps({
        "name": report.name,
        "root": str(report.root),
        "branch": report.branch,
        "branch_deleted": report.branch_deleted,
    }, ensure_ascii=False, indent=2))


@worktree_app.command("prune")
def worktree_prune(
    keep: Annotated[
        int, typer.Option("--keep", min=1, help="保留最近多少个托管 worktree"),
    ] = 15,
    root: Annotated[
        Path | None, typer.Option("--root", envvar="PRAXIS_WORKTREE_ROOT", help="worktree 根目录"),
    ] = None,
) -> None:
    """清理最旧的干净 worktree；脏的或有未合并提交的不会被删除。"""
    from agent_runtime.worktrees import WorktreeError, WorktreeManager

    try:
        reports = WorktreeManager(Path.cwd(), root=root).prune(keep=keep)
    except WorktreeError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from None
    typer.echo(json.dumps([
        {"name": report.name, "root": str(report.root), "branch": report.branch,
         "branch_deleted": report.branch_deleted}
        for report in reports
    ], ensure_ascii=False, indent=2))


@agent_app.command("permissions")
def agent_permissions(
    permissions: Annotated[Path | None, typer.Option("--permissions", help="权限策略 JSON/TOML 文件")] = None,
    workspace: Annotated[Path, typer.Option("--workspace")] = Path("."),
    path: Annotated[Path | None, typer.Option("--path", help="解释路径访问权限")] = None,
    write: Annotated[bool, typer.Option("--write", help="检查写入权限")] = False,
    tool: Annotated[str | None, typer.Option("--tool", help="解释工具匹配规则")] = None,
    url: Annotated[str | None, typer.Option("--url", help="解释网络目标规则")] = None,
) -> None:
    """查看有效权限、配置来源和规则决策；不调用模型或执行工具。"""
    from agent_runtime.permission_policy import resolve_permission_sources

    if sum(value is not None for value in (path, tool, url)) > 1:
        raise typer.BadParameter("--path、--tool、--url 只能选择一个")
    try:
        effective = resolve_permission_sources(workspace, permissions)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="--permissions") from None
    result = effective.explain()
    if path is not None:
        result.update(decision="allow" if effective.path_access(path, write=write) else "deny",
                      reason=effective.path_reason(path, write=write))
    elif tool is not None:
        result.update(decision=effective.tool_decision(tool) or "default",
                      reason="tool rule; actual effects and targets are checked at execution")
    elif url is not None:
        result.update(decision=effective.network_decision(url), reason="network destination rule")
    print(json.dumps(result, ensure_ascii=False, indent=2))
model_app.add_typer(model_trust_app, name="trust", help="管理冻结模型绑定的本地信任域。")
logger = logging.getLogger(__name__)


@agent_app.command("memory")
def agent_memory(
    action: Annotated[Literal["status", "configure", "update", "delete", "remember", "edit", "explain", "relocate"],
                      typer.Argument()] = "status",
    workspace: Annotated[Path, typer.Option("--workspace")] = Path("."),
    checkpoint_db: Annotated[Path, typer.Option("--checkpoint-db")] = DEFAULT_CHECKPOINT_PATH,
    use: Annotated[bool | None, typer.Option("--use/--no-use")] = None,
    generate: Annotated[bool | None, typer.Option("--generate/--no-generate")] = None,
    key: Annotated[str | None, typer.Option("--key")] = None,
    text: Annotated[str | None, typer.Option("--text")] = None,
    title: Annotated[str | None, typer.Option("--title")] = None,
    kind: Annotated[Literal["preference", "fact", "reference"] | None, typer.Option("--kind")] = None,
    expected_version: Annotated[int | None, typer.Option("--expected-version", min=0)] = None,
    from_workspace: Annotated[Path | None, typer.Option("--from-workspace")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
) -> None:
    """工作区跨聊天记忆：检查、配置、有限更新或删除；默认不开启。"""
    from agent_runtime import Agent, MemoryConfig

    agent = Agent(workspace_path=workspace, checkpoint_db=checkpoint_db, model=model,
                  model_session_path=None, enable_workspace_mcp=False)
    if action == "status":
        result = agent.memories()
    elif action == "relocate":
        if from_workspace is None:
            raise typer.BadParameter("relocate requires --from-workspace")
        try:
            result = agent.relocate_memory(from_workspace, expected_version=expected_version)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            raise typer.BadParameter(str(exc)) from None
    elif action == "configure":
        if use is None and generate is None:
            raise typer.BadParameter("configure requires --use/--no-use or --generate/--no-generate")
        with RuntimeState.open_sync(checkpoint_db) as store:
            state = store.read_memory_state(workspace)
        config = MemoryConfig(use_memories=state["use_memories"] if use is None else use,
                              generate_memories=state["generate_memories"] if generate is None else generate)
        state = agent.configure_memory(config)
        result = {"version": state["version"], **config.model_dump()}
    elif action == "delete":
        if key is None:
            raise typer.BadParameter("delete requires --key")
        try:
            state = agent.delete_memory(key)
        except KeyError:
            raise typer.BadParameter("unknown memory key") from None
        result = {"version": state["version"], "deleted_key": key}
    elif action in {"remember", "edit", "explain"}:
        if key is None:
            raise typer.BadParameter(f"{action} requires --key")
        try:
            if action == "explain":
                result = agent.explain_memory(key)
            else:
                if text is None:
                    raise typer.BadParameter(f"{action} requires --text")
                if action == "remember":
                    state = agent.remember_memory(key, text, title=title, kind=kind or "fact",
                                                  expected_version=expected_version)
                else:
                    state = agent.edit_memory(key, text, title=title, kind=kind, expected_version=expected_version)
                entries = cast(Sequence[Mapping[str, object]], state["entries"])
                result = {"version": state["version"], "entry": next(e for e in entries if e["key"] == key)}
        except (KeyError, ValueError, RuntimeError) as exc:
            raise typer.BadParameter(str(exc)) from None
    else:
        result = asyncio.run(agent.refresh_memory())
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "failed":
        raise typer.Exit(1)


@agent_app.command("history")
def agent_history(
    checkpoint_db: Annotated[Path, typer.Option("--checkpoint-db")] = DEFAULT_CHECKPOINT_PATH,
    turn_id: Annotated[str | None, typer.Option("--turn-id")] = None,
    call_id: Annotated[str | None, typer.Option("--call-id")] = None,
    item_id: Annotated[str | None, typer.Option("--item-id")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=100)] = 20,
) -> None:
    """只读历史记录：默认列出 Turn；按 Turn/调用/结果 ID 定位诊断，不调用模型。"""
    import sqlite3

    from agent_runtime.history import list_history_turns, read_history

    try:
        rows = (list_history_turns(checkpoint_db, limit=limit) if not any((turn_id, call_id, item_id)) else
                read_history(checkpoint_db, turn_id=turn_id, tool_call_id=call_id, item_id=item_id, limit=limit))
    except (OSError, ValueError, KeyError, sqlite3.Error):
        raise typer.BadParameter("历史数据库或指定记录不可读取；请检查路径和 ID。") from None
    typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))


class _ResumeOptions(TypedDict, total=False):
    files: Sequence[str]
    expected_request_id: str


class _PermissionOptions(TypedDict, total=False):
    permissions: Path
    approval_mode: ApprovalMode


def _permission_options(path: Path | None, approval_mode: ApprovalMode | None = None) -> _PermissionOptions:
    options: _PermissionOptions = {} if path is None else {"permissions": path}
    if approval_mode is not None:
        options["approval_mode"] = approval_mode
    return options


class _WebOptions(TypedDict, total=False):
    web_search_key_file: Path
    web_search_provider: str
    web_proxy_url: str
    web_limits_file: Path


@dataclass(frozen=True, slots=True)
class _CLIRuntimeBinding:
    model_id: str | None
    workspace_path: str
    knowledge: RAGKnowledgeConfig | None


@dataclass(frozen=True, slots=True)
class _CLITurn:
    turn_id: str
    status: str
    runtime: _CLIRuntimeBinding


_CLIToolEventDisplay = TerminalToolEventDisplay


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
            if tool_call.tool_name in {"web_search", "web_fetch"}:
                tool_info += f"：{web_error_message(tool_call.error_code)}"
            else:
                error_message = safe_terminal_text(tool_call.error_message or "unknown tool error")
                tool_info += f" ({tool_call.error_code or 'tool_error'}: {error_message})"
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
        return "模型服务暂时无法完成请求，请检查模型配置或稍后重试。"
    if stop_reason == "model_step_budget_exhausted":
        return "本次执行已达到尝试次数上限，本轮未完成收尾。"
    capacity_failures = {
        "model_token_budget_exhausted": "本轮累计 Token 预算不足，无法继续执行或完成收尾。",
        "model_context_window_exceeded": "模型单次请求超出上下文窗口，压缩恢复后仍无法继续。",
        "context_summary_incomplete": "上下文摘要输出被截断，原始历史已保留；本轮无法完成必要压缩。",
        "context_summary_invalid": "上下文摘要内容无效，原始历史已保留；本轮无法完成必要压缩。",
        "summary_context_overflow": "模型拒绝摘要请求：摘要输入超出可用上下文。",
        "byte_guard": "上下文达到配置的字节上限，本轮无法完成必要压缩。",
        "message_guard": "上下文达到配置的消息数上限，本轮无法完成必要压缩。",
        "context_source_changed": "上下文在压缩期间持续变化，本轮未能完成压缩。",
        "context_compaction_failed": "上下文压缩失败，原始历史已保留。",
        "no_compactable_context": "受保护的当前任务无法通过压缩历史缩小，本轮无法继续。",
        "summary_unit_exceeds_context": "一个完整的历史工作单元超出摘要输入容量，原始历史已保留。",
    }
    if stop_reason in capacity_failures:
        return capacity_failures[stop_reason]
    return "错误: Agent 运行失败"


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
    if verbose:
        if stop_reason:
            print(f"  stop_reason: {stop_reason}")
        for diagnostic in diagnostics:
            _print_diagnostic(diagnostic)


def _display_agent_result(
    result: AgentResult,
    *,
    verbose: bool,
    answer_streamed: bool = False,
    show_execution: bool = True,
    show_usage: bool = True,
    show_outcome: bool = True,
) -> None:
    if result.status == "failed" and show_outcome:
        _display_failure(
            stop_reason=result.stop_reason,
            diagnostics=result.diagnostics,
            verbose=verbose,
        )
    elif result.diagnostics and show_outcome:
        degraded = sum(1 for diagnostic in result.diagnostics if diagnostic.degraded)
        if degraded:
            print(f"\n警告: Agent 以降级模式运行（{degraded} 项诊断）")
        if verbose:
            for diagnostic in result.diagnostics:
                _print_diagnostic(diagnostic)

    if result.answer and not answer_streamed:
        print(f"\n{safe_terminal_text(result.answer)}")

    if result.status == "done" and show_outcome:
        scope = (
            f"（范围: {safe_terminal_text(result.validation_scope)}）"
            if result.validation_scope is not None else ""
        )
        print(f"\n验证状态: {result.validation_status}{scope}")
        if result.validation_status == "not_evaluated":
            print("本轮运行已结束；任务验收尚未评估。")
        elif result.validation_scope == "workspace_change":
            print("此状态仅表示工作区修改及修改后的验证检查。")
        elif result.validation_scope == "workspace_change_recorded":
            print("仅确认工作区发生了实际修改；任务验收尚未评估。")
        elif result.validation_scope == "acceptance_command":
            print("仅确认用户指定的验收命令执行成功。")

    if result.status == "paused" and result.pause is None and result.needs_user_input:
        print(f"\n暂停原因: {result.needs_user_input}")

    plan_summary = _format_plan_summary(result)
    if plan_summary and show_execution:
        print(plan_summary)
    if result.tool_calls and show_execution:
        print(_format_tool_summary(result))

    if show_usage:
        print("\n" + format_usage_summary(result.usage))

    if verbose:
        usage = result.usage
        def milliseconds(value: float | None) -> str:
            return "unknown" if value is None else f"{value:.0f}ms"

        print(
            "\n调用统计: "
            f"native={usage.native_calls}"
            f"({usage.native_errors}err/"
            f"{milliseconds(usage.native_latency_ms_total)}) "
            f"deferred={usage.deferred_calls if usage.deferred_calls is not None else 'unknown'} "
            f"mcp={usage.mcp_calls}"
            f"({usage.mcp_errors}err/"
            f"{milliseconds(usage.mcp_latency_ms_total)})"
        )
        print(
            "耗时: "
            f"total={milliseconds(usage.latency_ms)} "
            f"startup={milliseconds(usage.startup_ms)} "
            f"build={milliseconds(usage.build_service_ms)} "
            f"model_ready={milliseconds(usage.model_ready_ms)} "
            f"model={milliseconds(usage.model_latency_ms)} "
            f"tool={milliseconds(usage.tool_latency_ms)} "
            f"finalize={milliseconds(usage.finalize_latency_ms)} "
            f"prompt_bytes={usage.prompt_bytes if usage.prompt_bytes is not None else 'unknown'} "
            f"tool_schema_bytes={usage.tool_schema_bytes if usage.tool_schema_bytes is not None else 'unknown'}"
        )
        if result.evidence:
            print(f"证据: {len(result.evidence)} 条")
        if result.stop_reason:
            print(f"停止原因: {result.stop_reason}")

    if show_execution or verbose:
        print(f"Turn: {result.turn_id}")

    if verbose:
        print(f"状态: {result.status}")


async def _cancel_ui_execution(driver: Agent | Session, ui: TerminalChatApp, turn_id: str | None) -> AgentResult | None:
    ui.display.stopping()
    if turn_id is None:
        ui.display.interrupted()
        return None
    try:
        result: AgentResult = await ui.operation(driver.resume(turn_id, "abort", event_sink=ui.display))
    except BaseException:
        ui.display.interrupted()
        raise
    ui.display.present_result(result)
    ui.display.finish()
    return result


async def _handle_ui_pause(result: AgentResult, ui: TerminalChatApp) -> str | None:
    req = result.pause
    if req is None:
        return None
    if req.kind == "tool_approval":
        return await ui.choose_approval(req)
    print(f"\n⏸ 需要确认: {req.question}")
    for call in req.tool_calls:
        print(f"  {call.tool_name}: {call.args_preview}")
        if call.reason:
            print(f"  原因: {call.reason}")
    options = req.options
    text_answer = req.kind in {"clarification", "choice"}
    if not options and not text_answer:
        print("此暂停没有提供可选操作，已取消当前执行。")
        return None
    print(f"选项: {', '.join(options)}" if options else "输入你的回答；q 取消。")
    while True:
        choice = (await ui.prompt(approval=True)).strip()
        if choice.casefold() in {"q", "exit", "/exit"}:
            return None
        if text_answer:
            resolved = choice if choice and (not options or choice in options) else None
        else:
            resolved = resolve_pause_choice(choice, options)
        if resolved is not None:
            return resolved
        print(f"请输入: {', '.join(options)}；q 取消")


async def _handle_cli_pause(
    result: AgentResult, *, approval_mode: ApprovalMode | None = None,
    set_approval_mode: Callable[[ApprovalMode], None] | None = None,
) -> str | None:
    """TTY approval uses a selector; plain input remains available for pipes."""
    request = result.pause
    if request is not None and request.kind == "tool_approval" and request.options and _is_interactive_terminal():
        from prompt_toolkit.shortcuts import radiolist_dialog

        details = [request.question]
        if approval_mode is not None:
            details.insert(0, approval_mode_hint(approval_mode))
        details.extend(f"{call.tool_name}: {call.args_preview}" for call in request.tool_calls)
        scope = request.context.get("session_grant_label")
        if isinstance(scope, str):
            details.append("本会话授权范围：" + scope)
        values = [(choice, APPROVAL_CHOICES.get(choice, choice)) for choice in request.options]
        if set_approval_mode is not None:
            values.append(("permissions", APPROVAL_CHOICES["permissions"]))
        while True:
            action = await radiolist_dialog(
                title="操作审批", text=safe_terminal_text("\n".join(details)), values=values,
                default="deny" if "deny" in request.options else request.options[0],
                ok_text="确认", cancel_text="取消",
            ).run_async()
            if action != "permissions":
                return action
            selected = await radiolist_dialog(
                title="选择审批模式", text="应用到当前任务并按新策略重新检查；明确禁令和沙箱仍有效。",
                values=list(APPROVAL_MODES.items()), default=approval_mode,
                ok_text="应用", cancel_text="返回审批",
            ).run_async()
            if selected is not None and selected != approval_mode and set_approval_mode is not None:
                set_approval_mode(cast(ApprovalMode, selected))
                return "recheck"
    return _handle_pause(result)


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
    grant_label = req.context.get("session_grant_label")
    if "allow_session" in req.options and isinstance(grant_label, str):
        print("  本会话授权范围: " + safe_terminal_text(grant_label))

    options = req.options
    text_answer = req.kind in {"clarification", "choice"}
    if not options and not text_answer:
        print("此暂停没有提供可选操作，已取消当前执行。")
        return None
    if req.kind == "tool_approval":
        for number, option in enumerate(options, 1):
            print(f"  {number}. {APPROVAL_CHOICES.get(option, option)}")
    else:
        print(f"  选项: {', '.join(options)}" if options else "输入你的回答；q 取消。")

    while True:
        choice = input("> ").strip()
        if choice.casefold() in {"q", "exit", "/exit"}:
            return None
        if text_answer:
            resolved = choice if choice and (not options or choice in options) else None
        elif choice.isdecimal() and 1 <= int(choice) <= len(options):
            resolved = options[int(choice) - 1]
        else:
            resolved = resolve_pause_choice(choice, options)
        if resolved is not None:
            return resolved
        print("  请选择选项编号；q 取消。" if not text_answer else "  请输入你的回答。")


def _display_pending_recovery(
    request: AgentPause | None,
    *,
    turn_id: str,
    checkpoint_db: Path,
    recovery: Mapping[str, Any],
) -> None:
    print(f"\n⏸  待恢复 Turn: {turn_id}")
    if request is None:
        print("   未检测到待处理的人机请求。")
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
    options = recovery["actions"]
    print(f"   恢复依据: {recovery['reason_code']}")
    if recovery["evidence_ids"]:
        print("   记录: " + ", ".join(recovery["evidence_ids"]))
    if not options:
        print("   当前没有可直接执行的恢复动作；请先核对上述记录。")
        return
    print("   可用恢复命令:")
    for option in options:
        command = [
            "agent",
            "recover",
            turn_id,
            "--action",
            option,
            "--checkpoint-db",
            str(checkpoint_db),
        ]
        if recovery["request_id"] is not None:
            command.extend(["--request-id", recovery["request_id"]])
        print(f"     {shlex.join(command)}")
    if recovery["requires_input"]:
        if request is not None and request.kind == "choice":
            print("   可选输入: " + ", ".join(recovery["choices"]))
        print("   需要文本时，在 continue 命令后添加 --input '<text>'。")


def _load_web_limits(path: Path | None) -> WebLimits:
    if path is None:
        return WebLimits()
    try:
        with path.open("rb") as handle:
            data = handle.read(4097)
        if len(data) > 4096:
            raise ValueError("configuration exceeds 4096 bytes")
        return WebLimits.model_validate_json(data)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"无效网页预算配置: {error}", param_hint="--web-limits") from None


def _create_agent_facade(
    *,
    model: str | None = None,
    checkpoint_db: Path | None = None,
    workspace_path: Path | str | None = None,
    model_session_path: Path | None = None,
    knowledge: RAGKnowledgeConfig | None = None,
    enable_workspace_mcp: bool = True,
    web_search_key_file: Path | None = None,
    web_search_provider: str | None = None,
    web_proxy_url: str | None = None,
    web_limits_file: Path | None = None,
    permissions: Path | None = None,
    approval_mode: ApprovalMode | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    _selection_requester: ModelSwitchRequester = "system",
) -> Agent:
    from agent_runtime.agent import Agent

    try:
        return Agent(
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            checkpoint_db=checkpoint_db,
            workspace_path=workspace_path,
            model_session_path=model_session_path,
            knowledge=knowledge,
            enable_workspace_mcp=enable_workspace_mcp,
            web_search_key_file=web_search_key_file,
            web_search_provider=web_search_provider,
            web_proxy_url=web_proxy_url,
            web_limits=_load_web_limits(web_limits_file),
            **_permission_options(permissions, approval_mode),
            _selection_requester=_selection_requester,
        )
    except ValueError as error:
        if str(error).startswith("Web proxy must"):
            raise typer.BadParameter(str(error), param_hint="--web-proxy / proxy environment") from None
        if str(error).startswith("Web search"):
            raise typer.BadParameter(str(error), param_hint="--web-search-provider / --web-search-key-file") from None
        if reasoning_effort is not None or max_output_tokens is not None:
            raise typer.BadParameter(str(error), param_hint="--effort / --max-output-tokens") from None
        raise


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
    from agent_runtime.harness.tool_orchestrator import ToolResultCommitError

    try:
        return asyncio.run(awaitable)
    except ToolResultCommitError as exc:
        diagnostic = exc.result.metadata.get("diagnostic", {})
        detail = json.dumps(diagnostic, ensure_ascii=False, default=dict)
        raise click.ClickException(f"工具结果提交失败，未确认持久化；请勿自动重试。诊断: {detail}") from None
    except EndpointConflictError as exc:
        logger.debug("Local model endpoint conflict", exc_info=True)
        raise click.ClickException("本地模型端口已被其他服务占用，请检查模型服务配置。") from exc
    except LocalRuntimeError as exc:
        logger.debug("Local model provider unavailable", exc_info=True)
        raise click.ClickException("本地模型服务尚未就绪，请先启动模型服务，或切换到已配置的云端模型。") from exc
    except (
        FileNotFoundError,
        KeyError,
        ModelNotAvailableError,
        ModelPolicyError,
        UnknownModelIdError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc


def _resolve_cli_input_files(values: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    for value in values:
        try:
            path = resolve_local_file(value, base=Path.cwd(), resolve_symlinks=False)
        except (OSError, ValueError) as error:
            raise typer.BadParameter(str(error), param_hint="--file") from error
        resolved.append(str(path))
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
    acceptance_command: str | None = None,
    allow_write_tools: bool = False,
    allow_execute_tools: bool = False,
    allow_web_tools: bool = False,
    event_display: _CLIToolEventDisplay | None = None,
) -> AgentResult:
    """Run one Turn and optionally drive its approval lifecycle."""
    display = event_display or _CLIToolEventDisplay()
    try:
        display.begin_turn()
        result = await facade.run(
            task,
            previous_turn_id=previous_turn_id,
            files=files,
            max_turns=max_turns,
            max_tokens_total=max_tokens_total,
            require_workspace_change=require_workspace_change,
            acceptance_command=acceptance_command,
            allow_write_tools=allow_write_tools,
            allow_execute_tools=allow_execute_tools,
            allow_web_tools=allow_web_tools,
            event_sink=display,
        )
        while result.status == "paused" and interactive_approval:
            display.finish()
            action = await _handle_cli_pause(
                result, approval_mode=getattr(facade, "approval_mode", None) or "auto",
                set_approval_mode=lambda mode: setattr(facade, "approval_mode", mode),
            )
            if action is None:
                break
            display.begin_turn()
            if result.pause is not None and result.pause.kind in {"clarification", "choice"}:
                result = await facade.resume(
                    result.turn_id, "continue", user_input=action, event_sink=display,
                )
            else:
                result = await facade.resume(
                    result.turn_id, action, event_sink=display,
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
    expected_request_id: str | None = None,
) -> AgentResult:
    display = event_display or _CLIToolEventDisplay()
    display.begin_turn()
    try:
        request_options: _ResumeOptions = (
            {} if expected_request_id is None else {"expected_request_id": expected_request_id}
        )
        return await facade.resume(
            turn_id,
            action,
            user_input=user_input,
            event_sink=display,
            **request_options,
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
        return await facade.pending_input(turn_id)
    finally:
        display.finish()


def _print_startup_banner(model_id: str) -> None:
    print(f"Agent 就绪 (模型: {model_id})")
    print("输入查询，或输入 /help 查看交互命令。")
    print()


def _print_chat_help() -> None:
    print("交互命令:")
    for command, description in CHAT_COMMANDS.items():
        print(f"  {command:<18}{description}")
    print("  /model             交互终端内用 ↑↓ 选择模型，Enter 确认，Esc 取消")
    print("  输入 / 自动显示命令；继续输入前缀筛选；↑↓ 选择，Tab 补全，Enter 确认候选后再按一次执行")
    print("  执行记录：Ctrl+O 切换最近记录；Tab 进入记录，方向键移动，Enter 切换")
    print("  Ctrl+J / Alt+Enter 换行；PageUp/PageDown 滚动记录；Ctrl+End 回到底部；Ctrl+C 取消")
    print("  默认使用终端原生拖选、复制和粘贴（macOS：⌘C / ⌘V）；多行粘贴后按 Enter 发送")
    print("  F2 可开启鼠标交互，再按 F2 恢复终端原生选择")


def _chat_session_label(session: ChatSessionSummary) -> str:
    timestamp = datetime.fromtimestamp(session.updated_at_ms / 1000).astimezone().strftime("%Y-%m-%d %H:%M")
    state = {"completed": "已完成", "failed": "失败", "cancelled": "已取消",
             "running": "处理中", "paused": "暂停", "interrupted": "中断"}.get(session.status, session.status)
    return safe_terminal_text(f"{timestamp} · {state} · {session.title} · {session.thread_id}")


async def _select_chat_session(
    database: Path, workspace: Path, *, selector: str, ui: TerminalChatApp | None,
) -> ChatSessionSummary | None:
    try:
        sessions = await asyncio.to_thread(list_chat_sessions, database, workspace=workspace)
    except (OSError, ValueError, sqlite3.Error):
        raise click.ClickException("会话数据库不可读取；请检查路径。") from None
    if selector:
        selected = next((session for session in sessions if session.thread_id == selector), None)
        if selected is None:
            raise click.ClickException("当前工作区未找到这个会话 ID；使用 /resume 选择会话。")
        return selected
    if not sessions:
        print("当前工作区没有已保存的聊天。")
        return None
    print(f"打开旧聊天 · 工作区 {safe_terminal_text(str(workspace))}")
    labels = [_chat_session_label(session) for session in sessions]
    if ui:
        selected_label = await ui.choose_menu(labels, title="打开旧聊天")
        return None if selected_label is None else sessions[labels.index(selected_label)]
    for index, label in enumerate(labels, 1):
        print(f"  {index}. {label}")
    while True:
        try:
            choice = (await asyncio.to_thread(input, "会话编号或完整 ID（空行取消）: ")).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not choice:
            return None
        if choice.isdecimal() and 1 <= int(choice) <= len(sessions):
            return sessions[int(choice) - 1]
        selected = next((session for session in sessions if session.thread_id == choice), None)
        if selected is not None:
            return selected
        print("请选择列表中的编号或完整会话 ID。")


def _print_chat_recovery(session: ChatSessionSummary, pending: AgentResult | None = None) -> None:
    print(f"会话状态: {_chat_session_label(session)}")
    if session.active_turn_id:
        if pending is not None and pending.pause is not None:
            request = pending.pause
            if request.kind == "tool_reconciliation":
                print("上次操作的执行结果尚未确认，任务保持暂停。请先核对执行结果；/new 可开始其他聊天。")
                return
            print("上一轮等待你的回复: " + safe_terminal_text(request.question))
            for call in request.tool_calls:
                print(safe_terminal_text(f"  {call.tool_name}: {call.args_preview}"))
            if request.options:
                if request.kind == "tool_approval":
                    print("可选择: " + " / ".join(APPROVAL_CHOICES.get(choice, choice)
                                               for choice in request.options))
                    print("继续后会显示审批选择面板；/new 开始新聊天。")
                else:
                    print("可直接回复: " + ", ".join(request.options))
            else:
                print("直接输入你的回答即可继续；/new 开始新聊天。")
        else:
            print("历史已打开。直接发送消息可继续中断的任务；若原进程仍在执行，会提示回到原终端。")


async def _chat_facade_loop(
    facade: Agent,
    *,
    max_tokens_total: int | None,
    max_turns: int | None = None,
    previous_turn_id: str | None = None,
    resume_thread_id: str | None = None,
    allow_write_tools: bool = False,
    allow_execute_tools: bool = False,
    allow_web_tools: bool = False,
    ui_mode: Literal["inline", "fullscreen"] = "fullscreen",
    _ui: TerminalChatApp | None = None,
) -> None:
    if _ui is None and _is_tty(sys.stdin) and _is_tty(sys.stdout) and os.environ.get("TERM") != "dumb":
        from agent_runtime.terminal_app import TerminalChatApp

        initial_model = "历史聊天" if resume_thread_id is not None else facade.current_model().id
        ui = TerminalChatApp(model=initial_model, workspace=str(facade.workspace_path or Path.cwd()), ui_mode=ui_mode)
        await ui.run(lambda: _chat_facade_loop(
            facade, max_tokens_total=max_tokens_total, max_turns=max_turns,
            previous_turn_id=previous_turn_id, allow_write_tools=allow_write_tools,
            resume_thread_id=resume_thread_id,
            allow_execute_tools=allow_execute_tools, allow_web_tools=allow_web_tools, _ui=ui,
        ))
        return
    event_display = _ui.display if _ui else _CLIToolEventDisplay(interactive=False)
    composer = TerminalComposer() if _ui is None else None
    current_turn_id = previous_turn_id
    last_result: AgentResult | None = None
    last_answer = ""
    verbose = False
    default_chat_workspace = facade.workspace_path or Path.cwd()
    startup_approval_mode = getattr(facade, "approval_mode", None)
    chat_approval_mode = startup_approval_mode
    if chat_approval_mode is None and resume_thread_id is None and previous_turn_id is None:
        chat_approval_mode = "auto"
    if _ui:
        _ui.approval_mode = chat_approval_mode or "历史权限待加载"
        _ui.display.search_configuration_required = getattr(facade, "web_search_provider", False) is None
    pending_files: list[str] = []
    default_chat_knowledge = getattr(facade, "knowledge", None)
    chat_workspace = default_chat_workspace
    model_id = "历史聊天" if resume_thread_id is not None else facade.current_model().id
    if _ui is None:
        _print_startup_banner(model_id)
        if getattr(facade, "web_search_provider", False) is None:
            print("联网搜索未启用；输入 /web 设置搜索（隐藏输入 Key，无需退出聊天）。")
    from contextlib import AsyncExitStack

    async with AsyncExitStack() as stack:
        stack.callback(event_display.finish)
        session: Session | None = None
        selected_chat: ChatSessionSummary | None = None

        def refresh_request_status(control: Agent | Session) -> None:
            context = getattr(control, "tool_execution_context", None)
            if _ui and context is not None:
                _ui.approval_mode = context.approval_mode
            reader = getattr(control, "request_settings", None)
            if callable(reader):
                event_display.update_request_settings(reader())

        if resume_thread_id is None:
            refresh_request_status(facade)

        async def reopen(selector: str) -> bool:
            nonlocal session, selected_chat, current_turn_id, last_result, last_answer, model_id
            nonlocal chat_approval_mode
            database = facade.checkpoint_db or DEFAULT_CHECKPOINT_PATH
            selected = await _select_chat_session(
                database, Path(default_chat_workspace), selector=selector, ui=_ui,
            )
            if selected is None:
                return False
            messages = await asyncio.to_thread(read_chat_messages,
                database, thread_id=selected.thread_id, workspace=Path(default_chat_workspace),
            )
            await stack.aclose()
            stack.callback(event_display.finish)
            session = None
            selected_chat = selected
            chat_approval_mode = startup_approval_mode
            facade.approval_mode = startup_approval_mode
            current_turn_id = selected.head_turn_id
            last_result = None
            last_answer = next((text for kind, text in reversed(messages) if kind == "answer"), "")
            event_display.clear_context_status()
            pending_files.clear()
            model_id = selected.model_id or "未选择模型"
            facade.model = selected.model_id
            facade._followup_model_id = None
            if _ui:
                _ui.model = model_id
                _ui.approval_mode = "历史权限待加载"
                await _ui.replay_messages(messages)
            else:
                for kind, text in messages:
                    prefix = "❯ " if kind == "user" else "Praxis\n" if kind == "answer" else ""
                    print(prefix + safe_terminal_text(text) + "\n")
            pending = (await facade.read_result(selected.active_turn_id)
                       if selected.active_turn_id and selected.status in {"paused", "interrupted"} else None)
            _print_chat_recovery(selected, pending)
            return True

        async def open_session(*, frozen_turn_id: str | None = None) -> Session:
            nonlocal session, model_id, previous_turn_id
            changed = getattr(session, "connections_changed", None)
            if session is not None and callable(changed) and changed():
                facade.model = session.current_model().id
                previous_turn_id = current_turn_id
                await session.close()
                session = None
            if session is not None:
                return session
            if selected_chat is not None:
                anchor = selected_chat.head_turn_id
                if anchor is not None:
                    metadata = await asyncio.to_thread(
                        _cli_turn, facade.checkpoint_db or DEFAULT_CHECKPOINT_PATH, anchor,
                    )
                    facade.knowledge = metadata.runtime.knowledge
            opened = await stack.enter_async_context(facade.session(
                **({"_frozen_turn_id": frozen_turn_id} if frozen_turn_id is not None else {
                    "previous_turn_id": previous_turn_id if selected_chat is None else None,
                    "thread_id": selected_chat.thread_id if selected_chat is not None else None,
                }),
                max_turns=max_turns, max_tokens_total=max_tokens_total,
                require_workspace_change=False, allow_write_tools=allow_write_tools,
                allow_execute_tools=allow_execute_tools, allow_web_tools=allow_web_tools,
                **({"approval_mode": chat_approval_mode} if chat_approval_mode is not None else {}),
            ))
            session = opened
            model_id = opened.current_model().id
            if _ui:
                _ui.model = model_id
            refresh_request_status(opened)
            return opened

        async def change_request_settings(query: str) -> None:
            query = query.strip()
            control = session or facade
            if _ui and query == "/effort":
                capability = control.current_model().reasoning
                choices = ["default"]
                if capability is not None:
                    choices.extend(capability.supported_efforts or ())
                    if capability.thinking_toggle is not None:
                        choices.append("off")
                if len(choices) == 1:
                    print("当前模型未声明可选择的思考强度；设置未更改。")
                    return
                current = control.request_settings()["preferences"]["reasoning_effort"] or "default"
                selected = await _ui.choose_menu(choices, title="选择思考强度", current=current)
                if selected is None:
                    return
                query = f"/effort {selected}"
            _handle_request_settings_command(query, agent=control)
            refresh_request_status(control)

        def set_approval_mode(mode: str) -> bool:
            nonlocal chat_approval_mode, allow_write_tools, allow_execute_tools, allow_web_tools
            if mode not in APPROVAL_MODES:
                raise ValueError("审批模式必须为 ask、auto 或 full")
            previous = session.tool_execution_context.approval_mode if session else chat_approval_mode
            if session is not None:
                session.update_tool_policy(approval_mode=cast(ApprovalMode, mode),
                                           allow_write_tools=False, allow_execute_tools=False,
                                           allow_web_tools=False, auto_approve_sandboxed=False)
            chat_approval_mode = cast(ApprovalMode, mode)
            facade.approval_mode = chat_approval_mode
            if _ui:
                _ui.approval_mode = chat_approval_mode
            allow_write_tools = allow_execute_tools = allow_web_tools = False
            print("审批模式: " + APPROVAL_MODES[mode])
            return mode != previous

        async def choose_approval_mode() -> bool:
            current = session.tool_execution_context.approval_mode if session else chat_approval_mode or "ask"
            if _ui:
                choices = list(APPROVAL_MODES.values())
                chosen = await _ui.choose_menu(choices, title="选择审批模式", current=APPROVAL_MODES[current])
                if chosen is None:
                    return False
                return set_approval_mode(list(APPROVAL_MODES)[choices.index(chosen)])
            for key, label in APPROVAL_MODES.items():
                print(f"  {key}: {label}")
            chosen = (await asyncio.to_thread(input, "审批模式 ask/auto/full（空行取消）: ")).strip()
            return set_approval_mode(chosen) if chosen else False

        if _ui is not None:
            _ui.on_permissions = choose_approval_mode
            async def receive_input(text: str) -> str | None:
                assert _ui is not None
                if text.split(maxsplit=1)[0] in {"/effort", "/output"}:
                    await change_request_settings(text)
                    return None
                turn_id = _ui.display.turn_id
                if turn_id is None:
                    raise RuntimeError("当前 Turn 尚未启动，请保留输入稍后发送")
                receiver = session if session is not None else facade
                guidance, files = parse_file_input(text, base=Path(default_chat_workspace))
                if files:
                    if session is None:
                        raise RuntimeError("当前操作尚无可接收附件的会话；输入已保留")
                    receipt = await session.steer(guidance or "补充输入文件，请结合当前任务读取。",
                                                  expected_turn_id=turn_id, files=files)
                else:
                    receipt = await receiver.steer(text, expected_turn_id=turn_id)
                return receipt.item_id

            _ui.on_input = receive_input
            _ui.input_status = lambda identity: facade.input_status(identity)

        if resume_thread_id is not None and not await reopen(resume_thread_id):
            return
        while True:
            try:
                if _ui:
                    query = (await _ui.prompt()).strip()
                else:
                    assert composer is not None
                    query = (await asyncio.to_thread(composer.prompt, "> ")).strip()
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
            if query == "/web":
                from agent_runtime.web_cli import configure_search

                if ((last_result is not None and last_result.status in {"paused", "interrupted"})
                        or (selected_chat is not None and selected_chat.active_turn_id
                            and selected_chat.status in {"paused", "interrupted"})):
                    print("请先处理暂停的任务，或用 /new 开始新聊天后设置搜索。")
                    continue
                if _ui is None and not _is_tty(sys.stdin):
                    print("搜索设置需要交互终端隐藏输入 Key；请在终端运行 agent setup。")
                    continue
                try:
                    current_model = (session or facade).current_model()
                    provider = facade.web_search_provider or "deepseek"
                    key_file = facade.web_search_key_file
                    configuration = (await _ui.setup_search(current_model, provider=provider, key_file=key_file)
                                     if _ui else await asyncio.to_thread(
                                         configure_search, workspace=Path(default_chat_workspace), model=current_model,
                                         provider=provider, key_file=key_file,
                                     ))
                    # Tool definitions belong to the next Turn. Reopen idle
                    # resources while keeping the transcript anchor and model.
                    if session is not None:
                        facade.model = current_model.id
                        facade._followup_model_id = None
                        previous_turn_id = current_turn_id
                        await session.close()
                        session = None
                    facade.web_search_provider = configuration.provider
                    facade.web_search_key_file = configuration.key_file
                    if _ui:
                        _ui.display.search_configuration_required = configuration.provider is None
                        _ui.invalidate()
                    print(f"搜索已配置：{configuration.provider}；下一条消息即可使用，无需重启。")
                except (EOFError, KeyboardInterrupt, click.Abort):
                    print("已取消搜索设置，当前聊天未更改。")
                except (OSError, ValueError) as error:
                    from agent_runtime.core.runtime_diagnostics import redact_sensitive_text

                    print("搜索设置未完成：" + safe_terminal_text(redact_sensitive_text(error)))
                continue
            if query == "/attach" or query.startswith("/attach "):
                raw = query.removeprefix("/attach").strip()
                if raw:
                    remaining, files = parse_file_input(raw, base=Path(default_chat_workspace))
                    if remaining or not files:
                        print("附加失败：请输入存在的本地文件路径；含空格的多个路径请分别加引号。")
                        continue
                    pending_files[:] = list(dict.fromkeys((*pending_files, *files)))
                print("待发送附件（原地只读，不复制）：" + ("\n" + "\n".join(pending_files) if pending_files else "无"))
                continue
            if query == "/detach":
                pending_files.clear()
                print("已清除待发送附件。")
                continue
            if query == "/permissions" or query.startswith("/permissions "):
                from agent_runtime.permission_policy import PermissionPolicy

                parts = query.split(maxsplit=2)
                try:
                    mode = None
                    if query == "/permissions":
                        await choose_approval_mode()
                        continue
                    elif len(parts) == 3 and parts[1] == "mode":
                        mode = parts[2]
                    if mode is not None:
                        set_approval_mode(mode)
                        continue
                    if query == "/permissions grants":
                        grants = session.approval_grants() if session is not None else ()
                        print("当前会话授权：" + ("\n" + "\n".join(safe_terminal_text(grant.label) for grant in grants)
                                             if grants else "无"))
                        continue
                    if query == "/permissions revoke":
                        if session is not None:
                            session.revoke_approval_grants()
                        print("已撤销当前会话的全部命令授权。")
                        continue
                    if len(parts) == 3 and parts[1] == "load":
                        policy_path = Path(parts[2]).expanduser()
                        if session is not None:
                            session.update_permissions(policy_path)
                        else:
                            old = facade.permissions
                            facade.permissions = policy_path
                            try:
                                facade.effective_permissions()
                            except BaseException:
                                facade.permissions = old
                                raise
                        facade.permissions = policy_path
                    elif len(parts) == 3 and parts[1] == "profile":
                        policy = PermissionPolicy.model_validate({"profile": parts[2]})
                        if session is not None:
                            session.update_permissions(policy)
                        facade.permissions = policy
                    elif len(parts) == 2 and parts[1] == "reload":
                        if session is not None:
                            session.reload_permissions()
                    elif query != "/permissions show":
                        print("用法: /permissions [mode ask|auto|full | grants | revoke | show | "
                              "load <文件> | profile <名称> | reload]")
                        continue
                    effective = session.permissions() if session is not None else facade.effective_permissions()
                    print(json.dumps(effective.explain(), ensure_ascii=False, indent=2))
                except (OSError, ValueError) as error:
                    print("权限配置失败: " + safe_terminal_text(str(error)))
                continue
            if query == "/rewind" or query.startswith("/rewind "):
                argument = query.removeprefix("/rewind").strip()
                current_thread = (selected_chat.thread_id if selected_chat
                                  else getattr(session, "thread_id", None))
                try:
                    from agent_runtime.checkpoints import CheckpointError, _numeric

                    rows = await facade.list_checkpoints(thread_id=current_thread)
                    if not rows:
                        print("当前会话还没有检查点。首次修改工作区文件时会自动建立。")
                        continue
                    trigger_labels = {"pre_write": "写前", "restore": "回滚前"}
                    labels = [
                        "{time}  {trigger}  变更 {count} 个文件  {note}".format(
                            time=time.strftime("%m-%d %H:%M:%S", time.localtime(_numeric(row.get("wall_time")))),
                            trigger=trigger_labels.get(str(row["trigger"]), str(row["trigger"])),
                            count=row.get("changed_file_count"),
                            note=safe_terminal_text(str(row["turn_input"] or row["turn_id"])),
                        )
                        for row in rows
                    ]
                    selected_row: dict[str, object] | None = None
                    if argument:
                        if argument.isdigit():
                            index = int(argument)
                            selected_row = rows[index - 1] if 1 <= index <= len(rows) else None
                        else:
                            matches = [row for row in rows
                                       if str(row["checkpoint_id"]).startswith(argument)]
                            if len(matches) > 1:
                                print(f"检查点前缀匹配多个结果（{len(matches)} 个），请输入更长的前缀。")
                                continue
                            selected_row = matches[0] if len(matches) == 1 else None
                        if selected_row is None:
                            print("没有对应的检查点；/rewind 查看列表。")
                            continue
                    elif _ui:
                        chosen = await _ui.choose_menu(labels, title="选择要回滚的检查点（最新在前）")
                        if chosen is None:
                            continue
                        selected_row = rows[labels.index(chosen)]
                    else:
                        print("检查点（最新在前）：")
                        for index, label in enumerate(labels, start=1):
                            print(f"  {index:>3}. {label}")
                        answer = (await asyncio.to_thread(input, "输入序号回滚（空行取消）: ")).strip()
                        if not answer or not answer.isdigit() or not 1 <= int(answer) <= len(rows):
                            continue
                        selected_row = rows[int(answer) - 1]
                    if _ui:
                        confirmed = await _ui.choose_menu(
                            ["确认回滚", "取消"], title="回滚前会先把当前状态保存为检查点",
                        )
                        if confirmed != "确认回滚":
                            continue
                    else:
                        confirm = (await asyncio.to_thread(
                            input,
                            "回滚会把工作区文件恢复到该检查点（之后新增的文件会被删除；"
                            "当前状态会先保存为检查点）。确认回滚？[y/N] ",
                        )).strip()
                        if confirm.casefold() not in {"y", "yes"}:
                            continue
                    rewind_result = await facade.rewind(str(selected_row["checkpoint_id"]))
                    restored = [str(path) for path in cast("list[object]", rewind_result.get("restored", ()))]
                    deleted = [str(path) for path in cast("list[object]", rewind_result.get("deleted", ()))]
                    print(f"已回滚到 {str(rewind_result.get('target_checkpoint_id', ''))[:12]}："
                          f"还原 {len(restored)} 个文件，删除 {len(deleted)} 个文件。")
                    anchor = str(rewind_result.get("restore_anchor_id") or "")
                    if anchor:
                        print(f"回滚前状态已保存为检查点 {anchor[:12]}；再次 /rewind 可撤销本次回滚。")
                    for path in restored[:30]:
                        print(f"  还原 {safe_terminal_text(path)}")
                    for path in deleted[:30]:
                        print(f"  删除 {safe_terminal_text(path)}")
                except (CheckpointError, OSError, RuntimeError) as error:
                    print("回滚失败: " + safe_terminal_text(str(error)))
                continue
            if query == "/commit" or query.startswith("/commit "):
                message = query.removeprefix("/commit").strip()
                try:
                    from agent_runtime.git_delivery import DeliveryError, DeliveryInterruptedError

                    status = await facade.commit_status()
                    changed = cast("list[object]", status.get("changed", ()))
                    if not changed:
                        print("没有可提交的变更。")
                        continue
                    print(f"当前分支 {status.get('branch')}，待提交 {len(changed)} 项：")
                    for entry in changed[:20]:
                        print(f"  {safe_terminal_text(str(entry))}")
                    if len(changed) > 20:
                        print(f"  …另有 {len(changed) - 20} 项")
                    if not message:
                        if _ui:
                            print("请输入 /commit <提交说明> 后重试。")
                            continue
                        message = (await asyncio.to_thread(input, "提交说明: ")).strip()
                        if not message:
                            continue
                    if _ui:
                        confirmed = await _ui.choose_menu(
                            ["确认提交", "取消"],
                            title="提交会包含上面列出的全部变更",
                        )
                        if confirmed != "确认提交":
                            continue
                    else:
                        confirm = (await asyncio.to_thread(
                            input, "确认提交以上变更到 praxis/* 分支？[y/N] ",
                        )).strip()
                        if confirm.casefold() not in {"y", "yes"}:
                            continue
                    delivery = await facade.commit(message, expected_preview_id=str(status["preview_id"]))
                    print(f"已提交 {str(delivery.get('commit_id'))[:12]} 到分支 "
                          f"{delivery.get('branch')}（{delivery.get('changed_file_count')} 个文件）。")
                except DeliveryInterruptedError as error:
                    print(safe_terminal_text(str(error)))
                except (DeliveryError, OSError, RuntimeError) as error:
                    print("提交失败: " + safe_terminal_text(str(error)))
                continue
            if query == "/pull" or query.startswith("/pull "):
                title = query.removeprefix("/pull").strip()
                try:
                    from agent_runtime.checkpoints import _numeric
                    from agent_runtime.git_delivery import DeliveryError, DeliveryInterruptedError

                    pr_status = await facade.pull_request_status()
                    subjects = cast("list[object]", pr_status.get("commit_subjects", ()))
                    paths = cast("list[object]", pr_status.get("changed_paths", ()))
                    existing = pr_status.get("open_pull_request_url")
                    print(f"分支 {pr_status.get('branch')} → origin，基准 "
                          f"{pr_status.get('base_branch')}：")
                    for subject in subjects[:15]:
                        print(f"  提交 {safe_terminal_text(str(subject))}")
                    for changed_path in paths[:20]:
                        print(f"  文件 {safe_terminal_text(str(changed_path))}")
                    if len(paths) > 20:
                        print(f"  …另有 {len(paths) - 20} 个文件")
                    uncommitted = int(_numeric(pr_status.get("uncommitted_count")))
                    if uncommitted:
                        print(f"提示：工作区还有 {uncommitted} 项未提交变更，不会包含在本 PR 中"
                              "（可先 /commit）。")
                    if isinstance(existing, str) and existing:
                        print(f"该分支已有 Pull Request：{existing}")
                    if not title and _ui:
                        print("请输入 /pull <标题> 后重试（默认标题取最新提交说明）。")
                        continue
                    if _ui:
                        confirmed = await _ui.choose_menu(
                            ["确认推送并创建", "取消"],
                            title="推送分支并创建 Pull Request",
                        )
                        if confirmed != "确认推送并创建":
                            continue
                    else:
                        confirm = (await asyncio.to_thread(
                            input, "确认推送并创建 Pull Request？[y/N] ",
                        )).strip()
                        if confirm.casefold() not in {"y", "yes"}:
                            continue
                    pull_request = await facade.open_pull_request(
                        title=title or None, expected_preview_id=str(pr_status["preview_id"]),
                    )
                    verb = "已创建" if pull_request.get("created") else "已更新"
                    print(f"{verb} Pull Request：{pull_request.get('url')}")
                except DeliveryInterruptedError as error:
                    print(safe_terminal_text(str(error)))
                except (DeliveryError, OSError, RuntimeError) as error:
                    print("创建 Pull Request 失败: " + safe_terminal_text(str(error)))
                continue
            if query == "/checks" or query.startswith("/checks "):
                ci_message: str | None = None
                try:
                    from agent_runtime.git_delivery import DeliveryError, compose_ci_failure_message

                    tokens = query.split()[1:]
                    branch: str | None = None
                    wait_seconds = 0.0
                    index = 0
                    while index < len(tokens):
                        token = tokens[index]
                        if token == "--wait":
                            wait_seconds = 600.0
                            if index + 1 < len(tokens) and tokens[index + 1].isdigit():
                                wait_seconds = float(tokens[index + 1])
                                index += 1
                        elif token.startswith("--wait=") and token[7:].isdigit():
                            wait_seconds = float(token[7:])
                        elif branch is None:
                            branch = token
                        index += 1
                    if wait_seconds > 0:
                        print(f"等待 CI 完成（最长 {int(wait_seconds)} 秒，Ctrl+C 取消）…")
                    try:
                        report = await facade.pull_request_checks(
                            branch, include_failure_logs=True,
                            wait_seconds=wait_seconds,
                        )
                    except KeyboardInterrupt:
                        print("已取消等待；可再次运行 /checks 查看当前状态。")
                        continue
                    state_icons = {"pass": "✅", "fail": "❌", "pending": "⏳",
                                   "skipping": "⏸", "cancel": "⛔"}
                    checks = cast("list[dict[str, object]]", report.get("checks", ()))
                    print(f"分支 {report.get('branch')} 的 PR 检查"
                          f"（{report.get('pull_request_url') or 'PR'}）：")
                    for check in checks:
                        icon = state_icons.get(str(check.get("state")), "·")
                        print(f"  {icon} {safe_terminal_text(str(check.get('name', '')))}"
                              f"（{safe_terminal_text(str(check.get('workflow', '')))}）")
                    overall = str(report.get("overall"))
                    if overall == "pass":
                        print(f"CI 全部通过 ✅（{len(checks)} 项）。")
                        continue
                    if overall == "none":
                        print("该 Pull Request 没有配置检查。")
                        continue
                    if overall == "pending":
                        print("仍在进行中；可用 /checks --wait 继续等待。")
                        continue
                    failed = [check for check in checks
                              if str(check.get("state")) in {"fail", "cancel"}]
                    failure_logs = cast("list[dict[str, object]]", report.get("failure_logs", ()))
                    for entry in failure_logs:
                        logs = str(entry.get("logs", ""))
                        print(f"—— {entry.get('workflow')} 失败日志（截断）：")
                        for line in logs[-1200:].splitlines()[-25:]:
                            print("  " + safe_terminal_text(line))
                    if _ui:
                        confirmed = await _ui.choose_menu(
                            ["交给 agent 分析修复", "取消"],
                            title=f"CI 失败（{len(failed)} 项）",
                        )
                        if confirmed != "交给 agent 分析修复":
                            continue
                    else:
                        confirm = (await asyncio.to_thread(
                            input, "把失败详情交给 agent 分析修复？[y/N] ",
                        )).strip()
                        if confirm.casefold() not in {"y", "yes"}:
                            continue
                    ci_message = compose_ci_failure_message(
                        branch=str(report.get("branch")),
                        head_commit=str(report.get("head_commit") or ""),
                        pull_request_url=(
                            str(report["pull_request_url"])
                            if isinstance(report.get("pull_request_url"), str) else None
                        ),
                        failed=[(str(check.get("name", "")), str(check.get("workflow", "")))
                                for check in failed],
                        failure_logs=[(str(entry.get("workflow", "")), str(entry.get("logs", "")))
                                      for entry in failure_logs],
                    )
                except (DeliveryError, OSError, RuntimeError) as error:
                    print("读取 CI 状态失败: " + safe_terminal_text(str(error)))
                    continue
                if ci_message is None:
                    continue
                query = ci_message
            if query == "/connections":
                if _ui:
                    await _ui.open_connections()
                else:
                    print("请另开终端运行 uv run agent connections，在页面中选择服务并登录。")
                continue
            if query == "/status":
                print(f"会话: {selected_chat.thread_id if selected_chat else getattr(session, 'thread_id', '(new)')}")
                print(f"Previous Turn: {current_turn_id or '(none)'}")
                print(f"模型: {model_id}")
                print(f"工作区: {chat_workspace}")
                clock = capture_runtime_clock()
                print(f"当前时间: {clock['current_time']} · {clock['timezone']}")
                print(f"详细输出: {'开' if verbose else '关'}")
                print("网页网络: " + ("可信代理（上游解析域名，NO_PROXY 命中时严格直连）"
                                      if getattr(facade, "web_proxy_url", None) else "严格直连"))
                event_display.show_context_status()
                if last_result is not None:
                    print(f"验证状态: {last_result.validation_status}")
                continue
            if query == "/copy":
                if not last_answer:
                    print("暂无可复制的回答。")
                elif _ui:
                    await _ui.copy_answer(last_answer)
                else:
                    print(safe_terminal_text(last_answer))
                continue
            if query == "/resume" or query.startswith("/resume "):
                parts = query.split()
                if len(parts) > 2:
                    print("用法: /resume [完整会话ID]")
                    continue
                try:
                    await reopen(parts[1] if len(parts) == 2 else "")
                except (click.ClickException, OSError, ValueError, KeyError, sqlite3.Error) as error:
                    print(f"打开会话失败: {safe_terminal_text(str(error))}")
                continue
            if query in {"/new", "/clear"}:
                await stack.aclose()
                stack.callback(event_display.finish)
                if session is not None:
                    facade.model = model_id
                facade.knowledge = default_chat_knowledge
                session = None
                selected_chat = None
                previous_turn_id = None
                current_turn_id = None
                last_result = None
                last_answer = ""
                event_display.clear_context_status()
                pending_files.clear()
                chat_workspace = default_chat_workspace
                if _ui:
                    _ui.model = model_id
                refresh_request_status(facade)
                print("下一条消息将使用空历史。")
                continue
            if query == "/verbose":
                verbose = not verbose
                event_display.set_verbose(verbose)
                print(f"详细输出: {'开' if verbose else '关'}")
                if verbose:
                    event_display.show_context_status()
                continue
            if query.split(maxsplit=1)[0] in {"/effort", "/output"}:
                try:
                    await change_request_settings(query)
                except ValueError as error:
                    print(f"设置未更改：{error}")
                continue
            if query == "/model" or query.startswith("/model "):
                model_control = session or facade
                if _ui and query in {"/model", "/model list"}:
                    selected = await _ui.choose_model(
                        [spec.id for spec in model_control.models()], model_id,
                    )
                    if selected is None:
                        continue
                    query = f"/model switch {selected}"
                _handle_model_slash_command(
                    query,
                    agent=model_control,
                )
                model_id = model_control.current_model().id
                if _ui:
                    _ui.model = model_id
                refresh_request_status(model_control)
                continue
            query, dropped = parse_file_input(query, base=Path(default_chat_workspace))
            if dropped:
                pending_files[:] = list(dict.fromkeys((*pending_files, *dropped)))
                print(f"已附加 {len(pending_files)} 个本地文件（原地只读，不复制）。")
                if not query:
                    print("继续输入任务后发送；/attach 查看，/detach 清除待发送附件。")
                    continue
            if query.startswith("/"):
                print(f"未知命令: {query.split()[0]}；输入 /help 查看可用命令。")
                continue

            saved_result: AgentResult | None = None
            saved_action: str | None = None
            saved_input: str | None = None
            if selected_chat is not None:
                # Refresh the durable active slot before opening any resources. Another
                # process may have started a Turn since the conversation was selected.
                latest = next((chat for chat in await asyncio.to_thread(list_chat_sessions,
                    facade.checkpoint_db or DEFAULT_CHECKPOINT_PATH, workspace=Path(default_chat_workspace),
                ) if chat.thread_id == selected_chat.thread_id), None)
                if latest is None:
                    print("会话已不可读取；请重新 /resume 选择。")
                    continue
                selected_chat = latest
                if selected_chat.active_turn_id:
                    saved_result = await facade.read_result(selected_chat.active_turn_id)
                    request = saved_result.pause
                    if request is None:
                        saved_result = None
                    elif request.kind in {"clarification", "choice"}:
                        if request.options and query not in request.options:
                            _print_chat_recovery(selected_chat, saved_result)
                            continue
                        saved_action, saved_input = "continue", query
                    elif request.kind == "model_retry" and query not in {"retry", "abort"}:
                        saved_result = None
                    elif request.kind == "tool_reconciliation":
                        if query in {"continue", "retry", "abort"}:
                            saved_action = query
                        else:
                            # Session.submit reconciles under execution ownership
                            # before accepting this text as a follow-up Turn.
                            saved_result = None
                    else:
                        saved_action = resolve_pause_choice(query, request.options)
                        if saved_action is None and _ui:
                            saved_action = await _handle_ui_pause(saved_result, _ui)
                        if saved_action is None:
                            _print_chat_recovery(selected_chat, saved_result)
                            continue
                    if saved_result is not None and saved_action != "abort":
                        metadata = await asyncio.to_thread(
                            _cli_turn, facade.checkpoint_db or DEFAULT_CHECKPOINT_PATH, saved_result.turn_id,
                        )
                        facade.knowledge = metadata.runtime.knowledge
            if saved_result is None:
                session = await open_session()
            elif saved_action != "abort" and (_ui is not None or pending_files):
                session = await open_session(frozen_turn_id=saved_result.turn_id)
            resume_driver = session if session is not None else facade
            event_display.begin_turn()
            if saved_result is not None:
                event_display.update_usage(saved_result.usage)
            try:
                if saved_result is not None:
                    assert saved_action is not None
                    resume_options: _ResumeOptions = (
                        {"files": list(pending_files)} if pending_files and saved_action != "abort" else {}
                    )
                    submission = resume_driver.resume(saved_result.turn_id, saved_action,
                        user_input=saved_input, event_sink=event_display,
                        **resume_options)
                else:
                    assert session is not None
                    submission = session.submit(query, event_sink=event_display, **(
                        {"files": list(pending_files)} if pending_files else {}
                    ))
                result = await _ui.operation(submission) if _ui else await submission
                event_display.update_usage(result.usage)
                if saved_result is None or saved_action != "abort":
                    pending_files.clear()
            except (OSError, ValueError, ConversationBusyError) as error:
                print("发送失败：" + safe_terminal_text(str(error)))
                if _ui:
                    _ui.editor.text = query
                    _ui.invalidate()
                continue
            except asyncio.CancelledError:
                if _ui is None or ((task := asyncio.current_task()) is not None and task.cancelling()):
                    raise
                await _cancel_ui_execution(resume_driver, _ui, _ui.display.turn_id)
                continue
            current_turn_id = result.turn_id
            while result.status == "paused":
                event_display.finish()
                if result.pause is not None and result.pause.kind == "tool_reconciliation":
                    print("上次操作的执行结果尚未确认，任务保持暂停；核对前不会自动重跑或取消该操作。")
                    break
                if _ui:
                    _ui.display.present_result(result)
                else:
                    print("\n" + format_usage_summary(result.usage))
                text_answer = result.pause is not None and result.pause.kind in {"clarification", "choice"}
                pending_steering = bool(_ui and result.pause and result.pause.kind == "tool_approval"
                                        and any([await asyncio.to_thread(facade.input_status, identity) == "pending"
                                                 for identity in _ui._input_receipts]))
                action = ("continue" if pending_steering
                          else await _handle_ui_pause(result, _ui) if _ui else await _handle_cli_pause(result))
                if action is None:
                    if _ui:
                        cancelled = await _cancel_ui_execution(resume_driver, _ui, result.turn_id)
                        assert cancelled is not None
                        result = cancelled
                    else:
                        result = await resume_driver.resume(result.turn_id, "abort", event_sink=event_display)
                        print("已取消。")
                    break
                event_display.begin_turn(reset=False)
                if text_answer:
                    resumption = resume_driver.resume(
                        result.turn_id, "continue", user_input=action, event_sink=event_display,
                    )
                else:
                    resumption = resume_driver.resume(result.turn_id, action, event_sink=event_display)
                try:
                    result = await _ui.operation(resumption) if _ui else await resumption
                    event_display.update_usage(result.usage)
                except asyncio.CancelledError:
                    if _ui is None or ((task := asyncio.current_task()) is not None and task.cancelling()):
                        raise
                    cancelled = await _cancel_ui_execution(resume_driver, _ui, result.turn_id)
                    assert cancelled is not None
                    result = cancelled
            event_display.finish()
            event_display.update_usage(result.usage)
            if _ui:
                _ui.display.present_result(result)
            last_result = result
            last_answer = result.answer or ""
            _display_agent_result(
                result,
                verbose=verbose,
                answer_streamed=event_display.answer_streamed,
                show_execution=_ui is None,
                show_usage=_ui is None,
                show_outcome=_ui is None,
            )


def _print_current_model(spec: ModelSpec) -> None:
    print(f"{spec.id}")
    print(f"provider: {spec.provider}")
    print(f"context_window: {spec.context_window}")
    print(f"location: {spec.location}")


def _print_model_menu(agent: Agent | Session) -> None:
    current = agent.current_model()
    print(f"当前模型: {current.id}")
    print("可用模型:")
    for line in format_model_rows(
        agent.models(),
        current_model_id=current.id,
    ):
        print(line)
    print("切换: /model <model_id>")


class _RequestSettingsOptions(TypedDict, total=False):
    reasoning_effort: str
    max_output_tokens: int


def _handle_request_settings_command(query: str, *, agent: Any) -> None:
    parts = query.split()
    if len(parts) not in (1, 2) or parts[0] not in {"/effort", "/output"}:
        raise ValueError("用法：/effort <支持的强度|default|off>；/output <正整数|default>")
    if len(parts) == 2:
        if parts[0] == "/effort":
            agent.configure_request(reasoning_effort=parts[1])
        else:
            value = parts[1]
            if value != "default" and (not value.isascii() or not value.isdecimal() or int(value) < 1):
                raise ValueError("输出额度必须为正整数或 default")
            agent.configure_request(max_output_tokens=value if value == "default" else int(value))
    snapshot = agent.request_settings()
    spec = agent.current_model()
    preferences = snapshot["preferences"]
    defaults = snapshot["model_defaults"]
    effort = preferences["reasoning_effort"] or defaults["reasoning_effort"]
    requested = preferences["max_output_tokens"] or defaults["max_output_tokens"]
    output = min(requested, spec.max_output_tokens) if spec.max_output_tokens is not None else requested
    print(f"模型: {spec.id}；思考强度: {effort or '未声明'}；单次输出额度: {output:,}")
    if spec.reasoning is not None and spec.reasoning.supported_efforts is not None:
        print("支持强度: " + (", ".join(spec.reasoning.supported_efforts) or "不支持"))
    else:
        print("支持强度: 未知（需在模型声明中配置）")
    if requested != output:
        print(f"申请 {requested:,}，受模型上限限制为 {output:,}；请求时还会校验上下文和累计预算。")
    if len(parts) == 2:
        print("设置已保存，从下一次模型请求生效；已发出的请求和恢复重试保持原配置。")


def _handle_model_slash_command(
    query: str,
    *,
    agent: Agent | Session | None,
) -> None:
    if agent is None:
        print("模型控制平面不可用。")
        return
    parts = query.split()
    if len(parts) == 1:
        _print_model_menu(agent)
        return
    action = parts[1] if len(parts) > 1 else "current"
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
    except (ModelPolicyError, UnknownModelIdError) as exc:
        print(f"模型切换失败: {exc}")
        _print_model_menu(agent)
        return
    print("用法: /model [current|list|switch <model_id>]")


# ── CLI Commands ──


@model_app.command(name="list")
def model_list(
    source: Annotated[
        bool,
        typer.Option("--source", help="显示 builtin/user/override 来源"),
    ] = False,
    session_path: Annotated[
        Path,
        typer.Option("--session-path", help="模型 session state 文件"),
    ] = DEFAULT_MODEL_SESSION_PATH,
) -> None:
    """列出可用模型，并标记当前会话模型。"""
    service = _model_admin_service(session_path)
    selection = _model_admin_call(service.current_selection)
    _print_model_session_diagnostics(selection.diagnostics)
    current_id = selection.spec.id
    for entry in _model_admin_call(service.list_models):
        marker = "*" if entry.model_id == current_id else " "
        suffix = f" source={entry.origin}" if source else ""
        print(f"{marker} {entry.model_id} provider={entry.spec.provider}{suffix}")


@model_app.command(name="show")
def model_show(
    model_id: Annotated[str, typer.Argument(help="模型 id")],
    session_path: Annotated[
        Path,
        typer.Option("--session-path", help="模型 session state 文件"),
    ] = DEFAULT_MODEL_SESSION_PATH,
) -> None:
    """显示一个模型的规范化定义摘要，不解析凭据值。"""
    entry = _model_admin_call(lambda: _model_admin_service(session_path).show(model_id))
    print(f"model_id: {entry.model_id}")
    print(f"source: {entry.origin}")
    print(f"provider: {entry.definition.provider.value}")
    print(f"provider_name: {entry.definition.provider_name or '<none>'}")
    print(f"protocol: {entry.spec.protocol or '<none>'}")
    print(f"location: {entry.spec.location}")
    print(f"base_url: {entry.spec.base_url or '<default>'}")
    print(f"api_key_env: {entry.spec.api_key_env or '<none>'}")
    print(f"tokenizer_model: {entry.definition.tokenizer_model or '<default>'}")
    print(
        f"context_window_tokens: "
        f"{entry.definition.context_window_tokens}"
    )


    print(
        f"max_output_tokens: "
        f"{entry.definition.max_output_tokens}"
    )
    print(f"timeout_seconds: {entry.definition.timeout_seconds}")
    print(f"supports_tools: {str(entry.spec.supports_tools).lower()}")
    print(
        "supports_structured_output: "
        f"{str(entry.spec.supports_structured_output).lower()}"
    )
    print(f"definition_revision: {entry.definition_revision}")


@model_app.command(name="current")
def model_current(
    session_path: Annotated[
        Path,
        typer.Option("--session-path", help="模型 session state 文件"),
    ] = DEFAULT_MODEL_SESSION_PATH,
) -> None:
    """显示当前会话模型。"""
    selection = _model_admin_call(_model_admin_service(session_path).current_selection)
    _print_model_session_diagnostics(selection.diagnostics)
    _print_current_model(selection.spec)


@model_app.command(name="switch")
def model_switch(
    model_id: Annotated[str, typer.Argument(help="要切换到的模型 id")],
    session_path: Annotated[
        Path,
        typer.Option("--session-path", help="模型 session state 文件"),
    ] = DEFAULT_MODEL_SESSION_PATH,
) -> None:
    """切换当前模型 session state，不修改 models.yaml。"""
    spec = _model_admin_call(lambda: _model_admin_service(session_path).switch(model_id))
    print(f"已切换模型: {spec.id}")


@model_app.command(name="probe")
def model_probe(
    model_id: Annotated[str, typer.Argument(help="模型 id")],
    level: Annotated[
        ProbeLevel,
        typer.Option("--level", help="connectivity、stream 或 full"),
    ] = ProbeLevel.FULL,
    session_path: Annotated[
        Path,
        typer.Option("--session-path", help="模型 session state 文件"),
    ] = DEFAULT_MODEL_SESSION_PATH,
) -> None:
    """探测已注册模型，不修改注册表或 session。"""
    service = _model_admin_service(session_path)
    evidence = _model_admin_call(lambda: _run_cli_async(service.probe(model_id, level=level)))
    _print_probe_evidence(evidence)


@model_app.command(name="add")
def model_add(
    model_id: Annotated[str, typer.Argument(help="新模型 id")],
    provider: Annotated[ModelProvider | None, typer.Option("--provider")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    api_key_env: Annotated[str | None, typer.Option("--api-key-env")] = None,
    tokenizer_model: Annotated[str | None, typer.Option("--tokenizer-model")] = None,
    provider_name: Annotated[str | None, typer.Option("--provider-name")] = None,
    protocol: Annotated[str | None, typer.Option("--protocol")] = None,
    timeout_seconds: Annotated[float | None, typer.Option("--timeout-seconds")] = None,
    context_window_tokens: Annotated[
        int | None,
        typer.Option("--context-window-tokens"),
    ] = None,
    max_output_tokens: Annotated[
        int | None,
        typer.Option("--max-output-tokens"),
    ] = None,
    supports_tools: Annotated[
        bool | None,
        typer.Option("--tools/--no-tools"),
    ] = None,
    supports_structured_output: Annotated[
        bool | None,
        typer.Option("--structured-output/--no-structured-output"),
    ] = None,
    location: Annotated[Literal["local", "cloud"] | None, typer.Option("--location")] = None,
    from_path: Annotated[Path | None, typer.Option("--from", help="单模型 YAML 定义")] = None,
    skip_probe: Annotated[
        bool,
        typer.Option("--skip-probe", help="离线登记；明确标记为未验证"),
    ] = False,
    session_path: Annotated[
        Path,
        typer.Option("--session-path", help="模型 session state 文件"),
    ] = DEFAULT_MODEL_SESSION_PATH,
) -> None:
    """验证并登记一个用户模型；默认在提交前执行 full probe。"""
    service = _model_admin_service(session_path)
    arguments = _model_definition_arguments(
        provider=provider,
        base_url=base_url,
        api_key_env=api_key_env,
        tokenizer_model=tokenizer_model,
        provider_name=provider_name,
        protocol=protocol,
        timeout_seconds=timeout_seconds,
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
        supports_tools=supports_tools,
        supports_structured_output=supports_structured_output,
        location=location,
    )
    outcome = _model_admin_call(
        lambda: _run_cli_async(
            service.add(
                model_id,
                arguments=arguments,
                from_path=from_path,
                skip_probe=skip_probe,
            )
        )
    )
    _print_model_mutation(outcome)


@model_app.command(name="update")
def model_update(
    model_id: Annotated[str, typer.Argument(help="用户模型 id")],
    provider: Annotated[ModelProvider | None, typer.Option("--provider")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    api_key_env: Annotated[str | None, typer.Option("--api-key-env")] = None,
    tokenizer_model: Annotated[str | None, typer.Option("--tokenizer-model")] = None,
    provider_name: Annotated[str | None, typer.Option("--provider-name")] = None,
    protocol: Annotated[str | None, typer.Option("--protocol")] = None,
    timeout_seconds: Annotated[float | None, typer.Option("--timeout-seconds")] = None,
    context_window_tokens: Annotated[
        int | None,
        typer.Option("--context-window-tokens"),
    ] = None,

    max_output_tokens: Annotated[
        int | None,
        typer.Option("--max-output-tokens"),
    ] = None,
    supports_tools: Annotated[
        bool | None,
        typer.Option("--tools/--no-tools"),
    ] = None,
    supports_structured_output: Annotated[
        bool | None,
        typer.Option("--structured-output/--no-structured-output"),
    ] = None,
    location: Annotated[Literal["local", "cloud"] | None, typer.Option("--location")] = None,
    unset: Annotated[list[str] | None, typer.Option("--unset", help="清除可空字段路径")] = None,
    from_path: Annotated[Path | None, typer.Option("--from", help="完整替换定义 YAML")] = None,
    skip_probe: Annotated[bool, typer.Option("--skip-probe")] = False,
    session_path: Annotated[
        Path,
        typer.Option("--session-path", help="模型 session state 文件"),
    ] = DEFAULT_MODEL_SESSION_PATH,
) -> None:
    """以类型化 patch 或完整定义更新用户模型。"""
    service = _model_admin_service(session_path)
    arguments = _model_definition_arguments(
        provider=provider,
        base_url=base_url,
        api_key_env=api_key_env,
        tokenizer_model=tokenizer_model,
        provider_name=provider_name,
        protocol=protocol,
        timeout_seconds=timeout_seconds,
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
        supports_tools=supports_tools,
        supports_structured_output=supports_structured_output,
        location=location,
    )
    outcome = _model_admin_call(
        lambda: _run_cli_async(
            service.update(
                model_id,
                arguments=arguments,
                from_path=from_path,
                unset_paths=tuple(unset or ()),
                skip_probe=skip_probe,
            )
        )
    )
    _print_model_mutation(outcome)


@model_app.command(name="remove")
def model_remove(
    model_id: Annotated[str, typer.Argument(help="用户模型 id")],
    session_path: Annotated[
        Path,
        typer.Option("--session-path", help="模型 session state 文件"),
    ] = DEFAULT_MODEL_SESSION_PATH,
) -> None:
    """删除非当前会话选中的用户模型。"""
    result = _model_admin_call(lambda: _model_admin_service(session_path).remove(model_id))
    print(f"removed: {model_id}")
    print(f"registry_revision: {result.snapshot.document.revision}")


@model_trust_app.command(name="init")
def model_trust_init(
    session_path: Annotated[
        Path,
        typer.Option("--session-path", help="模型 session state 文件"),
    ] = DEFAULT_MODEL_SESSION_PATH,
) -> None:
    """幂等初始化本地模型绑定信任域。"""
    status = _model_admin_call(lambda: _model_admin_service(session_path).trust_init())
    _print_trust_status(status.trust_domain_id, status.signing_key_id)


@model_trust_app.command(name="status")
def model_trust_status(
    session_path: Annotated[
        Path,
        typer.Option("--session-path", help="模型 session state 文件"),
    ] = DEFAULT_MODEL_SESSION_PATH,
) -> None:
    """显示信任域标识与签名 key id，不显示密钥。"""
    status = _model_admin_call(lambda: _model_admin_service(session_path).trust_status())
    _print_trust_status(status.trust_domain_id, status.signing_key_id)


def _model_admin_service(session_path: Path) -> ModelAdminService:
    return _model_admin_call(lambda: build_model_admin_service(session_path=session_path))


def _model_admin_call[T](operation: Callable[[], T]) -> T:
    try:
        return operation()
    except ValidationError as exc:
        fields = sorted(
            ".".join(str(part) for part in error.get("loc", ())) or "configuration"
            for error in exc.errors(include_url=False, include_input=False)
        )
        raise click.ClickException(
            "Invalid model configuration fields: " + ", ".join(fields)
        ) from None
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from None


def _model_definition_arguments(
    *,
    provider: ModelProvider | None,
    base_url: str | None,
    api_key_env: str | None,
    tokenizer_model: str | None,
    provider_name: str | None,
    protocol: str | None,
    timeout_seconds: float | None,
    context_window_tokens: int | None,
    max_output_tokens: int | None,
    supports_tools: bool | None,
    supports_structured_output: bool | None,
    location: Literal["local", "cloud"] | None,
) -> ModelDefinitionArguments:
    return ModelDefinitionArguments(
        provider=provider,
        base_url=base_url,
        api_key_env=api_key_env,
        tokenizer_model=tokenizer_model,
        provider_name=provider_name,
        protocol=protocol,
        timeout_seconds=timeout_seconds,
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
        supports_tools=supports_tools,
        supports_structured_output=supports_structured_output,
        location=location,
    )


def _print_probe_evidence(evidence: ModelProbeEvidence) -> None:
    print(f"level: {evidence.level.value}")
    print(f"connectivity_ok: {str(evidence.connectivity_ok).lower()}")
    print(f"text_delta_count: {evidence.text_delta_count}")
    print(f"completion_ok: {str(evidence.completion_ok).lower()}")
    print(f"tool_call_ok: {_optional_bool(evidence.tool_call_ok)}")
    print(f"structured_output_ok: {_optional_bool(evidence.structured_output_ok)}")


def _print_model_mutation(outcome: ModelMutationOutcome) -> None:
    print(f"model_id: {outcome.model_id}")
    print("source: user")
    print(f"changed: {str(outcome.changed).lower()}")
    print(f"definition_revision: {outcome.definition_revision}")
    print(f"registry_revision: {outcome.registry_revision}")
    print(f"registry_fingerprint: {outcome.registry_fingerprint}")
    if outcome.unverified:
        print("probe: skipped (unverified)")
    elif outcome.probe_evidence is not None:
        print("probe: verified")
        _print_probe_evidence(outcome.probe_evidence)
    else:
        print("probe: skipped (normalized no-op)")


def _print_trust_status(trust_domain_id: str, signing_key_id: str) -> None:
    print("initialized: true")
    print(f"trust_domain_id: {trust_domain_id}")
    print(f"signing_key_id: {signing_key_id}")


def _optional_bool(value: bool | None) -> str:
    return "not_run" if value is None else str(value).lower()


def _print_model_session_diagnostics(diagnostics: tuple[str, ...]) -> None:
    for diagnostic in diagnostics:
        typer.echo(f"model session diagnostic: {diagnostic}", err=True)


def _latest_cli_turn(
    checkpoint_db: Path,
    *,
    workspace_path: Path | None,
) -> _CLITurn | None:
    with RuntimeState.open_sync(checkpoint_db, read_only=True) as store:
        turn = _latest_harness_turn(
            store,
            statuses={"paused", "interrupted", "running"},
            workspace_path=workspace_path,
        )
        return None if turn is None else _project_cli_turn(store, turn)


def _latest_abortable_cli_turn(
    checkpoint_db: Path,
    *,
    workspace_path: Path | None,
) -> _CLITurn | None:
    with RuntimeState.open_sync(checkpoint_db, read_only=True) as store:
        turn = _latest_harness_turn(
            store,
            statuses={"paused", "interrupted", "running"},
            workspace_path=workspace_path,
        )
        return None if turn is None else _project_abortable_cli_turn(store, turn)


def _latest_completed_cli_turn(
    checkpoint_db: Path,
    *,
    workspace_path: Path | None,
) -> _CLITurn | None:
    with RuntimeState.open_sync(checkpoint_db, read_only=True) as store:
        turn = _latest_harness_turn(
            store,
            statuses={"completed", "failed", "cancelled"},
            workspace_path=workspace_path,
        )
        return None if turn is None else _project_cli_turn(store, turn)


def _cli_turn(checkpoint_db: Path, turn_id: str) -> _CLITurn:
    with RuntimeState.open_sync(checkpoint_db, read_only=True) as store:
        try:
            turn = store.read_turn(turn_id)
        except KeyError as exc:
            raise KeyError(f"Turn not found: {turn_id}") from exc
        return _project_cli_turn(store, turn)


def _abortable_cli_turn(checkpoint_db: Path, turn_id: str) -> _CLITurn:
    with RuntimeState.open_sync(checkpoint_db, read_only=True) as store:
        try:
            turn = store.read_turn(turn_id)
        except KeyError as exc:
            raise KeyError(f"Turn not found: {turn_id}") from exc
        return _project_abortable_cli_turn(store, turn)


def _latest_harness_turn(
    store: RuntimeState,
    *,
    statuses: set[str],
    workspace_path: Path | None,
) -> TurnSnapshot | None:
    expected_workspace = (
        None if workspace_path is None else workspace_path.expanduser().resolve()
    )
    for turn in reversed(store.list_turns()):
        if turn.status not in statuses:
            continue
        thread = store.read_thread(turn.thread_id)
        if (
            expected_workspace is not None
            and Path(thread.workspace).resolve() != expected_workspace
        ):
            continue
        return turn
    return None


def _project_cli_turn(store: RuntimeState, turn: TurnSnapshot) -> _CLITurn:
    thread = store.read_thread(turn.thread_id)
    if "model_alias" in turn.binding_manifest:
        raise RuntimeError("legacy model_alias binding is unsupported")
    model_id = turn.binding_manifest.get("model_id")
    knowledge_value = turn.binding_manifest.get("knowledge_config")
    knowledge = (
        RAGKnowledgeConfig.model_validate(knowledge_value)
        if isinstance(knowledge_value, Mapping)
        else None
    )
    return _CLITurn(
        turn_id=turn.turn_id,
        status=turn.status,
        runtime=_CLIRuntimeBinding(
            model_id=model_id if isinstance(model_id, str) else None,
            workspace_path=thread.workspace,
            knowledge=knowledge,
        ),
    )


def _project_abortable_cli_turn(
    store: RuntimeState,
    turn: TurnSnapshot,
) -> _CLITurn:
    thread = store.read_thread(turn.thread_id)
    return _CLITurn(
        turn_id=turn.turn_id,
        status=turn.status,
        runtime=_CLIRuntimeBinding(
            model_id=None,
            workspace_path=thread.workspace,
            knowledge=None,
        ),
    )


@agent_app.command(name="chat")
def agent_chat(
    ui_mode: Annotated[
        Literal["inline", "fullscreen"],
        typer.Option("--ui", help="终端界面：inline 清屏后原生滚动；fullscreen 全屏固定输入框"),
    ] = "fullscreen",
    previous_turn_id: Annotated[
        str | None,
        typer.Option(
            "--previous-turn-id",
            help="从指定 Turn 继续；省略则从空上下文开始。",
        ),
    ] = None,
    last: Annotated[
        bool,
        typer.Option("--last", hidden=True, help="兼容别名：使用 --continue"),
    ] = False,
    continue_chat: Annotated[bool, typer.Option("--continue", help="打开当前工作区最近聊天，回放后继续")] = False,
    resume_thread_id: Annotated[
        str | None, typer.Option("--resume", help="打开指定会话 ID；交互选择请使用 agent resume"),
    ] = None,
    checkpoint_db: Annotated[
        Path,
        typer.Option("--rollout-db", "--checkpoint-db", help="完整 SQLite Rollout／事件数据库"),
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
        typer.Option("--model", help="主生成模型 ID，对应 configs/models.yaml 中 capability=chat 的条目"),
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
        typer.Option("--max-steps", "--max-turns", min=1, help="每条消息允许的最大执行 step 数（旧名 --max-turns）"),
    ] = None,
    allow_write_tools: Annotated[
        bool,
        typer.Option("--allow-write-tools", help="预授权 workspace 写入类调用"),
    ] = False,
    allow_execute_tools: Annotated[
        bool,
        typer.Option("--allow-execute-tools", help="预授权进程执行类调用"),
    ] = False,
    allow_web_tools: Annotated[
        bool, typer.Option("--allow-web-tools/--no-web-tools", help="启用公网搜索和网页读取（默认开启）"),
    ] = True,
    web_limits_file: Annotated[
        Path | None, typer.Option("--web-limits", envvar="PRAXIS_WEB_LIMITS",
                                 help="有界网页预算JSON；恢复使用原Turn配置"),
    ] = None,
    web_proxy_url: Annotated[
        str | None, typer.Option("--web-proxy", help="可信 HTTP(S) 网页代理；direct 强制直连；默认读取启动环境"),
    ] = None,
    web_search_key_file: Annotated[
        Path | None, typer.Option("--web-search-key-file",
                                  help="工作区外的受保护搜索服务密钥文件"),
    ] = None,
    web_search_provider: Annotated[
        str | None, typer.Option("--web-search-provider",
                                 help="deepseek | brave | tavily；默认未配置则搜索不可用；仅提供密钥时用 Brave"),
    ] = None,
    permissions: Annotated[Path | None, typer.Option("--permissions", help="当前权限策略 JSON/TOML 文件")] = None,
    approval_mode: Annotated[
        ApprovalMode | None, typer.Option(
            "--approval-mode", help="ask 请求批准 | auto 帮我批准（新聊天默认）| full 范围内全自动",
        ),
    ] = None,
    reasoning_effort: Annotated[
        str | None, typer.Option("--effort", help="模型声明支持的思考强度；default/off 需匹配模型能力"),
    ] = None,
    max_output_tokens: Annotated[
        int | None, typer.Option("--max-output-tokens", min=1, help="单次输出额度（包含思考 token），与累计预算分开"),
    ] = None,
    worktree: Annotated[
        bool, typer.Option("--worktree", help="在独立 Git worktree 中开始新会话，自动生成 worktree/ 分支"),
    ] = False,
    worktree_name: Annotated[
        str | None, typer.Option("--worktree-name", help="命名 worktree 分支（worktree/<name>）；隐含 --worktree"),
    ] = None,
) -> None:
    """交互式 Agent 对话。暂停时支持工具审批。"""
    if approval_mode == "ask":
        allow_write_tools = allow_execute_tools = allow_web_tools = False
    if sum((previous_turn_id is not None, last or continue_chat, resume_thread_id is not None)) > 1:
        raise typer.BadParameter("--previous-turn-id、--continue/--last 和 --resume 只能选择一个")
    effective_previous_turn_id = previous_turn_id
    continued_turn: _CLITurn | None = None
    if last or continue_chat:
        sessions = list_chat_sessions(checkpoint_db, workspace=Path.cwd())
        if not sessions:
            raise typer.BadParameter("当前工作区没有已保存的聊天")
        resume_thread_id = sessions[0].thread_id
    if resume_thread_id is not None and knowledge_config is not None:
        raise typer.BadParameter("打开旧聊天时使用保存的知识库配置，不能传 --knowledge-config")
    if effective_previous_turn_id is not None and knowledge_config is not None:
        raise typer.BadParameter("继续已有 Turn 时不能传 --knowledge-config；Turn 的 RuntimeBinding 是唯一配置来源")
    if effective_previous_turn_id is not None and model is not None:
        raise typer.BadParameter(
            "--previous-turn-id 仅继承上下文；新 Turn 使用当前 session 模型。"
            "不能同时传 --model，请先运行 agent model switch <model-id>。"
        )
    if effective_previous_turn_id is not None and continued_turn is None:
        continued_turn = _cli_turn(checkpoint_db, effective_previous_turn_id)
    created_worktree: CreateReport | None = None
    if worktree or worktree_name is not None:
        if effective_previous_turn_id is not None or resume_thread_id is not None:
            raise typer.BadParameter(
                "--worktree 只用于新会话；恢复旧会话会自动回到其记录的工作区。"
            )
        created_worktree = _open_cli_worktree(worktree_name)
    if continued_turn is None:
        facade_model = model
        facade_workspace: Path | str = (
            created_worktree.worktree.root if created_worktree is not None else Path.cwd()
        )
        facade_knowledge = _load_knowledge_config(knowledge_config)
    else:
        binding = continued_turn.runtime
        facade_model = None
        facade_workspace = binding.workspace_path or Path.cwd()
        facade_knowledge = binding.knowledge
    if created_worktree is not None:
        # The worktree carries its own rollout database so its sessions are as
        # self-contained as its checkout: cd into it and --continue works.
        checkpoint_db = created_worktree.worktree.root / ".praxis" / "checkpoints.sqlite"
    web_options: _WebOptions = {}
    if web_limits_file is not None:
        web_options["web_limits_file"] = web_limits_file
    if web_search_key_file is not None:
        web_options["web_search_key_file"] = web_search_key_file
    if web_search_provider is not None:
        web_options["web_search_provider"] = web_search_provider
    if web_proxy_url is not None:
        web_options["web_proxy_url"] = web_proxy_url
    request_options: _RequestSettingsOptions = {}
    if reasoning_effort is not None:
        request_options["reasoning_effort"] = reasoning_effort
    if max_output_tokens is not None:
        request_options["max_output_tokens"] = max_output_tokens
    facade = _create_agent_facade(
        **request_options,
        model=facade_model,
        checkpoint_db=checkpoint_db,
        workspace_path=facade_workspace,
        model_session_path=DEFAULT_MODEL_SESSION_PATH,
        knowledge=facade_knowledge,
        **_permission_options(permissions, approval_mode),
        _selection_requester="user",
        **web_options,
    )
    _run_cli_async(
        _chat_facade_loop(
            facade,
            max_tokens_total=max_tokens_total,
            max_turns=max_turns,
            previous_turn_id=effective_previous_turn_id,
            resume_thread_id=resume_thread_id,
            allow_write_tools=allow_write_tools,
            allow_execute_tools=allow_execute_tools,
            allow_web_tools=allow_web_tools,
            ui_mode=ui_mode,
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
        typer.Option("--rollout-db", "--checkpoint-db", help="完整 SQLite Rollout／事件数据库；支持跨进程 resume"),
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
        typer.Option("--max-steps", "--max-turns", min=1, help="本次运行允许的最大执行 step 数（旧名 --max-turns）"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="主生成模型 ID，对应 configs/models.yaml 中 capability=chat 的条目"),
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
            "--require-workspace-change/--no-require-workspace-change",
            help=(
                "要求产生真实 workspace 变更；需要机器验收时另设 --acceptance-command"
            ),
        ),
    ] = False,
    acceptance_command: Annotated[
        str | None, typer.Option("--acceptance-command", help="用户指定的验收命令；必须在最后修改后真实执行成功"),
    ] = None,
    allow_execute_tools: Annotated[
        bool,
        typer.Option("--allow-execute-tools", help="预授权本次进程执行类调用"),
    ] = False,
    allow_web_tools: Annotated[
        bool, typer.Option("--allow-web-tools/--no-web-tools", help="启用公网搜索和网页读取（默认开启）"),
    ] = True,
    web_proxy_url: Annotated[
        str | None, typer.Option("--web-proxy", help="可信 HTTP(S) 网页代理；direct 强制直连；默认读取启动环境"),
    ] = None,
    web_search_key_file: Annotated[
        Path | None, typer.Option("--web-search-key-file",
                                  help="工作区外的受保护搜索服务密钥文件"),
    ] = None,
    web_search_provider: Annotated[
        str | None, typer.Option("--web-search-provider",
                                 help="deepseek | brave | tavily；默认未配置则搜索不可用；仅提供密钥时用 Brave"),
    ] = None,
    web_limits_file: Annotated[
        Path | None, typer.Option("--web-limits", envvar="PRAXIS_WEB_LIMITS",
                                 help="有界网页预算JSON；Turn请求与浏览器资源分别配置，恢复使用原Turn配置"),
    ] = None,
    disable_workspace_mcp: Annotated[
        bool,
        typer.Option(
            "--disable-workspace-mcp",
            help="隔离运行：不发现或启动工作区中的 MCP 服务。",
        ),
    ] = False,
    permissions: Annotated[Path | None, typer.Option("--permissions", help="当前权限策略 JSON/TOML 文件")] = None,
    approval_mode: Annotated[
        ApprovalMode, typer.Option("--approval-mode", help="ask 请求批准 | auto 帮我批准 | full 权限范围内全自动"),
    ] = "auto",
    reasoning_effort: Annotated[
        str | None, typer.Option("--effort", help="模型声明支持的思考强度；default/off 需匹配模型能力"),
    ] = None,
    max_output_tokens: Annotated[
        int | None, typer.Option("--max-output-tokens", min=1, help="单次输出额度（包含思考 token），与累计预算分开"),
    ] = None,
    worktree: Annotated[
        bool, typer.Option("--worktree", help="在独立 Git worktree 中执行本任务，自动生成 worktree/ 分支"),
    ] = False,
    worktree_name: Annotated[
        str | None, typer.Option("--worktree-name", help="命名 worktree 分支（worktree/<name>）；隐含 --worktree"),
    ] = None,
) -> None:
    """执行一个新 Turn；可从已完成 Turn 继续上下文。"""
    if approval_mode == "ask":
        allow_write_tools = allow_execute_tools = allow_web_tools = False
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
            "--previous-turn-id 仅继承上下文；新 Turn 使用当前 session 模型。"
            "不能同时传 --model，请先运行 agent model switch <model-id>。"
        )
    created_worktree: CreateReport | None = None
    if worktree or worktree_name is not None:
        if effective_previous_turn_id is not None:
            raise typer.BadParameter(
                "--worktree 只用于新任务；继续旧 Turn 请使用其原工作区。"
            )
        created_worktree = _open_cli_worktree(worktree_name)
    effective_database = (
        created_worktree.worktree.root / ".praxis" / "checkpoints.sqlite"
        if created_worktree is not None else checkpoint_db
    ).expanduser().resolve()
    facade = _create_agent_facade(
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        model=model,
        checkpoint_db=effective_database,
        workspace_path=(
            created_worktree.worktree.root if created_worktree is not None else Path.cwd()
        ),
        model_session_path=model_session_path,
        knowledge=_load_knowledge_config(knowledge_config),
        **_permission_options(permissions, approval_mode),
        enable_workspace_mcp=not disable_workspace_mcp,
        web_search_key_file=web_search_key_file,
        web_search_provider=web_search_provider,
        web_proxy_url=web_proxy_url,
        web_limits_file=web_limits_file,
        _selection_requester="user",
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
            acceptance_command=acceptance_command,
            allow_write_tools=allow_write_tools,
            allow_execute_tools=allow_execute_tools,
            allow_web_tools=allow_web_tools,
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
            "recover",
            result.turn_id,
            "--checkpoint-db",
            str(effective_database),
        ]
        print(f"   {shlex.join(resume_args)}")

        if result.pause is not None:
            print(f"\n   待处理: {result.pause.question}")

        raise typer.Exit(code=2)

    if result.status == "failed":
        raise typer.Exit(code=1)


@agent_app.command(name="resume")
def agent_resume(
    thread_id: Annotated[str | None, typer.Argument(help="稳定会话 ID；省略则选择当前工作区旧聊天")] = None,
    checkpoint_db: Annotated[
        Path, typer.Option("--rollout-db", "--checkpoint-db", help="已有 SQLite Rollout 数据库"),
    ] = DEFAULT_CHECKPOINT_PATH,
    web_proxy_url: Annotated[str | None, typer.Option("--web-proxy", help="当前调用的可信网页代理")] = None,
    web_limits_file: Annotated[
        Path | None, typer.Option("--web-limits", envvar="PRAXIS_WEB_LIMITS", help="新 Turn 的有界网页预算 JSON"),
    ] = None,
    web_search_key_file: Annotated[
        Path | None, typer.Option("--web-search-key-file",
                                 help="工作区外的受保护搜索服务密钥文件"),
    ] = None,
    web_search_provider: Annotated[
        str | None, typer.Option("--web-search-provider",
                                 help="deepseek | brave | tavily；默认未配置则搜索不可用；仅提供密钥时用 Brave"),
    ] = None,
    permissions: Annotated[Path | None, typer.Option("--permissions", help="当前权限上限；旧授权不能扩大它")] = None,
    approval_mode: Annotated[
        ApprovalMode | None, typer.Option("--approval-mode", help="覆盖聊天保存的审批模式：ask | auto | full"),
    ] = None,
) -> None:
    """重新打开旧聊天：只读回放历史，下一条消息创建新 Turn。原任务恢复使用 recover。"""
    agent_chat(resume_thread_id=thread_id or "", checkpoint_db=checkpoint_db,
               web_proxy_url=web_proxy_url, web_limits_file=web_limits_file,
               web_search_key_file=web_search_key_file, web_search_provider=web_search_provider,
               **_permission_options(permissions, approval_mode))


@agent_app.command(name="recover")
def agent_recover(
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
        typer.Option("--rollout-db", "--checkpoint-db", help="完整 SQLite Rollout／事件数据库"),
    ] = DEFAULT_CHECKPOINT_PATH,
    action: Annotated[
        str | None,
        typer.Option(
            "--action",
            help=("不带 --action 运行 recover，可查看当前可用动作"),
        ),
    ] = None,
    user_input: Annotated[
        str | None,
        typer.Option("--input", help="clarification/choice 恢复时的用户输入"),
    ] = None,
    request_id: Annotated[
        str | None, typer.Option("--request-id", help="只响应指定的持久请求；请求改变时先重新查看"),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="详细输出")] = False,
    web_limits_file: Annotated[
        Path | None, typer.Option("--web-limits", envvar="PRAXIS_WEB_LIMITS",
                                 help="有界网页预算JSON；恢复使用原Turn配置"),
    ] = None,
    web_proxy_url: Annotated[
        str | None, typer.Option("--web-proxy", help="可信 HTTP(S) 网页代理；direct 强制直连；默认读取启动环境"),
    ] = None,
    web_search_key_file: Annotated[
        Path | None,
        typer.Option("--web-search-key-file",
                     help="工作区外、仅所有者可读的搜索服务密钥文件"),
    ] = None,
    web_search_provider: Annotated[
        str | None, typer.Option("--web-search-provider",
                                 help="deepseek | brave | tavily；恢复时须与原 Turn 的搜索服务一致"),
    ] = None,
    permissions: Annotated[Path | None, typer.Option("--permissions", help="当前权限上限；旧授权不能扩大它")] = None,
    approval_mode: Annotated[
        ApprovalMode | None, typer.Option("--approval-mode", help="恢复时使用的当前审批模式：ask | auto | full"),
    ] = None,
) -> None:
    """先读取持久化 Turn 元数据，再恢复未完成的 Turn。"""
    if turn_id is not None and last:
        raise typer.BadParameter("Turn ID 与 --last 不能同时使用")
    if all_workspaces and not last:
        raise typer.BadParameter("--all 只能与 --last 一起使用")
    effective_turn_id = turn_id
    if last:
        latest = (
            _latest_abortable_cli_turn(
                checkpoint_db,
                workspace_path=None if all_workspaces else Path.cwd(),
            )
            if action == "abort"
            else _latest_cli_turn(
                checkpoint_db,
                workspace_path=None if all_workspaces else Path.cwd(),
            )
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
    turn_metadata = (
        _abortable_cli_turn(checkpoint_db, effective_turn_id)
        if action == "abort"
        else _cli_turn(checkpoint_db, effective_turn_id)
    )
    web_options: _WebOptions = {}
    if web_limits_file is not None:
        web_options["web_limits_file"] = web_limits_file
    if web_search_key_file is not None:
        web_options["web_search_key_file"] = web_search_key_file
    if web_search_provider is not None:
        web_options["web_search_provider"] = web_search_provider
    if web_proxy_url is not None:
        web_options["web_proxy_url"] = web_proxy_url
    facade = _create_agent_facade(
        model=None,
        checkpoint_db=checkpoint_db,
        workspace_path=turn_metadata.runtime.workspace_path,
        knowledge=turn_metadata.runtime.knowledge,
        **_permission_options(permissions, approval_mode),
        **web_options,
    )
    event_display = _CLIToolEventDisplay()
    if action is None:
        recovery = _run_cli_async(facade.recovery_state(effective_turn_id))
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
            recovery=recovery,
        )
        raise typer.Exit(code=2)
    result = _run_cli_async(
        _resume_facade_command(
            facade,
            turn_id=effective_turn_id,
            action=action,
            user_input=user_input,
            event_display=event_display,
            expected_request_id=request_id,
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


async def _watch_command(
    *,
    workspace: Path,
    checkpoint_db: Path,
    permissions: Path | None,
    approval_mode: ApprovalMode,
    branch: str | None,
    thread_id: str | None,
    timeout_seconds: float,
    poll_interval_seconds: float,
    fix: bool,
) -> int:
    """Wait for CI on a delivery branch; on failure hand the report to the agent."""
    from agent_runtime.git_delivery import (
        BRANCH_PREFIX,
        DeliveryError,
        GitDelivery,
        compose_ci_failure_message,
    )

    delivery = GitDelivery(workspace)
    try:
        resolved_branch = branch
        if resolved_branch is None:
            resolved_branch = delivery.current_branch()
            if not resolved_branch.startswith(BRANCH_PREFIX):
                resolved_branch = delivery.latest_delivery_branch() or ""
        if not resolved_branch.startswith(BRANCH_PREFIX):
            print("没有可观察的 praxis/* 分支；请先 /commit，或用 --branch 指定。")
            return 2
        print(f"观察分支 {resolved_branch} 的 CI（最长 {int(timeout_seconds)} 秒，"
              f"每 {int(poll_interval_seconds)} 秒轮询；Ctrl+C 取消）…")
        report = await delivery.pull_request_checks(
            resolved_branch,
            include_failure_logs=True,
            wait_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        failed = [check for check in report.checks if check.state in {"fail", "cancel"}]
        if report.overall == "pass":
            print(f"CI 全部通过 ✅（{len(report.checks)} 项）。")
            return 0
        if report.overall == "none":
            print("该 Pull Request 没有配置检查。")
            return 2
        if report.overall == "pending":
            print("等待超时，CI 仍在进行中；可再次运行 agent watch 继续。")
            return 2
        for check in failed:
            print(f"  ❌ {check.name}（{check.workflow}）")
        if not fix:
            print("已按 --no-fix 停止；去掉该参数可把失败交给 agent 分析修复。")
            return 1
        resolved_thread = thread_id
        if resolved_thread is None:
            if not checkpoint_db.exists():
                print("当前工作区没有历史会话可续写；用 --thread 指定，或使用 --no-fix。")
                return 2
            sessions = await asyncio.to_thread(list_chat_sessions, checkpoint_db, workspace=workspace)
            if not sessions:
                print("当前工作区没有历史会话可续写；用 --thread 指定，或使用 --no-fix。")
                return 2
            resolved_thread = sessions[0].thread_id
        message = compose_ci_failure_message(
            branch=report.branch,
            head_commit=report.head_commit,
            pull_request_url=report.pull_request_url,
            failed=[(check.name, check.workflow) for check in failed],
            failure_logs=report.failure_logs,
        )
        facade = _create_agent_facade(
            checkpoint_db=checkpoint_db,
            workspace_path=workspace,
            permissions=permissions,
            approval_mode=approval_mode,
            enable_workspace_mcp=False,
        )
        print(f"把失败详情交给会话 {resolved_thread} 的 agent 分析…")
        try:
            async with facade.session(
                thread_id=resolved_thread, approval_mode=approval_mode,
            ) as session:
                result = await session.submit(message)
        except ConversationBusyError as error:
            print(f"会话正由其他进程使用，未能续写：{error}")
            print("失败详情已打印在上方，可稍后在聊天中手动交给 agent。")
            return 2
        if result.status == "paused":
            print("agent 等待审批或补充输入；用 agent resume 选择会话继续。")
            return 1
        print(result.answer or "（agent 未返回文字结论）")
        print(f"验证状态: {result.validation_status}")
        print("修复完成后请检查工作区，再运行聊天内 /commit 提交、/pull 推送。")
        return 1
    except DeliveryError as error:
        print("观察失败: " + safe_terminal_text(str(error)))
        return 2


@agent_app.command(name="watch")
def agent_watch(
    branch: Annotated[
        str | None, typer.Option("--branch", help="要观察的分支；默认当前分支或最新的 praxis/* 分支")
    ] = None,
    thread_id: Annotated[
        str | None, typer.Option("--thread", help="CI 失败时要续写的会话 ID；默认当前工作区最近会话")
    ] = None,
    timeout_seconds: Annotated[
        float, typer.Option("--timeout", min=10, max=7200, help="最长等待秒数")
    ] = 1800.0,
    poll_interval_seconds: Annotated[
        float, typer.Option("--interval", min=5, max=600, help="轮询间隔秒数")
    ] = 30.0,
    fix: Annotated[
        bool, typer.Option("--fix/--no-fix", help="CI 失败时是否把失败详情交给 agent 分析修复")
    ] = True,
    checkpoint_db: Annotated[
        Path, typer.Option("--rollout-db", "--checkpoint-db", help="完整 SQLite Rollout／事件数据库")
    ] = DEFAULT_CHECKPOINT_PATH,
    permissions: Annotated[Path | None, typer.Option("--permissions", help="当前权限上限")] = None,
    approval_mode: Annotated[
        ApprovalMode, typer.Option("--approval-mode", help="续写使用的审批模式：ask | auto | full")
    ] = "auto",
) -> None:
    """等待交付分支的 CI；失败时自动把详情交给该工作区会话的 agent 修复。

    交付动作保持用户确认：watch 只完成观察与分析，/commit 与 /pull 仍由你在聊天中执行。
    """
    raise typer.Exit(_run_cli_async(_watch_command(
        workspace=Path.cwd(),
        checkpoint_db=checkpoint_db,
        permissions=permissions,
        approval_mode=approval_mode,
        branch=branch,
        thread_id=thread_id,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        fix=fix,
    )))
