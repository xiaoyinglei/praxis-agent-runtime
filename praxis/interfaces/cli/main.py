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
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

import click
import typer
from typer.core import TyperGroup

from praxis.interfaces.cli.mcp_cli import app as mcp_app
from praxis.interfaces.cli.models_cli import app as models_app
from praxis.interfaces.cli.render import render_item
from praxis.interfaces.cli.slash import REGISTRY, ExitChat
from praxis.interfaces.cli.web_cli import app as web_app
from praxis.interfaces.sdk import (
    UNLIMITED,
    ConversationBusy,
    Limits,
    McpBundle,
    ModelResponse,
    ModelToolCall,
    PermissionMode,
    PermissionNotice,
    PermissionPolicy,
    ResumeNotConfirmed,
    ScriptedAdapter,
    Session,
    TurnResult,
    TurnStatus,
    Usage,
    connect_mcp_servers,
    open_session,
    resolve_declared_policy,
    resolve_mcp_config_path,
)

EXIT_COMPLETED = 0
EXIT_FAILED = 1
EXIT_PAUSED = 2
EXIT_USAGE = 3

_DEFAULT_THREAD = "default"

_WORKSPACE_OPT = typer.Option(None, "--workspace", help="Workspace directory (default: cwd).")
_MODEL_OPT = typer.Option(
    None,
    "--model",
    help="Registered model id (else the default; PRAXIS_MODEL_ID is the unregistered escape hatch).",
)
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
# 阶段4 T3:预算树 root limits(约定预算,耗尽 → PAUSED 可补预算续跑;
# 与 max_steps 硬护栏语义不同——护栏是失控防御,FAILED 不可续)。
_MAX_TOKENS_OPT = typer.Option(
    None, "--max-tokens", help="Budget cap on total tokens; exhaustion pauses (resume --extend-tokens)."
)
_MAX_COST_MICROS_OPT = typer.Option(
    None,
    "--max-cost-micros",
    help="Budget cap on cost in micros; needs model pricing, else pauses pricing_unavailable.",
)
_MAX_STEPS_OPT = typer.Option(
    None, "--max-steps", help="Budget cap on tool rounds; exhaustion pauses (resumable with budget extend)."
)
_EXTEND_TOKENS_OPT = typer.Option(
    None, "--extend-tokens", help="Add tokens to the turn's budget and resume (budget_extension item)."
)
_EXTEND_COST_MICROS_OPT = typer.Option(
    None, "--extend-cost-micros", help="Add cost micros to the turn's budget and resume (budget_extension item)."
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
    help="Praxis workspace agent — minimal console (run/chat/resume/status/log/steer/verify/models/web/mcp).",
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")
app.add_typer(web_app, name="web")
app.add_typer(mcp_app, name="mcp")


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
    reasoning = entry.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise typer.BadParameter(f"scripted entry [{index}] 'reasoning' must be a string")
    return ModelResponse(
        content=content,
        tool_calls=tuple(calls),
        usage=Usage(0, 0),
        finish_reason=finish_reason,
        reasoning=reasoning,
    )


def _load_scripted(path: Path) -> list[ModelResponse]:
    """剧本格式:``[{content, tool_calls?: [{call_id, name, arguments?}], reasoning?, finish_reason?}]``。

    ``reasoning`` 可选(缺省空串),用于端到端驱动思考通道(R-T2-6)。
    """
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


def _nonneg_flag(value: int | None, flag: str) -> int | None:
    """预算旗标边界拒绝:负值不是限额(预算只增不减),BadParameter → 退出码 3。"""
    if value is not None and value < 0:
        raise typer.BadParameter(f"{flag} must be >= 0, got {value}")
    return value


def _budget_from_flags(max_tokens: int | None, max_cost_micros: int | None, max_steps: int | None) -> Limits:
    """run --max-* → root limits;全缺省 = UNLIMITED(历史路径零扰动)。"""
    return Limits(
        tokens=_nonneg_flag(max_tokens, "--max-tokens"),
        cost_micros=_nonneg_flag(max_cost_micros, "--max-cost-micros"),
        steps=_nonneg_flag(max_steps, "--max-steps"),
    )


def _extend_from_flags(extend_tokens: int | None, extend_cost_micros: int | None) -> dict[str, int]:
    """resume --extend-* → Limits 维度增量映射;空映射 = 不补预算。"""
    extend: dict[str, int] = {}
    tokens = _nonneg_flag(extend_tokens, "--extend-tokens")
    if tokens is not None:
        extend["tokens"] = tokens
    cost = _nonneg_flag(extend_cost_micros, "--extend-cost-micros")
    if cost is not None:
        extend["cost_micros"] = cost
    return extend


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
    model: str | None = None,
    policy: PermissionPolicy | None,
    mode: PermissionMode | None,
    thread_id: str,
    extra_tools: Sequence[Any] = (),
    integrations: Sequence[Any] = (),
    budget: Limits = UNLIMITED,
) -> Session:
    """R12:open_session 的 ValueError(损坏的既有状态,如 .praxis/permissions.json
    无法解析)是运行时失败 → 干净错误消息 + EXIT_FAILED(1),绝不裸 traceback(C4)。
    KeyError 来自模型注册表(未注册/无默认),属于显式配置选择错误 → EXIT_USAGE(3)。
    用法错误(--policy/--scripted 坏参)在此之前已以 BadParameter 抛出,保持退出码 3。"""
    try:
        return open_session(
            workspace,
            adapter,
            model=model,
            policy=policy,
            mode=mode,
            thread_id=thread_id,
            extra_tools=extra_tools,
            integrations=integrations,
            budget=budget,
        )
    except KeyError as exc:
        message = exc.args[0] if exc.args else str(exc)
        typer.echo(f"error: {message}", err=True)
        raise typer.Exit(EXIT_USAGE) from None
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_FAILED) from None


def _open_from_options(
    workspace: Path | None,
    model: str | None,
    scripted: Path | None,
    thread: str | None,
    policy: Path | None = None,
    mode: str | None = None,
    budget: Limits = UNLIMITED,
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
            budget=budget,
        )
    # R-T6-2:CLI 不再直接构造 provider adapter;模型解析与唯一分派都在
    # composition(_adapter_and_spec_from_registry → _adapter_for)。
    return _open_session_guarded(
        _resolve_workspace(workspace),
        None,
        model=model,
        policy=resolved_policy,
        mode=resolved_mode,
        thread_id=_resolve_thread(thread),
        budget=budget,
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


def _echo_notices(notices: tuple[PermissionNotice, ...]) -> None:
    """权限更新披露的单行渲染(D9 等);空元组零输出,退出码不变。"""
    for notice in notices:
        typer.echo(f"notice: {notice.text}")


def _echo_budget_hint(result: TurnResult) -> None:
    """budget_exhausted 暂停的一行文案(T3):点名原因与补预算续跑出路。

    其余暂停原因(approval_required/outcome_unknown)沿用既有渲染,零变化。
    """
    if result.reason == "budget_exhausted":
        typer.echo(
            f"  budget exhausted: praxis resume {result.turn_id} --extend-tokens N (or --extend-cost-micros N)"
        )


def _report_turn(result: TurnResult) -> int:
    """turn 终态 → 打印 + 统一退出码:0 完成 / 1 失败或取消 / 2 暂停待输入。

    权限披露(如 NetworkRule 授权的 D9 联动)在终态之前打印:它是更新生效
    的事实,不是 turn 结局的一部分,绝不改变退出码。
    """
    _echo_notices(result.notices)
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
        _echo_budget_hint(result)
        return EXIT_PAUSED
    typer.echo(f"unexpected turn status: {status.value}", err=True)
    return EXIT_FAILED


def _fmt_time(timestamp: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def _print_turn_ref(session: Session, result: TurnResult) -> None:
    """R15:run/resume 打印可复制的 turn/thread 参照行(`praxis resume <turn_id>` 的入据)。"""
    typer.echo(f"turn: {result.turn_id} (thread: {session.thread_id})")


# ---------------------------------------------------------------------------
# MCP 单 loop 装配(Q9)
# ---------------------------------------------------------------------------


def _mcp_configured(workspace: Path) -> bool:
    """是否存在 MCP 配置;配置损坏时返回 True,让 connect 路径给出干净错误。

    无配置时走**逐字不变的**既有同步装配路径(空 bundle 不引入任何行为变化;
    既有 CLI 测试不因 T19 改写)。显式路径/信任门错误属于运行时失败,交给
    ``_connect_bundle`` 统一打印并退出 1。
    """
    try:
        return resolve_mcp_config_path(workspace) is not None
    except ValueError:
        return True


def _policy_snapshot(workspace: Path, policy: Path | None) -> PermissionPolicy:
    """声明快照的唯一加载点;同一对象同时交给 connect 与 open_session。"""
    explicit = _load_policy_file(Path(policy)) if policy is not None else None
    try:
        return resolve_declared_policy(workspace, explicit)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_FAILED) from None


def _adapter_for_options(model: str | None, scripted: Path | None) -> Any:
    if scripted is not None:
        return ScriptedAdapter(_load_scripted(Path(scripted)))
    return None


async def _connect_bundle(workspace: Path, policy: PermissionPolicy) -> McpBundle:
    try:
        return await connect_mcp_servers(workspace, policy=policy)
    except (OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_FAILED) from None


def _report_mcp_statuses(bundle: McpBundle) -> None:
    """把跳过/不可用诊断如实打到 stderr(绝不静默吞掉一个 server)。"""
    for status in bundle.statuses:
        if status.diagnostic_code is None:
            continue
        typer.echo(
            f"warning: mcp server {status.server_name} ({status.diagnostic_code}): "
            f"{status.diagnostic_message}",
            err=True,
        )


def _open_mcp_session(
    workspace: Path,
    bundle: McpBundle,
    *,
    model: str | None,
    scripted: Path | None,
    policy: PermissionPolicy,
    mode: str | None,
    thread: str | None,
    budget: Limits = UNLIMITED,
) -> Session:
    return _open_session_guarded(
        workspace,
        _adapter_for_options(model, scripted),
        model=model,
        policy=policy,
        mode=cast("PermissionMode | None", mode),
        thread_id=_resolve_thread(thread),
        extra_tools=bundle.tools,
        integrations=(bundle,),
        budget=budget,
    )


# ---------------------------------------------------------------------------
# run / resume
# ---------------------------------------------------------------------------


@app.command()
def run(
    prompt: str,
    workspace: Path | None = _WORKSPACE_OPT,
    model: str | None = _MODEL_OPT,
    scripted: Path | None = _SCRIPTED_OPT,
    thread: str | None = _THREAD_OPT,
    policy: Path | None = _POLICY_OPT,
    mode: str | None = _MODE_OPT,
    max_tokens: int | None = _MAX_TOKENS_OPT,
    max_cost_micros: int | None = _MAX_COST_MICROS_OPT,
    max_steps: int | None = _MAX_STEPS_OPT,
) -> None:
    """一次性 turn:提交 prompt,打印结果,按 turn 终态决定退出码。

    --max-tokens/--max-cost-micros/--max-steps 是预算树 root limits(约定预算):
    耗尽 → PAUSED budget_exhausted,可 resume 补预算续跑;与内部 max_steps
    硬护栏(失控防御,FAILED)语义不同(R-T3-1)。
    """
    budget = _budget_from_flags(max_tokens, max_cost_micros, max_steps)
    root = _resolve_workspace(workspace)
    if not _mcp_configured(root):
        # 无 MCP:既有同步装配路径逐字保留(T19 零行为变化)。
        session = _open_from_options(workspace, model, scripted, thread, policy, mode, budget)
        try:
            result = asyncio.run(session.submit(prompt))
        except ConversationBusy as exc:
            typer.echo(f"busy: {exc}", err=True)
            raise typer.Exit(EXIT_FAILED) from None
        finally:
            session.close()
        _print_turn_ref(session, result)
        raise typer.Exit(_report_turn(result))
    raise typer.Exit(
        asyncio.run(
            _run_with_mcp(
                root,
                prompt,
                model=model,
                scripted=scripted,
                thread=thread,
                policy=policy,
                mode=mode,
                budget=budget,
            )
        )
    )


async def _run_with_mcp(
    root: Path,
    prompt: str,
    *,
    model: str | None,
    scripted: Path | None,
    thread: str | None,
    policy: Path | None,
    mode: str | None,
    budget: Limits = UNLIMITED,
) -> int:
    """Q9:connect → open_session → submit → ``finally`` aclose 都在同一 loop。

    MCP owner task 活在连接它的 loop 里;``asyncio.run`` 之外再 close 会让
    进程组逃逸。``session.aclose`` 先 integrations 后 store,bundle aclose
    幂等(与 Session.close 双触达无害)。
    """
    snapshot = _policy_snapshot(root, policy)
    bundle = await _connect_bundle(root, snapshot)
    _report_mcp_statuses(bundle)
    session: Session | None = None
    result: TurnResult | None = None
    try:
        session = _open_mcp_session(
            root,
            bundle,
            model=model,
            scripted=scripted,
            policy=snapshot,
            mode=mode,
            thread=thread,
            budget=budget,
        )
        try:
            result = await session.submit(prompt)
        except ConversationBusy as exc:
            typer.echo(f"busy: {exc}", err=True)
            return EXIT_FAILED
    finally:
        if session is not None:
            await session.aclose()
        else:
            await bundle.aclose()
    if result is None or session is None:
        return EXIT_FAILED
    _print_turn_ref(session, result)
    return _report_turn(result)


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
    scripted: Path | None = _SCRIPTED_OPT,
    thread: str | None = _THREAD_OPT,
    policy: Path | None = _POLICY_OPT,
    mode: str | None = _MODE_OPT,
    confirm: bool = _CONFIRM_OPT,
    allow: list[str] = _ALLOW_OPT,
    deny: list[str] = _DENY_OPT,
    reconcile: list[str] = _RECONCILE_OPT,
    extend_tokens: int | None = _EXTEND_TOKENS_OPT,
    extend_cost_micros: int | None = _EXTEND_COST_MICROS_OPT,
) -> None:
    """续跑 paused/interrupted turn:--confirm 人工确认,--allow/--deny 审批,--reconcile 对账。

    --extend-tokens/--extend-cost-micros 补预算(budget_extension item)后走
    同一续跑路径:budget_exhausted 的 PAUSED 无需审批即可续跑(T3)。
    """
    approval_map = _parse_approval(allow, deny)
    reconcile_map = _parse_reconcile(reconcile)
    extend_map = _extend_from_flags(extend_tokens, extend_cost_micros)
    root = _resolve_workspace(workspace)
    if not _mcp_configured(root):
        session = _open_from_options(workspace, model, scripted, thread, policy, mode)
        try:
            result = asyncio.run(
                session.resume(
                    turn_id,
                    confirm=confirm,
                    approval=approval_map or None,
                    reconcile=reconcile_map or None,
                    extend=extend_map or None,
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
    raise typer.Exit(
        asyncio.run(
            _resume_with_mcp(
                root,
                turn_id,
                confirm=confirm,
                approval=approval_map or None,
                reconcile=reconcile_map or None,
                extend=extend_map or None,
                model=model,
                scripted=scripted,
                thread=thread,
                policy=policy,
                mode=mode,
            )
        )
    )


async def _resume_with_mcp(
    root: Path,
    turn_id: str,
    *,
    confirm: bool,
    approval: Mapping[str, Literal["allow", "deny"]] | None,
    reconcile: Mapping[str, Literal["ok", "failed"]] | None,
    extend: Mapping[str, int] | None = None,
    model: str | None,
    scripted: Path | None,
    thread: str | None,
    policy: Path | None,
    mode: str | None,
) -> int:
    """与 ``_run_with_mcp`` 同一条单 loop 纪律(connect → resume → aclose)。"""
    snapshot = _policy_snapshot(root, policy)
    bundle = await _connect_bundle(root, snapshot)
    _report_mcp_statuses(bundle)
    session: Session | None = None
    result: TurnResult | None = None
    try:
        session = _open_mcp_session(
            root,
            bundle,
            model=model,
            scripted=scripted,
            policy=snapshot,
            mode=mode,
            thread=thread,
        )
        try:
            result = await session.resume(
                turn_id, confirm=confirm, approval=approval, reconcile=reconcile, extend=extend
            )
        except (ResumeNotConfirmed, ValueError) as exc:
            typer.echo(f"error: {exc}", err=True)
            return EXIT_USAGE
        except ConversationBusy as exc:
            typer.echo(f"busy: {exc}", err=True)
            return EXIT_FAILED
    finally:
        if session is not None:
            await session.aclose()
        else:
            await bundle.aclose()
    if result is None or session is None:
        return EXIT_FAILED
    _print_turn_ref(session, result)
    return _report_turn(result)


# ---------------------------------------------------------------------------
# chat(C1 错误边界 + slash 注册表分派)
# ---------------------------------------------------------------------------


def _read_answer(prompt_text: str) -> str | None:
    """input() 一行;EOF/Ctrl+C → None(调用方决定退出路径)。"""
    try:
        return input(prompt_text)
    except (EOFError, KeyboardInterrupt):
        return None


def _parse_verdict(answer: str, choices: Mapping[str, str]) -> str | None:
    """一行回答 → verdict(大小写/首尾空白不敏感);不匹配返回 None。"""
    return choices.get(answer.strip().lower())


def _ask_verdict(question: str, choices: Mapping[str, str]) -> str | None:
    while True:
        answer = _read_answer(question)
        if answer is None:
            return None
        verdict = _parse_verdict(answer, choices)
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


def _resolve_future(future: asyncio.Future[str | None], value: str | None) -> None:
    """loop 线程内的 future 结算;已被取消/已结算时静默忽略。"""
    if not future.done():
        future.set_result(value)


async def _read_answer_async(prompt_text: str) -> str | None:
    """R-T19-2:daemon 线程读 stdin,**不占默认 executor**。

    ``asyncio.to_thread`` 的 worker 会阻塞在 ``input()`` 上,而首次 Ctrl+C
    之后 ``Runner.close()`` 会 join 默认 executor 的工作线程 ⇒ 进程挂住。
    专起 daemon 线程并用 ``call_soon_threadsafe`` 结算 future:第一次 Ctrl+C
    取消 ``await`` 即可干净返回,阻塞中的 daemon 线程随进程退出(它可能仍
    握着终端提示符;这是本方案已知的提示符回显角落,不阻塞退出)。

    同一机制服务全部 REPL 提问:主提示 ``> ``、审批 y/n 与对账 ok/failed
    (见 ``_ask_verdict_async``)。EOF/Ctrl+C 一律结算 ``None``。
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str | None] = loop.create_future()

    def read() -> None:
        try:
            value = _read_answer(prompt_text)
        except BaseException:  # noqa: BLE001 — T20:吊销 tty 的 OSError 等异常
            # daemon 线程无法把异常带出:若 future 不结算,``await`` 会永久挂起。
            # 一律以 None 哨兵结算(_chat_loop_async 把 None 当 EOF 退出),异常
            # 类型不进入控制流。
            value = None
        try:
            loop.call_soon_threadsafe(_resolve_future, future, value)
        except RuntimeError:
            return  # loop 已关闭:线程随进程退出

    threading.Thread(target=read, name="praxis-chat-input", daemon=True).start()
    return await future


async def _ask_verdict_async(question: str, choices: Mapping[str, str]) -> str | None:
    """R-T19-2 同族:审批/对账提示经 daemon 读,**不占默认 executor**。

    审批提示曾把 ``_ask_verdict`` 丢进默认 executor:turn PAUSED 后 Ctrl+C 取消
    ``await``,``asyncio.run`` 收尾仍会 join 默认 executor 里阻塞在 ``input()``
    的 worker ⇒ 与主 REPL 同一挂死。此函数逐次提问走 ``_read_answer_async``,
    语义与同步 ``_ask_verdict`` 一致:EOF/Ctrl+C 时 daemon 线程结算 ``None`` →
    返回 ``None``(调用方输出 ``cancelled`` 并返回,不 resume)。
    """
    while True:
        answer = await _read_answer_async(question)
        if answer is None:
            return None
        verdict = _parse_verdict(answer, choices)
        if verdict is not None:
            return verdict
        typer.echo(f"please answer one of: {', '.join(choices)}")


async def _chat_turn_async(session: Session, text: str) -> None:
    """异步版单 turn 错误边界(与 ``_chat_turn`` 同一 C1 语义)。"""
    try:
        result = await session.submit(text)
    except KeyboardInterrupt:
        typer.echo("cancelled")
        return
    except ConversationBusy as exc:
        typer.echo(f"busy: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 — C1 错误边界
        typer.echo(f"error: {type(exc).__name__}: {exc}")
        return
    await _chat_result_async(session, result)


async def _chat_result_async(session: Session, result: TurnResult) -> None:
    """异步版审批循环;提问走 daemon 读(不占默认 executor),resume 留在 owner loop。"""
    while True:
        code = _report_turn(result)
        if code != EXIT_PAUSED:
            return
        approval: dict[str, Literal["allow", "deny"]] = {}
        for summary in result.pending_approvals:
            verdict = await _ask_verdict_async(
                f"allow {summary.tool_name} (call {summary.call_id}, digest {summary.args_digest[:12]})? [y/n] ",
                _VERDICTS_YES_NO,
            )
            if verdict is None:
                typer.echo("cancelled")
                return
            approval[summary.call_id] = cast("Literal['allow', 'deny']", verdict)
        reconcile: dict[str, Literal["ok", "failed"]] = {}
        for call_id in result.unknown_calls:
            outcome = await _ask_verdict_async(
                f"call {call_id} side effect already happened? [ok/failed] ",
                _VERDICTS_OUTCOME,
            )
            if outcome is None:
                typer.echo("cancelled")
                return
            reconcile[call_id] = cast("Literal['ok', 'failed']", outcome)
        if not approval and not reconcile:
            return
        try:
            result = await session.resume(
                result.turn_id, approval=approval or None, reconcile=reconcile or None
            )
        except KeyboardInterrupt:
            typer.echo("cancelled")
            return
        except Exception as exc:  # noqa: BLE001 — C1 错误边界
            typer.echo(f"error: {type(exc).__name__}: {exc}")
            return


async def _chat_loop_async(session: Session) -> None:
    """异步 REPL:stdin 读取在 daemon 线程(R-T19-2),模型/MCP 调用留在同一循环。"""
    while True:
        line = await _read_answer_async("> ")
        if line is None:
            return
        text = line.strip()
        if not text:
            continue
        if text.startswith("/"):
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
        await _chat_turn_async(session, text)


async def _chat_with_mcp(
    root: Path,
    *,
    model: str | None,
    scripted: Path | None,
    thread: str | None,
    policy: Path | None,
    mode: str | None,
) -> int:
    """Q9:整个 REPL 活在一个 ``asyncio.run`` 里(MCP owner task 不跨 loop)。"""
    snapshot = _policy_snapshot(root, policy)
    bundle = await _connect_bundle(root, snapshot)
    _report_mcp_statuses(bundle)
    session: Session | None = None
    try:
        session = _open_mcp_session(
            root,
            bundle,
            model=model,
            scripted=scripted,
            policy=snapshot,
            mode=mode,
            thread=thread,
        )
        typer.echo(f"praxis chat — thread {session.thread_id} — /help for commands, /exit to leave")
        await _chat_loop_async(session)
        return EXIT_COMPLETED
    finally:
        if session is not None:
            await session.aclose()
        else:
            await bundle.aclose()


@app.command()
def chat(
    workspace: Path | None = _WORKSPACE_OPT,
    model: str | None = _MODEL_OPT,
    scripted: Path | None = _SCRIPTED_OPT,
    thread: str | None = _THREAD_OPT,
    policy: Path | None = _POLICY_OPT,
    mode: str | None = _MODE_OPT,
) -> None:
    """交互式 REPL:每 turn 错误边界(C1);/help /status /exit 查表分派。"""
    root = _resolve_workspace(workspace)
    if not _mcp_configured(root):
        session = _open_from_options(workspace, model, scripted, thread, policy, mode)
        typer.echo(f"praxis chat — thread {session.thread_id} — /help for commands, /exit to leave")
        try:
            _chat_loop(session)
        finally:
            session.close()
        return
    raise typer.Exit(
        asyncio.run(
            _chat_with_mcp(
                root,
                model=model,
                scripted=scripted,
                thread=thread,
                policy=policy,
                mode=mode,
            )
        )
    )


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
        if turn.status == TurnStatus.PAUSED:
            _echo_budget_hint(result)
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
