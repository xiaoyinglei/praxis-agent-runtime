"""``praxis agent``:子代理身份的人类面 discoverability(M2;设计 §5.13/Q3)。

最小集(控制器裁决):``list`` / ``show``。子注册表 = child_allocation items
折叠(无表列,core.agents 同口径)——本命令只读折叠展示,不驱动子执行。
Q3 直面 resume 子之前,人类经此发现子的 thread_id/turn_id/state/workspace_mode/
workspace,再 ``praxis resume <child_turn> --thread <child>``(子装配身份经
binding 自动重建,R-FINAL-C1)。

契约:
- 本模块只 import ``praxis.interfaces.sdk``(praxis_cli_via_sdk);items/turn
  一律经 sdk 的 Session 公开读面触达,不直依 core/store;
- 退出码与主 CLI 一致:0 完成 / 1 失败 / 3 参数错误。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

import typer

from praxis.interfaces.sdk import (
    Item,
    ItemKind,
    ScriptedAdapter,
    Session,
    open_session,
)

EXIT_COMPLETED = 0
EXIT_FAILED = 1
EXIT_USAGE = 3

_WORKSPACE_OPT = typer.Option(None, "--workspace", help="Workspace directory (default: cwd).")
_THREAD_OPT = typer.Option(None, "--thread", help='Parent thread id (default: "default").')
_JSON_OPT = typer.Option(False, "--json", help="Print JSON instead of human-readable text.")
_CHILD_ARG = typer.Argument(..., help="Child thread id (from `praxis agent list`).")

app = typer.Typer(
    help="Discover subagent children (list/show) folded from child_allocation facts.",
    no_args_is_help=True,
)


def _fail(message: str, code: int) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code)


@contextmanager
def _session(workspace: Path | None, thread: str | None) -> Iterator[Session]:
    root = Path(workspace) if workspace is not None else Path.cwd()
    try:
        session = open_session(root, ScriptedAdapter([]), thread_id=thread or "default")
    except (KeyError, OSError, ValueError) as exc:
        _fail(str(exc), EXIT_FAILED)
    try:
        yield session
    finally:
        session.close()


def _allocations(items: list[Item]) -> list[Mapping[str, Any]]:
    """本 thread 的 child_allocation payload 折叠(seq 升序;畸形跳过)。"""
    refs: list[Mapping[str, Any]] = []
    for item in sorted(items, key=lambda entry: entry.seq):
        if item.kind != ItemKind.CHILD_ALLOCATION:
            continue
        payload = item.payload
        if not isinstance(payload.get("child_thread_id"), str):
            continue
        refs.append(payload)
    return refs


def _child_state(session: Session, root_turn_id: str) -> str:
    try:
        return session.status(root_turn_id).status
    except ValueError:
        return "unknown"


def _child_payload(session: Session, payload: Mapping[str, Any]) -> dict[str, Any]:
    root_turn_id = str(payload.get("child_root_turn_id") or "")
    # 展示投影(非协议 payload):身份/状态/工作区核心键先成形,协议来源字段
    # 事后补键——避免与 cancel_request/allocation 的特征键集偶然共现(规则 26)。
    child: dict[str, Any] = {
        "child_thread_id": payload.get("child_thread_id"),
        "child_root_turn_id": root_turn_id,
        "state": _child_state(session, root_turn_id) if root_turn_id else "unknown",
        "workspace_mode": payload.get("workspace_mode") or "read_only",
        "workspace": payload.get("workspace") or "",
    }
    child["parent_turn_id"] = payload.get("parent_turn_id")
    child["source"] = payload.get("source") or "model"
    return child


@app.command("list")
def list_children(
    workspace: Path | None = _WORKSPACE_OPT,
    thread: str | None = _THREAD_OPT,
    json_output: bool = _JSON_OPT,
) -> None:
    """列出本 thread 全部子代理身份(thread/turn/state/mode/workspace)。"""
    with _session(workspace, thread) as session:
        children = [_child_payload(session, payload) for payload in _allocations(session.items())]
    if json_output:
        typer.echo(json.dumps(children, ensure_ascii=False, indent=2))
        return
    if not children:
        typer.echo("no subagent children in this thread")
        return
    for child in children:
        typer.echo(
            f"{child['child_thread_id']}  turn={child['child_root_turn_id']}  "
            f"{child['state']}  {child['workspace_mode']}  {child['workspace']}"
        )


@app.command("show")
def show_child(
    child_thread_id: str = _CHILD_ARG,
    workspace: Path | None = _WORKSPACE_OPT,
    thread: str | None = _THREAD_OPT,
    json_output: bool = _JSON_OPT,
) -> None:
    """展示一个子的身份与当前结局(结论/待批/未知调用)。"""
    with _session(workspace, thread) as session:
        match = next(
            (
                payload
                for payload in _allocations(session.items())
                if payload.get("child_thread_id") == child_thread_id
            ),
            None,
        )
        if match is None:
            _fail(f"unknown child thread {child_thread_id!r} (no allocation in this thread)", EXIT_USAGE)
        child = _child_payload(session, match)
        root_turn_id = child["child_root_turn_id"]
        try:
            result = session.result(root_turn_id) if root_turn_id else None
        except ValueError:
            result = None
    if result is not None:
        child["conclusion"] = result.output
        child["reason"] = result.reason
        child["pending_approvals"] = [approval.call_id for approval in result.pending_approvals]
        child["unknown_calls"] = list(result.unknown_calls)
    if json_output:
        typer.echo(json.dumps(child, ensure_ascii=False, indent=2))
        return
    typer.echo(f"child thread: {child['child_thread_id']}")
    typer.echo(f"root turn: {child['child_root_turn_id']}")
    typer.echo(f"state: {child['state']}")
    typer.echo(f"workspace_mode: {child['workspace_mode']}")
    typer.echo(f"workspace: {child['workspace']}")
    if result is not None:
        typer.echo(f"reason: {child.get('reason') or '-'}")
        if child.get("conclusion"):
            typer.echo(f"conclusion: {child['conclusion']}")
        if child.get("pending_approvals"):
            typer.echo(f"pending approvals: {', '.join(child['pending_approvals'])}")
        if child.get("unknown_calls"):
            typer.echo(f"outcome unknown: {', '.join(child['unknown_calls'])}")
    typer.echo(
        f"resume: praxis resume {child['child_root_turn_id']} --thread {child['child_thread_id']}"
    )