"""``praxis memory``:工作区 memory 的人类面 CLI(阶段4 T13;设计 §5.14)。

最小集(控制器裁决,旧 ``agent_runtime/cli.py:274-330`` 语义):
``status`` / ``remember`` / ``edit`` / ``delete`` / ``explain``;全部支持
``--json``。configure/update/relocate 不移植:update 归后台 driver(Q6),
relocate 因每 workspace 一库(scope 机制消亡)不存在。

契约:
- 本模块只 import ``praxis.interfaces.sdk``(praxis_cli_via_sdk);core/store
  一律经 sdk 的公开助手触达;
- edit/delete 的 CAS + record_revision + prune 全部在 ``core.memory`` 内,
  CLI 只把 KeyError/ValueError/RuntimeError 映射为统一退出码;
- 退出码与主 CLI 一致:0 完成 / 1 失败 / 2 暂停待输入 / 3 参数错误。

实现说明:memory 状态在 workspace 库里,CLI 经 ``sdk.open_session``(唯一装配
根,空剧本 adapter 只是 open_session 的必填参数)取得 store/tools,再调用
sdk 的 memory 助手。打开会话会走既有 sweep,与 ``praxis status/log`` 同口径。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

import typer

from praxis.interfaces.sdk import (
    MemoryEntryView,
    MemoryExplanation,
    MemoryMutation,
    MemoryStateConflict,
    MemoryStatusView,
    ScriptedAdapter,
    Session,
    delete_memory,
    edit_memory,
    explain_memory,
    open_session,
    read_memory_status,
    remember_memory,
)

EXIT_COMPLETED = 0
EXIT_FAILED = 1
EXIT_PAUSED = 2
EXIT_USAGE = 3

_WORKSPACE_OPT = typer.Option(None, "--workspace", help="Workspace directory (default: cwd).")
_JSON_OPT = typer.Option(False, "--json", help="Print JSON instead of human-readable text.")
_TEXT_OPT = typer.Option(..., "--text", help="Entry text (1..2000 chars).")
_TITLE_OPT = typer.Option(None, "--title", help="Short navigation title (default: the key).")
_KIND_OPT = typer.Option(
    None, "--kind", help="Entry kind: preference | fact | reference (default: fact)."
)
_EXPECTED_VERSION_OPT = typer.Option(
    None, "--expected-version", help="Reject the write unless state is still at this version (CAS)."
)
_VERSION_OPT = typer.Option(None, "--version", help="Read this historical state snapshot instead of current.")
_OFFSET_OPT = typer.Option(0, "--offset", help="History page offset (>= 0).")
_LIMIT_OPT = typer.Option(8, "--limit", help="History page size (1..8).")

app = typer.Typer(
    help="Inspect and edit workspace memory (status/remember/edit/delete/explain).",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# 会话打开与错误/退出码映射
# ---------------------------------------------------------------------------


def _fail(message: str, code: int) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code)


@contextmanager
def _session(workspace: Path | None) -> Iterator[Session]:
    """打开 workspace 会话(空剧本 adapter);装配错误 → 退出码 1。"""
    root = Path(workspace) if workspace is not None else Path.cwd()
    try:
        session = open_session(root, ScriptedAdapter([]), thread_id="default")
    except (KeyError, OSError, ValueError) as exc:
        _fail(str(exc), EXIT_FAILED)
    try:
        yield session
    finally:
        session.close()


def _domain_error(exc: Exception) -> NoReturn:
    """memory 边界错误 → 统一退出码(3 参数错误 / 1 运行时失败)。"""
    if isinstance(exc, MemoryStateConflict):
        _fail(str(exc), EXIT_FAILED)
    if isinstance(exc, KeyError):
        _fail(str(exc.args[0]) if exc.args else str(exc), EXIT_USAGE)
    if isinstance(exc, ValueError):
        _fail(str(exc), EXIT_USAGE)
    if isinstance(exc, (OSError, sqlite3.Error, RuntimeError)):
        _fail(str(exc), EXIT_FAILED)
    raise exc


def _entry_payload(entry: MemoryEntryView) -> dict[str, Any]:
    return {
        "key": entry.key,
        "title": entry.title,
        "text": entry.text,
        "kind": entry.kind,
        "entry_id": entry.entry_id,
        "origin": entry.origin,
        "updated_at": entry.updated_at,
        "source_item_ids": list(entry.source_item_ids),
        "supersedes": entry.supersedes,
        "evidence": list(entry.evidence),
    }


def _status_payload(view: MemoryStatusView) -> dict[str, Any]:
    job = view.job
    return {
        "version": view.version,
        "total_entries": view.total_entries,
        "use_memories": view.use_memories,
        "generate_memories": view.generate_memories,
        "entries": [_entry_payload(entry) for entry in view.entries],
        "job": {
            "status": job.status,
            "in_flight": job.in_flight,
            "lease_active": job.lease_active,
            "turn_ids": list(job.turn_ids),
            "retry_after": job.retry_after,
            "error_type": job.error_type,
            "error_message": job.error_message,
            "source_thread": job.source_thread,
        },
    }


def _print_status(view: MemoryStatusView) -> None:
    typer.echo(f"memory version: {view.version}")
    typer.echo(f"entries: {view.total_entries}")
    typer.echo(f"use_memories: {str(view.use_memories).lower()}")
    typer.echo(f"generate_memories: {str(view.generate_memories).lower()}")
    for entry in view.entries:
        typer.echo(f"  {entry.key}\t{entry.kind}\t{entry.title}")
    job = view.job
    if job.status is None:
        typer.echo("job: none")
        return
    line = f"job: {job.status}"
    if job.status in ("running", "needs_reconciliation"):
        # run_update 只读判定:B4 在途 = 未终态 turn 或未闭合事实窗口。
        # T13-81:租约未过期 = worker 可能仍活,不过度声明 takeover。
        if job.in_flight:
            transit = "worker active" if job.lease_active else "awaiting lease takeover"
            line += f" (in transit — {transit}; unresolved maintenance turns)"
        else:
            line += " (no in-flight maintenance work)"
    typer.echo(line)
    if job.source_thread is not None:
        typer.echo(f"  source: {job.source_thread}")
    if job.turn_ids:
        typer.echo(f"  turn_ids: {', '.join(job.turn_ids)}")
    if job.retry_after is not None:
        typer.echo(f"  retry_after: {job.retry_after:.0f}")
    if job.error_message is not None:
        typer.echo(f"  error: {job.error_type}: {job.error_message}")


def _mutation_payload(action: str, mutation: MemoryMutation) -> dict[str, Any]:
    return {
        "action": action,
        "version": mutation.version,
        "key": mutation.entry.key if mutation.entry is not None else mutation.deleted_key,
        "entry": _entry_payload(mutation.entry) if mutation.entry is not None else None,
    }


def _print_mutation(action: str, mutation: MemoryMutation) -> None:
    if mutation.entry is not None:
        typer.echo(f"ok: {action} {mutation.entry.key!r} (version {mutation.version})")
    else:
        typer.echo(f"ok: deleted {mutation.deleted_key!r} (version {mutation.version})")


def _explanation_payload(key: str, explanation: MemoryExplanation) -> dict[str, Any]:
    return {
        "version": explanation.version,
        "key": key,
        "entry": _entry_payload(explanation.entry) if explanation.entry is not None else None,
        "forgotten": explanation.forgotten,
        "history": [dict(record) for record in explanation.history],
        "total_changes": explanation.total_changes,
        "next_offset": explanation.next_offset,
        "evidence_policy": explanation.evidence_policy,
    }


def _print_explanation(key: str, explanation: MemoryExplanation) -> None:
    typer.echo(f"memory version: {explanation.version}")
    typer.echo(f"key: {key}")
    typer.echo(f"forgotten: {str(explanation.forgotten).lower()}")
    if explanation.entry is not None:
        typer.echo(f"entry: {explanation.entry.title} ({explanation.entry.kind})")
        typer.echo(f"  text: {explanation.entry.text}")
    typer.echo(f"history ({explanation.total_changes} change(s)):")
    for record in explanation.history:
        typer.echo(
            f"  v{record.get('version')} {record.get('action')} "
            f"{record.get('origin')} entry_id={record.get('entry_id')}"
        )
    if explanation.next_offset is not None:
        typer.echo(f"next offset: {explanation.next_offset}")


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------


@app.command("status")
def status(
    workspace: Path | None = _WORKSPACE_OPT,
    json_output: bool = _JSON_OPT,
) -> None:
    """显示 state 版本/条目/两旗标与作业行(在途作业如实标注 in transit)。"""
    with _session(workspace) as session:
        try:
            view = read_memory_status(session.store, session.tools)
        except Exception as exc:  # noqa: BLE001 — 边界统一映射(见 _domain_error)
            _domain_error(exc)
    if json_output:
        typer.echo(json.dumps(_status_payload(view), ensure_ascii=False, indent=2))
    else:
        _print_status(view)
    raise typer.Exit(EXIT_COMPLETED)


@app.command("remember")
def remember(
    key: str = typer.Argument(..., help="Memory key (stable subject key, 1..120 chars)."),
    text: str = _TEXT_OPT,
    title: str | None = _TITLE_OPT,
    kind: str | None = _KIND_OPT,
    expected_version: int | None = _EXPECTED_VERSION_OPT,
    workspace: Path | None = _WORKSPACE_OPT,
    json_output: bool = _JSON_OPT,
) -> None:
    """新建显式 memory entry(origin=explicit);重复 key 拒绝。"""
    with _session(workspace) as session:
        try:
            mutation = remember_memory(
                session.store,
                key=key,
                text=text,
                title=title,
                kind=kind or "fact",
                expected_version=expected_version,
            )
        except Exception as exc:  # noqa: BLE001 — 边界统一映射
            _domain_error(exc)
    if json_output:
        typer.echo(json.dumps(_mutation_payload("remember", mutation), ensure_ascii=False, indent=2))
    else:
        _print_mutation("remembered", mutation)
    raise typer.Exit(EXIT_COMPLETED)


@app.command("edit")
def edit(
    key: str = typer.Argument(..., help="Memory key to edit."),
    text: str = _TEXT_OPT,
    title: str | None = _TITLE_OPT,
    kind: str | None = _KIND_OPT,
    expected_version: int | None = _EXPECTED_VERSION_OPT,
    workspace: Path | None = _WORKSPACE_OPT,
    json_output: bool = _JSON_OPT,
) -> None:
    """改写既有 entry(省略 title/kind 时保留旧值);record_revision 记修订。"""
    with _session(workspace) as session:
        try:
            mutation = edit_memory(
                session.store,
                key=key,
                text=text,
                title=title,
                kind=kind,
                expected_version=expected_version,
            )
        except Exception as exc:  # noqa: BLE001 — 边界统一映射
            _domain_error(exc)
    if json_output:
        typer.echo(json.dumps(_mutation_payload("edit", mutation), ensure_ascii=False, indent=2))
    else:
        _print_mutation("edited", mutation)
    raise typer.Exit(EXIT_COMPLETED)


@app.command("delete")
def delete(
    key: str = typer.Argument(..., help="Memory key to delete."),
    expected_version: int | None = _EXPECTED_VERSION_OPT,
    workspace: Path | None = _WORKSPACE_OPT,
    json_output: bool = _JSON_OPT,
) -> None:
    """删除 entry(deleted_keys/deleted_sources 防复生;entry_history 记 delete)。"""
    with _session(workspace) as session:
        try:
            mutation = delete_memory(session.store, key=key, expected_version=expected_version)
        except Exception as exc:  # noqa: BLE001 — 边界统一映射
            _domain_error(exc)
    if json_output:
        typer.echo(
            json.dumps(
                {"version": mutation.version, "deleted_key": mutation.deleted_key},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        _print_mutation("deleted", mutation)
    raise typer.Exit(EXIT_COMPLETED)


@app.command("explain")
def explain(
    key: str = typer.Argument(..., help="Memory key to explain (history is kept after delete)."),
    version: int | None = _VERSION_OPT,
    offset: int = _OFFSET_OPT,
    limit: int = _LIMIT_OPT,
    workspace: Path | None = _WORKSPACE_OPT,
    json_output: bool = _JSON_OPT,
) -> None:
    """读 entry + 分页修订历史;--version 读历史快照(memory:state:version:<n>)。"""
    with _session(workspace) as session:
        try:
            explanation = explain_memory(
                session.store, key=key, version=version, offset=offset, limit=limit
            )
        except Exception as exc:  # noqa: BLE001 — 边界统一映射
            _domain_error(exc)
    if json_output:
        typer.echo(json.dumps(_explanation_payload(key, explanation), ensure_ascii=False, indent=2))
    else:
        _print_explanation(key, explanation)
    raise typer.Exit(EXIT_COMPLETED)


__all__ = ["app"]