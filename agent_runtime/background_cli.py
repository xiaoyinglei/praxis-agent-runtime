"""CLI attachments use the same model-free SDK as Python callers."""
from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from agent_runtime.background import BackgroundHandle
from agent_runtime.harness.host_ops import HostCommand
from agent_runtime.host_worker import serve

background_app = typer.Typer(help="查看、接入和控制后台编码任务。")


def _handle(database: Path, workspace: Path, thread_id: str, turn_id: str) -> BackgroundHandle:
    return BackgroundHandle(database=database, workspace=workspace, thread_id=thread_id, turn_id=turn_id)


def _receipt(command: HostCommand) -> None:
    typer.echo(json.dumps({
        "command_id": command.command_id, "thread_id": command.thread_id,
        "turn_id": command.turn_id, "status": command.status,
    }, ensure_ascii=False))


@background_app.command("status")
def status(
    thread_id: str, turn_id: str,
    database: Annotated[Path, typer.Option("--checkpoint-db", "--rollout-db")],
    workspace: Annotated[Path, typer.Option("--workspace")] = Path("."),
) -> None:
    typer.echo(json.dumps(asyncio.run(_handle(database, workspace, thread_id, turn_id).status()), ensure_ascii=False))


@background_app.command("events")
def events(
    thread_id: str, turn_id: str,
    database: Annotated[Path, typer.Option("--checkpoint-db", "--rollout-db")],
    workspace: Annotated[Path, typer.Option("--workspace")] = Path("."),
    after: Annotated[str | None, typer.Option("--after")] = None,
) -> None:
    typer.echo(json.dumps(asyncio.run(_handle(database, workspace, thread_id, turn_id).events(after=after)),
                          ensure_ascii=False))


@background_app.command("wait")
def wait(
    thread_id: str, turn_id: str,
    database: Annotated[Path, typer.Option("--checkpoint-db", "--rollout-db")],
    workspace: Annotated[Path, typer.Option("--workspace")] = Path("."),
    timeout: Annotated[float | None, typer.Option("--timeout", min=0)] = None,
) -> None:
    """等待终态或需要人处理的暂停；超时不取消执行。"""
    handle = _handle(database, workspace, thread_id, turn_id)
    typer.echo(json.dumps(asyncio.run(handle.wait(timeout=timeout)), ensure_ascii=False))


@background_app.command("result")
def result(
    thread_id: str, turn_id: str,
    database: Annotated[Path, typer.Option("--checkpoint-db", "--rollout-db")],
    workspace: Annotated[Path, typer.Option("--workspace")] = Path("."),
) -> None:
    value = asyncio.run(_handle(database, workspace, thread_id, turn_id).result())
    typer.echo(json.dumps({
        "thread_id": value.thread_id, "turn_id": value.turn_id, "status": value.status,
        "answer": value.answer, "stop_reason": value.stop_reason, "usage": asdict(value.usage),
        "diagnostics": [asdict(d) for d in value.diagnostics],
        "delivery": copy.deepcopy(value.delivery),
    }, ensure_ascii=False))


@background_app.command("send")
def send(
    thread_id: str, turn_id: str, text: str,
    database: Annotated[Path, typer.Option("--checkpoint-db", "--rollout-db")],
    workspace: Annotated[Path, typer.Option("--workspace")] = Path("."),
    input_id: Annotated[str | None, typer.Option("--input-id")] = None,
) -> None:
    _receipt(asyncio.run(_handle(database, workspace, thread_id, turn_id).send(text, input_id=input_id)))


@background_app.command("respond")
def respond(
    thread_id: str, turn_id: str, request_id: str, action: str,
    database: Annotated[Path, typer.Option("--checkpoint-db", "--rollout-db")],
    workspace: Annotated[Path, typer.Option("--workspace")] = Path("."),
    text: Annotated[str | None, typer.Option("--text")] = None,
    input_id: Annotated[str | None, typer.Option("--input-id")] = None,
) -> None:
    _receipt(asyncio.run(_handle(database, workspace, thread_id, turn_id).respond(
        request_id, action, text=text, input_id=input_id
    )))


@background_app.command("resume")
def resume(
    thread_id: str, turn_id: str,
    database: Annotated[Path, typer.Option("--checkpoint-db", "--rollout-db")],
    workspace: Annotated[Path, typer.Option("--workspace")] = Path("."),
    action: Annotated[str, typer.Option("--action")] = "continue",
    input_id: Annotated[str | None, typer.Option("--input-id")] = None,
) -> None:
    _receipt(asyncio.run(_handle(database, workspace, thread_id, turn_id).resume(action=action, input_id=input_id)))


@background_app.command("cancel")
def cancel(
    thread_id: str, turn_id: str,
    database: Annotated[Path, typer.Option("--checkpoint-db", "--rollout-db")],
    workspace: Annotated[Path, typer.Option("--workspace")] = Path("."),
    input_id: Annotated[str | None, typer.Option("--input-id")] = None,
) -> None:
    _receipt(asyncio.run(_handle(database, workspace, thread_id, turn_id).cancel(input_id=input_id)))


@background_app.command("serve")
def host_serve(
    database: Annotated[Path, typer.Option("--checkpoint-db", "--rollout-db")],
    concurrency: Annotated[int, typer.Option("--concurrency", min=1, max=32)] = 4,
) -> None:
    """在当前进程托管队列；普通提交会自动启动独立后台进程。"""
    asyncio.run(serve(database, concurrency=concurrency))
