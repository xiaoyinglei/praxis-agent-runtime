"""Inspect or cancel a crashed Agent's command without opening a model."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Literal

import typer

from agent_runtime.workspace import DEFAULT_CHECKPOINT_PATH


def commands(
    turn_id: Annotated[str, typer.Argument(help="原命令所属的 Turn ID")],
    session_id: Annotated[str, typer.Argument(help="run_command 返回的 session_id")],
    action: Annotated[Literal["status", "wait", "cancel", "read"], typer.Option()] = "status",
    wait_seconds: Annotated[float, typer.Option(min=0, max=60)] = 10,
    stream: Annotated[Literal["stdout", "stderr"], typer.Option()] = "stdout",
    offset: Annotated[int, typer.Option(min=0)] = 0,
    max_bytes: Annotated[int, typer.Option(min=1, max=64000)] = 16000,
    rollout_db: Annotated[Path, typer.Option()] = DEFAULT_CHECKPOINT_PATH,
    permissions: Annotated[Path | None, typer.Option(help="当前权限上限")] = None,
) -> None:
    """查询、等待、取消或读取崩溃前命令的日志；不调用模型，不重跑命令。"""
    from agent_runtime.agent import Agent
    agent = Agent(workspace_path=Path.cwd(), checkpoint_db=rollout_db, permissions=permissions,
                  enable_workspace_mcp=False)
    try:
        result = asyncio.run(agent.command(turn_id, session_id, action=action, wait_seconds=wait_seconds,
                                          stream=stream, offset=offset, max_bytes=max_bytes))
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(result.model_dump_json())
