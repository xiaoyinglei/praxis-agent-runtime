"""CLI entry points for model-free Rollout maintenance."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from agent_runtime.maintenance import backup_rollout, restore_rollout, upgrade_rollout, verify_rollout

rollout_app = typer.Typer(help="检查、备份、恢复和升级 Rollout 数据库及工件。")
DatabaseOption = Annotated[Path, typer.Option("--rollout-db", "--checkpoint-db")]


def _maintenance[T](operation: Callable[..., T], *args: Path) -> T:
    try:
        return operation(*args)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@rollout_app.command()
def verify(database: DatabaseOption) -> None:
    """只读检查 SQLite、历史事件、投影和工件。"""
    report = _maintenance(verify_rollout, database)
    typer.echo(json.dumps(asdict(report), ensure_ascii=False))
    if not report.valid:
        raise typer.Exit(code=1)


@rollout_app.command()
def backup(destination: Path, database: DatabaseOption) -> None:
    """将一致的数据库和工件快照写入新目录。"""
    typer.echo(_maintenance(backup_rollout, database, destination))


@rollout_app.command()
def restore(backup_directory: Path, destination: Path) -> None:
    """检查备份并将其恢复到新目录，保留备份的 schema。"""
    typer.echo(_maintenance(restore_rollout, backup_directory, destination))


@rollout_app.command()
def upgrade(destination: Path, database: DatabaseOption) -> None:
    """从已停止的宿主数据库复制和升级到新目录。"""
    typer.echo(_maintenance(upgrade_rollout, database, destination))
