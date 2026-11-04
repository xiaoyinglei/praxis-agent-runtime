"""Local channel administration; HTTP has no equivalent privileged endpoints."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from agent_runtime.channels import Channels
from agent_runtime.channels.providers import Source

channel_app = typer.Typer(help="配置认证渠道、精确绑定、授权和持久投递。", no_args_is_help=True)
Database = Annotated[Path, typer.Option("--checkpoint-db", "--rollout-db")]
Workspace = Annotated[Path, typer.Option("--workspace")]


def _client(database: Path, workspace: Path) -> Channels:
    return Channels(database=database, workspace=workspace, autostart=True)


def _show(value: Any) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False))


@channel_app.command("configure")
def configure(thread_id: str, config: Path, database: Database, workspace: Workspace = Path(".")) -> None:
    """JSON 只存凭证环境变量引用；同 source 更新会使旧投递授权失效。"""
    source = Source.model_validate_json(config.read_bytes())
    _show(asyncio.run(_client(database, workspace).configure(thread_id, source)))


@channel_app.command("bind")
def bind(
    source_id: str,
    conversation_id: str,
    thread_id: str,
    destination: str,
    database: Database,
    workspace: Workspace = Path("."),
    enabled: bool = True,
) -> None:
    _show(
        asyncio.run(
            _client(database, workspace).bind(
                source_id=source_id,
                conversation_id=conversation_id,
                thread_id=thread_id,
                destination=destination,
                enabled=enabled,
            )
        )
    )


@channel_app.command("grant")
def grant(
    source_id: str,
    principal: str,
    conversation_id: str,
    thread_id: str,
    database: Database,
    operations: Annotated[str, typer.Option("--operations")],
    workspace: Workspace = Path("."),
    role: str = "member",
    enabled: bool = True,
    local_identity: str | None = None,
) -> None:
    """明确赋予身份/角色/操作；--no-enabled 撤销，版本递增不复活旧队列。"""
    _show(
        asyncio.run(
            _client(database, workspace).grant(
                source_id=source_id,
                principal=principal,
                conversation_id=conversation_id,
                thread_id=thread_id,
                operations=operations.split(","),
                role=role,
                enabled=enabled,
                local_identity=local_identity,
            )
        )
    )


@channel_app.command("status")
def status(database: Database, workspace: Workspace = Path(".")) -> None:
    _show(asyncio.run(_client(database, workspace).inspect()))


@channel_app.command("receive")
def receive(source_id: str, body: Path, headers: Path, database: Database, workspace: Workspace = Path(".")) -> None:
    """从文件接收原生签名事件，沿与 Webhook 相同的认证和事务。"""
    _show(
        asyncio.run(_client(database, workspace).receive(source_id, json.loads(headers.read_text()), body.read_bytes()))
    )


@channel_app.command("deliver")
def deliver(database: Database, workspace: Workspace = Path(".")) -> None:
    """收集当前交付/交互摘要，尝试一个当前已授权的 outbox。"""

    async def run() -> bool:
        channels = _client(database, workspace)
        await channels.collect()
        return await channels.deliver()

    _show({"attempted": asyncio.run(run())})


@channel_app.command("outbox")
def outbox(
    outbox_id: str, action: str, database: Database, workspace: Workspace = Path("."), grant_id: str | None = None
) -> None:
    """retry / reconcile / abandon；未知结果只能核对，不能盲重投。"""
    asyncio.run(_client(database, workspace).outbox_action(outbox_id, action, grant_id=grant_id))
    _show({"outbox_id": outbox_id, "action": action})


@channel_app.command("serve")
def serve(database: Database, workspace: Workspace = Path("."), host: str = "127.0.0.1", port: int = 8766) -> None:
    """只开放 /events/{source_id} 签名入口；没有授权/管理 HTTP 后门。"""
    import uvicorn

    from agent_runtime.channels.webhook import application

    uvicorn.run(application(_client(database, workspace)), host=host, port=port)


@channel_app.command("ci-subscribe")
def ci_subscribe(config: Path, database: Database, workspace: Workspace = Path(".")) -> None:
    """JSON 明确指定 repo/PR/head/check 身份、主体、轮询及有限续跑额度。"""
    _show(asyncio.run(_client(database, workspace).subscribe_ci(**json.loads(config.read_text()))))


@channel_app.command("ci-status")
def ci_status(subscription_id: str, database: Database, workspace: Workspace = Path(".")) -> None:
    _show(asyncio.run(_client(database, workspace).ci_status(subscription_id)))


@channel_app.command("ci-disable")
def ci_disable(subscription_id: str, database: Database, workspace: Workspace = Path(".")) -> None:
    asyncio.run(_client(database, workspace).disable_ci(subscription_id))
    _show({"subscription_id": subscription_id, "enabled": False})


@channel_app.command("publication-preview")
def publication_preview(workspace: Workspace = Path(".")) -> None:
    from dataclasses import asdict

    from agent_runtime.git_delivery import GitDelivery

    _show(asdict(asyncio.run(GitDelivery(workspace).pull_request_status())))


@channel_app.command("publication-prepare")
def publication_prepare(config: Path, database: Database, workspace: Workspace = Path(".")) -> None:
    """冻结精确预览、范围及关联键；准备不会发送。"""
    _show(asyncio.run(_client(database, workspace).prepare_publication(**json.loads(config.read_text()))))


@channel_app.command("publication-execute")
def publication_execute(
    operation_id: str, database: Database, workspace: Workspace = Path("."), reconcile: bool = False
) -> None:
    """执行本地显式准备的单项副作用；--reconcile 只核对远端。"""
    _show(asyncio.run(_client(database, workspace).execute_publication(operation_id, reconcile=reconcile)))


@channel_app.command("ci-head")
def ci_head(subscription_id: str, head_sha: str, database: Database, workspace: Workspace = Path(".")) -> None:
    """显式选择新的精确 head；不会重置轮询/续跑额度。"""
    asyncio.run(_client(database, workspace).ci_head(subscription_id, head_sha))
    _show({"subscription_id": subscription_id, "head_sha": head_sha})
