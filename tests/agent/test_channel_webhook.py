import asyncio
import time

import httpx
import pytest

from agent_runtime.channels import Channels
from agent_runtime.channels.providers import Source
from agent_runtime.channels.webhook import application
from agent_runtime.harness import RuntimeStore, host_ops


@pytest.mark.anyio
async def test_webhook_body_absolute_deadline_and_no_admin_routes(tmp_path, monkeypatch):
    channels = Channels(database=tmp_path / "r.db", workspace=tmp_path)
    async with await RuntimeStore.open(channels.database) as store:
        start = await store.run_async(
            host_ops.receive_command, store, workspace=tmp_path, kind="start", payload={"text": "work"}
        )
    await channels.configure(
        start.thread_id,
        Source(source_id="gh", kind="github", repository="o/r", webhook_secret_env="SECRET", timeout_seconds=0.08),
    )
    app = application(channels)

    async def slow():
        for _ in range(40):
            yield b" "
            await asyncio.sleep(0.01)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        started = time.monotonic()
        response = await client.post("/events/gh", content=slow())
        assert response.status_code == 408
        assert time.monotonic() - started < 0.3
        assert (await client.post("/grant", json={"enabled": True})).status_code == 404
        assert (await client.get("/openapi.json")).status_code == 404
    async with await RuntimeStore.open(channels.database) as store:
        from agent_runtime.harness import channel_ops

        assert await store.run_async(channel_ops.entries, store, "inbox") == ()
