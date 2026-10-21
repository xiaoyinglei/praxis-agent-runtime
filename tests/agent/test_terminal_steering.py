import asyncio

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from agent_runtime.terminal_app import TerminalChatApp


@pytest.mark.anyio
async def test_busy_enter_submits_ordered_messages_without_finishing_operation():
    received = []
    release = asyncio.Event()
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="test", workspace="/tmp", input=pipe, output=DummyOutput())

        async def receive(text):
            received.append(text)
            return str(len(received))

        ui.on_input = receive
        ui.input_status = lambda identity: "pending"
        task = asyncio.create_task(ui.run(lambda: ui.operation(release.wait())))
        try:
            for _ in range(100):
                if ui._operation is not None:
                    break
                await asyncio.sleep(.01)
            pipe.send_text("first addition\rsecond addition\r")
            for _ in range(100):
                if len(received) == 2:
                    break
                await asyncio.sleep(.01)
            assert received == ["first addition", "second addition"]
            assert not task.done()
            assert "等待应用" in ui.view.plain_text()
            ui.input_status = lambda identity: "applied"
            async with asyncio.timeout(3):
                while "已应用" not in ui.view.plain_text():
                    await asyncio.sleep(.01)
            assert "已应用" in ui.view.plain_text()
        finally:
            release.set()
            await asyncio.wait_for(task, 3)


@pytest.mark.anyio
async def test_failed_busy_submission_keeps_draft():
    release = asyncio.Event()
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="test", workspace="/tmp", input=pipe, output=DummyOutput())

        async def receive(text):
            raise RuntimeError("expected Turn already finished")

        ui.on_input = receive
        task = asyncio.create_task(ui.run(lambda: ui.operation(release.wait())))
        try:
            for _ in range(100):
                if ui._operation is not None:
                    break
                await asyncio.sleep(.01)
            pipe.send_text("preserve this\r")
            await asyncio.sleep(.1)
            assert ui.editor.text == "preserve this"
            assert "未接收" in ui.view.plain_text()
        finally:
            release.set()
            await asyncio.wait_for(task, 3)


@pytest.mark.anyio
async def test_real_chat_cli_steers_and_continues_same_conversation(tmp_path, monkeypatch):
    from agent_runtime import Agent
    from agent_runtime.cli import _chat_facade_loop
    from tests.agent.harness.test_steering import SteeringModel

    model = SteeringModel()
    agent = Agent(checkpoint_db=tmp_path / "run.db", workspace_path=tmp_path)
    monkeypatch.setattr(agent, "_model_factory", lambda: model)
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="test", workspace=str(tmp_path), input=pipe, output=DummyOutput())
        task = asyncio.create_task(ui.run(lambda: _chat_facade_loop(agent, max_tokens_total=None, _ui=ui)))
        try:
            for _ in range(200):
                if ui.waiting:
                    break
                await asyncio.sleep(.01)
            pipe.send_text("initial task\r")
            await asyncio.wait_for(model.started.wait(), 5)
            pipe.send_text("new CLI requirement\r")
            for _ in range(200):
                if "已接收" in ui.view.plain_text():
                    break
                await asyncio.sleep(.01)
            assert "已接收" in ui.view.plain_text()
            model.release.set()
            for _ in range(200):
                if ui.waiting:
                    break
                await asyncio.sleep(.01)
            assert len(model.requests) == 2
            pipe.send_text("explain the result\r")
            for _ in range(200):
                if len(model.requests) == 3 and ui.waiting:
                    break
                await asyncio.sleep(.01)
            assert len(model.requests) == 3
            assert [m.content for m in model.requests[-1].messages if m.role == "user"] == [
                "initial task", "new CLI requirement", "explain the result",
            ]
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, 5)
        finally:
            model.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_approval_editor_text_never_becomes_an_approval_shortcut():
    from agent_runtime.result import AgentPause

    received, choices = [], []
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="test", workspace="/tmp", input=pipe, output=DummyOutput())

        async def receive(text):
            received.append(text)
            return "input-receipt"

        ui.on_input = receive

        async def conversation():
            choices.append(await ui.choose_approval(AgentPause(
                request_id="approval", kind="tool_approval", question="allow read?", options=("allow_once", "deny"),
            )))

        task = asyncio.create_task(ui.run(conversation))
        try:
            for _ in range(100):
                if ui._approval is not None:
                    break
                await asyncio.sleep(.01)
            pipe.send_text("\tyou should explain instead\r")
            await asyncio.wait_for(task, 3)
            assert received == ["you should explain instead"]
            assert choices == ["continue"]
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_busy_settings_command_is_not_rendered_as_steering_receipt():
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model='test', workspace='/tmp', input=pipe, output=DummyOutput())
        async def receive(text):
            assert text == '/effort low'
            return None
        ui.on_input = receive
        await ui._send_input('/effort low')
        assert not ui._input_receipts
        assert '等待应用' not in ui.view.plain_text()
