import asyncio
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime import Agent
from agent_runtime.core.model_request import toolset_revision_for_tools
from agent_runtime.harness import HarnessModelResponse, ModelToolCall, RuntimeStore
from agent_runtime.permission_policy import PathRule, PermissionPolicy
from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel


@pytest.mark.parametrize("style", ["prose", "backticks", "quoted", "escaped", "uri", "line"])
def test_user_path_recognition_preserves_boundaries(tmp_path, style):
    from urllib.parse import quote

    from agent_runtime.local_files import user_path_references

    source = tmp_path / ("source" if style == "prose" else "source files")
    source.mkdir()
    (tmp_path / "source").mkdir(exist_ok=True)
    text = {
        "prose": f"源码参考这个：{source},继续分析",
        "backticks": f"Read `{source}` please",
        "quoted": f'Read "{source}" please',
        "escaped": "Read " + str(source).replace(" ", "\\ ") + " please",
        "uri": f"Read file://{quote(str(source))} please",
        "line": str(source),
    }[style]
    assert user_path_references(text) == ({
        "path": str(source), "resolved_path": str(source.resolve()), "kind": "directory",
    },)
    assert user_path_references(f"https://example.com{source}") == ()
    assert user_path_references(f"file://remotehost{source}") == ()


class ReadModel(PublicHarnessModel):
    def __init__(self, path):
        self.path = path
        self.requests = []

    def prepare(self, request):
        self.requests.append(request)
        prepared = super().prepare(request)
        return replace(prepared, request_ref={
            **prepared.request_ref,
            "request_id": f"{request.turn_id}:{request.step}",
            "toolset_revision": toolset_revision_for_tools(request.tools),
            "exposed_tool_names": [tool.definition.name for tool in request.tools],
        })

    async def dispatch(self, prepared, *, event_sink=None):
        if len(self.requests) == 1:
            return HarnessModelResponse.from_parts(text="", provider_response_id=None, usage={}, tool_calls=(
                ModelToolCall(id="read", name="read_file", arguments={"path": str(self.path)}),
            ))
        return HarnessModelResponse.from_parts(text="Read complete.", provider_response_id=None, usage={})


def local_agent(tmp_path, model, **options):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    agent = Agent(workspace_path=workspace, checkpoint_db=tmp_path / "run.sqlite",
                  model_session_path=None, enable_workspace_mcp=False,
                  permissions=options.pop("permissions", PermissionPolicy()), **options)
    agent._model_factory = lambda: model
    return agent


@pytest.mark.anyio
@pytest.mark.parametrize("directory", [False, True])
async def test_user_pasted_path_grants_real_read_and_reopens_without_copy(tmp_path, directory):
    source = tmp_path / "reference source"
    source.mkdir()
    file = source / "query.ts"
    file.write_text("ORIGINAL_SOURCE_671")
    selected = source if directory else file
    model = ReadModel(file)
    agent = local_agent(tmp_path, model)
    async with agent.session() as session:
        result = await session.submit(f"请读取源码：`{selected}`，保持原文件不变。")
        assert result.status == "done"
        assert result.tool_calls and not result.tool_calls[0].is_error
        assert "ORIGINAL_SOURCE_671" in str(model.requests[-1].messages)
        assert session.permissions().path_access(file)
        assert not session.permissions().path_access(file, write=True)
        assert not session.permissions().path_access(tmp_path / "unmentioned.txt")
        assert session.store.verify().valid
    fresh = local_agent(tmp_path, PublicHarnessModel())
    async with fresh.session(thread_id=result.thread_id) as reopened:
        assert reopened.permissions().path_access(file)
        assert not reopened.permissions().path_access(file, write=True)
    assert file.read_text() == "ORIGINAL_SOURCE_671"
    assert list((tmp_path / "workspace").glob("**/query.ts")) == []


@pytest.mark.anyio
async def test_model_output_and_unmentioned_targets_never_grant_reads(tmp_path):
    file = tmp_path / "not-selected.txt"
    file.write_text("private")
    model = ReadModel(file)
    agent = local_agent(tmp_path, model)
    async with agent.session() as session:
        result = await session.submit("Inspect the workspace only")
        assert result.tool_calls[0].is_error
        assert not session.permissions().path_access(file)


@pytest.mark.anyio
async def test_grant_does_not_follow_retargeted_link_or_file_to_directory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    file = tmp_path / "file.txt"
    file.write_text("file")
    link = tmp_path / "link"
    link.symlink_to(source)
    agent = local_agent(tmp_path, PublicHarnessModel())
    async with agent.session() as session:
        result = await session.submit(f"Read `{link}` and `{file}`")
    link.unlink()
    link.symlink_to(other)
    file.unlink()
    file.mkdir()
    fresh = local_agent(tmp_path, PublicHarnessModel())
    async with fresh.session(thread_id=result.thread_id) as session:
        assert not session.permissions().path_access(link / "child")
        assert not session.permissions().path_access(other / "child")
        assert not session.permissions().path_access(file / "child")


@pytest.mark.anyio
@pytest.mark.parametrize("managed", [False, True])
async def test_reopening_respects_new_denial_over_saved_user_grant(tmp_path, managed):
    source = tmp_path / "selected"
    source.mkdir()
    file = source / "notes.txt"
    file.write_text("original")
    agent = local_agent(tmp_path, PublicHarnessModel())
    async with agent.session() as session:
        first = await session.submit(f"Read `{source}`")
        assert session.permissions().path_access(file)
    options = {"managed_permissions": PermissionPolicy()} if managed else {
        "permissions": PermissionPolicy(filesystem=(PathRule(path=str(source), access="deny"),))}
    fresh = local_agent(tmp_path, ReadModel(file), **options)
    async with fresh.session(thread_id=first.thread_id) as reopened:
        assert not reopened.permissions().path_access(file)
        result = await reopened.submit("Read the previous source again")
        assert result.tool_calls[0].is_error


@pytest.mark.anyio
async def test_rejected_or_replayed_input_does_not_grant_new_paths(tmp_path):
    from agent_runtime.harness import turn_ops

    file = tmp_path / "not-created.txt"
    agent = local_agent(tmp_path, PublicHarnessModel())
    async with agent.session() as session:
        turn = await session.store.run_async(
            lambda: turn_ops.start_turn(session.store, thread_id=session.thread_id,
                                        user_message="initial", binding_manifest={}))
        text = f"Read `{file}`"
        receipt = await session.steer(text, expected_turn_id=turn.turn_id, input_id="selected-input")
        file.write_text("created later")
        assert await session.steer(text, expected_turn_id=turn.turn_id, input_id="selected-input") == receipt
        assert not session.permissions().path_access(file)
        with pytest.raises(RuntimeError, match="active"):
            await session.steer(text, expected_turn_id="wrong-turn")
        assert not session.permissions().path_access(file)


@pytest.mark.anyio
@pytest.mark.parametrize("ceiling", [False, True])
async def test_pasted_path_cannot_override_denial_or_managed_ceiling(tmp_path, ceiling):
    file = tmp_path / "forbidden.txt"
    file.write_text("DENIED_CONTENT")
    options = ({"managed_permissions": PermissionPolicy()} if ceiling else {
        "permissions": PermissionPolicy(filesystem=(PathRule(path=str(file), access="deny"),))})
    agent = local_agent(tmp_path, ReadModel(file), **options)
    async with agent.session() as session:
        result = await session.submit(f"Read `{file}`")
        assert result.tool_calls[0].is_error
        assert not session.permissions().path_access(file)


@pytest.mark.anyio
@pytest.mark.parametrize("entry", ["session", "agent"])
async def test_steered_path_is_effective_at_next_step_without_reapproval(tmp_path, entry):
    file = tmp_path / "steered.txt"
    file.write_text("STEERED_CONTENT")

    class WaitingModel(ReadModel):
        def __init__(self):
            super().__init__(file)
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def dispatch(self, prepared, *, event_sink=None):
            if len(self.requests) == 1:
                self.started.set()
                await self.release.wait()
                return HarnessModelResponse.from_parts(text="Old answer", provider_response_id=None, usage={})
            if len(self.requests) == 2:
                return HarnessModelResponse.from_parts(text="", provider_response_id=None, usage={}, tool_calls=(
                    ModelToolCall(id="read", name="read_file", arguments={"path": str(file)}),))
            return await super().dispatch(prepared, event_sink=event_sink)

    model = WaitingModel()
    agent = local_agent(tmp_path, model)
    async with agent.session() as session:
        task = asyncio.create_task(session.submit("Wait for my source."))
        await asyncio.wait_for(model.started.wait(), 5)
        turn = session.store.read_thread(session.thread_id).active_turn_id
        try:
            public = session if entry == "session" else agent
            await public.steer(f"Read this file: {file}", expected_turn_id=turn, input_id="source-input")
        finally:
            model.release.set()
        result = await task
        assert result.turn_id == turn and result.status == "done"
        assert result.tool_calls and not result.tool_calls[0].is_error
        assert "STEERED_CONTENT" in str(model.requests[-1].messages)


def _cli_main():
    from agent_runtime import cli

    recovering = sys.argv[1] == "recover"
    source = Path(os.environ["PRAXIS_TEST_SOURCE"])

    class ProcessModel(ReadModel):
        def prepare(self, request):
            prepared = super().prepare(request)
            if not recovering and len(self.requests) == 2:
                os._exit(75)  # The first real tool result committed; no second dispatch.
            return prepared

        async def dispatch(self, prepared, *, event_sink=None):
            if len(self.requests) == 1:
                name, arguments = ("read_file", {"path": str(source)}) if recovering else (
                    "run_command", {"command": f"cat {shlex.quote(str(source))}"})
                return HarnessModelResponse.from_parts(text="", provider_response_id=None, usage={}, tool_calls=(
                    ModelToolCall(id="reopened-read" if recovering else "shell-read", name=name, arguments=arguments),
                ))
            assert "CROSS_PROCESS_SOURCE_281" in str(self.requests[-1].messages)
            return HarnessModelResponse.from_parts(
                text="Source read after reopen.", provider_response_id=None, usage={})

    model = ProcessModel(source)
    create = cli._create_agent_facade

    def facade(**options):
        agent = create(**options)
        agent._model_factory = lambda: model
        return agent

    cli._create_agent_facade = facade
    cli.agent_app()


@pytest.mark.skipif(sys.platform != "darwin", reason="real macOS process sandbox")
def test_public_cli_external_directory_shell_read_then_same_turn_reopen(tmp_path):
    source = tmp_path / "external source"
    source.mkdir()
    file = source / "query.ts"
    file.write_text("CROSS_PROCESS_SOURCE_281")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "cli.sqlite"
    config = tmp_path / "permissions.toml"
    config.write_text('profile = "workspace"\n')
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2]), "PRAXIS_TEST_SOURCE": str(file)}
    entry = [sys.executable, "-c", f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r})['_cli_main']()"]

    def invoke(*args):
        return subprocess.run([*entry, *args, "--checkpoint-db", str(database), "--permissions", str(config)],
                              cwd=workspace, env=env, text=True, capture_output=True, timeout=45)

    first = invoke("run", f"请读取 `{source}` 中的源码。", "--non-interactive", "--disable-workspace-mcp")
    assert first.returncode == 75, first.stdout + first.stderr
    with RuntimeStore.open_sync(database, read_only=True) as store:
        [turn] = store.list_turns()
        [result] = store.list_items(turn.turn_id, kind="tool_result")
        assert result.payload["structured_content"]["exit_code"] == 0
        assert result.payload["structured_content"]["stdout"] == "CROSS_PROCESS_SOURCE_281"
        assert store.list_tool_operations(turn.turn_id)[0].approval_request_id is None
    second = invoke("recover", turn.turn_id, "--action", "continue")
    assert second.returncode == 0, second.stdout + second.stderr
    with RuntimeStore.open_sync(database, read_only=True) as store:
        [finished] = store.list_turns()
        assert finished.turn_id == turn.turn_id and finished.status == "completed"
        assert all(not i.payload["is_error"] for i in store.list_items(turn.turn_id, kind="tool_result"))
        assert store.verify().valid
    assert file.read_text() == "CROSS_PROCESS_SOURCE_281"
    assert not list(workspace.glob("**/query.ts"))
