from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_runtime import Agent, MemoryConfig
from agent_runtime.cli import agent_app
from agent_runtime.harness import RuntimeState, Session
from agent_runtime.harness.ownership import ConversationBusyError


@pytest.fixture(autouse=True)
def no_model_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args, **_kwargs):
        pytest.fail("memory relocation must not construct a model or provider")

    monkeypatch.setattr(Agent, "_get_model_control_plane", forbidden)


def seed_memory(workspace: Path, database: Path | None) -> tuple[dict, dict]:
    workspace.mkdir(parents=True, exist_ok=True)
    agent = Agent(workspace_path=workspace, checkpoint_db=database, enable_workspace_mcp=False)
    agent.configure_memory(MemoryConfig(use_memories=True, generate_memories=False))
    memory = agent.remember_memory("tests", "Use uv run pytest.", title="Test runner", kind="preference")
    with RuntimeState.open_sync(agent._harness_database()) as state:
        scope = state.memory_scope(workspace)
    assert scope is not None
    return memory, scope


def test_sdk_relocation_refuses_missing_database_without_creating_it(tmp_path: Path) -> None:
    new = tmp_path / "new"
    new.mkdir()
    agent = Agent(workspace_path=new, checkpoint_db=None, enable_workspace_mcp=False)
    with pytest.raises(FileNotFoundError, match="existing.*database|database.*exist"):
        agent.relocate_memory(tmp_path / "old")
    assert not agent._harness_database().exists()
    assert not (new / ".praxis").exists()


def test_sdk_relocation_preserves_scope_versions_and_frozen_history_after_real_move(tmp_path: Path) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    memory, scope = seed_memory(old, None)
    old_database = old / ".praxis" / "runtime" / "rollout.sqlite3"
    with RuntimeState.open_sync(old_database) as state:
        thread = state.create_thread(workspace=old)
        turn = state.start_turn(
            thread_id=thread.thread_id, user_message="frozen old workspace",
            binding_manifest={"workspace": str(old), "approval_mode": "ask", "memory_scope_id": scope["scope_id"]},
        )
        old_turn = state.read_turn(turn.turn_id)
        before_turns = state.list_turns()
        old_records = state.list_records(thread.thread_id)
    old.rename(new)
    agent = Agent(workspace_path=new, checkpoint_db=None, enable_workspace_mcp=False)
    result = agent.relocate_memory(old, expected_version=memory["version"])
    assert result["scope_id"] == scope["scope_id"]
    assert result["previous_workspace"] == str(old)
    assert result["workspace"] == str(new)
    assert result["version"] == memory["version"]
    assert result["binding_generation"] > scope["binding_generation"]
    assert agent.memories()["entries"] == memory["entries"]
    with RuntimeState.open_sync(agent._harness_database()) as state:
        assert state.memory_scope(new)["scope_id"] == scope["scope_id"]
        assert state.read_memory_state(new, version=memory["version"])["entries"] == memory["entries"]
        assert state.read_turn(turn.turn_id) == old_turn
        assert state.read_thread(thread.thread_id).workspace == str(old)
        assert state.list_turns() == before_turns
        assert state.list_records(thread.thread_id) == old_records


def test_cli_relocation_uses_the_existing_explicit_database_and_sdk_scope(tmp_path: Path) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    shared_database = tmp_path / "shared.sqlite3"
    memory, scope = seed_memory(old, shared_database)
    old.rename(new)
    result = CliRunner().invoke(agent_app, [
        "memory", "relocate", "--from-workspace", str(old), "--workspace", str(new),
        "--checkpoint-db", str(shared_database), "--expected-version", str(memory["version"]),
    ])
    assert result.exit_code == 0, result.output
    relocated = json.loads(result.output)
    assert relocated["scope_id"] == scope["scope_id"]
    assert relocated["version"] == memory["version"]
    assert relocated["workspace"] == str(new)
    sdk = Agent(workspace_path=new, checkpoint_db=shared_database, enable_workspace_mcp=False)
    assert sdk.memories()["entries"] == memory["entries"]


def test_cli_default_database_moves_with_project_when_invoked_from_new_directory(tmp_path: Path, monkeypatch) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    memory, scope = seed_memory(old, old / ".praxis" / "checkpoints.sqlite")
    old.rename(new)
    monkeypatch.chdir(new)
    result = CliRunner().invoke(agent_app, [
        "memory", "relocate", "--from-workspace", str(old), "--workspace", str(new),
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["scope_id"] == scope["scope_id"]
    assert Agent(workspace_path=new).memories()["version"] == memory["version"]


def test_memory_status_keeps_controls_content_and_jobs_in_one_readonly_snapshot(tmp_path: Path, monkeypatch) -> None:
    from agent_runtime.harness.memory_management import remember_memory

    old, new = tmp_path / "old", tmp_path / "new"
    database = tmp_path / "shared.sqlite3"
    memory, scope = seed_memory(old, database)
    new.mkdir()
    with RuntimeState.open_sync(database) as writer:
        source = writer.read_item(memory["entries"][0]["source_item_ids"][0]).thread_id
        writer.write_memory_job(old, source, {"status": "pending", "turn_ids": [], "candidates": ["private"]})
        expected_jobs = {source: {"status": "pending", "turn_ids": [], "revision": 1}}
        original_read = RuntimeState.read_memory_state
        moved = False

        def forbidden(*_args, **_kwargs):
            pytest.fail("memory status must not write through its readonly connection")

        def move_after_first_read(store, *args, **kwargs):
            nonlocal moved
            if store is not writer:
                monkeypatch.setattr(store.persistence, "put_metadata", forbidden)
                monkeypatch.setattr(store.persistence, "delete_metadata", forbidden)
            state = original_read(store, *args, **kwargs)
            if store is not writer and not moved:
                moved = True
                writer.relocate_memory(previous_workspace=old, workspace=new)
                writer.commit_memory_state(old, expected_version=0, state=writer.read_memory_state(old))
                remember_memory(writer, old, key="tests", title="Tests", text="Use make.")
            return state

        monkeypatch.setattr(RuntimeState, "read_memory_state", move_after_first_read)
        status = Agent(workspace_path=old, checkpoint_db=database, enable_workspace_mcp=False).memories()
        assert moved
        assert status["scope_id"] == scope["scope_id"]
        assert status["version"] == memory["version"]
        assert status["entries"] == memory["entries"]
        assert status["use_memories"] is True and status["generate_memories"] is False
        assert status["maintenance_jobs"] == expected_jobs
        assert writer.memory_scope(old)["scope_id"] != scope["scope_id"]
        assert writer.read_memory_state(old)["use_memories"] is False


@pytest.mark.parametrize("failure", ["unknown_source", "target_collision", "stale_version"])
def test_sdk_relocation_rejects_conflicts_without_rebinding(tmp_path: Path, failure: str) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    database = tmp_path / "shared.sqlite3"
    memory, scope = seed_memory(old, database)
    new.mkdir()
    previous = old
    expected_version = memory["version"]
    if failure == "unknown_source":
        previous = tmp_path / "unknown"
    elif failure == "target_collision":
        seed_memory(new, database)
    else:
        expected_version -= 1
    agent = Agent(workspace_path=new, checkpoint_db=database, enable_workspace_mcp=False)
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        agent.relocate_memory(previous, expected_version=expected_version)
    with RuntimeState.open_sync(database) as state:
        assert state.memory_scope(old)["scope_id"] == scope["scope_id"]
        assert state.read_memory_state(old)["version"] == memory["version"]
        assert state.read_memory_state(old)["entries"] == memory["entries"]


def test_sdk_relocation_rejects_same_canonical_workspace_and_missing_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    database = tmp_path / "shared.sqlite3"
    seed_memory(workspace, database)
    link = tmp_path / "alias"
    link.symlink_to(workspace, target_is_directory=True)
    with pytest.raises(ValueError, match="different|distinct|same"):
        Agent(workspace_path=link, checkpoint_db=database).relocate_memory(workspace)
    with pytest.raises(ValueError, match="directory|exist"):
        Agent(workspace_path=tmp_path / "missing", checkpoint_db=database).relocate_memory(workspace)


def test_cli_reports_missing_source_option_and_stale_revision(tmp_path: Path) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    database = tmp_path / "shared.sqlite3"
    memory, _scope = seed_memory(old, database)
    new.mkdir()
    runner = CliRunner()
    common = ["--workspace", str(new), "--checkpoint-db", str(database)]
    missing = runner.invoke(agent_app, ["memory", "relocate", *common])
    assert missing.exit_code != 0
    assert "--from-workspace" in missing.output
    stale = runner.invoke(agent_app, ["memory", "relocate", "--from-workspace", str(old),
                                     "--expected-version", str(memory["version"] - 1), *common])
    assert stale.exit_code != 0
    assert "Traceback" not in stale.output
    assert "version" in stale.output.lower() or "changed" in stale.output.lower()


@pytest.mark.anyio
@pytest.mark.parametrize("active_workspace", ["source", "destination"])
async def test_open_session_in_either_workspace_blocks_relocation(tmp_path: Path, active_workspace: str) -> None:
    from tests.agent.harness.test_thread_manager import EchoHistoryModel

    old, new = tmp_path / "old", tmp_path / "new"
    database = tmp_path / "shared.sqlite3"
    _memory, scope = await asyncio.to_thread(seed_memory, old, database)
    new.mkdir()
    model = EchoHistoryModel()
    active = old if active_workspace == "source" else new
    async with await Session.open(database=database, workspace=active, model=model, workspace_mcp_enabled=False):
        with pytest.raises(ConversationBusyError, match="Session|session|workspace|工作区|会话"):
            Agent(workspace_path=new, checkpoint_db=database).relocate_memory(old)
        assert model.requests == []
    async with await RuntimeState.open(database) as state:
        current = await state.run_async(state.memory_scope, old)
        assert current["scope_id"] == scope["scope_id"]
