"""The ordinary Turn path does not capture workspace delivery snapshots."""

import pytest

from agent_runtime import Agent
from agent_runtime.harness import RuntimeStore, delivery
from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel
from tests.agent.harness.test_runtime_efficiency import PlanningModel
from tests.agent.harness.test_subagent_budget import SmallAnswerModel


@pytest.mark.anyio
@pytest.mark.parametrize("model_factory", [SmallAnswerModel, lambda: PlanningModel(reconcile=False)])
async def test_answers_and_advisory_plans_need_no_workspace_snapshot(tmp_path, monkeypatch, model_factory):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source.txt").write_text("source")
    database = tmp_path / "rollout.sqlite3"
    agent = Agent(workspace_path=workspace, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", model_factory)
    captures = []
    original = delivery._capture

    def observe(*args, **kwargs):
        captures.append(kwargs.get("retain_objects", True))
        return original(*args, **kwargs)

    monkeypatch.setattr(delivery, "_capture", observe)
    result = await agent.run("Inspect source", max_steps=3)

    assert result.status == "done"
    assert captures == []
    assert result.delivery["status"] == "unverified"
    assert result.delivery["next_actions"] == ()
    assert "delivery_id" not in result.delivery
    if result.plan is not None:
        assert result.plan.status == "needs_replan"
    async with await RuntimeStore.open(database, read_only=True) as store:
        items = await store.run_async(store.list_items, result.turn_id)
        assert not any(item.kind.startswith("delivery_") for item in items)


@pytest.mark.anyio
async def test_first_admitted_write_keeps_preimage_for_automatic_delivery(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path = workspace / "value.txt"
    path.write_text("before\n")
    database = tmp_path / "rollout.sqlite3"
    agent = Agent(workspace_path=workspace, checkpoint_db=database, enable_workspace_mcp=False)

    class WriteModel(PatchThenAnswerModel):
        async def dispatch(self, prepared, *, event_sink=None):
            if prepared.request_ref["request_id"].endswith(":step:1"):
                assert captures == []
            return await super().dispatch(prepared, event_sink=event_sink)

    monkeypatch.setattr(agent, "_model_factory", WriteModel)
    captures = []
    original = delivery._capture

    def observe(*args, **kwargs):
        captures.append(path.read_text().strip())
        return original(*args, **kwargs)

    monkeypatch.setattr(delivery, "_capture", observe)
    paused = await agent.run("Update the marker", max_steps=3)
    assert paused.status == "paused"
    assert captures == []
    result = await agent.resume(paused.turn_id, "allow_once")

    assert result.status == "done"
    assert path.read_text() == "after\n"
    assert captures[0] == "before"
    assert captures[-1] == "after"
    assert result.delivery["status"] == "unverified"
    assert result.delivery["changed_paths"] == ("value.txt",)
    assert result.delivery["baseline"]["archive_artifact_id"]
    assert result.delivery["final"]["archive_artifact_id"]
    async with await RuntimeStore.open(database, read_only=True) as store:
        patch = await store.run_async(store.read_artifact, result.delivery["diff_artifact_id"])
        assert b"-before\n+after\n" in patch
        assert (await store.run_async(store.verify)).valid


@pytest.mark.anyio
@pytest.mark.parametrize("unreadable", [False, True])
async def test_required_workspace_change_uses_committed_patch(tmp_path, monkeypatch, unreadable):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "value.txt"
    target.write_text("before\n")
    unrelated = workspace / "unrelated.txt"
    unrelated.write_text("unrelated")
    if unreadable:
        unrelated.chmod(0)
    database = tmp_path / "rollout.sqlite3"
    agent = Agent(workspace_path=workspace, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", PatchThenAnswerModel)

    paused = await agent.run("Update the marker", max_steps=3, require_workspace_change=True)
    assert paused.status == "paused"
    result = await agent.resume(paused.turn_id, "allow_once")

    assert target.read_text() == "after\n"
    assert result.status == "done"
    async with await RuntimeStore.open(database, read_only=True) as store:
        results = [item for item in store.list_items(result.turn_id) if item.kind == "tool_result"]
        [patch] = [item for item in results if item.payload["tool_name"] == "apply_patch"]
        metadata = patch.payload["metadata"]
        assert metadata["runtime_workspace_changed"] is True
        assert metadata.get("workspace_tree_before_sha256") is None
        assert metadata.get("workspace_tree_after_sha256") is None
        [change] = metadata["runtime_workspace_file_changes"]
        assert change["path"] == "value.txt"
        assert change["before_sha256"] != change["after_sha256"]
        assert store.verify().valid
