from __future__ import annotations

import pytest

from agent_runtime import Agent
from agent_runtime.harness import RuntimeStore
from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel


@pytest.mark.anyio
async def test_registered_pre_hook_cannot_replace_approval_and_can_deny_after_resume(tmp_path, monkeypatch):
    from agent_runtime import HookDecision, LifecycleHook

    (tmp_path / "value.txt").write_text("before")
    seen = []

    async def deny(request):
        seen.append(request)
        return HookDecision(block=True, feedback="The host blocked this patch.")

    hook = LifecycleHook("PreToolUse", "patch-policy", deny)
    db = tmp_path / "run.db"

    def open_agent():
        agent = Agent(workspace_path=tmp_path, checkpoint_db=db, enable_workspace_mcp=False, hooks=(hook,))
        monkeypatch.setattr(agent, "_model_factory", PatchThenAnswerModel)
        return agent

    paused = await open_agent().run("Update value.txt", max_steps=5)
    assert paused.status == "paused"
    assert not seen
    result = await open_agent().resume(paused.turn_id, "allow_once")

    assert result.status == "done"
    assert (tmp_path / "value.txt").read_text() == "before"
    assert len(seen) == 1
    assert seen[0].tool_call_id == "patch-1"
    assert result.tool_calls[0].error_code == "pre_tool_hook_blocked"
    async with await RuntimeStore.open(db) as store:
        assert (await store.run_async(store.verify)).valid


class ScriptModel(PatchThenAnswerModel):
    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    def prepare(self, request):
        self.requests.append(request)
        return super().prepare(request)

    async def dispatch(self, prepared, *, event_sink=None):
        step = int(prepared.request_ref["request_id"].rsplit(":", 1)[1])
        return self.responses[step - 1]


def registered_agent(monkeypatch, workspace, database, model, hooks):
    agent = Agent(workspace_path=workspace, checkpoint_db=database, enable_workspace_mcp=False, hooks=hooks)
    monkeypatch.setattr(agent, "_model_factory", lambda: model)
    return agent


@pytest.mark.parametrize("first_status", ["blocked", "timeout", "error"])
def test_cli_command_stop_blocks_and_feeds_the_next_request(tmp_path, monkeypatch, first_status):
    import json
    import sys

    from typer.testing import CliRunner

    from agent_runtime import EventType, cli
    from agent_runtime.harness import HarnessModelResponse, RolloutEventReader

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    host = tmp_path / "host"
    host.mkdir()
    program = host / "review.py"
    first_action = {
        "blocked": "print(json.dumps({'block': True, 'feedback': 'Add the missing detail.'}))",
        "timeout": "time.sleep(10)",
        "error": "print('x' * 1000000)",
    }[first_status]
    program.write_text(
        "import json, sys, time\n"
        "request = json.load(sys.stdin)\n"
        "assert request['event'] == 'Stop' and request['source_item_id']\n"
        "assert request['completion_revision']\n"
        "assert request['completion']['decision']['action'] == 'accept'\n"
        "assert request['completion']['stop_hook_active'] == (request['answer'] == 'revised answer')\n"
        "if request['answer'] == 'first proposal':\n"
        f"    {first_action}\n"
    )
    config = host / "hooks.json"
    config.write_text(json.dumps([{
        "event": "Stop", "name": "review", "argv": [sys.executable, str(program)], "timeout_seconds": 0.8,
    }]))
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))
    monkeypatch.chdir(workspace)
    model = ScriptModel([
        HarnessModelResponse.from_parts(text="first proposal"),
        HarnessModelResponse.from_parts(text="revised answer"),
    ])
    original = cli._create_agent_facade

    def create(**kwargs):
        agent = original(**kwargs)
        monkeypatch.setattr(agent, "_model_factory", lambda: model)
        return agent

    monkeypatch.setattr(cli, "_create_agent_facade", create)
    db = tmp_path / "run.db"
    result = CliRunner().invoke(cli.agent_app, [
        "run", "Explain", "--max-steps", "4", "--non-interactive",
        "--disable-workspace-mcp", "--rollout-db", str(db),
    ])
    assert result.exit_code == 0, result.output
    assert "revised answer" in result.output
    expected_feedback = {
        "blocked": "Add the missing detail.",
        "timeout": "The hook exceeded its time limit.",
        "error": "The hook failed or returned an invalid decision.",
    }[first_status]
    assert any(expected_feedback in m.content for m in model.requests[-1].messages if m.role == "context")
    with RuntimeStore.open_sync(db) as store:
        [turn] = store.list_turns()
        assert store.verify().valid
        statuses = [
            entry.event.data["status"] for entry in RolloutEventReader(store).read(turn.thread_id)
            if entry.event.type is EventType.LIFECYCLE_HOOK_UPDATED
        ]
        assert statuses == ["started", first_status, "started", "allowed"]


@pytest.mark.anyio
async def test_parallel_post_hooks_observe_committed_results_and_separate_feedback(tmp_path, monkeypatch):
    import asyncio

    from agent_runtime import HookDecision, LifecycleHook
    from agent_runtime.harness import HarnessModelResponse, ModelToolCall

    for name in ("first", "second"):
        (tmp_path / name).write_text(name)
    db = tmp_path / "run.db"
    model = ScriptModel([
        HarnessModelResponse.from_parts(tool_calls=tuple(
            ModelToolCall(name, "read_file", {"path": name}) for name in ("first", "second")
        )),
        HarnessModelResponse.from_parts(text="both read"),
    ])
    seen = []
    both_running = asyncio.Event()

    async def observe(request):
        async with await RuntimeStore.open(db, read_only=True) as store:
            result = await store.run_async(store.read_item, request.source_item_id)
        assert result.kind == "tool_result"
        assert result.status == "completed"
        assert result.payload["tool_call_id"] == request.tool_call_id
        assert request.result["model_content"] == result.payload["model_content"]
        seen.append(request.tool_call_id)
        if len(seen) == 2:
            both_running.set()
        await asyncio.wait_for(both_running.wait(), 2)
        return HookDecision(feedback=f"observed {request.tool_call_id}")

    agent = registered_agent(monkeypatch, tmp_path, db, model, (LifecycleHook("PostToolUse", "audit", observe),))
    result = await agent.run("Read both files", max_steps=4)

    assert result.status == "done"
    assert sorted(seen) == ["first", "second"]
    last = model.requests[-1].messages
    assert [m.tool_call_id for m in last if m.role == "tool"] == ["first", "second"]
    assert all("Host hook observation" not in m.content for m in last if m.role == "tool")
    assert sum("Host hook observation" in m.content for m in last if m.role == "context") == 2


@pytest.mark.anyio
@pytest.mark.parametrize("boundary", ["post_started", "stop_settled"])
async def test_reopen_preserves_hook_outcomes_without_replaying_handlers(tmp_path, monkeypatch, boundary):
    from agent_runtime import LifecycleHook
    from agent_runtime.harness import HarnessModelResponse, ModelToolCall
    from agent_runtime.harness.stop import TurnStop

    class Crash(BaseException):
        pass

    (tmp_path / "source").write_text("one")
    db = tmp_path / "run.db"
    model = ScriptModel([
        HarnessModelResponse.from_parts(tool_calls=(ModelToolCall("read", "read_file", {"path": "source"}),)),
        HarnessModelResponse.from_parts(text="answer"),
    ])
    seen = []

    async def observe(request):
        seen.append("post")
        if boundary == "post_started":
            raise Crash()

    async def stop(request):
        seen.append("stop")

    original = TurnStop._finish
    if boundary == "stop_settled":
        def crash(*args, **kwargs):
            raise Crash()

        monkeypatch.setattr(TurnStop, "_finish", crash)

    hooks = (LifecycleHook("PostToolUse", "observe", observe), LifecycleHook("Stop", "check", stop))
    agent = registered_agent(monkeypatch, tmp_path, db, model, hooks)
    with pytest.raises(Crash):
        await agent.run("Read the source", max_steps=4)
    monkeypatch.setattr(TurnStop, "_finish", original)
    async with await RuntimeStore.open(db) as store:
        [turn] = await store.run_async(store.list_turns)
    fresh = registered_agent(monkeypatch, tmp_path, db, model, hooks)
    result = await fresh.resume(turn.turn_id, "continue")

    assert result.status == "done"
    assert seen == ["post", "stop"]
    async with await RuntimeStore.open(db) as store:
        items = await store.run_async(store.list_items, turn.turn_id)
        post = next(i for i in items if i.kind == "lifecycle_hook" and i.payload["hook_event"] == "PostToolUse")
        assert post.payload["status"] == ("unknown" if boundary == "post_started" else "observed")
        [stop_item] = [i for i in items if i.kind == "lifecycle_hook" and i.payload["hook_event"] == "Stop"]
        [decision] = [i for i in items if i.kind == "completion_decision"]
        assert decision.payload["hook_evidence_revision"] == stop_item.payload["completion_revision"]
        assert len(await store.run_async(store.list_tool_operations, turn.turn_id)) == 1
        assert (await store.run_async(store.verify)).valid


@pytest.mark.anyio
@pytest.mark.parametrize("settled", [True, False])
async def test_pre_hook_reopens_before_tool_claim(tmp_path, monkeypatch, settled):
    from agent_runtime import LifecycleHook
    from agent_runtime.harness import tool_ops

    class Crash(BaseException):
        pass

    (tmp_path / "value.txt").write_text("before")
    db = tmp_path / "run.db"
    seen = []

    async def pre(request):
        seen.append(request.invocation_id)
        if not settled:
            raise Crash()

    hooks = (LifecycleHook("PreToolUse", "pre", pre),)
    model = PatchThenAnswerModel()
    first = registered_agent(monkeypatch, tmp_path, db, model, hooks)
    paused = await first.run("Update value.txt", max_steps=4)
    assert paused.status == "paused"
    if settled:
        original = tool_ops.claim_tool_operation

        def crash(*args, **kwargs):
            raise Crash()

        monkeypatch.setattr(tool_ops, "claim_tool_operation", crash)
    with pytest.raises(Crash):
        await first.resume(paused.turn_id, "allow_once")
    if settled:
        monkeypatch.setattr(tool_ops, "claim_tool_operation", original)
    fresh = registered_agent(monkeypatch, tmp_path, db, model, hooks)
    result = await fresh.resume(paused.turn_id, "continue")

    assert result.status == "done"
    assert len(seen) == 1
    assert (tmp_path / "value.txt").read_text() == ("after" if settled else "before")
    async with await RuntimeStore.open(db) as store:
        hook = await store.run_async(store.read_item, seen[0])
        assert hook.payload["status"] == ("allowed" if settled else "unknown")
        [operation] = await store.run_async(store.list_tool_operations, result.turn_id)
        assert operation.attempt_count == int(settled)
        assert (await store.run_async(store.verify)).valid


@pytest.mark.anyio
async def test_stop_rechecks_new_input_after_handler_returns(tmp_path, monkeypatch):
    import asyncio

    from agent_runtime import LifecycleHook
    from agent_runtime.harness import HarnessModelResponse

    entered = asyncio.Event()
    release = asyncio.Event()
    proposals = []
    db = tmp_path / "run.db"
    model = ScriptModel([
        HarnessModelResponse.from_parts(text="stale proposal"),
        HarnessModelResponse.from_parts(text="revised answer"),
    ])

    async def stop(request):
        proposals.append(request)
        if len(proposals) == 1:
            entered.set()
            await release.wait()

    agent = registered_agent(monkeypatch, tmp_path, db, model, (LifecycleHook("Stop", "check", stop),))
    pending = asyncio.create_task(agent.run("Explain", max_steps=4))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await agent.steer("Use the updated requirement", expected_turn_id=proposals[0].turn_id)
    finally:
        release.set()
    result = await pending

    assert result.answer == "revised answer"
    assert proposals[1].input_revision == proposals[0].input_revision + 1
    assert proposals[0].source_item_id != proposals[1].source_item_id
    assert any("Use the updated requirement" in m.content for m in model.requests[-1].messages)


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["timeout", "exception", "missing_registration", "changed_selector"])
async def test_failed_pre_hook_blocks_without_a_tool_attempt(tmp_path, monkeypatch, failure):
    import asyncio

    from agent_runtime import LifecycleHook
    from agent_runtime.harness import HarnessModelResponse, ModelToolCall

    db = tmp_path / "run.db"
    (tmp_path / "value.txt").write_text("before")

    async def pre(request):
        if failure == "timeout":
            await asyncio.sleep(1)
        if failure == "exception":
            raise RuntimeError("handler private details")
        raise AssertionError("unregistered hook must never be invoked")

    hook = LifecycleHook("PreToolUse", "guard", pre, timeout_seconds=0.01)
    if failure in {"missing_registration", "changed_selector"}:
        model = PatchThenAnswerModel()
        first = registered_agent(monkeypatch, tmp_path, db, model, (hook,))
        paused = await first.run("Update value.txt", max_steps=4)
        assert paused.status == "paused"
        fresh_hooks = (
            (LifecycleHook("PreToolUse", "guard", pre, timeout_seconds=0.01, tool_names=("read_file",)),)
            if failure == "changed_selector" else ()
        )
        fresh = registered_agent(monkeypatch, tmp_path, db, model, fresh_hooks)
        result = await fresh.resume(paused.turn_id, "allow_once")
    else:
        model = ScriptModel([
            HarnessModelResponse.from_parts(
                tool_calls=(ModelToolCall("read", "read_file", {"path": "value.txt"}),),
            ),
            HarnessModelResponse.from_parts(text="ended"),
        ])
        agent = registered_agent(monkeypatch, tmp_path, db, model, (hook,))
        result = await agent.run("Read the file", max_steps=4)

    assert result.status == "done"
    assert (tmp_path / "value.txt").read_text() == "before"
    assert result.tool_calls[0].error_code == "pre_tool_hook_blocked"
    async with await RuntimeStore.open(db) as store:
        [operation] = await store.run_async(store.list_tool_operations, result.turn_id)
        assert operation.attempt_count == 0
        items = await store.run_async(store.list_items, result.turn_id)
        [outcome] = [i.payload for i in items if i.kind == "lifecycle_hook"]
        assert outcome["status"] == {
            "timeout": "timeout", "exception": "error",
            "missing_registration": "unavailable", "changed_selector": "unavailable",
        }[failure]
        assert "private details" not in outcome["feedback"]


def test_workspace_file_cannot_self_register_hooks(tmp_path, monkeypatch):
    config = tmp_path / "hooks.json"
    config.write_text("[]")
    assert Agent(workspace_path=tmp_path).hooks == ()
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))
    with pytest.raises(ValueError, match="outside the workspace"):
        Agent(workspace_path=tmp_path)


@pytest.mark.anyio
async def test_pre_hook_allow_does_not_admit_a_superseded_write(tmp_path, monkeypatch):
    import asyncio

    from agent_runtime import LifecycleHook

    (tmp_path / "value.txt").write_text("before")
    db = tmp_path / "run.db"
    entered = asyncio.Event()
    release = asyncio.Event()

    async def pre(request):
        entered.set()
        await release.wait()

    agent = registered_agent(
        monkeypatch, tmp_path, db, PatchThenAnswerModel(), (LifecycleHook("PreToolUse", "gate", pre),),
    )
    paused = await agent.run("Update value.txt", max_steps=4)
    assert paused.status == "paused"
    pending = asyncio.create_task(agent.resume(paused.turn_id, "allow_once"))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        await agent.steer("Leave the file as it is", expected_turn_id=paused.turn_id)
    finally:
        release.set()
    result = await pending

    assert result.status == "done"
    assert (tmp_path / "value.txt").read_text() == "before"
    async with await RuntimeStore.open(db) as store:
        [operation] = await store.run_async(store.list_tool_operations, result.turn_id)
        assert operation.attempt_count == 0
        assert operation.error_code == "input_superseded"
        assert (await store.run_async(store.verify)).valid


@pytest.mark.anyio
async def test_stop_continuations_are_bounded_even_across_tool_batches(tmp_path, monkeypatch):
    from agent_runtime import HookDecision, LifecycleHook
    from agent_runtime.harness import HarnessModelResponse, ModelToolCall

    (tmp_path / "source").write_text("data")
    responses = []
    for index in range(9):
        if index:
            responses.append(HarnessModelResponse.from_parts(
                tool_calls=(ModelToolCall(f"read-{index}", "read_file", {"path": "source"}),),
            ))
        responses.append(HarnessModelResponse.from_parts(text=f"proposal {index}"))

    async def stop(request):
        return HookDecision(block=True, feedback="Still needs review.")

    model = ScriptModel(responses)
    db = tmp_path / "run.db"
    agent = registered_agent(monkeypatch, tmp_path, db, model, (LifecycleHook("Stop", "review", stop),))
    result = await agent.run("Explain", max_steps=20)

    assert result.status == "failed"
    assert result.stop_reason == "stop_hook_continuation_limit"
    assert len(result.tool_calls) == 8
    async with await RuntimeStore.open(db) as store:
        items = await store.run_async(store.list_items, result.turn_id)
        assert sum(i.payload.get("reason_code") == "stop_hook_blocked" for i in items) == 8
        assert (await store.run_async(store.verify)).valid


@pytest.mark.anyio
async def test_reopen_commits_a_pre_hook_denial_and_result_together(tmp_path, monkeypatch):
    from agent_runtime import HookDecision, LifecycleHook
    from agent_runtime.harness import tool_ops

    class Crash(BaseException):
        pass

    (tmp_path / "value.txt").write_text("before")
    db = tmp_path / "run.db"
    seen = []

    async def deny(request):
        seen.append(request.invocation_id)
        return HookDecision(block=True, feedback="Denied by host.")

    hooks = (LifecycleHook("PreToolUse", "guard", deny),)
    model = PatchThenAnswerModel()
    first = registered_agent(monkeypatch, tmp_path, db, model, hooks)
    paused = await first.run("Update value.txt", max_steps=4)
    original = tool_ops.record_tool_result

    def crash(*args, **kwargs):
        if kwargs["result"]["error_code"] == "pre_tool_hook_blocked":
            raise Crash()
        return original(*args, **kwargs)

    monkeypatch.setattr(tool_ops, "record_tool_result", crash)
    with pytest.raises(Crash):
        await first.resume(paused.turn_id, "allow_once")
    monkeypatch.setattr(tool_ops, "record_tool_result", original)

    fresh = registered_agent(monkeypatch, tmp_path, db, model, hooks)
    result = await fresh.resume(paused.turn_id, "continue")

    assert result.status == "done"
    assert len(seen) == 1
    assert result.tool_calls[0].error_code == "pre_tool_hook_blocked"
    assert (tmp_path / "value.txt").read_text() == "before"
    async with await RuntimeStore.open(db) as store:
        [operation] = await store.run_async(store.list_tool_operations, result.turn_id)
        assert operation.attempt_count == 0
        assert operation.result_item_id
        assert (await store.run_async(store.verify)).valid


@pytest.mark.anyio
async def test_post_hook_own_cancellation_preserves_turn_and_result(tmp_path, monkeypatch):
    import asyncio

    from agent_runtime import LifecycleHook
    from agent_runtime.harness import HarnessModelResponse, ModelToolCall

    (tmp_path / "source").write_text("data")
    db = tmp_path / "run.db"
    model = ScriptModel([
        HarnessModelResponse.from_parts(tool_calls=(ModelToolCall("read", "read_file", {"path": "source"}),)),
        HarnessModelResponse.from_parts(text="answer after observation"),
    ])

    async def observe(request):
        future = asyncio.get_running_loop().create_future()
        future.cancel()
        await future

    agent = registered_agent(monkeypatch, tmp_path, db, model, (LifecycleHook("PostToolUse", "audit", observe),))
    result = await agent.run("Read source", max_steps=4)

    assert result.status == "done"
    assert not result.tool_calls[0].is_error
    async with await RuntimeStore.open(db) as store:
        items = await store.run_async(store.list_items, result.turn_id)
        [hook] = [i for i in items if i.kind == "lifecycle_hook"]
        assert hook.payload["status"] == "error"
        assert (await store.run_async(store.verify)).valid


@pytest.mark.anyio
async def test_stop_cannot_finalize_after_host_authority_is_revoked(tmp_path, monkeypatch):
    from agent_runtime import LifecycleHook, Session
    from agent_runtime.harness import HarnessModelResponse
    from agent_runtime.harness.operation_values import TurnCancellationPendingError

    revoked = False

    def guard(store):
        if revoked:
            raise TurnCancellationPendingError("lost background authority")

    async def stop(request):
        nonlocal revoked
        revoked = True

    db = tmp_path / "run.db"
    model = ScriptModel([HarnessModelResponse.from_parts(text="must not complete")])
    agent = registered_agent(monkeypatch, tmp_path, db, model, (LifecycleHook("Stop", "check", stop),))

    async with await Session.open(agent=agent, dispatch_guard=guard) as session:
        result = await session.submit("Explain")

    assert result.status == "failed"
    assert result.stop_reason == "cancelled"
    assert result.answer is None
    async with await RuntimeStore.open(db) as store:
        turn = await store.run_async(store.read_turn, result.turn_id)
        assert turn.status == "cancelled"
        assert (await store.run_async(store.verify)).valid


@pytest.mark.anyio
async def test_pre_hook_verdict_does_not_transfer_to_a_changed_filesystem_target(tmp_path, monkeypatch):
    from agent_runtime import LifecycleHook
    from agent_runtime.harness import HarnessModelResponse, ModelToolCall, tool_ops

    class Crash(BaseException):
        pass

    source = tmp_path / "source"
    replacement = tmp_path / "replacement"
    source.write_text("allowed data")
    replacement.write_text("different data")
    alias = tmp_path / "alias"
    alias.symlink_to(source)
    db = tmp_path / "run.db"
    model = ScriptModel([
        HarnessModelResponse.from_parts(tool_calls=(ModelToolCall("read", "read_file", {"path": "alias"}),)),
        HarnessModelResponse.from_parts(text="ended"),
    ])
    seen = []

    async def pre(request):
        seen.append(request)

    hooks = (LifecycleHook("PreToolUse", "guard", pre),)
    agent = registered_agent(monkeypatch, tmp_path, db, model, hooks)
    original = tool_ops.claim_tool_operation

    def crash(*args, **kwargs):
        raise Crash()

    monkeypatch.setattr(tool_ops, "claim_tool_operation", crash)
    with pytest.raises(Crash):
        await agent.run("Read alias", max_steps=4)
    monkeypatch.setattr(tool_ops, "claim_tool_operation", original)
    alias.unlink()
    alias.symlink_to(replacement)
    fresh = registered_agent(monkeypatch, tmp_path, db, model, hooks)

    result = await fresh.resume(seen[0].turn_id, "continue")

    assert result.status == "done"
    assert result.tool_calls[0].error_code == "pre_tool_hook_blocked"
    assert len(seen) == 1
    assert seen[0].resources == ({"kind": "filesystem", "identity": str(source), "access": "read"},)
    async with await RuntimeStore.open(db) as store:
        [operation] = await store.run_async(store.list_tool_operations, result.turn_id)
        assert operation.attempt_count == 0
        assert (await store.run_async(store.verify)).valid


@pytest.mark.parametrize("entrypoint", ["cli", "sdk"])
def test_selective_tool_hooks_admit_only_matching_calls(tmp_path, monkeypatch, entrypoint):
    import asyncio
    import json
    import sys

    from typer.testing import CliRunner

    from agent_runtime import LifecycleHook, cli
    from agent_runtime.harness import HarnessModelResponse, ModelToolCall
    from agent_runtime.tools.tool import ToolEffect

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "value.txt").write_text("before")
    database = tmp_path / "run.db"
    model = ScriptModel([
        HarnessModelResponse.from_parts(tool_calls=(
            ModelToolCall("read", "read_file", {"path": "value.txt"}),
            ModelToolCall("patch", "apply_patch", {
                "file_path": "value.txt", "old_string": "before", "new_string": "after",
            }),
        )),
        HarnessModelResponse.from_parts(text="finished"),
    ])
    seen = []
    if entrypoint == "sdk":
        async def observe(request):
            seen.append((request.event, request.tool_call_id))

        hooks = (
            LifecycleHook("PreToolUse", "writes", observe, effects=(ToolEffect.WRITE_WORKSPACE,)),
            LifecycleHook("PostToolUse", "patch", observe, tool_names=("apply_patch",)),
            LifecycleHook("PreToolUse", "both", observe, tool_names=("read_file",),
                          effects=(ToolEffect.WRITE_WORKSPACE,)),
        )
        agent = registered_agent(monkeypatch, workspace, database, model, hooks)
        agent.approval_mode = "full"
        assert asyncio.run(agent.run("Update value", max_steps=4)).status == "done"
    else:
        log = tmp_path / "calls.jsonl"
        program = tmp_path / "hook.py"
        program.write_text(
            "import json, sys\n"
            "request = json.load(sys.stdin)\n"
            "with open(sys.argv[1], 'a') as log:\n"
            "    print(json.dumps([request['event'], request['tool_call_id']]), file=log)\n"
        )
        argv = [sys.executable, str(program), str(log)]
        config = tmp_path / "hooks.json"
        config.write_text(json.dumps([
            {"event": "PreToolUse", "name": "writes", "argv": argv, "effects": ["write_workspace"]},
            {"event": "PostToolUse", "name": "patch", "argv": argv, "tool_names": ["apply_patch"]},
            {"event": "PreToolUse", "name": "both", "argv": argv,
             "tool_names": ["read_file"], "effects": ["write_workspace"]},
        ]))
        monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))
        monkeypatch.chdir(workspace)
        original = cli._create_agent_facade

        def create(**kwargs):
            agent = original(**kwargs)
            monkeypatch.setattr(agent, "_model_factory", lambda: model)
            return agent

        monkeypatch.setattr(cli, "_create_agent_facade", create)
        result = CliRunner().invoke(cli.agent_app, [
            "run", "Update value", "--max-steps", "4", "--approval-mode", "full",
            "--non-interactive", "--disable-workspace-mcp", "--rollout-db", str(database),
        ])
        assert result.exit_code == 0, result.output
        seen = [tuple(json.loads(line)) for line in log.read_text().splitlines()]
    assert seen == [("PreToolUse", "patch"), ("PostToolUse", "patch")]
    assert (workspace / "value.txt").read_text() == "after"
    with RuntimeStore.open_sync(database) as store:
        [turn] = store.list_turns()
        items = [i for i in store.list_items(turn.turn_id) if i.kind == "lifecycle_hook"]
        assert [i.payload["tool_call_id"] for i in items] == ["patch", "patch"]
        assert store.verify().valid


@pytest.mark.parametrize("legacy", [False, True])
def test_cli_recover_reuses_a_settled_stop_with_unchanged_evidence(tmp_path, monkeypatch, legacy):
    import json
    import sys

    from typer.testing import CliRunner

    from agent_runtime import cli
    from agent_runtime.harness import HarnessModelResponse, turn_ops
    from agent_runtime.harness.stop import TurnStop

    class Crash(BaseException):
        pass

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    db = tmp_path / "run.db"
    log = tmp_path / "calls"
    script = tmp_path / "hook.py"
    script.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "with open(sys.argv[1], 'a') as calls:\n"
        "    print(request.get('completion_revision', 'legacy'), file=calls)\n"
    )
    config = tmp_path / "hooks.json"
    config.write_text(json.dumps([{
        "event": "Stop", "name": "review", "argv": [sys.executable, str(script), str(log)],
    }]))
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))
    monkeypatch.chdir(workspace)
    model = ScriptModel([HarnessModelResponse.from_parts(text="already reviewed")])
    original_create = cli._create_agent_facade

    def create(**kwargs):
        agent = original_create(**kwargs)
        monkeypatch.setattr(agent, "_model_factory", lambda: model)
        return agent

    def crash(*args, **kwargs):
        raise Crash()

    monkeypatch.setattr(cli, "_create_agent_facade", create)
    if legacy:
        start = turn_ops.start_turn

        def start_legacy(store, **kwargs):
            kwargs["binding_manifest"].pop("lifecycle_hook_protocol", None)
            return start(store, **kwargs)

        monkeypatch.setattr(turn_ops, "start_turn", start_legacy)
    original_finish = TurnStop._finish
    monkeypatch.setattr(TurnStop, "_finish", crash)
    runner = CliRunner()
    with pytest.raises(Crash):
        runner.invoke(cli.agent_app, [
            "run", "Explain", "--max-steps", "2", "--non-interactive", "--disable-workspace-mcp",
            "--rollout-db", str(db),
        ])
    monkeypatch.setattr(TurnStop, "_finish", original_finish)
    with RuntimeStore.open_sync(db) as store:
        [turn] = store.list_turns()

    result = runner.invoke(cli.agent_app, ["recover", turn.turn_id, "--action", "continue", "--rollout-db", str(db)])

    assert result.exit_code == 0, (result.output, result.exception)
    [revision] = log.read_text().splitlines()
    with RuntimeStore.open_sync(db) as store:
        assert store.read_turn(turn.turn_id).status == "completed"
        [hook] = [i for i in store.list_items(turn.turn_id) if i.kind == "lifecycle_hook"]
        if legacy:
            assert revision == "legacy"
            assert "completion_revision" not in hook.payload
        else:
            assert hook.payload["completion_revision"] == revision
            assert revision != "legacy"
        assert store.verify().valid


@pytest.mark.anyio
async def test_pre_hook_cannot_allow_a_target_changed_by_the_callback(tmp_path, monkeypatch):
    from agent_runtime import LifecycleHook
    from agent_runtime.harness import HarnessModelResponse, ModelToolCall

    source = tmp_path / "source"
    source.write_text("old")
    replacement = tmp_path / "replacement"
    replacement.write_text("new")
    alias = tmp_path / "alias"
    alias.symlink_to(source)
    model = ScriptModel([
        HarnessModelResponse.from_parts(tool_calls=(ModelToolCall("read", "read_file", {"path": "alias"}),)),
        HarnessModelResponse.from_parts(text="ended"),
    ])

    async def pre(request):
        alias.unlink()
        alias.symlink_to(replacement)

    agent = registered_agent(
        monkeypatch, tmp_path, tmp_path / "run.db", model, (LifecycleHook("PreToolUse", "guard", pre),),
    )
    result = await agent.run("Read alias", max_steps=4)

    assert result.status == "done"
    assert result.tool_calls[0].error_code == "pre_tool_hook_blocked"


@pytest.mark.anyio
async def test_recovery_cannot_bypass_effect_selector_when_tool_use_changes(tmp_path, monkeypatch):
    from dataclasses import replace

    from agent_runtime import LifecycleHook, Session
    from agent_runtime.harness import HarnessModelResponse, ModelToolCall, tool_ops
    from agent_runtime.tools.builtins.filesystem import create_read_file_tool
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.tools.tool import ToolEffect
    from agent_runtime.workspace import open_workspace

    class Crash(BaseException):
        pass

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source").write_text("original")
    db = tmp_path / "run.db"
    original_tool = create_read_file_tool(open_workspace(workspace))
    write = False
    seen = []

    def resolve(arguments):
        use = original_tool.resolve_use(arguments)
        return replace(use, effects=use.effects | {ToolEffect.WRITE_WORKSPACE}) if write else use

    async def pre(request):
        seen.append(request)

    options = {
        "workspace": workspace,
        "database": db,
        "model": ScriptModel([
            HarnessModelResponse.from_parts(tool_calls=(ModelToolCall("read", "read_file", {"path": "source"}),)),
            HarnessModelResponse.from_parts(text="ended"),
        ]),
        "tools": {"read_file": replace(original_tool, resolve_use=resolve)},
        "hooks": (LifecycleHook("PreToolUse", "writes", pre, effects=(ToolEffect.WRITE_WORKSPACE,)),),
        "tool_execution_context": ToolExecutionContext(workspace_root=workspace, cwd=workspace, approval_mode="full"),
    }
    original_claim = tool_ops.claim_tool_operation

    def crash(*args, **kwargs):
        raise Crash()

    monkeypatch.setattr(tool_ops, "claim_tool_operation", crash)
    async with await Session.open(**options) as session:
        with pytest.raises(Crash):
            await session.submit("Read source")
    monkeypatch.setattr(tool_ops, "claim_tool_operation", original_claim)
    async with await RuntimeStore.open(db) as store:
        [turn] = await store.run_async(store.list_turns)
    write = True

    async with await Session.open(**options, frozen_turn_id=turn.turn_id) as session:
        result = await session.resume(turn.turn_id, "continue")

    assert result.status == "done"
    assert result.tool_calls[0].error_code == "pre_tool_hook_blocked"
    assert not seen
