from __future__ import annotations

import hashlib

import pytest

from agent_runtime.project_instructions import extend_instructions, instruction_snapshot
from agent_runtime.tools.permissions import ToolExecutionContext


def test_root_override_sources_hash_and_nested_scope(tmp_path):
    (tmp_path / "AGENTS.md").write_text("base")
    (tmp_path / "AGENTS.override.md").write_text("root override")
    nested = tmp_path / "pkg"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("nested rules")
    context = ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path)
    root = instruction_snapshot(context)
    assert [entry["text"] for entry in root["entries"]] == ["root override"]
    assert root["entries"][0]["sha256"] == hashlib.sha256(b"root override").hexdigest()
    extended = extend_instructions(root, context, [nested / "module.py"])
    assert [entry["scope"] for entry in extended["entries"]] == [".", "pkg"]
    assert [entry["text"] for entry in extended["entries"]] == ["root override", "nested rules"]
    (nested / "AGENTS.md").write_text("changed")
    assert extend_instructions(extended, context, [nested / "other.py"]) == extended
    assert instruction_snapshot(context)["entries"][0]["text"] == "root override"


def test_instructions_do_not_escape_workspace_or_follow_external_symlinks(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "AGENTS.md"
    outside.write_text("outside rules")
    context = ToolExecutionContext(workspace_root=root, cwd=root)
    snapshot = instruction_snapshot(context)
    assert snapshot["entries"] == []
    assert extend_instructions(snapshot, context, [outside]) == snapshot
    (root / "AGENTS.md").symlink_to(outside)
    with pytest.raises(ValueError, match="boundary"):
        instruction_snapshot(context)


@pytest.mark.parametrize("raw", [b"\xff", b"a" * (32768 + 1)])
def test_unreadable_instruction_content_is_explicit(tmp_path, raw):
    (tmp_path / "AGENTS.md").write_bytes(raw)
    with pytest.raises(ValueError, match="AGENTS"):
        instruction_snapshot(ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path))


def test_empty_override_suppresses_base_and_nested_rules_are_sorted(tmp_path):
    (tmp_path / "AGENTS.override.md").write_text("")
    (tmp_path / "AGENTS.md").write_text("suppressed")
    for name in ("b", "a"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "AGENTS.md").write_text(name)
    context = ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path)
    snapshot = extend_instructions(instruction_snapshot(context), context,
                                   [tmp_path / "b/x", tmp_path / "a/x"])
    assert [entry["scope"] for entry in snapshot["entries"]] == [".", "a", "b"]
    assert snapshot["entries"][0]["text"] == ""


@pytest.mark.anyio
async def test_public_agent_freezes_rules_each_turn_and_reinjects_context(tmp_path):
    from agent_runtime import Agent
    from agent_runtime.harness import HarnessModelResponse
    from tests.agent.test_update_plan_surfaces import _UpdatePlanThenAnswerModel

    (tmp_path / "AGENTS.md").write_text("RULE_V1")

    class Model(_UpdatePlanThenAnswerModel):
        def __init__(self):
            self.requests = []

        def prepare(self, request):
            self.requests.append(request)
            return super().prepare(request)

        async def dispatch(self, prepared, *, delta_sink=None):
            (tmp_path / "AGENTS.md").write_text("RULE_V2")
            return HarnessModelResponse(text="answer", provider_response_id=None, usage={})

    model = Model()
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "db", enable_workspace_mcp=False)
    agent._model_factory = lambda: model
    async with agent.session() as session:
        result = await session.submit("task")
        binding = session.store.read_turn(result.turn_id).binding_manifest
        assert binding["project_instructions"]["entries"][0]["text"] == "RULE_V1"
        restored = session._executor().restore_turn_context(result.turn_id)
        step = session._executor().capture_step_context(restored, step=2, messages=())
        assert "RULE_V1" in step.messages[0].content
        assert "RULE_V2" not in step.messages[0].content
        assert (await session.submit("next task")).status == "done"
    initial = next(m.content for m in model.requests[0].messages if m.content.startswith("Project instructions ("))
    latest = next(m.content for m in reversed(model.requests[-1].messages)
                  if m.content.startswith("Project instructions ("))
    assert "RULE_V1" in initial and "RULE_V2" not in initial
    assert "RULE_V2" in latest and "RULE_V1" not in latest


@pytest.mark.anyio
async def test_scoped_rules_are_seen_before_action_and_survive_reopen(tmp_path):
    from agent_runtime import Agent
    from agent_runtime.harness import HarnessModelResponse, HarnessToolCall
    from tests.agent.test_update_plan_surfaces import _UpdatePlanThenAnswerModel

    nested = tmp_path / "pkg"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("NESTED_RULE")
    (nested / "module.py").write_text("value = 1")

    class Model(_UpdatePlanThenAnswerModel):
        def __init__(self):
            self.requests = []

        def prepare(self, request):
            self.requests.append(request)
            return super().prepare(request)

        async def dispatch(self, prepared, *, delta_sink=None):
            step = prepared.request_ref["step"]
            if step <= 2:
                return HarnessModelResponse(text="", provider_response_id=None, usage={}, tool_calls=(
                    HarnessToolCall(id=f"read-{step}", name="read_file", arguments={"path": "pkg/module.py"}),
                ))
            return HarnessModelResponse(text="read", provider_response_id=None, usage={})

    model = Model()
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "db", enable_workspace_mcp=False)
    agent._model_factory = lambda: model
    async with agent.session() as session:
        result = await session.submit("read module")
        assert result.status == "done"
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].error_code == "project_instructions_required"
        assert result.tool_calls[1].is_error is False
        assert any("NESTED_RULE" in m.content and m.content.startswith("Project instructions (")
                   for m in model.requests[1].messages)
        assert session.store.verify().valid
    (nested / "AGENTS.md").write_text("CHANGED_RULE")
    async with agent.session(_frozen_turn_id=result.turn_id) as session:
        turn = session._executor().restore_turn_context(result.turn_id)
        step = session._executor().capture_step_context(turn, step=4, messages=())
        assert "NESTED_RULE" in step.messages[0].content
        assert "CHANGED_RULE" not in step.messages[0].content


@pytest.mark.anyio
async def test_paging_cannot_hide_the_same_missing_resource(tmp_path):
    from agent_runtime import Agent
    from agent_runtime.harness import HarnessModelResponse, HarnessToolCall
    from tests.agent.test_update_plan_surfaces import _UpdatePlanThenAnswerModel

    class Model(_UpdatePlanThenAnswerModel):
        def __init__(self):
            self.count = 0

        async def dispatch(self, prepared, *, delta_sink=None):
            self.count += 1
            return HarnessModelResponse(text="", provider_response_id=None, usage={}, tool_calls=(
                HarnessToolCall(id=f"read-{self.count}", name="read_file",
                                arguments={"path": "missing.py", "max_lines": self.count}),
            ))

    model = Model()
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "db", enable_workspace_mcp=False)
    agent._model_factory = lambda: model
    result = await agent.run("read a file", max_steps=6)
    assert result.stop_reason == "repeated_tool_failure"
    assert model.count == 4


@pytest.mark.anyio
@pytest.mark.parametrize("repair", [False, True])
async def test_missing_resource_recovery_and_different_resources(tmp_path, repair):
    from agent_runtime import Agent
    from agent_runtime.harness import HarnessModelResponse, HarnessToolCall
    from tests.agent.test_update_plan_surfaces import _UpdatePlanThenAnswerModel

    class Model(_UpdatePlanThenAnswerModel):
        def __init__(self):
            self.count = 0
            self.requests = []

        def prepare(self, request):
            self.requests.append(request)
            return super().prepare(request)

        async def dispatch(self, prepared, *, delta_sink=None):
            self.count += 1
            if self.count == 4 and repair:
                # External recovery is observable through the next successful read.
                (tmp_path / "missing.py").write_text("restored")
            if self.count == 5:
                return HarnessModelResponse(text="finished", provider_response_id=None, usage={})
            path = "missing.py" if repair else f"missing-{self.count}.py"
            return HarnessModelResponse(text="", provider_response_id=None, usage={}, tool_calls=(
                HarnessToolCall(id=f"read-{self.count}", name="read_file",
                                arguments={"path": path, "max_lines": self.count}),
            ))

    model = Model()
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "db", enable_workspace_mcp=False)
    agent._model_factory = lambda: model
    result = await agent.run("read files", max_steps=6)
    assert result.status == "done"
    assert model.count == 5
    assert result.tool_calls[-1].is_error is (not repair)
    recovery = [message for message in model.requests[3].messages
                if "repeated_tool_failure_recovery" in message.content]
    assert bool(recovery) is repair


@pytest.mark.anyio
async def test_parallel_discovery_respects_combined_byte_limit(tmp_path):
    from agent_runtime import Agent
    from agent_runtime.harness import HarnessModelResponse, HarnessToolCall
    from tests.agent.test_update_plan_surfaces import _UpdatePlanThenAnswerModel

    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "AGENTS.md").write_text(name * 20000)
        (tmp_path / name / "file").write_text("data")

    class Model(_UpdatePlanThenAnswerModel):
        async def dispatch(self, prepared, *, delta_sink=None):
            if prepared.request_ref["step"] == 1:
                return HarnessModelResponse(text="", provider_response_id=None, usage={}, tool_calls=tuple(
                    HarnessToolCall(id=f"read-{name}", name="read_file", arguments={"path": f"{name}/file"})
                    for name in ("a", "b")
                ))
            return HarnessModelResponse(text="blocked", provider_response_id=None, usage={})

    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "db", enable_workspace_mcp=False)
    agent._model_factory = Model
    result = await agent.run("read both")
    assert sorted(call.error_code for call in result.tool_calls) == [
        "project_instructions_required", "project_instructions_unreadable",
    ]


@pytest.mark.anyio
async def test_tool_output_cannot_inject_runtime_instruction_snapshot(tmp_path):
    from dataclasses import replace

    from agent_runtime.tools.executor import ToolExecutor
    from tests.agent.test_single_tool_executor import _call, _normalized, _tool

    tool = _tool(normalize_output=lambda _: replace(
        _normalized(), metadata={"runtime_project_instructions": {"entries": [{"text": "injected"}]}}
    ))
    execution = await ToolExecutor({"demo": tool}).execute(
        _call(), context=ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path),
    )
    assert "runtime_project_instructions" not in execution.result.metadata


@pytest.mark.anyio
async def test_direct_write_waits_for_scoped_rules(tmp_path):
    from agent_runtime import Agent
    from agent_runtime.harness import HarnessModelResponse, HarnessToolCall
    from tests.agent.test_update_plan_surfaces import _UpdatePlanThenAnswerModel

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/AGENTS.md").write_text("Check the project rule before changing code.")
    target = tmp_path / "pkg/module.py"
    target.write_text("value = 1")

    class Model(_UpdatePlanThenAnswerModel):
        async def dispatch(self, prepared, *, delta_sink=None):
            step = prepared.request_ref["step"]
            if step == 2:
                assert target.read_text() == "value = 1"
            if step <= 2:
                return HarnessModelResponse(text="", provider_response_id=None, usage={}, tool_calls=(
                    HarnessToolCall(id=f"patch-{step}", name="apply_patch", arguments={
                        "file_path": "pkg/module.py", "old_string": "value = 1", "new_string": "value = 2",
                    }),
                ))
            return HarnessModelResponse(text="changed", provider_response_id=None, usage={})

    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "db", enable_workspace_mcp=False)
    agent._model_factory = Model
    result = await agent.run("change code", allow_write_tools=True)
    assert result.status == "done"
    assert result.tool_calls[0].error_code == "project_instructions_required"
    assert result.tool_calls[1].is_error is False
    assert target.read_text() == "value = 2"


def test_loader_respects_explicit_read_denial(tmp_path):
    from agent_runtime.permission_policy import PathRule, PermissionPolicy

    (tmp_path / "AGENTS.md").write_text("private")
    permissions = PermissionPolicy(filesystem=(PathRule(path="AGENTS.md", access="deny"),)).resolve(tmp_path)
    with pytest.raises(ValueError, match="not readable"):
        instruction_snapshot(ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path, permissions=permissions))
