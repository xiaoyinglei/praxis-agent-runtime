"""Request settings are capability checked, durable, and frozen per request."""

from dataclasses import replace

import pytest

from agent_runtime.core.generation_settings import ModelRequestPreferences, ReasoningCapabilities
from agent_runtime.harness import GatewayHarnessModel, HarnessMessage, HarnessModelRequest
from agent_runtime.model_definition import RequestDefaultsDefinition
from agent_runtime.modeling.contracts import LLMCallStage
from agent_runtime.modeling.gateway import LLMGateway
from agent_runtime.modeling.openai_wire import serialize_openai_request
from tests.agent.harness.test_model_adapter import CharacterAccounting, _resolved_model


def test_unknown_and_unsupported_reasoning_are_distinct():
    unknown = ReasoningCapabilities()
    unsupported = ReasoningCapabilities(supported_efforts=())
    for capability, diagnostic in [(unknown, "unknown"), (unsupported, "unsupported")]:
        with pytest.raises(ValueError, match=diagnostic):
            capability.validate_effort("high")
    capable = ReasoningCapabilities(supported_efforts=("low", "high"), default_effort="high")
    capable.validate_effort("low")
    with pytest.raises(ValueError, match="low, high"):
        capable.validate_effort("max")
    with pytest.raises(ValueError, match="disable"):
        capable.validate_effort("off")


def test_model_aware_gateway_does_not_recapture_explicit_output_under_stage_cap():
    gateway = LLMGateway(
        generator=object(),
        token_accounting=CharacterAccounting(),
        model_context_tokens=1048576,
        agent_output_policy="model-aware-v1",
        model_max_output_tokens=131072,
    )
    assert (
        gateway.effective_stage_budget(LLMCallStage.AGENT_STEP, kwargs={"max_tokens": 65536}).max_output_tokens == 65536
    )
    assert (
        gateway.effective_stage_budget(LLMCallStage.CONTEXT_COMPACTION, kwargs={"max_tokens": 65536}).max_output_tokens
        == gateway.stage_budget(LLMCallStage.CONTEXT_COMPACTION).max_output_tokens
    )


def test_legacy_gateway_keeps_old_stage_ceiling():
    gateway = LLMGateway(generator=object(), token_accounting=CharacterAccounting(), model_context_tokens=1048576)
    assert (
        gateway.effective_stage_budget(LLMCallStage.AGENT_STEP, kwargs={"max_tokens": 65536}).max_output_tokens == 32768
    )


def test_adapter_freezes_effort_and_requested_effective_output_diagnostics():
    gateway = LLMGateway(
        generator=object(),
        token_accounting=CharacterAccounting(),
        model_context_tokens=1048576,
        agent_output_policy="model-aware-v1",
        model_max_output_tokens=65536,
    )
    resolved = _resolved_model(
        gateway=gateway, model_id="test-model", context_window_tokens=1048576, max_output_tokens=65536
    )
    resolved = replace(
        resolved,
        capabilities=replace(
            resolved.capabilities,
            reasoning=ReasoningCapabilities(
                supported_efforts=("low", "high"), default_effort="high", thinking_toggle="thinking"
            ),
        ),
        request_defaults=RequestDefaultsDefinition(max_output_tokens=49152),
    )
    model = GatewayHarnessModel(model_id="test-model", resolved=resolved, instructions=("Follow task",))
    request = HarnessModelRequest(
        thread_id="thread",
        turn_id="turn",
        messages=(HarnessMessage(role="user", content="Work"),),
        binding_manifest={
            "request_settings": {
                "revision": 2,
                "model_id": "test-model",
                "preferences": ModelRequestPreferences(reasoning_effort="low", max_output_tokens=98304).model_dump(),
            }
        },
    )
    prepared = model.prepare(request)
    wire = serialize_openai_request(prepared.dispatch_payload.request)
    assert wire.payload["reasoning_effort"] == "low"
    assert wire.payload["max_completion_tokens"] == 65536
    settings = prepared.request_ref["effective_settings"]
    assert settings["requested_output_tokens"] == 98304
    assert settings["max_output_tokens"] == 65536
    assert settings["output_constraints"] == ["model_max_output_tokens"]
    assert settings["revision"] == 2
    assert prepared.resource_request.output_tokens == 65536


def _control_plane(tmp_path):
    from agent_runtime.core.llm_config import AgentModelsConfig, ModelSpec
    from agent_runtime.core.llm_registry import ModelRegistry
    from agent_runtime.modeling.config import GenerationConfig
    from agent_runtime.models import ModelControlPlane

    capable = ModelSpec(
        provider="openai_compatible",
        context_window_tokens=1048576,
        max_output_tokens=131072,
        base_url="http://127.0.0.1:8080/v1",
        location="local",
        reasoning=ReasoningCapabilities(
            supported_efforts=("low", "high"), default_effort="high", thinking_toggle="thinking"
        ),
        defaults={"max_output_tokens": 65536},
    )
    config = AgentModelsConfig(
        models={
            "capable": capable,
            "plain": capable.model_copy(
                update={"reasoning": ReasoningCapabilities(supported_efforts=()), "defaults": {}}
            ),
        },
        default_model="capable",
        fallback_model=None,
        generation=GenerationConfig(),
    )
    return ModelControlPlane.from_registry(ModelRegistry(config), session_path=tmp_path / "model-session.json")


def test_preferences_validate_before_mutation_and_survive_switch_and_reload(tmp_path):
    plane = _control_plane(tmp_path)
    plane.configure_request(reasoning_effort="low", max_output_tokens=98304)
    assert plane.request_settings()["preferences"]["reasoning_effort"] == "low"
    before = (tmp_path / "model-session.json").read_bytes()
    with pytest.raises(ValueError, match="supported"):
        plane.configure_request(reasoning_effort="max")
    assert (tmp_path / "model-session.json").read_bytes() == before
    plane.switch_model("plain", requested_by="user")
    assert plane.request_settings()["preferences"]["reasoning_effort"] is None
    plane.switch_model("capable", requested_by="user")
    assert plane.request_settings()["preferences"]["max_output_tokens"] == 98304
    reopened = _control_plane(tmp_path)
    assert reopened.request_settings() == plane.request_settings()


def test_concurrent_request_preference_write_rejects_stale_state(tmp_path):
    from agent_runtime.model_config_io import ConfigVersionConflict

    first = _control_plane(tmp_path)
    stale = _control_plane(tmp_path)
    first.configure_request(reasoning_effort="low")
    with pytest.raises(ConfigVersionConflict):
        stale.configure_request(reasoning_effort="high")
    assert stale.request_settings()["preferences"]["reasoning_effort"] is None


@pytest.mark.anyio
async def test_live_settings_freeze_at_next_request_and_restore_from_rollout(tmp_path):
    from agent_runtime.harness import Session
    from tests.agent.harness.test_thread_manager import EchoHistoryModel

    plane = _control_plane(tmp_path)
    model = EchoHistoryModel()
    async with await Session.open(database=tmp_path / "rollout.db", workspace=tmp_path, model=model) as session:
        session.model_control_plane = plane
        session.binding_provider._request_settings = plane.request_settings
        session.configure_request(reasoning_effort="high", max_output_tokens=65536)
        turn = session.store.start_turn(
            thread_id=session.thread_id,
            user_message="work",
            binding_manifest=session.binding_provider.snapshot(thread_id=session.thread_id, turn_id="turn-x"),
            turn_id="turn-x",
        )
        runner = session._executor()
        context = runner.restore_turn_context(turn.turn_id)
        old = runner.capture_step_context(context, step=1)
        session.configure_request(reasoning_effort="low")
        new = runner.capture_step_context(context, step=2)
        assert old.binding_manifest["request_settings"]["preferences"]["reasoning_effort"] == "high"
        assert new.binding_manifest["request_settings"]["preferences"]["reasoning_effort"] == "low"
        assert (
            new.binding_manifest["request_settings"]["revision"] > old.binding_manifest["request_settings"]["revision"]
        )
        assert (
            session.store.read_thread(session.thread_id).settings["request_preferences"]["capable"]["reasoning_effort"]
            == "low"
        )
        assert session.store.verify().valid


def test_cli_request_settings_commands_use_shared_validator_without_model_io(tmp_path, capsys):
    from agent_runtime.cli import _handle_request_settings_command

    plane = _control_plane(tmp_path)
    _handle_request_settings_command("/effort low", agent=plane)
    assert plane.request_settings()["preferences"]["reasoning_effort"] == "low"
    _handle_request_settings_command("/output 65536", agent=plane)
    assert plane.request_settings()["preferences"]["max_output_tokens"] == 65536
    with pytest.raises(ValueError):
        _handle_request_settings_command("/effort max", agent=plane)
    _handle_request_settings_command("/effort", agent=plane)
    assert "low" in capsys.readouterr().out


@pytest.mark.anyio
async def test_saved_request_reprepares_with_original_wire_after_live_change(tmp_path):
    from types import SimpleNamespace

    from agent_runtime.harness import RolloutContextManager, RolloutStore, TurnExecutor
    from tests.agent.harness.test_thread_manager import AcceptPlainAnswer

    plane = _control_plane(tmp_path)
    plane.configure_request(reasoning_effort="high", max_output_tokens=65536)
    gateway = LLMGateway(
        generator=object(),
        token_accounting=CharacterAccounting(),
        model_context_tokens=1048576,
        agent_output_policy="model-aware-v1",
        model_max_output_tokens=131072,
    )
    resolved = _resolved_model(
        gateway=gateway, model_id="capable", context_window_tokens=1048576, max_output_tokens=131072
    )
    resolved = replace(
        resolved, capabilities=replace(resolved.capabilities, reasoning=plane.catalog.definition("capable").reasoning)
    )
    model = GatewayHarnessModel(model_id="capable", resolved=resolved, instructions=("Work",))
    with RolloutStore(tmp_path / "rollout.db") as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="Work",
            binding_manifest={"model_id": "capable", "request_settings": plane.request_settings()},
        )
        runner = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=AcceptPlainAnswer(),
        )
        context = runner.restore_turn_context(turn.turn_id)
        first = await runner._prepare_step(runner.capture_step_context(context, step=1))
        plane.configure_request(reasoning_effort="low", max_output_tokens=16384)
        saved = runner.restore_step_context(context, SimpleNamespace(request_ref=first.request_ref))
        replay = await runner._prepare_step(saved)
        assert replay.wire_hash == first.wire_hash
        assert replay.request_hash == first.request_hash
        assert replay.request_ref["effective_settings"]["reasoning_effort"] == "high"
        assert replay.request_ref["effective_settings"]["max_output_tokens"] == 65536


def test_new_optional_definition_fields_do_not_change_legacy_canonical_bytes(tmp_path):
    import json

    from agent_runtime.model_definition import ModelExecutionDefinition, canonical_definition_json

    definition = _control_plane(tmp_path).catalog.definition("capable")
    payload = json.loads(canonical_definition_json(definition))
    payload.pop("reasoning")
    payload.pop("agent_output_policy")
    payload["defaults"].pop("max_output_tokens")
    legacy = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert canonical_definition_json(ModelExecutionDefinition.model_validate_json(legacy)) == legacy


@pytest.mark.parametrize(
    "effort,default,requested,expected",
    [(None, 49152, None, 49152), ("off", 49152, 65536, 65536), ("low", None, None, 32768)],
)
def test_settings_precedence_and_thinking_off_are_independent_from_output(effort, default, requested, expected):
    gateway = LLMGateway(
        generator=object(),
        token_accounting=CharacterAccounting(),
        model_context_tokens=1048576,
        agent_output_policy="model-aware-v1",
    )
    resolved = _resolved_model(gateway=gateway, context_window_tokens=1048576, max_output_tokens=131072)
    resolved = replace(
        resolved,
        capabilities=replace(
            resolved.capabilities,
            reasoning=ReasoningCapabilities(
                supported_efforts=("low", "high"), default_effort="high", thinking_toggle="thinking"
            ),
        ),
        request_defaults=RequestDefaultsDefinition(max_output_tokens=default),
    )
    model = GatewayHarnessModel(model_id="provider-model", resolved=resolved, instructions=("Work",))
    request = HarnessModelRequest(
        thread_id="t",
        turn_id="u",
        messages=(HarnessMessage(role="user", content="Work"),),
        binding_manifest={
            "request_settings": {"preferences": {"reasoning_effort": effort, "max_output_tokens": requested}}
        },
    )
    prepared = model.prepare(request)
    wire = serialize_openai_request(prepared.dispatch_payload.request)
    assert wire.payload["max_completion_tokens"] == expected
    if effort == "off":
        assert wire.payload["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in wire.payload
    else:
        assert wire.payload["reasoning_effort"] == (effort or "high")
    summary = model.prepare(replace(request, purpose="context_summary"))
    summary_wire = serialize_openai_request(summary.dispatch_payload.request)
    assert (
        summary_wire.payload["max_completion_tokens"]
        == gateway.stage_budget(LLMCallStage.CONTEXT_COMPACTION).max_output_tokens
    )
    assert "reasoning_effort" not in summary_wire.payload
    assert summary_wire.payload["thinking"] == {"type": "disabled"}


@pytest.mark.anyio
async def test_reopening_chat_keeps_live_preference_over_initial_cli_choice(tmp_path, monkeypatch):
    from agent_runtime import Agent
    from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel

    agent = Agent(
        model="deepseek-flash",
        workspace_path=tmp_path,
        checkpoint_db=tmp_path / "chat.db",
        model_session_path=None,
        enable_workspace_mcp=False,
        reasoning_effort="high",
    )
    agent._model_factory = PublicHarnessModel
    async with agent.session(require_workspace_change=False) as session:
        thread_id = session.thread_id
        session.configure_request(reasoning_effort="low")
    async with agent.session(thread_id=thread_id, require_workspace_change=False) as reopened:
        assert reopened.request_settings()["preferences"]["reasoning_effort"] == "low"


@pytest.mark.anyio
async def test_two_open_sessions_cannot_silently_overwrite_request_preferences(tmp_path):
    from agent_runtime.harness import Session
    from agent_runtime.model_config_io import ConfigVersionConflict
    from tests.agent.harness.test_thread_manager import EchoHistoryModel

    async with await Session.open(
        database=tmp_path / "concurrent.db", workspace=tmp_path, model=EchoHistoryModel()
    ) as first:
        async with await Session.open(
            database=tmp_path / "concurrent.db", workspace=tmp_path, thread_id=first.thread_id, model=EchoHistoryModel()
        ) as stale:
            first.model_control_plane = _control_plane(tmp_path)
            stale.model_control_plane = _control_plane(tmp_path)
            first.configure_request(reasoning_effort="high")
            with pytest.raises(ConfigVersionConflict):
                stale.configure_request(reasoning_effort="low")
            assert stale.request_settings()["preferences"]["reasoning_effort"] is None


def test_invalid_cli_effort_lists_choices_before_creating_session_file(tmp_path):
    from typer.testing import CliRunner

    from agent_runtime.cli import agent_app

    session_path = tmp_path / "selection.json"
    result = CliRunner().invoke(
        agent_app,
        [
            "run",
            "do not dispatch",
            "--model",
            "deepseek-flash",
            "--effort",
            "ultra",
            "--model-session-path",
            str(session_path),
            "--non-interactive",
            "--disable-workspace-mcp",
            "--no-web-tools",
        ],
    )
    assert result.exit_code != 0
    assert "low, high, max" in result.output
    assert not session_path.exists()
