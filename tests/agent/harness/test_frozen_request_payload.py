import pytest

from agent_runtime.harness import GatewayHarnessModel, HarnessModelRequest, ModelMessage
from tests.agent.harness.test_model_adapter import CapturingGateway, _resolved_model


def prepare(instructions=("original instructions",)):
    model = GatewayHarnessModel(
        model_id="test-model", resolved=_resolved_model(gateway=CapturingGateway()), instructions=instructions
    )
    request = HarnessModelRequest(
        thread_id="thread", turn_id="turn", binding_manifest={}, messages=(ModelMessage(role="user", body="task"),)
    )
    return model, request, model.prepare(request)


def test_frozen_request_replay_ignores_current_prompt_formatter(monkeypatch):
    from agent_runtime.harness.prepared_requests import frozen_request_payload

    original, request, prepared = prepare()
    artifact = frozen_request_payload(prepared)
    changed, _, changed_prepared = prepare(("changed instructions",))
    assert changed_prepared.request_hash != prepared.request_hash
    monkeypatch.setattr(
        "agent_runtime.harness.model_adapter.build_stable_context",
        lambda **kw: (_ for _ in ()).throw(AssertionError("current formatter called")),
    )
    restored = changed.replay(artifact, request=request)
    assert (restored.request_hash, restored.context_hash, restored.tool_hash, restored.wire_hash) == (
        prepared.request_hash,
        prepared.context_hash,
        prepared.tool_hash,
        prepared.wire_hash,
    )
    assert restored.dispatch_payload.request == prepared.dispatch_payload.request
    assert restored.dispatch_payload.wire == prepared.dispatch_payload.wire


@pytest.mark.parametrize("field", ["request_hash", "context_hash", "tool_hash", "wire_hash", "protocol_revision"])
def test_frozen_payload_validates_all_hashes_and_revision(field):
    from agent_runtime.harness.prepared_requests import frozen_request_payload

    model, request, prepared = prepare()
    artifact = dict(frozen_request_payload(prepared))
    artifact[field] = "changed"
    with pytest.raises(ValueError):
        model.replay(artifact, request=request)


@pytest.mark.anyio
async def test_real_gateway_dispatch_sends_frozen_wire_without_current_serializer(monkeypatch):
    from agent_runtime.harness.prepared_requests import frozen_request_payload
    from agent_runtime.modeling.gateway import LLMGateway
    from tests.agent.harness.test_model_adapter import CharacterAccounting, NonStreamingProvider

    class Provider(NonStreamingProvider):
        def generate_with_tools(self, **kwargs):
            self.received = kwargs
            return super().generate_with_tools(**kwargs)

    provider = Provider()
    accounting = CharacterAccounting()
    gateway = LLMGateway(generator=provider, token_accounting=accounting, model_context_tokens=8192)
    model = GatewayHarnessModel(
        model_id="test-model",
        resolved=_resolved_model(gateway=gateway, generator=provider, token_accounting=accounting),
        instructions=("original instructions",),
    )
    request = HarnessModelRequest(
        thread_id="thread", turn_id="turn", binding_manifest={}, messages=(ModelMessage(role="user", body="task"),)
    )
    prepared = model.prepare(request)
    artifact = frozen_request_payload(prepared)
    restored = model.replay(artifact, request=request)

    def reject(*args, **kwargs):
        raise AssertionError("current wire formatter called during replay dispatch")

    monkeypatch.setattr("agent_runtime.modeling.gateway.serialize_openai_request", reject)
    response = await model.dispatch(restored)
    assert response.text == "one complete response"
    assert provider.received["messages"] == [dict(message) for message in artifact["wire_payload"]["messages"]]


@pytest.mark.parametrize("kind", ["canonical_request", "wire_payload"])
def test_payload_mutation_is_detected_before_replay(kind):
    from agent_runtime.harness.prepared_requests import frozen_request_payload
    from agent_runtime.tools.tool import _thaw_json

    model, request, prepared = prepare()
    artifact = _thaw_json(frozen_request_payload(prepared))
    if kind == "canonical_request":
        artifact[kind]["messages"][-1]["body"] = "replaced task"
    else:
        artifact[kind]["messages"][-1]["content"] = "replaced task"
    with pytest.raises(ValueError, match="hash mismatch"):
        model.replay(artifact, request=request)
