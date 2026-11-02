"""Portable, validated canonical and provider-wire artifacts for exact replay.

Credentials live in the resolved endpoint, outside this representation. Replay
reconstructs value objects directly and never invokes prompt or wire formatters.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from agent_runtime.budget import ResourceUsage
from agent_runtime.core.messages import canonical_json_text, restore_model_message
from agent_runtime.core.model_request import (
    CANONICAL_REQUEST_REVISION,
    ModelRequest,
    ModelSettings,
    StableModelContext,
    ToolChoice,
    ToolChoiceMode,
    canonical_hash,
    canonical_model_request_json,
    freeze_json_mapping,
    stable_context_json,
)
from agent_runtime.harness.protocol import PreparedModelCall
from agent_runtime.modeling.local_agent_wire import LOCAL_AGENT_WIRE_REVISION, LocalAgentWireRequest
from agent_runtime.modeling.openai_wire import OPENAI_WIRE_REVISION, OpenAIWireRequest
from agent_runtime.tools.tool import ToolDefinition, _thaw_json

FROZEN_REQUEST_REVISION = "frozen-model-request-v1"
_SECRET_FIELDS = frozenset({"api_key", "authorization", "access_token", "secret_key", "password", "api_token"})


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _no_credentials(canonical: Mapping[str, Any], wire: Mapping[str, Any]) -> None:
    # Endpoint authentication must never enter model generation settings. History
    # and tool schemas are data: a user-defined field named password is not an
    # endpoint credential and must not prevent a request from being prepared.
    options = canonical["settings"]["provider_options"]
    generation = wire.get("generation_options", wire)
    for values in (options, generation):
        if any(str(key).casefold() in _SECRET_FIELDS for key in values):
            raise ValueError("provider credentials cannot be persisted in frozen requests")
        extra = values.get("extra_body")
        if isinstance(extra, Mapping) and any(str(key).casefold() in _SECRET_FIELDS for key in extra):
            raise ValueError("provider credentials cannot be persisted in frozen requests")


def encode_frozen_request(
    *,
    request: ModelRequest,
    context: StableModelContext,
    wire: OpenAIWireRequest | LocalAgentWireRequest,
    prepared: PreparedModelCall,
    provider: str,
    stage: str,
) -> Mapping[str, Any]:
    canonical = json.loads(canonical_model_request_json(request))
    wire_payload = json.loads(wire.serialized_json)
    _no_credentials(canonical, wire_payload)
    payload = {
        "protocol_revision": FROZEN_REQUEST_REVISION,
        "canonical_revision": CANONICAL_REQUEST_REVISION,
        "canonical_request": canonical,
        "stable_context": json.loads(stable_context_json(context)),
        "wire_kind": "local" if isinstance(wire, LocalAgentWireRequest) else "openai",
        "wire_payload": wire_payload,
        "serializer_revision": wire.serializer_revision,
        "provider": provider,
        "stage": stage,
        "request_hash": prepared.request_hash,
        "context_hash": prepared.context_hash,
        "tool_hash": prepared.tool_hash,
        "wire_hash": prepared.wire_hash,
        "request_ref": _thaw_json(prepared.request_ref),
        "resource_request": {
            field.name: getattr(prepared.resource_request, field.name) for field in fields(ResourceUsage)
        },
    }
    return freeze_json_mapping(payload)


def frozen_request_payload(prepared: PreparedModelCall) -> Mapping[str, Any] | None:
    """Return JSON artifact content; custom models may have no frozen codec."""
    if prepared.frozen_payload is None:
        return None
    artifact = {
        **dict(prepared.frozen_payload),
        "request_ref": _thaw_json(prepared.request_ref),
        "resource_request": {
            field.name: getattr(prepared.resource_request, field.name) for field in fields(ResourceUsage)
        },
    }
    plain = _thaw_json(freeze_json_mapping(artifact))
    if not isinstance(plain, dict):
        raise TypeError("frozen request artifact must be a JSON object")
    return plain


def decode_frozen_request(payload: Mapping[str, Any]) -> tuple[ModelRequest, OpenAIWireRequest | LocalAgentWireRequest]:
    if payload.get("protocol_revision") != FROZEN_REQUEST_REVISION:
        raise ValueError("unsupported frozen request protocol revision")
    if payload.get("canonical_revision") != CANONICAL_REQUEST_REVISION:
        raise ValueError("unsupported canonical request revision")
    canonical = payload["canonical_request"]
    _no_credentials(canonical, payload["wire_payload"])
    if _digest(canonical_json_text(canonical)) != payload["request_hash"]:
        raise ValueError("frozen request hash mismatch")
    context = payload["stable_context"]
    revision_payload = {
        key: context[key] for key in ("serializer_revision", "instructions", "frozen_run_context", "initial_user_task")
    }
    expected_revision = "context_" + canonical_hash(revision_payload)
    if context["context_revision"] != expected_revision or _digest(expected_revision) != payload["context_hash"]:
        raise ValueError("frozen context hash mismatch")
    if _digest(canonical["toolset_revision"]) != payload["tool_hash"]:
        raise ValueError("frozen tool hash mismatch")
    request = ModelRequest(
        request_id=canonical["request_id"],
        messages=tuple(restore_model_message(message) for message in canonical["messages"]),
        tools=tuple(ToolDefinition(**tool) for tool in canonical["tools"]),
        exposed_tool_names=tuple(canonical["exposed_tool_names"]),
        tool_choice=ToolChoice(ToolChoiceMode(canonical["tool_choice"]["mode"]), canonical["tool_choice"]["name"]),
        settings=ModelSettings(**canonical["settings"]),
        prompt_revision=canonical["prompt_revision"],
        toolset_revision=canonical["toolset_revision"],
    )
    revision = payload["serializer_revision"]
    data = payload["wire_payload"]
    wire: OpenAIWireRequest | LocalAgentWireRequest
    if payload["wire_kind"] == "openai":
        if revision != OPENAI_WIRE_REVISION:
            raise ValueError("unsupported frozen OpenAI serializer revision")
        actual = "wire_" + canonical_hash({"serializer_revision": revision, "payload": data})
        wire = OpenAIWireRequest(
            payload=freeze_json_mapping(data),
            serialized_json=canonical_json_text(data),
            provider_wire_hash=actual,
            serializer_revision=revision,
        )
    elif payload["wire_kind"] == "local":
        if revision != LOCAL_AGENT_WIRE_REVISION or data["serializer_revision"] != revision:
            raise ValueError("unsupported frozen local serializer revision")
        actual = "wire_" + canonical_hash(data)
        wire = LocalAgentWireRequest(
            provider=data["provider"],
            prompt=data["prompt"],
            generation_options=freeze_json_mapping(data["generation_options"]),
            serialized_json=canonical_json_text(data),
            provider_wire_hash=actual,
            serializer_revision=revision,
        )
    else:
        raise ValueError("unsupported frozen wire kind")
    if actual != payload["wire_hash"]:
        raise ValueError("frozen wire hash mismatch")
    return request, wire


def restore_prepared_call(payload: Mapping[str, Any], *, dispatch_payload: object) -> PreparedModelCall:
    # The adapter must decode first, so every replay validates all four hashes.
    return PreparedModelCall(
        request_hash=payload["request_hash"],
        context_hash=payload["context_hash"],
        tool_hash=payload["tool_hash"],
        wire_hash=payload["wire_hash"],
        request_ref=freeze_json_mapping(payload["request_ref"]),
        resource_request=ResourceUsage(**payload["resource_request"]),
        dispatch_payload=dispatch_payload,
        frozen_payload=freeze_json_mapping(dict(payload)),
    )
