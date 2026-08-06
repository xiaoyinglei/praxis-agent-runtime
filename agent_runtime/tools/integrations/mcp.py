from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolTarget,
    json_schema_input,
    json_schema_output,
)

_INVALID_NAME = re.compile(r"[^a-z0-9_]+")
_MAX_TEXT_CHARS = 100_000
_MAX_IMAGE_DATA_CHARS = 250_000
_MAX_CONTENT_BLOCKS = 20

type MCPCaller = Callable[
    [str, str, Mapping[str, JsonValue]],
    object | Awaitable[object],
]


@dataclass(frozen=True, slots=True)
class MCPToolDescriptor:
    """One discovered MCP definition, detached from its transport lifecycle."""

    server_name: str
    tool_name: str
    description: str
    input_schema: Mapping[str, JsonValue]
    read_only_hint: bool = False
    destructive_hint: bool = False
    idempotent_hint: bool = False
    timeout_seconds: float = 30.0
    execution_revision: str = "server-v1"

    def __post_init__(self) -> None:
        canonical_mcp_name(self.server_name, self.tool_name)
        if not isinstance(self.description, str):
            raise TypeError("MCP tool description must be a string")
        if not isinstance(self.input_schema, Mapping):
            raise TypeError("MCP input_schema must be a mapping")
        if not isinstance(self.execution_revision, str) or not self.execution_revision:
            raise ValueError("MCP execution_revision must be non-empty")
        for name in (
            "read_only_hint",
            "destructive_hint",
            "idempotent_hint",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")


def normalize_mcp_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("MCP name must be a string")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _INVALID_NAME.sub("_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


def canonical_mcp_name(server_name: str, tool_name: str) -> str:
    server = normalize_mcp_name(server_name)
    tool = normalize_mcp_name(tool_name)
    if not server or not tool:
        raise ValueError("MCP server and tool names must normalize to non-empty values")
    return f"mcp__{server}__{tool}"


def create_mcp_tools(
    descriptors: Sequence[MCPToolDescriptor],
    call_tool: MCPCaller,
) -> tuple[Tool, ...]:
    """Project concrete descriptions and an externally owned caller into Tools."""

    if not callable(call_tool):
        raise TypeError("call_tool must be callable")
    tools: list[Tool] = []
    names: set[str] = set()
    for descriptor in descriptors:
        if not isinstance(descriptor, MCPToolDescriptor):
            raise TypeError("descriptors must contain MCPToolDescriptor values")
        canonical_name = canonical_mcp_name(
            descriptor.server_name,
            descriptor.tool_name,
        )
        if canonical_name in names:
            raise ValueError(
                f"duplicate canonical MCP tool name: {canonical_name}"
            )
        names.add(canonical_name)
        tools.append(_create_mcp_tool(descriptor, call_tool))
    return tuple(tools)


def _create_mcp_tool(
    descriptor: MCPToolDescriptor,
    call_tool: MCPCaller,
) -> Tool:
    name = canonical_mcp_name(descriptor.server_name, descriptor.tool_name)
    validate_input = json_schema_input(descriptor.input_schema)
    effects = {ToolEffect.NETWORK}
    if descriptor.destructive_hint:
        effects.add(ToolEffect.DESTRUCTIVE)

    async def run(arguments: Mapping[str, JsonValue]) -> object:
        result = call_tool(
            descriptor.server_name,
            descriptor.tool_name,
            arguments,
        )
        return await result if inspect.isawaitable(result) else result

    return Tool(
        definition=ToolDefinition(
            name=name,
            description=(
                descriptor.description.strip()
                or (
                    f"Call {descriptor.tool_name} on the configured MCP server "
                    f"{descriptor.server_name}."
                )
            ),
            input_schema=descriptor.input_schema,
        ),
        validate_input=validate_input,
        run=run,
        normalize_output=lambda raw: _normalize_mcp_output(raw, descriptor),
        output_schema=None,
        static_effects=frozenset(effects),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset(effects),
            targets=(ToolTarget(kind="mcp_tool", value=name),),
        ),
        execution_revision=(
            f"integration-mcp-v1:{descriptor.execution_revision}"
        ),
        idempotent=(
            not descriptor.destructive_hint
            and (descriptor.read_only_hint or descriptor.idempotent_hint)
        ),
        concurrency_safe=(
            descriptor.read_only_hint and not descriptor.destructive_hint
        ),
        cancellation_mode=CancellationMode.REMOTE_BEST_EFFORT,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=descriptor.timeout_seconds,
        max_model_output_bytes=1_000_000,
    )


def _normalize_mcp_output(
    raw: object,
    descriptor: MCPToolDescriptor,
) -> NormalizedToolOutput:
    blocks: list[ToolContentBlock] = []
    content = _field(raw, "content", default=())
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        for item in content[:_MAX_CONTENT_BLOCKS]:
            block = _content_block(item)
            if block is not None:
                blocks.append(block)

    if not blocks:
        text = _field(raw, "text", default="")
        if isinstance(text, str) and text:
            blocks.append(
                ToolContentBlock(
                    type="text",
                    data={"text": text[:_MAX_TEXT_CHARS]},
                )
            )
        images = _field(raw, "images", default=())
        if isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
            for image in images[:_MAX_CONTENT_BLOCKS - len(blocks)]:
                if isinstance(image, str):
                    blocks.append(
                        ToolContentBlock(
                            type="image",
                            data={"data": image[:_MAX_IMAGE_DATA_CHARS]},
                        )
                    )
        resources = _field(raw, "resources", default=())
        if isinstance(resources, Sequence) and not isinstance(
            resources,
            (str, bytes),
        ):
            for resource in resources[:_MAX_CONTENT_BLOCKS - len(blocks)]:
                if isinstance(resource, str):
                    blocks.append(
                        ToolContentBlock(
                            type="resource",
                            data={"uri": resource[:4096]},
                        )
                    )

    structured = _field(
        raw,
        "structuredContent",
        "structured_content",
        default=None,
    )
    if structured is not None:
        structured = json_schema_output(None, cast(JsonValue, structured))
    is_error = bool(_field(raw, "isError", "is_error", default=False))
    error_text = next(
        (
            str(block.data.get("text", ""))
            for block in blocks
            if block.type == "text" and block.data.get("text")
        ),
        "MCP tool reported an error",
    )
    return NormalizedToolOutput(
        content=tuple(blocks),
        structured_content=structured,
        is_error=is_error,
        error_code="mcp_tool_error" if is_error else None,
        error_message=error_text[:512] if is_error else None,
        retryable=False,
        metadata={
            "mcp_server": descriptor.server_name,
            "mcp_tool": descriptor.tool_name,
        },
    )


def _content_block(raw: object) -> ToolContentBlock | None:
    block_type = _field(raw, "type", default="")
    if block_type == "text":
        text = _field(raw, "text", default="")
        if isinstance(text, str):
            return ToolContentBlock(
                type="text",
                data={"text": text[:_MAX_TEXT_CHARS]},
            )
    if block_type == "image":
        data = _field(raw, "data", default="")
        media_type = _field(raw, "mimeType", "mime_type", default="")
        if isinstance(data, str):
            payload: dict[str, JsonValue] = {
                "data": data[:_MAX_IMAGE_DATA_CHARS]
            }
            if isinstance(media_type, str) and media_type:
                payload["mime_type"] = media_type[:200]
            return ToolContentBlock(type="image", data=payload)
    if block_type == "resource":
        uri = _field(raw, "uri", default=None)
        if uri is None:
            resource = _field(raw, "resource", default={})
            uri = _field(resource, "uri", default="")
        if isinstance(uri, str):
            return ToolContentBlock(
                type="resource",
                data={"uri": uri[:4096]},
            )
    return None


def _field(raw: object, *names: str, default: object) -> object:
    for name in names:
        if isinstance(raw, Mapping) and name in raw:
            return raw[name]
        if hasattr(raw, name):
            return getattr(raw, name)
    return default


__all__ = [
    "MCPCaller",
    "MCPToolDescriptor",
    "canonical_mcp_name",
    "create_mcp_tools",
    "normalize_mcp_name",
]
