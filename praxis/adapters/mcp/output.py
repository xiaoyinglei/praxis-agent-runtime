"""MCP 调用结果 → praxis ``ToolResult`` 的归一(legacy ``:172-241`` 逐条移植)。

**输入形状**:mcp SDK 的 ``CallToolResult``(pydantic 模型)或等价的
mapping/dataclass。字段读取一律经 ``_field``,与旧实现同源,因此 content 块
既可以是 SDK 模型也可以是 JSON dict。

**content 块映射(逐字 legacy)**:
- text ⇒ ``ToolContentBlock("text", {"text": ...})``;
- image ⇒ ``ToolContentBlock("image", {"data": <base64>, "mime_type": <mime>})``
  (键名 ``mime_type`` 与旧实现一致;输入字段名是 ``mimeType``);
- resource ⇒ 有 ``resource.text`` 时折成 **text** 块
  ``{"text", "resource_uri", "truncated": False}``(旧实现如此,便于模型直接
  阅读文本资源);只有 blob/uri 时是 ``resource`` 块 ``{"uri"}``。
- content 为空时保留旧 fallback:顶层 ``text`` / ``images`` / ``resources``
  三个兼容字段(旧 adapter 的非标准输入形状)。

**structuredContent**:旧 ``NormalizedToolOutput.structured_content`` 是独立
字段,praxis protocol 有意删除了该字段(见 protocol/tools.py 模块 docstring),
因此这里把它放进 ``metadata["structured_content"]``(JSON 规范化后),与
``mcp_server``/``mcp_tool`` 同仓。T21 对照表应把此落点记为有意差异。

**isError**:``status=ERROR``、``error_code="mcp_tool_error"``、
``error_message`` 取第一条非空 text 块的前 512 字符(无文本则固定文案);
``metadata`` 恒为 ``{mcp_server, mcp_tool}``(旧实现同源,成功失败都有)。
其余异常(畸形 structured 值等)不在此吞掉:由 descriptors 的 run 统一折算
``mcp_call_failed``,保持"输出归一不制造第二套错误码"。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import AnyUrl

from praxis.adapters.mcp.names import canonical_tool_name
from praxis.protocol.json import JsonValue, freeze_json
from praxis.protocol.tools import ResultStatus, ToolContentBlock, ToolResult

_PENDING_TOOL_CALL_ID = "pending"
_ERROR_MESSAGE_LIMIT = 512
_FALLBACK_ERROR_MESSAGE = "MCP tool reported an error"


def normalize_mcp_output(
    raw: object,
    *,
    server_name: str,
    tool_name: str,
) -> ToolResult:
    """把一次 ``call_tool`` 结果归一为 ``ToolResult``(tool_call_id 用占位符)。

    ``server_name``/``tool_name`` 是**配置里的原始名字**(metadata 口径与旧
    descriptor 一致);``tool_name`` 字段取 canonical 名(``mcp__server__tool``)。
    """
    blocks: list[ToolContentBlock] = []
    content = _field(raw, "content", default=())
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        for item in content:
            block = _content_block(item)
            if block is not None:
                blocks.append(block)

    if not blocks:
        text = _field(raw, "text", default="")
        if isinstance(text, str) and text:
            blocks.append(ToolContentBlock(type="text", data={"text": text}))
        images = _field(raw, "images", default=())
        if isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
            for image in images:
                if isinstance(image, str):
                    blocks.append(ToolContentBlock(type="image", data={"data": image}))
        resources = _field(raw, "resources", default=())
        if isinstance(resources, Sequence) and not isinstance(resources, (str, bytes)):
            for resource in resources:
                if isinstance(resource, str):
                    blocks.append(ToolContentBlock(type="resource", data={"uri": resource}))

    structured = _field(raw, "structuredContent", "structured_content", default=None)
    metadata: dict[str, JsonValue] = {
        "mcp_server": server_name,
        "mcp_tool": tool_name,
    }
    if structured is not None:
        metadata["structured_content"] = freeze_json(structured, path="structuredContent")

    is_error = bool(_field(raw, "isError", "is_error", default=False))
    error_text = next(
        (
            str(block.data.get("text", ""))
            for block in blocks
            if block.type == "text" and block.data.get("text")
        ),
        _FALLBACK_ERROR_MESSAGE,
    )
    return ToolResult(
        tool_call_id=_PENDING_TOOL_CALL_ID,
        tool_name=canonical_tool_name(server_name, tool_name),
        content=tuple(blocks),
        status=ResultStatus.ERROR if is_error else ResultStatus.OK,
        is_error=is_error,
        error_code="mcp_tool_error" if is_error else None,
        error_message=error_text[:_ERROR_MESSAGE_LIMIT] if is_error else None,
        retryable=False,
        metadata=metadata,
    )


def _content_block(raw: object) -> ToolContentBlock | None:
    block_type = _field(raw, "type", default="")
    if block_type == "text":
        text = _field(raw, "text", default="")
        if isinstance(text, str):
            return ToolContentBlock(type="text", data={"text": text})
    if block_type == "image":
        data = _field(raw, "data", default="")
        media_type = _field(raw, "mimeType", "mime_type", default="")
        if isinstance(data, str):
            payload: dict[str, JsonValue] = {"data": data}
            if isinstance(media_type, str) and media_type:
                payload["mime_type"] = media_type
            return ToolContentBlock(type="image", data=payload)
    if block_type == "resource":
        resource = _field(raw, "resource", default={})
        uri = _field(raw, "uri", default=None)
        if uri is None:
            uri = _field(resource, "uri", default="")
        text = _field(resource, "text", default=None)
        if isinstance(text, str):
            return ToolContentBlock(
                type="text",
                data={
                    "text": text,
                    "resource_uri": str(uri),
                    "truncated": False,
                },
            )
        if isinstance(uri, (str, AnyUrl)):
            return ToolContentBlock(type="resource", data={"uri": str(uri)})
    return None


def _field(raw: object, *names: str, default: object) -> object:
    for name in names:
        if isinstance(raw, Mapping) and name in raw:
            return raw[name]
        if hasattr(raw, name):
            return getattr(raw, name)
    return default


__all__ = ["normalize_mcp_output"]