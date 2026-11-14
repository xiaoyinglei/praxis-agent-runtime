"""read_artifact 内置工具(设计 §5.7;D10)。

超限工具结果被写时外部化到 ``<workspace>/.praxis/artifacts``;``.praxis`` 是
地板条目(read/write/bash 均不可达),所以模型回读**必须**走这个显式工具——
不给 path target 开口子。

目标形状是 ``ToolTarget(kind="artifact", value=digest)``:kind 非 ``"path"``,
因此不参与地板/路径裁决;工具本身也只按 digest 取,无任何路径成分。

错误码:``invalid_input``(digest 形状/参数)、``artifact_not_found``、
``artifact_integrity``(读后 sha256 不符)。一切畸形输入都返回结果,不抛穿管道。
"""

from __future__ import annotations

import codecs
from collections.abc import Mapping
from typing import cast

from praxis.protocol.artifacts import ArtifactIntegrityError, ArtifactReader
from praxis.protocol.json import JsonValue
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolResult,
    ToolTarget,
    ToolValidationError,
    json_schema_input,
)

PENDING_TOOL_CALL_ID = "pending"

_DEFAULT_MAX_BYTES = 65_536
# 必须严格小于本工具 max_model_output_bytes(200_000):否则取满一页会被管道
# 再次外部化,模型要的一页变成又一张指针。
_MAX_BYTES_LIMIT = 196_608
_TOOL_NAME = "read_artifact"


def _text(text: str) -> ToolContentBlock:
    return ToolContentBlock(type="text", data={"text": text})


def _error(error_code: str, message: str) -> ToolResult:
    return ToolResult(
        tool_call_id=PENDING_TOOL_CALL_ID,
        tool_name=_TOOL_NAME,
        content=(_text(f"{error_code}: {message}"),),
        status=ResultStatus.ERROR,
        is_error=True,
        error_code=error_code,
        error_message=message,
    )


def _decode_page(page: bytes, *, final: bool) -> tuple[str, int]:
    """UTF-8 分页解码:页尾不完整的码点留给下一页,不产出替换字符。

    ``final=True``(已到文件尾)时由解码器用 ``replace`` 兜底非法字节,
    这就是"fail-open 读":坏字节不会让工具炸掉。
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text = decoder.decode(page, final=final)
    pending = len(decoder.getstate()[0])
    return text, len(page) - pending


def create_read_artifact_tool(store: ArtifactReader) -> Tool:
    """按 digest 回读 artifact。``store`` 由组合根注入 ``ArtifactStore``(结构匹配)。"""
    validate_input = json_schema_input(
        {
            "type": "object",
            "properties": {
                "digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "offset": {"type": "integer", "minimum": 0},
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _MAX_BYTES_LIMIT,
                },
            },
            "required": ("digest",),
            "additionalProperties": False,
        }
    )

    async def run(
        arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext  # 进程内工具:不消费沙箱
    ) -> ToolResult:
        try:
            args = validate_input(arguments)
        except ToolValidationError as exc:
            return _error("invalid_input", f"{exc.path}: {exc.message}")
        digest = cast("str", args["digest"])
        offset = cast("int", args.get("offset", 0))
        max_bytes = cast("int", args.get("max_bytes", _DEFAULT_MAX_BYTES))
        try:
            data = store.get(digest)
        except ArtifactIntegrityError as exc:
            return _error("artifact_integrity", str(exc))
        except ValueError as exc:  # digest 形状(直接调用 run 时的防御)
            return _error("invalid_input", str(exc))
        if data is None:
            return _error("artifact_not_found", f"no artifact for digest {digest}")
        page = data[offset : offset + max_bytes]
        has_more = offset + len(page) < len(data)
        text, consumed = _decode_page(page, final=not has_more)
        return ToolResult(
            tool_call_id=PENDING_TOOL_CALL_ID,
            tool_name=_TOOL_NAME,
            content=(_text(text),),
            status=ResultStatus.OK,
            truncated=has_more,
            metadata={
                "digest": digest,
                "offset": offset,
                "size_bytes": len(data),
                "next_offset": offset + consumed if has_more else None,
                "truncated": has_more,
            },
        )

    def resolve_use(arguments: Mapping[str, JsonValue]) -> ResolvedUse:
        value = arguments.get("digest")
        # 管道在 resolve_use 之前已校验;这里的兜底只为直接调用不产生空 target。
        digest = value if isinstance(value, str) and value else "invalid-input"
        return ResolvedUse(
            effects=frozenset({ToolEffect.READ_WORKSPACE}),
            targets=(ToolTarget(kind="artifact", value=digest),),
        )

    return Tool(
        definition=ToolDefinition(
            name=_TOOL_NAME,
            description=(
                "Read a stored tool-output artifact by digest, in byte pages; "
                "returns next_offset when truncated."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "digest": {
                        "type": "string",
                        "description": "64-char lowercase hex sha256 digest from a stored artifact pointer.",
                    },
                    "offset": {"type": "integer", "description": "Byte offset to start reading (default 0)."},
                    "max_bytes": {
                        "type": "integer",
                        "description": (
                            f"Maximum bytes to return, 1..{_MAX_BYTES_LIMIT} "
                            f"(default {_DEFAULT_MAX_BYTES})."
                        ),
                    },
                },
                "required": ("digest",),
                "additionalProperties": False,
            },
        ),
        validate_input=validate_input,
        run=run,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=resolve_use,
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=10,
        max_model_output_bytes=200_000,
    )
