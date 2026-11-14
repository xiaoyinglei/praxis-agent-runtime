"""update_plan 内置工具（B16：无截断计划投影）。

写侧规范化工具；读侧投影 praxis.protocol.plan.latest_plan。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from praxis.protocol.json import JsonValue, thaw_json
from praxis.protocol.plan import (
    MAX_PLAN_STEPS,
    MAX_PLAN_SUMMARY_CHARS,
    MAX_STEP_TITLE_CHARS,
)
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolResult,
    ToolValidationError,
    json_schema_input,
)

# Status values for JSON Schema enum (statically known).
_STATUS_VALUES = ("pending", "in_progress", "completed")


# ---------------------------------------------------------------------------
# 输入校验器
# ---------------------------------------------------------------------------

_UPDATE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "plan": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_PLAN_STEPS,
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "string", "maxLength": 80},
                    "step": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_STEP_TITLE_CHARS,
                    },
                    "status": {"enum": _STATUS_VALUES},
                },
                "required": ["status", "step"],
                "additionalProperties": False,
            },
        },
        "explanation": {
            "type": "string",
            "maxLength": MAX_PLAN_SUMMARY_CHARS,
        },
    },
    "required": ["plan"],
    "additionalProperties": False,
}

_validate_schema = json_schema_input(_UPDATE_INPUT_SCHEMA)


def _validate_plan_input(
    arguments: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    args = _validate_schema(arguments)
    # Thaw before mutating: json_schema_input returns frozen mappingproxy/tuple
    raw_plan = list(thaw_json(args["plan"]))  # type: ignore[arg-type]
    if not isinstance(raw_plan, list):
        raise ToolValidationError(path="$.plan", message="plan must be a list")

    # Count in_progress steps (on the thawed view)
    in_progress_count = 0
    for entry in raw_plan:
        s = entry.get("status") if isinstance(entry, dict) else None
        if s == "in_progress":
            in_progress_count += 1
    if in_progress_count > 1:
        raise ToolValidationError(
            path="$.plan",
            message="At most one step can be in_progress at a time",
        )

    # Assign deterministic step_ids where missing
    for idx, entry in enumerate(raw_plan):
        if isinstance(entry, dict) and entry.get("step_id") is None:
            entry["step_id"] = f"step_{idx + 1}"

    # Replace the plan key with our thawed/mutated version
    result: dict[str, JsonValue] = dict(args)
    result["plan"] = raw_plan  # type: ignore[assignment]
    return result


# ---------------------------------------------------------------------------
# 工具构造
# ---------------------------------------------------------------------------

def create_update_plan_tool(workspace: object) -> Tool:
    """创建 update_plan 内置工具。

    workspace 参数保留以匹配现有工具工厂签名，本工具本身不使用文件系统。
    """

    async def run(
        arguments: Mapping[str, JsonValue],
        exec_ctx: ToolExecContext,  # noqa: ARG001 — unused, in-progress tool
    ) -> ToolResult:
        validated = _validate_plan_input(arguments)
        plan = cast("list[dict[str, Any]]", validated["plan"])

        step_dicts: list[dict[str, str]] = []
        for e in plan:
            entry: dict[str, str] = {
                "step_id": cast("str", e.get("step_id")),
                "step": cast("str", e.get("step")),
                "status": cast("str", e.get("status")),
            }
            step_dicts.append(entry)

        metadata: dict[str, JsonValue] = {
            "accepted": True,
            "revision": len(plan),
            "authority": "advisory",
            "step_count": len(plan),
            "plan": step_dicts,  # type: ignore[dict-item]
        }
        explanation = validated.get("explanation")
        if isinstance(explanation, str) and explanation.strip():
            metadata["explanation"] = explanation[:MAX_PLAN_SUMMARY_CHARS]

        content_lines = [f"[update_plan] plan v{metadata['revision']} ({metadata['step_count']} steps):"]
        for i, step in enumerate(plan, 1):
            status_icon = (
                "▶" if step.get("status") == "in_progress" else "○"
            )
            sid = step.get("step_id", "")
            content_lines.append(f"  {i}. [{sid}] {status_icon} {step.get('step', '')}")

        text = "\n".join(content_lines)

        return ToolResult(
            tool_call_id="pending",
            tool_name="update_plan",
            content=(ToolContentBlock(type="text", data={"text": text}),),
            status=ResultStatus.OK,
            metadata=metadata,
        )

    return Tool(
        definition=ToolDefinition(
            name="update_plan",
            description=(
                "Update the current execution plan: record progress, "
                "mark steps as completed, and set at most one step to in_progress. "
                "Returns an advisory snapshot; revision equals step count."
            ),
            input_schema=_UPDATE_INPUT_SCHEMA,
        ),
        validate_input=lambda a: _validate_plan_input(a),
        run=run,
        static_effects=frozenset(),
        resolve_use=lambda _a: ResolvedUse(frozenset(), ()),
        idempotent=True,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.FINISH_CURRENT,
        timeout_seconds=5.0,
        max_model_output_bytes=50_000,
    )
