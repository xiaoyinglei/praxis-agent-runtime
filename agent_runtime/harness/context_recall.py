"""Read canonical context Items using the executing Turn's visibility boundary."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, cast

from agent_runtime.harness.rollout import RolloutStore
from agent_runtime.harness.tool_orchestrator import current_tool_turn_id
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolTarget,
    json_schema_input,
)

READ_CONTEXT_NAME = "read_context"
MAX_CONTEXT_PAGE_CHARS = 4_000
MAX_CONTEXT_SEARCH_ITEMS = 64
_INPUT_SCHEMA: Mapping[str, JsonValue] = {
    "type": "object",
    "properties": {
        "item_id": {"type": "string", "minLength": 1, "maxLength": 256},
        "query": {"type": "string", "minLength": 1, "maxLength": 200},
        "offset": {"type": "integer", "minimum": 0},
        "max_chars": {"type": "integer", "minimum": 1, "maximum": MAX_CONTEXT_PAGE_CHARS},
        "start_index": {"type": "integer", "minimum": 0},
        "view": {"type": "string", "enum": ("content", "raw")},
    },
    "required": ("item_id",),
    "additionalProperties": False,
}


def _item_text(kind: str, payload: Mapping[str, Any], *, raw: bool) -> tuple[str, str]:
    """Expose evidence once; the canonical payload remains available explicitly."""
    if kind == "tool_result" and not raw:
        if not payload.get("is_error"):
            structured = payload.get("structured_content")
            if isinstance(structured, Mapping) and isinstance(structured.get("content"), str):
                return structured["content"], "tool_content"
            if structured is not None:
                return json.dumps(structured, ensure_ascii=False, sort_keys=True), "tool_content"
            blocks = payload.get("content", [])
            texts = [b["data"]["text"] for b in blocks
                     if b.get("type") == "text" and isinstance(b.get("data", {}).get("text"), str)]
            if texts:
                return "\n".join(texts), "tool_content"
        if isinstance(payload.get("model_content"), str):
            return payload["model_content"], "tool_content"
    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")), "json_payload"


def create_context_recall_tool(store: RolloutStore) -> Tool:
    """Create a resident read-only tool; never accept caller-provided Turn IDs or paths."""

    def run(arguments: Mapping[str, JsonValue]) -> NormalizedToolOutput:
        items = store.list_context_items(current_tool_turn_id())
        history_positions = {entry.item_id: index for index, entry in enumerate(items)}
        item = next(
            (item for item in items if item.item_id == arguments["item_id"] and item.status == "completed"), None
        )
        catalog = arguments["item_id"] == "history"
        if item is None and not catalog:
            return NormalizedToolOutput(
                is_error=True,
                error_code="context_item_unavailable",
                error_message="No completed context Item with this ID is visible to the current Turn.",
            )
        raw = arguments.get("view") == "raw"
        value: object
        if catalog:
            archived = {
                item_id
                for entry in items
                if entry.kind == "context_compaction"
                for item_id in (
                    entry.payload["tool_result_overrides"]
                    if "tool_result_overrides" in entry.payload
                    and entry.payload.get("algorithm_revision") != "semantic-compaction-v5"
                    else (
                        set(entry.payload.get("covered_item_ids", ()))
                        - set(entry.payload.get("preserved_item_ids", ()))
                        | set(entry.payload.get("tool_result_overrides", {}))
                    )
                )
            }
            if isinstance(arguments.get("query"), str):
                start_index = cast(int, arguments.get("start_index", 0))
                query = cast(str, arguments["query"])
                pattern = re.compile(re.escape(query), re.IGNORECASE)
                max_chars = cast(int, arguments.get("max_chars", 2_000))
                searchable = [
                    entry
                    for entry in items
                    if entry.item_id in archived
                    and entry.status == "completed"
                    and entry.kind
                    in {
                        "user_message",
                        "agent_message",
                        "context_message",
                        "completion_feedback",
                        "tool_result",
                        "input_file",
                        "verification",
                    }
                    and entry.payload.get("tool_name") != READ_CONTEXT_NAME
                    and history_positions[entry.item_id] >= start_index
                ]
                page: dict[str, JsonValue] = {
                    "item_id": "history",
                    "query": query,
                    "matched": False,
                    "text": "",
                    "search_scope": "visible_archived_originals",
                    "searched_from_index": start_index,
                    "searched_to_index": start_index,
                    "next_index": None,
                    "search_complete": True,
                }
                for index, entry in enumerate(searchable[:MAX_CONTEXT_SEARCH_ITEMS]):
                    position = history_positions[entry.item_id]
                    payload, content_format = _item_text(entry.kind, entry.payload, raw=raw)
                    match = pattern.search(payload)
                    has_more = index + 1 < len(searchable)
                    page.update(
                        searched_to_index=position,
                        next_index=position + 1 if has_more else None,
                        search_complete=not has_more,
                    )
                    if match is not None:
                        offset = max(0, match.start() - max_chars // 4)
                        page.update(
                            matched=True,
                            matched_item_id=entry.item_id,
                            history_index=position,
                            kind=entry.kind,
                            offset=offset,
                            match_offset=match.start(),
                            next_match_offset=match.end(),
                            total_chars=len(payload),
                            text=payload[offset : offset + max_chars],
                            format=content_format,
                        )
                        break
                return NormalizedToolOutput(
                    content=(
                        ToolContentBlock(
                            type="text",
                            data={"text": json.dumps(page, ensure_ascii=False)},
                        ),
                    )
                )
            value = [
                {
                    "item_id": entry.item_id,
                    "kind": entry.kind,
                    "turn_id": entry.turn_id,
                    "history_index": history_positions[entry.item_id],
                }
                for entry in items
                if entry.item_id in archived
                and entry.status == "completed"
                and entry.kind not in {"model_request", "context_summary_response"}
            ]
            payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            content_format = "json_payload"
        else:
            assert item is not None
            payload, content_format = _item_text(item.kind, item.payload, raw=raw)
        offset = cast(int, arguments.get("offset", 0))
        max_chars = cast(int, arguments.get("max_chars", 2_000))
        if offset > len(payload):
            return NormalizedToolOutput(
                is_error=True,
                error_code="context_offset_out_of_range",
                error_message="offset exceeds the serialized Item payload length.",
            )
        item_query = arguments.get("query")
        search_metadata: dict[str, JsonValue] = {}
        if isinstance(item_query, str):
            match = re.compile(re.escape(item_query), re.IGNORECASE).search(payload, offset)
            search_metadata = {
                "query": item_query,
                "matched": match is not None,
                "searched_from": offset,
                "searched_to": len(payload),
            }
            if match is None:
                offset = len(payload)
            else:
                search_metadata["match_offset"] = match.start()
                search_metadata["next_match_offset"] = match.end()
                offset = max(0, match.start() - max_chars // 4)
        text = payload[offset : offset + max_chars]
        end = offset + len(text)
        page = {
            "item_id": arguments["item_id"],
            "history_index": history_positions[item.item_id] if item is not None else None,
            "format": content_format,
            "offset": offset,
            "total_chars": len(payload),
            "text": text,
            "truncated": end < len(payload),
            "next_offset": end if end < len(payload) else None,
            **search_metadata,
        }
        return NormalizedToolOutput(
            content=(
                ToolContentBlock(
                    type="text",
                    data={"text": json.dumps(page, ensure_ascii=False)},
                ),
            )
        )

    return Tool(
        definition=ToolDefinition(
            name=READ_CONTEXT_NAME,
            description=(
                "Recall original content from a compacted context reference by canonical item_id. "
                "Use item_id=history to page through the archived Item ID catalog. "
                "With item_id=history AND query, search archived original CONTENT without knowing its ID; "
                "returns the first matching item and excerpt. start_index (default 0) is a history index "
                "for this search mode; use next_index to continue. Each search examines up to 64 archived items. "
                "matched=false only describes searched_from_index through searched_to_index; "
                "only search_complete=true means there are no further archived items in this view. "
                "For more matches within the same item use matched_item_id and next_match_offset. "
                "history_index gives chronological order in this visible history; higher means later. "
                "The catalog contains metadata only; search a specific item for its content. "
                "For a specific fact, supply query to search the ENTIRE original payload case-insensitively "
                "and return a bounded excerpt around the first match; use next_match_offset to find another. "
                "Never infer absence from skipped pages. A matched=false search covers searched_from to searched_to. "
                "Tool results default to the readable original content, without duplicated storage envelopes. "
                "Use view=raw for the full canonical JSON payload. Non-tool Items and the catalog use JSON text. "
                "Offsets and search matches refer to the selected view; keep view unchanged when paging. "
                "Concatenate JSON text pages before parsing. offset is a Unicode character offset (default 0); "
                "max_chars defaults to 2000, maximum 4000. If truncated, continue at next_offset. "
                "Only completed Items visible to this Turn, including its fork ancestry, can be read."
            ),
            input_schema=_INPUT_SCHEMA,
        ),
        validate_input=json_schema_input(_INPUT_SCHEMA),
        run=run,
        normalize_output=_normalize_output,
        output_schema=None,
        static_effects=frozenset(),
        resolve_use=lambda arguments: ResolvedToolUse(
            effects=frozenset(),
            targets=(ToolTarget(kind="context_item", value=str(arguments["item_id"])),),
        ),
        execution_revision="context-recall-v5",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=5.0,
        # Even control characters escaped by both JSON layers fit without executor truncation.
        max_model_output_bytes=65_536,
    )


def _normalize_output(raw: object) -> NormalizedToolOutput:
    if not isinstance(raw, NormalizedToolOutput):
        raise TypeError("context recall must return NormalizedToolOutput")
    return raw
