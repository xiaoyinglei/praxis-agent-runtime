"""Read-only history projections for CLI inspection without creating an Agent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from agent_runtime.core.runtime_diagnostics import redact_sensitive_text
from agent_runtime.harness.records import ItemSnapshot
from agent_runtime.harness.state import RuntimeState


@dataclass(frozen=True, slots=True)
class ChatSessionSummary:
    thread_id: str
    title: str
    updated_at_ms: int
    status: str
    head_turn_id: str | None
    active_turn_id: str | None
    model_id: str | None


def list_chat_sessions(database: Path, *, workspace: Path) -> tuple[ChatSessionSummary, ...]:
    """List workspace conversations, excluding runtime-created child agent Threads."""
    if not database.exists():
        return ()
    expected_workspace = workspace.expanduser().resolve()
    with RuntimeState.open_sync(database, read_only=True) as store:
        sessions = []
        for thread in store.list_threads():
            if Path(thread.workspace).resolve() != expected_workspace:
                continue
            turns = store.list_turns(thread.thread_id)
            if not turns:
                continue
            # Child runs have a durable parent binding, unlike user conversation forks.
            if turns[0].binding_manifest.get("budget_parent_turn_id") is not None:
                continue
            if turns[0].binding_manifest.get("memory_maintenance"):
                continue
            first = next((item.payload.get("text") for item in store.list_items(turns[0].turn_id)
                          if item.kind == "user_message"), "")
            latest = store.read_turn(thread.active_turn_id or thread.head_turn_id or turns[-1].turn_id)
            record = store.last_record(thread.thread_id)
            model_id = thread.settings.get("model_id") or latest.binding_manifest.get("model_id")
            sessions.append(ChatSessionSummary(
                thread.thread_id, " ".join(str(first).split())[:100],
                record.committed_at_ms if record else 0, latest.status,
                thread.head_turn_id, thread.active_turn_id, model_id if isinstance(model_id, str) else None,
            ))
        return tuple(sorted(sessions, key=lambda session: session.updated_at_ms, reverse=True))


def read_chat_messages(
    database: Path, *, thread_id: str, workspace: Path,
) -> tuple[tuple[str, str], ...]:
    """Replay saved human/assistant text, including fork ancestors and no tool results.

    Original Items survive context compaction. Pre-v3 runtime answers can be
    aliases of model output; compatibility deduplication is limited to those Turns.
    Incomplete output is not presented as a finished assistant answer.
    """
    with RuntimeState.open_sync(database, read_only=True) as store:
        thread = store.read_thread(thread_id)
        if Path(thread.workspace).resolve() != workspace.expanduser().resolve():
            raise ValueError("会话属于其他工作区")
        anchor = thread.active_turn_id or thread.head_turn_id
        if anchor is None:
            return ()
        items = store.list_conversation_items(anchor)
        final_answers = {item.turn_id: item.payload.get("text") for item in items
                         if item.kind == "agent_message" and item.status == "completed"}
        legacy_turns = {turn_id for turn_id in final_answers
                        if (completion := store.read_turn_completion(turn_id)) is not None
                        and completion.payload_schema_version < 3}
        committed_responses = {operation.response_item_id: operation for turn_id in {item.turn_id for item in items}
                               for operation in store.list_model_operations(turn_id)
                               if operation.status == "completed"
                               and operation.request_ref.get("purpose") != "context_summary"}
        messages = []
        for item in items:
            text = item.payload.get("text")
            if item.status != "completed" or not isinstance(text, str) or not text.strip():
                continue
            if item.kind == "user_message":
                messages.append(("user", text))
                if "input_revision" in item.payload:
                    status = store.turn_input_status(item.item_id)
                    label = {"pending": "已接收 · 待生效", "applied": "已生效（已进入模型请求）",
                             "unconsumed": "本次执行已结束，补充尚未生效"}[status]
                    messages.append(("notice", f"补充 {item.item_id}：{label}"))
            elif item.kind == "agent_message":
                messages.append(("answer", text))
            elif item.kind == "model_response":
                if item.item_id not in committed_responses or item.payload.get("response_status") == "incomplete":
                    continue
                if (item.turn_id in legacy_turns and not item.payload.get("tool_calls")
                        and text == final_answers.get(item.turn_id)):
                    continue
                operation = committed_responses[item.item_id]
                # Reading retained facts grants no execution authority. Only
                # explicit valid revisions participate in supersession filtering.
                revision = operation.request_ref.get("input_revision")
                if (not item.payload.get("tool_calls") and type(revision) is int and revision > 0
                        and revision < store.user_input_revision(item.turn_id)):
                    continue  # Retained in the ledger, superseded as a completion.
                messages.append(("answer", text))
        return tuple(messages)


def list_history_turns(database: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    """List recent Turn identities and statuses without prompts or arguments."""

    _validate_limit(limit)
    with RuntimeState.open_sync(database, read_only=True) as store:
        return [
            {
                "thread_id": turn.thread_id,
                "turn_id": turn.turn_id,
                "status": turn.status,
                "terminal_reason_code": turn.terminal_reason_code,
                "tool_result_count": sum(item.kind == "tool_result" for item in store.list_items(turn.turn_id)),
            }
            for turn in reversed(store.list_turns()[-limit:])
        ]


def read_history(
    database: Path,
    *,
    turn_id: str | None = None,
    tool_call_id: str | None = None,
    item_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Locate canonical ToolResults; omit tool arguments, content and arbitrary metadata."""

    _validate_limit(limit)
    with RuntimeState.open_sync(database, read_only=True) as store:
        items: tuple[ItemSnapshot, ...]
        if item_id is not None:
            items = (store.read_item(item_id),)
            if items[0].kind != "tool_result":
                raise ValueError("history item must be a tool_result")
        elif turn_id is not None:
            store.read_turn(turn_id)
            items = store.list_items(turn_id)
        else:
            items = tuple(
                item
                for turn in store.list_turns()
                for item in store.list_items(turn.turn_id)
                if item.kind == "tool_result"
            )
        rows = [
            _result_row(item)
            for item in items
            if item.kind == "tool_result"
            and (turn_id is None or item.turn_id == turn_id)
            and (tool_call_id is None or item.payload.get("tool_call_id") == tool_call_id)
        ]
        return rows[-limit:]


def _result_row(item: ItemSnapshot) -> dict[str, Any]:
    payload = item.payload
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    row: dict[str, Any] = {
        "thread_id": item.thread_id,
        "turn_id": item.turn_id,
        "item_id": item.item_id,
        "sequence": item.sequence,
        "tool_call_id": payload.get("tool_call_id"),
        "tool_name": payload.get("tool_name"),
        "operation_id": metadata.get("operation_id"),
        "status": item.status,
        "is_error": payload.get("is_error"),
        "error_code": payload.get("error_code"),
        "error_message": payload.get("error_message"),
        "retryable": payload.get("retryable"),
    }
    diagnostic = metadata.get("diagnostic")
    if isinstance(diagnostic, Mapping):
        row["diagnostic"] = _fields(diagnostic, (
            "stage", "exception_type", "message", "traceback_truncated", "turn_id", "tool_call_id", "operation_id",
        ))
        frames = diagnostic.get("traceback")
        if isinstance(frames, (tuple, list)):
            row["diagnostic"]["traceback"] = [_fields(frame, ("filename", "lineno", "function"))
                                               for frame in frames[-12:]]
    structured = payload.get("structured_content")
    if payload.get("tool_name") in {"web_fetch", "web_search"} and isinstance(structured, Mapping):
        row["web"] = _web_details(structured)
    return {key: _redact(value) for key, value in row.items()}


def _fields(value: object, names: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {key: (item[:512] if isinstance(item, str) else item)
            for key in names if key in value and isinstance(item := value[key], (str, int, float, bool, type(None)))}


def _safe_url(value: object) -> str:
    if not isinstance(value, str):
        return "[invalid URL]"
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return "[invalid URL]"
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        if parsed.port is not None:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))[:512]
    except ValueError:
        return "[invalid URL]"


def _web_details(value: Mapping[str, Any]) -> dict[str, Any]:
    details = _fields(value, (
        "source_id", "static_source_id", "diagnostics_source_id", "fetched_at", "content_hash", "render_mode",
        "failure_stage", "failure_origin", "network_bytes",
    ))
    if "url" in value:
        details["url"] = _safe_url(value["url"])
    if isinstance(budget := value.get("request_budget"), Mapping):
        details["request_budget"] = _fields(budget, (
            "limit", "used", "remaining", "unit", "in_flight_counted", "item_id", "reservation_item_id",
            "last_reservation_item_id", "unknown_attempts_counted",
        ))
    if isinstance(render := value.get("render_diagnostics"), Mapping):
        projected = _fields(render, ("requests", "wire_bytes", "decoded_bytes", "dom_bytes", "snapshot_available",
                                    "failed_resources_omitted"))
        for key in ("limits", "remaining", "blocked_actions", "skipped_resources", "resource_failures"):
            if isinstance(counts := render.get(key), Mapping):
                projected[key] = {str(name)[:120]: count for name, count in list(counts.items())[:32]
                                  if isinstance(count, (int, float, bool))}
        if isinstance(timeouts := render.get("load_timeouts"), (tuple, list)):
            projected["load_timeouts"] = [item[:80] for item in timeouts[:16] if isinstance(item, str)]
        if isinstance(resources := render.get("failed_resources"), (tuple, list)):
            projected["failed_resources"] = [
                {**_fields(resource, ("method", "resource_type", "reason")), "url": _safe_url(resource.get("url"))}
                for resource in resources[:16] if isinstance(resource, Mapping)
            ]
        details["render_diagnostics"] = projected
    if isinstance(comparison := value.get("comparison"), Mapping):
        projected = _fields(comparison, ("status", "reason", "scope", "limitation"))
        for key in ("baseline", "current"):
            if isinstance(source := comparison.get(key), Mapping):
                projected[key] = _fields(source, ("source_id", "fetched_at", "content_hash"))
                if "url" in source:
                    projected[key]["url"] = _safe_url(source["url"])
        details["comparison"] = projected
    if isinstance(origin := value.get("failure_origin"), Mapping):
        details["failure_origin"] = _fields(origin, ("item_id", "tool_call_id"))
    return details


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, Mapping):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_redact(item) for item in value]
    return value


def _validate_limit(limit: int) -> None:
    if not 1 <= limit <= 100:
        raise ValueError("history limit must be between 1 and 100")
