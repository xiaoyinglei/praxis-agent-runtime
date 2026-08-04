from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from agent_runtime.loop.state import LoopState


def normalize_loop_state(
    state: LoopState,
    *,
    observed: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Project a loop state onto stable, user-visible runtime invariants."""

    terminal = state.get("terminal")
    pause = state.get("pause")
    finish = state["finish_state"]
    normalized: dict[str, object] = {
        "status": "done" if state["status"] == "completed" else state["status"],
        "final_answer": finish.final_answer,
        "final_output": _plain(finish.final_output),
        "stop_reason": None if terminal is None else terminal.stop_reason,
        "pause_reason": None if pause is None else pause.reason,
        "iteration": state["iteration"],
        "tool_results": [
            {
                "tool_call_id": result.tool_call_id,
                "tool_name": result.tool_name,
                "outcome": "error" if result.is_error else "ok",
                "text": _result_text(result.content),
                "structured_content": _plain(result.structured_content),
                "error_code": result.error_code,
                "retryable": result.retryable,
            }
            for result in state["tool_results"]
        ],
        "pending_tool_names": [
            pending.tool_name for pending in state["pending_tool_calls"]
        ],
        "active_tool_names": list(state["active_tool_names"]),
        "transcript_roles": [
            message.role for message in state["turn_transcript"]
        ],
        "call_origins": [
            {
                "tool_call_id": call.tool_call_id,
                "request_id": call.origin.request_id,
                "toolset_revision": call.origin.toolset_revision,
                "exposed_tool_names": list(call.origin.exposed_tool_names),
            }
            for call in state["canonical_tool_calls"].values()
        ],
        "messages": [_message(message) for message in state["messages"]],
        "working_summary": _plain(state["memory_state"].working_summary),
        "memory_ref_count": len(state["memory_state"].memory_refs),
        "feedback_codes": [item.code for item in finish.feedback],
    }
    if observed is not None:
        normalized["observed"] = _plain(observed)
    return normalized


def _result_text(content: object) -> str:
    blocks = content if isinstance(content, tuple) else ()
    values: list[str] = []
    for block in blocks:
        data = getattr(block, "data", None)
        if isinstance(data, Mapping):
            text = data.get("text")
            if isinstance(text, str):
                values.append(text)
    return "\n".join(values)


def _message(message: object) -> dict[str, object]:
    if not isinstance(message, BaseMessage):
        return {"value": _plain(message)}
    return {
        "type": message.type,
        "content": _plain(message.content),
    }


def _plain(value: object) -> object:
    if isinstance(value, BaseModel):
        return _plain(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value
