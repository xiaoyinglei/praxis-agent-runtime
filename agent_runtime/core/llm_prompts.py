from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_runtime.loop.state import LoopState


def build_loop_turn_prompt(
    state: LoopState,
    *,
    budget_remaining: int | None = None,
    allowed_tools: Sequence[str] = (),
) -> str:
    """Build the model contract for the ordinary Python loop kernel.

    ``allowed_tools`` is the already-selected model-facing tool-name list.
    """

    current_message = state["current_message"]
    iteration = state.get("iteration", 0)
    tool_results = state.get("tool_results", [])
    ok_count = sum(1 for result in tool_results if getattr(result, "status", None) == "ok")
    error_count = sum(1 for result in tool_results if getattr(result, "status", None) == "error")
    visible_names = list(allowed_tools)
    budget_line = ""
    if budget_remaining is not None:
        budget_text = "unbounded" if budget_remaining < 0 else str(budget_remaining)
        budget_line = f"\nBudget remaining: {budget_text}"

    return f"""Current user message: {current_message}
Iteration: {iteration}
Tools completed: {ok_count} ok, {error_count} failed
Available tools: {", ".join(visible_names) if visible_names else "none"}{budget_line}

Analyze the current message and conversation context, then decide your next action.

If a tool can advance the request → return action="execute" with concrete tool_calls.
For simple questions, greetings, literal reply requests, or tasks answerable
from the current conversation → return action="finish" without calling tools.
If you have enough context to answer → return action="finish" with a complete
final_answer. Preserve citations when you used evidence.
If you need external input → return action="pause" with a clear pause_reason.

Do not repeat completed tool calls. Preserve citations, scores, and artifact paths.
Keep tool arguments bounded — no full documents or logs in arguments.

Return JSON:
{{
    "action": "execute" | "finish" | "pause",
    "tool_calls": [{{"tool_call_id": "tc_xxx", "tool_name": "...", "arguments": {{...}}}}],
    "final_answer": "...",
    "pause_reason": "..."
}}""".strip()
