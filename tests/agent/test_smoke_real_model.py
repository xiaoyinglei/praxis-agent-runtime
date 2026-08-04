"""Real model smoke tests — verify the full agent loop works with a live LLM.

Uses DeepSeek (cheapest available model) for minimal end-to-end checks.
Requires DEEPSEEK_API_KEY in .env and network access.

Run:
    RUN_REAL_MODEL_SMOKE=1 DEEPSEEK_API_KEY=... uv run pytest tests/agent/test_smoke_real_model.py -q -v
"""

from __future__ import annotations

import os

import pytest

requires_real_model = pytest.mark.skipif(
    os.environ.get("RUN_REAL_MODEL_SMOKE") != "1" or not os.environ.get("DEEPSEEK_API_KEY"),
    reason="Set RUN_REAL_MODEL_SMOKE=1 and DEEPSEEK_API_KEY to run real model smoke tests",
)


def _deepseek_service():
    from agent_runtime.runtime.builder import build_agent_service

    return build_agent_service(
        None,
        model_alias="deepseek_chat",
    )


@pytest.mark.anyio
@requires_real_model
class TestRealModelSmoke:
    async def test_hello(self) -> None:
        """Agent returns a simple text response via DeepSeek."""
        from agent_runtime.service import AgentRunRequest

        svc = _deepseek_service()
        result = await svc.run(
            AgentRunRequest(
                message='Say exactly: "OK"',
                max_turns=10,
            )
        )

        assert result.status == "done", f"status={result.status}, stop_reason={result.stop_reason}"
        assert result.final_answer is not None
        assert "ok" in result.final_answer.lower()

    async def test_simple_math(self) -> None:
        """Model answers 2+2 correctly."""
        from agent_runtime.service import AgentRunRequest

        svc = _deepseek_service()
        result = await svc.run(
            AgentRunRequest(
                message="What is 2 + 2? Answer with just the number.",
                max_turns=10,
            )
        )

        assert result.status == "done", f"status={result.status}, stop_reason={result.stop_reason}"
        assert result.final_answer is not None
        assert "4" in result.final_answer
