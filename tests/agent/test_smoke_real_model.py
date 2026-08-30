"""Real model smoke tests — verify the full agent loop works with a live LLM.

Uses DeepSeek (cheapest available model) for minimal end-to-end checks.
Requires DEEPSEEK_API_KEY in .env and network access.

Run:
    RUN_REAL_MODEL_SMOKE=1 DEEPSEEK_API_KEY=... uv run pytest tests/agent/test_smoke_real_model.py -q -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

requires_real_model = pytest.mark.skipif(
    os.environ.get("RUN_REAL_MODEL_SMOKE") != "1" or not os.environ.get("DEEPSEEK_API_KEY"),
    reason="Set RUN_REAL_MODEL_SMOKE=1 and DEEPSEEK_API_KEY to run real model smoke tests",
)


def _deepseek_agent(workspace: Path):
    from agent_runtime.agent import Agent

    return Agent(
        model="deepseek_chat",
        workspace_path=workspace,
        checkpoint_db=workspace / ".praxis" / "runtime" / "rollout.sqlite3",
        model_session_path=workspace / ".praxis" / "runtime" / "model-session.json",
    )


@pytest.mark.anyio
@requires_real_model
class TestRealModelSmoke:
    async def test_hello(self, tmp_path: Path) -> None:
        """Agent returns a simple text response via DeepSeek."""
        agent = _deepseek_agent(tmp_path)
        result = await agent.arun(
            'Say exactly: "OK"',
            max_turns=10,
            require_workspace_change=False,
        )

        assert result.status == "done", f"status={result.status}, stop_reason={result.stop_reason}"
        assert result.answer is not None
        assert "ok" in result.answer.lower()

    async def test_simple_math(self, tmp_path: Path) -> None:
        """Model answers 2+2 correctly."""
        agent = _deepseek_agent(tmp_path)
        result = await agent.arun(
            "What is 2 + 2? Answer with just the number.",
            max_turns=10,
            require_workspace_change=False,
        )

        assert result.status == "done", f"status={result.status}, stop_reason={result.stop_reason}"
        assert result.answer is not None
        assert "4" in result.answer
