from __future__ import annotations

import inspect

from agent_runtime import Agent


def test_public_agent_execution_surface_is_async_only() -> None:
    assert inspect.iscoroutinefunction(Agent.run)
    assert inspect.iscoroutinefunction(Agent.resume)
    assert inspect.iscoroutinefunction(Agent.read_result)
    assert inspect.iscoroutinefunction(Agent.pending_input)
    assert inspect.isasyncgenfunction(Agent.stream)

    for removed_name in (
        "arun",
        "aresume",
        "aread_result",
        "apending_input",
        "astream",
    ):
        assert not hasattr(Agent, removed_name)
