from agent_runtime.builtin.generic import GENERIC_SYSTEM_PROMPT


def test_system_prompt_exposes_completion_as_a_final_response_not_a_tool() -> None:
    normalized = " ".join(GENERIC_SYSTEM_PROMPT.split())

    assert "There is no `finish` tool" in normalized
    assert "return a non-empty final answer with zero tool calls" in normalized
    assert (
        "running a pytest file with `python` does not execute its tests"
        in normalized.lower()
    )
    assert (
        "use `pytest -q` from the workspace virtual environment"
        in normalized.lower()
    )
