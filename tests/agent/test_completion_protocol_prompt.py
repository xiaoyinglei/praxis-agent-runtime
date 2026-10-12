from agent_runtime.builtin.generic import GENERIC_SYSTEM_PROMPT
from agent_runtime.tools.builtins.shell import create_run_command_tool
from agent_runtime.workspace import WorkspaceRuntime


def test_system_prompt_exposes_completion_as_a_final_response_not_a_tool() -> None:
    normalized = " ".join(GENERIC_SYSTEM_PROMPT.split())

    assert "There is no `finish` tool" in normalized
    assert "return a non-empty final answer with zero tool calls" in normalized


def test_verifier_invocation_guidance_is_owned_by_the_tool(tmp_path) -> None:
    description = create_run_command_tool(WorkspaceRuntime(root=tmp_path, is_temporary=False)).definition.description
    assert "Running a pytest file with python does not execute its tests" in description
    assert "execute_python" in description
    assert "pytest -q" not in GENERIC_SYSTEM_PROMPT
