"""The live-task judge must link verification to the exact executed command."""

from types import SimpleNamespace

from evals.context_management.session_code_live import verification_succeeded


def test_successful_directory_listing_does_not_hide_failed_verification():
    items = [
        SimpleNamespace(kind="tool_call", payload={"tool_call_id": "ls", "arguments": {"command": "ls -la"}}),
        SimpleNamespace(kind="tool_call", payload={
            "tool_call_id": "verify", "arguments": {"command": "python3 verify.py"},
        }),
    ]
    operations = [
        SimpleNamespace(tool_name="run_command", tool_call_id="ls", status="succeeded"),
        SimpleNamespace(tool_name="run_command", tool_call_id="verify", status="failed"),
    ]
    assert not verification_succeeded(items, operations)
    operations[1].status = "succeeded"
    assert verification_succeeded(items, operations)


def test_echoing_verification_command_is_not_execution():
    items = [SimpleNamespace(kind="tool_call", payload={
        "tool_call_id": "echo", "arguments": {"command": "echo python3 verify.py"},
    })]
    operations = [SimpleNamespace(tool_name="run_command", tool_call_id="echo", status="succeeded")]
    assert not verification_succeeded(items, operations)
