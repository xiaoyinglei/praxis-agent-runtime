"""Model-free host controls for ledger-authorized detached commands."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from agent_runtime.harness.rollout import RolloutStore
from agent_runtime.harness.tool_orchestrator import ToolOrchestrator
from agent_runtime.permission_policy import ResolvedPermissions
from agent_runtime.tools.builtins.command_sessions import DurableCommandSessions, ManageCommandInput
from agent_runtime.tools.builtins.shell import RunCommandOutput, _normalize_command_output, create_run_command_tool
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import ToolContinuation
from agent_runtime.workspace import open_workspace

if TYPE_CHECKING:
    from agent_runtime.agent import Agent


async def cancel_commands(agent: Agent, turn_id: str) -> None:
    """Abort must reconcile detached processes before declaring the Turn cancelled."""
    async with await RolloutStore.open(agent._harness_database(), read_only=True) as store:
        keys = []
        for operation in await store.run_async(store.list_tool_operations, turn_id):
            if operation.tool_name != "run_command" or operation.status != "succeeded" or not operation.result_item_id:
                continue
            output = (await store.run_async(store.read_item, operation.result_item_id)).payload.get(
                "structured_content", {}
            )
            if (isinstance(output.get("session_id"), str) and isinstance(output.get("request_digest"), str)
                    and output.get("status") == "running"):
                keys.append(output["session_id"])
    for key in keys:
        await manage_command(agent, turn_id, key, action="cancel")


async def manage_command(agent: Agent, turn_id: str, session_id: str, *,
                         action: Literal["status", "wait", "cancel", "read"] = "status",
                         wait_seconds: float = 10, stream: Literal["stdout", "stderr"] = "stdout",
                         offset: int = 0, max_bytes: int = 16000) -> RunCommandOutput:
    request = ManageCommandInput(session_id=session_id, action=action, wait_seconds=wait_seconds,
                                 stream=stream, offset=offset, max_bytes=max_bytes)
    async with await RolloutStore.open(agent._harness_database()) as store:
        if not (await store.run_async(store.verify)).valid:
            raise RuntimeError("Rollout integrity check failed")
        turn = await store.run_async(store.read_turn, turn_id)
        if (await store.run_async(store.read_thread, turn.thread_id)).workspace != str(agent.workspace_path.resolve()):
            raise ValueError("Turn belongs to a different workspace")
        saved = turn.binding_manifest.get("tool_execution_policy", {}).get("permissions")
        ceiling = None if saved is None else ResolvedPermissions.model_validate(saved)
        workspace = open_workspace(agent.workspace_path)

        def permissions() -> ResolvedPermissions:
            current = agent.effective_permissions()
            return current if ceiling is None else current.intersection(ceiling)

        workspace.permission_provider = permissions
        owner = DurableCommandSessions(workspace)
        tools = {"run_command": create_run_command_tool(workspace, command_sessions=owner)}
        orchestrator = ToolOrchestrator(store=store, tools=tools, execution_context=ToolExecutionContext())
        try:
            await orchestrator.restore_command_sessions(owner, turn_id, include_terminal=True)
            if request.action == "read":
                result = await owner.read_log(session_id, request.stream, request.offset, request.max_bytes)
            else:
                result = await owner.manage(session_id, action=request.action, wait_seconds=request.wait_seconds)
            if result.status != "running" and result.session_id == session_id:
                normalized = _normalize_command_output(result)
                await orchestrator.close_continuation(ToolContinuation(owner, session_id, "observe", True), {
                    "structured_content": result.model_dump(mode="json"), "is_error": normalized.is_error,
                    "error_code": normalized.error_code, "error_message": normalized.error_message,
                })
            return result
        finally:
            await owner.detach()
