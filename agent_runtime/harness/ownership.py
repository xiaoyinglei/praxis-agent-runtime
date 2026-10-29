"""Local execution ownership and conservative recovery after owner loss."""
from __future__ import annotations

import fcntl
import hashlib
import os
import stat
import time
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from agent_runtime.harness.protocol import restore_model_response
from agent_runtime.harness.state import RuntimeState, _state_operation
from agent_runtime.model_config_io import ensure_durable_directory
from agent_runtime.user_config import user_config_directory

OWNERSHIP_REVISION = "thread-flock-v1"


class ConversationBusyError(RuntimeError):
    """Another worker still owns this conversation's execution."""


class ConversationRecoveryRequiredError(ValueError):
    """An uncertain tool effect must be reconciled before a new Turn."""


@contextmanager
def execution_owner(database: Path, thread_id: str) -> Iterator[None]:
    """Hold a nonblocking per-Thread lock; process exit releases it automatically.

    Never unlink lock files: other open descriptors must refer to the same inode.
    This lock covers execution, not read-only history or intentional live steering.
    """
    database = database.resolve()
    directory = database.with_name(database.name + ".locks")
    with _file_owner(directory, hashlib.sha256(thread_id.encode()).hexdigest(), exclusive=True):
        yield


@contextmanager
def workspace_owner(workspace: Path, *, exclusive: bool = False) -> Iterator[None]:
    """Share an external workspace lock while Sessions are open; deletion excludes them."""
    config = user_config_directory()
    ensure_durable_directory(config)
    directory = config / "workspace-locks"
    workspace = workspace.resolve()
    # A Session can use a subdirectory (or nested repository) of a managed
    # checkout and keep its DB elsewhere. Every containing checkout must
    # therefore remain alive, not just the exact workspace path.
    roots = {workspace}
    if not exclusive:
        roots.update(parent for parent in workspace.parents if (parent / ".git").exists())
    with ExitStack() as locks:
        for root in sorted(roots):
            key = hashlib.sha256(os.fsencode(root)).hexdigest()
            locks.enter_context(_file_owner(directory, key, exclusive=exclusive))
        yield


@contextmanager
def _file_owner(directory: Path, key: str, *, exclusive: bool) -> Iterator[None]:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError("unsafe conversation lock directory")
    path = directory / (key + ".lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise ValueError("unsafe conversation lock file")
        try:
            fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ConversationBusyError("这个会话仍由另一进程执行；请回到原终端，或等待它停止后再继续。") from None
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def assert_recovery_leases(*, owned: bool, leases: Iterable[float | None]) -> None:
    if not owned and any(lease is None or lease >= time.time() for lease in leases):
        raise ConversationBusyError("旧会话的执行租约尚未过期，暂不能接管；请确认原进程状态后再试。")


@_state_operation
def recover_owner_loss(store: RuntimeState, turn_id: str, *, abort: bool = False) -> None:
    """Reconcile dispatches while holding execution_owner, without redoing effects.

    New Turns promise the OS lock. Legacy workers did not hold it, so only their
    expired operation leases authorize fencing; an unexpired lease stays intact.
    """
    turn = store.read_turn(turn_id)
    if turn.status not in {"running", "paused", "interrupted"}:
        return
    owned = turn.binding_manifest.get("execution_ownership") == OWNERSHIP_REVISION
    now = time.time()
    models = [op for op in store.list_model_operations(turn_id) if op.status == "dispatched"]
    tools = [op for op in store.list_tool_operations(turn_id) if op.status == "running"]
    attempts = {op.operation_id: store.list_model_attempts(op.operation_id)[-1] for op in models}
    assert_recovery_leases(owned=owned, leases=(
        *(attempt.lease_expires_at for attempt in attempts.values()),
        *(op.lease_expires_at for op in tools),
    ))
    with store.persistence.transaction():
        for operation in models:
            attempt = attempts[operation.operation_id]
            store.mark_model_attempt_unknown(operation_id=operation.operation_id,
                attempt_id=attempt.attempt_id, generation=attempt.generation,
                error_type="ExecutionOwnerLost",
                error_message="Previous model worker stopped before committing a response")
        # Reconcile one uncertain effect at a time. Resolving it resumes the Turn;
        # the next owned recovery then exposes the next effect in the batch.
        waiting = any(item.kind == "tool_reconciliation" and item.status == "pending"
                      for item in store.list_interactions(turn_id))
        for tool_operation in (() if waiting else tools[:1]):
            if tool_operation.lease_expires_at is not None and tool_operation.lease_expires_at < now:
                store.expire_tool_operation_claim(operation_id=tool_operation.operation_id)
            else:
                assert owned and tool_operation.fencing_token is not None
                if not store.commit_tool_operation_outcome(operation_id=tool_operation.operation_id,
                    claim_generation=tool_operation.claim_generation, fencing_token=tool_operation.fencing_token,
                    status="unknown", attempt_count=tool_operation.attempt_count,
                    error_code="execution_owner_lost", requires_reconciliation=True):
                    raise RuntimeError("tool execution changed during owner recovery")
        if (abort and owned and store.read_turn(turn_id).status == "running"
                and not any(op.status in {"running", "unknown"} for op in store.list_tool_operations(turn_id))):
            _close_unstarted_calls(store, turn_id)
            if store.read_turn(turn_id).status == "running":
                store.pause_turn(turn_id=turn_id, reason="Previous execution owner stopped; abort requested")


def _close_unstarted_calls(store: RuntimeState, turn_id: str) -> None:
    """Close provider tool groups without executing calls abandoned before start."""
    operations = {op.tool_call_id: op for op in store.list_tool_operations(turn_id)}
    results = {item.payload.get("tool_call_id") for item in store.list_items(turn_id) if item.kind == "tool_result"}
    for model in store.list_model_operations(turn_id):
        if model.status != "completed" or model.response_item_id is None:
            continue
        response = restore_model_response(store.read_item(model.response_item_id).payload)
        if response.status == "incomplete":
            continue
        for call in response.tool_calls:
            if call.id in results:
                continue
            operation = operations.get(call.id)
            if operation is not None and operation.result_item_id is not None:
                continue
            if operation is not None and operation.attempt_count:
                store.mark_tool_result_missing(operation_id=operation.operation_id)
                return
            if operation is not None:
                if operation.status == "prepared":
                    store.record_tool_execution_state(turn_id=turn_id, operation_id=operation.operation_id,
                        tool_call_id=operation.tool_call_id, tool_name=operation.tool_name,
                        arguments_digest=operation.arguments_digest, execution_revision=operation.execution_revision,
                        idempotent=operation.idempotent, status="ready", attempt_count=0, error_code=None,
                        requires_reconciliation=False, effects=operation.effects, resources=operation.resources)
                if operation.status in {"prepared", "ready"}:
                    store.reject_ready_tool_operation(operation_id=operation.operation_id, error_code="not_executed")
            store.record_tool_result(turn_id=turn_id,
                operation_id=None if operation is None else operation.operation_id,
                result={"tool_call_id": call.id, "tool_name": call.name, "is_error": True,
                    "error_code": "not_executed", "error_message": "Execution stopped before this tool started",
                    "model_content": "Tool was not executed: the previous worker stopped before it started.",
                    "content": [{"type": "text", "data": {"text": "操作未执行：原执行进程已停止。"}}],
                    "attachments": []})
            results.add(call.id)
