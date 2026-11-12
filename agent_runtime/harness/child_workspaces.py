"""Attached child environments and reviewed integration over canonical Items.

Environment facts belong to the parent Turn; no separate task lifecycle exists.
Filesystem writes have a durable intent. A crash leaves an unresolved integration
and both workspaces for explicit reconciliation, never a guessed rollback.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_runtime.budget import ResourceUsage
from agent_runtime.harness import artifact_ops, tool_ops, turn_ops
from agent_runtime.harness.child_workspace_ops import KIND, record, state
from agent_runtime.harness.ownership import execution_owner, workspace_owner
from agent_runtime.permission_policy import PermissionPolicy, ResolvedPermissions
from agent_runtime.permission_policy import relocate_workspace_permissions as relocated_permissions
from agent_runtime.tools.builtins.file_access import atomic_bytes, atomic_symlink, remove_empty_directory, unlink_file
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import ToolEffect
from agent_runtime.user_config import command_state_directory, user_config_directory
from agent_runtime.workspace_snapshot import archive_tree, capture_snapshot, git_bytes, tree_entries, working_entries
from agent_runtime.worktrees import WorktreeManager

if TYPE_CHECKING:
    from agent_runtime.harness.session import Session


def child_context(
    context: ToolExecutionContext, source: Path, destination: Path, *, read_only: bool
) -> ToolExecutionContext:
    def live() -> ResolvedPermissions:
        policy = context.refreshed().effective_permissions or PermissionPolicy().resolve(source)
        result = relocated_permissions(policy, source, destination)
        readonly = (
            PermissionPolicy(profile="read-only").resolve(destination).model_copy(update={"network": result.network})
        )
        return result.intersection(readonly) if read_only else result

    def live_context() -> ToolExecutionContext:
        return replace(
            context.refreshed(),
            workspace_root=destination,
            cwd=destination,
            permissions=live(),
            permission_provider=None,
            context_provider=None,
            command_grants=(),
        )

    return replace(
        context,
        workspace_root=destination,
        cwd=destination,
        permissions=live(),
        permission_provider=live,
        context_provider=live_context,
        command_grants=(),
        approved_tool_call_ids=frozenset(),
        deny_effects=context.deny_effects
        | (frozenset({ToolEffect.WRITE_WORKSPACE, ToolEffect.DESTRUCTIVE}) if read_only else frozenset()),
    )


def prepare(session: Session, *, parent: str | None, child: str, thread: str, mode: str) -> dict[str, Any] | None:
    if mode not in {"auto", "write", "read-only"}:
        raise ValueError("workspace_mode must be auto, write, or read-only")
    tools = session._service_options.get("tools") or {}
    mutable = any(
        tool.static_effects & {ToolEffect.WRITE_WORKSPACE, ToolEffect.EXECUTE_PROCESS} for tool in tools.values()
    )
    write = mode == "write" or (mode == "auto" and mutable)
    if parent is None:
        if write:
            raise ValueError("isolated writing children require an attached parent Turn")
        return None
    identity = {
        "database": str(session.database.resolve()),
        "parent_turn_id": parent,
        "child_turn_id": child,
        "thread_id": thread,
        "mode": "write" if write else "read-only",
    }
    manager = WorktreeManager(session.workspace_path)

    def persist(environment: dict[str, object]) -> None:
        # Git archive is immutable and retained independently of worktree cleanup.
        archive = archive_tree(Path(str(environment["repository"])), str(environment["base_commit"]))
        artifact = artifact_ops.commit_artifact(
            session.store, turn_id=parent, content=archive, name=f"{child}-source.tar", media_type="application/x-tar"
        )
        environment["source_artifact_id"] = artifact.artifact_id
        record(session.store, parent, {**environment, "status": "preparing"})

    excluded = (
        user_config_directory(),
        command_state_directory(),
        *tuple(
            Path(str(session.database.resolve()) + suffix)
            for suffix in (
                "",
                "-wal",
                "-shm",
                ".artifacts",
                ".locks",
                ".host.lock",
                ".host.log",
            )
        ),
    )

    def authorize(path: Path) -> None:
        if not session.permissions().path_access(path):
            raise PermissionError(f"source snapshot path denied: {path}")

    report, snapshot = manager.create_child_snapshot(
        child,
        identity=identity,
        before_create=persist,
        excluded=excluded,
        authorize=authorize,
    )
    environment = state(session.store, parent, child)
    assert environment is not None
    # Keep exact staged/unstaged provenance in an artifact, outside the model binding.
    source = artifact_ops.commit_artifact(
        session.store,
        turn_id=parent,
        content=json.dumps(snapshot).encode(),
        name=f"{child}-source.json",
        media_type="application/json",
    )
    environment.update(source_manifest_id=source.artifact_id, workspace=str(report.worktree.root), status="ready")
    record(session.store, parent, environment)
    return environment


def integrate(
    session: Session,
    parent: str,
    child: str,
    action: str,
    artifact_id: str | None,
    verification_tool_call_id: str | None = None,
    resolution_note: str | None = None,
    diff_offset: int = 0,
    diff_limit: int = 100_000,
) -> dict[str, Any]:
    store = session.store
    current = state(store, parent, child)
    if current is None:
        raise ValueError("child has no isolated environment")
    turn_ops._assert_no_pending_cancellation(store, parent)
    repository = Path(current["repository"])
    excluded = tuple(Path(p) for p in current.get("excluded_paths", ()))

    def parent_snapshot() -> dict[str, Any]:
        return capture_snapshot(session.workspace_path, object_root=repository, excluded=excluded)

    def parent_entries(
        paths: tuple[str, ...],
        observations: dict[str, tuple[int, int, int, int, int] | None] | None = None,
    ) -> dict[str, dict[str, str]]:
        return working_entries(
            session.workspace_path,
            object_root=repository,
            excluded=excluded,
            paths=paths,
            observations=observations,
            include_directories=True,
        )

    def diff_page() -> dict[str, Any]:
        if (
            type(diff_offset) is not int
            or type(diff_limit) is not int
            or diff_offset < 0
            or not 1 <= diff_limit <= 200_000
        ):
            raise ValueError("diff page offset/limit is invalid")
        content = store.read_artifact(current["artifact_id"])
        end = min(len(content), diff_offset + diff_limit)
        while end < len(content) and content[end] & 0xC0 == 0x80:
            end += 1
        through = current.get("reviewed_through", 0)
        if diff_offset <= through:
            current["reviewed_through"] = max(through, end)
        current["diff_size_bytes"] = len(content)
        record(store, parent, current)
        return {
            **current,
            "diff": content[diff_offset:end].decode("utf-8", errors="backslashreplace"),
            "diff_next_offset": end if end < len(content) else None,
        }

    if action == "verify":
        from agent_runtime.workspace import workspace_tree_sha256

        if current["status"] != "integrated":
            raise RuntimeError("child artifact must be integrated before parent verification")
        operation = next(
            (op for op in store.list_tool_operations(parent) if op.tool_call_id == verification_tool_call_id), None
        )
        if (
            operation is None
            or operation.tool_name not in {"run_command", "manage_command"}
            or not operation.result_item_id
        ):
            raise ValueError("verification requires a parent command tool_call_id")
        result = store.read_item(operation.result_item_id)
        output = result.payload.get("structured_content", {})
        metadata = result.payload.get("metadata", {})
        from agent_runtime.harness.delivery import receipt_issue

        receipt = output.get("verification_receipt")
        versioned = (
            operation.execution_revision == "builtin-run-command-v13-versioned-verification"
            and isinstance(receipt, dict)
            and receipt.get("source_workspace") == str(session.workspace_path)
            and receipt_issue(receipt, parent_snapshot())[0] is None
        )
        integrated = max(
            i.sequence
            for i in store.list_items(parent)
            if i.kind == KIND
            and i.payload.get("child_turn_id") == child
            and i.payload.get("status") == "integrated"
            and i.payload.get("verification") == "required"
        )
        if (
            operation.status != "succeeded"
            or result.producer != "tool"
            or result.sequence <= integrated
            or output.get("exit_code") != 0
            or output.get("status") in {"running", "cancelled"}
            or output.get("timed_out")
            or output.get("sandbox_error")
            or not versioned and (
                metadata.get("workspace_tree_after_sha256") is None
                or metadata["workspace_tree_after_sha256"] != workspace_tree_sha256(session.workspace_path)
            )
        ):
            raise RuntimeError(
                "parent verification is unsuccessful, stale, or lacks a workspace fingerprint; "
                "rerun with workspace_write=true"
            )
        evidence = tool_ops.record_verification(
            store,
            turn_id=parent,
            operation_id=operation.operation_id,
            kind="assertion",
            verifier="explicit-parent-integration-check",
            verified_resources=tuple(current.get("changed_paths", ())),
        )
        verified_snapshot = parent_snapshot()
        verified_entries = parent_entries(tuple(current.get("changed_paths", ())))
        if (
            versioned and receipt_issue(receipt, parent_snapshot())[0] is not None
            or not versioned
            and metadata["workspace_tree_after_sha256"] != workspace_tree_sha256(session.workspace_path)
        ):
            raise RuntimeError("workspace changed while recording parent verification")
        current.update(
            verification="recorded",
            verification_item_id=evidence.item_id,
            verified_tree=verified_snapshot["tree"],
            verified_entries=verified_entries,
        )
        record(store, parent, current)
        return current
    if action == "discard":
        try:
            child_status = store.read_turn(child).status
        except KeyError:
            child_status = "not_allocated"
        if child_status not in {"completed", "failed", "cancelled", "not_allocated"}:
            raise RuntimeError("active child cannot be discarded")
        current.update(status="discarded", verification="not_evaluated")
        record(store, parent, current)
        return current
    if action not in {"inspect", "apply", "reconcile", "resolve"}:
        raise ValueError("integration action must be inspect, apply, reconcile, verify, or discard")
    if current["status"] == "applying" and action == "apply":
        raise RuntimeError("integration outcome unknown; inspect retained before/after images and reconcile explicitly")
    checkout = Path(current["workspace"])
    child_turn = store.read_turn(child)
    if child_turn.status not in {"completed", "failed", "cancelled"}:
        raise RuntimeError("child must be terminal before artifact inspection or integration")
    budget = store.read_budget_state(child)
    if budget.reserved + budget.uncertain + budget.child_reserved != ResourceUsage():
        raise RuntimeError("child has unresolved or unknown exposure; preserve the environment")
    if action == "inspect":
        if current.get("artifact_id"):
            return diff_page()
        with workspace_owner(checkout, exclusive=True), execution_owner(session.database, child_turn.thread_id):
            snapshot = capture_snapshot(checkout)
            final_tree = snapshot["tree"]
            patch = git_bytes(
                checkout, "diff", "--binary", "--no-ext-diff", "--no-textconv", current["base_tree"], final_tree
            )
            artifact = artifact_ops.commit_artifact(
                store, turn_id=parent, content=patch, name=f"{child}.patch", media_type="text/x-diff"
            )
            archive = artifact_ops.commit_artifact(
                store,
                turn_id=parent,
                content=archive_tree(checkout, final_tree),
                name=f"{child}-result.tar",
                media_type="application/x-tar",
            )
            current.update(
                status="reviewed",
                artifact_id=artifact.artifact_id,
                result_archive_id=archive.artifact_id,
                final_tree=final_tree,
                verification="required",
            )
            record(store, parent, current)
            return diff_page()
    if artifact_id is None or current.get("artifact_id") != artifact_id:
        raise ValueError("apply requires the exact reviewed artifact_id")
    if current.get("reviewed_through", 0) < store.read_artifact_metadata(artifact_id).size_bytes:
        raise ValueError("read all diff pages before applying the reviewed artifact")
    if current["mode"] != "write":
        raise PermissionError("read-only child artifacts cannot be applied")
    if current["status"] == "integrated":
        return current
    base = tree_entries(repository, current["base_tree"])
    final = tree_entries(repository, current["final_tree"])
    changed = sorted(p for p in base.keys() | final.keys() if base.get(p) != final.get(p))
    if action == "resolve":
        if not resolution_note or not resolution_note.strip():
            raise ValueError("resolve requires a resolution_note describing the reviewed manual integration")
        snapshot = parent_snapshot()
        patch = git_bytes(
            repository, "diff", "--binary", "--no-ext-diff", "--no-textconv", current["base_tree"], snapshot["tree"]
        )
        artifact = artifact_ops.commit_artifact(
            store, turn_id=parent, content=patch, name=f"{child}-resolution.patch", media_type="text/x-diff"
        )
        current.update(
            status="integrated",
            verification="required",
            changed_paths=changed,
            integrated_tree=snapshot["tree"],
            resolution_note=resolution_note,
            resolution_artifact_id=artifact.artifact_id,
        )
        record(store, parent, current)
        return current
    original_versions: dict[str, tuple[int, int, int, int, int] | None] = {}
    live = parent_entries(tuple(changed), original_versions)
    replaced_directories: dict[str, dict[str, tuple[int, int]]] = {}
    for path in changed:
        if live.get(path, {}).get("mode") != "040000" or path not in final:
            continue
        expected = {p: entry for p, entry in base.items() if p.startswith(path + "/")}
        if not expected:
            continue
        names: list[str] = []
        for directory, directories, files in os.walk(session.workspace_path / path, followlinks=False):
            names.extend(
                str((Path(directory) / name).relative_to(session.workspace_path)) for name in (*directories, *files)
            )
        versions: dict[str, tuple[int, int, int, int, int] | None] = {}
        contents = parent_entries(tuple(names), versions)
        actual_files = {p: entry for p, entry in contents.items() if entry["mode"] != "040000"}
        expected_directories = {
            str(parent) for p in expected for parent in Path(p).parents if str(parent).startswith(path + "/")
        }
        actual_directories = {p for p, entry in contents.items() if entry["mode"] == "040000"}
        allow_missing = current.get("reconciled") or action == "reconcile"
        valid_files = (
            all(expected.get(p) == entry for p, entry in actual_files.items())
            if allow_missing
            else actual_files == expected
        )
        if valid_files and actual_directories <= expected_directories:
            directories_to_remove = {}
            for directory in actual_directories | {path}:
                version = original_versions[path] if directory == path else versions[directory]
                assert version is not None
                directories_to_remove[directory] = version[:2]
            replaced_directories[path] = directories_to_remove
    conflicts = [
        p
        for p in changed
        if p not in replaced_directories
        and live.get(p) != base.get(p)
        and not ((current.get("reconciled") or action == "reconcile") and live.get(p) == final.get(p))
    ]
    permissions = session._resource_permissions()
    for path in changed:
        target = session.workspace_path / path
        if not permissions.path_access(target, write=True):
            raise PermissionError(f"integration path denied: {path}")
    if conflicts:
        return {**current, "status": "conflict", "conflicts": conflicts}
    if action == "reconcile":
        current.update(status="reviewed", reconciled=True)
        record(store, parent, current)
        return current
    if changed:
        from agent_runtime.harness.delivery import begin_before_write

        begin_before_write(store, parent)
    current.update(status="applying", changed_paths=changed)
    record(store, parent, current)  # Durable write intent before first filesystem mutation.
    with tarfile.open(fileobj=io.BytesIO(store.read_artifact(current["result_archive_id"]))) as result_tar:
        members = {member.name: member for member in result_tar.getmembers()}
        removals = sorted((p for p in changed if p not in final), key=lambda p: (-len(Path(p).parts), p))
        additions = sorted((p for p in changed if p in final), key=lambda p: (len(Path(p).parts), p))
        for path in (*removals, *additions):
            target = session.workspace_path / path
            if current.get("reconciled") and parent_entries((path,)).get(path) == final.get(path):
                continue
            if path in replaced_directories:
                for directory, identity in sorted(
                    replaced_directories[path].items(), key=lambda pair: -len(Path(pair[0]).parts)
                ):
                    remove_empty_directory(session.workspace_path / directory, identity=identity)
            # Repeat preimage checks immediately before each write. Partial success
            # is retained as applying/unknown, not reported as an atomic rollback.
            file_versions: dict[str, tuple[int, int, int, int, int] | None] = {}
            if parent_entries((path,), file_versions).get(path) != base.get(path):
                raise RuntimeError(f"integration preimage changed: {path}")
            version = file_versions.get(path)
            if path not in final:
                assert version is not None
                unlink_file(target, expected=version)
                continue
            member = members[path]
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.issym():
                atomic_symlink(target, member.linkname, expected=version, create_only=version is None)
            else:
                stream = result_tar.extractfile(member)
                if stream is None:
                    raise ValueError("invalid artifact file")
                atomic_bytes(
                    target,
                    stream.read(),
                    mode=0o755 if final[path]["mode"] == "100755" else 0o644,
                    expected=version,
                    create_only=version is None,
                    create_parents=True,
                )
    current.update(status="integrated", verification="required", integrated_tree=parent_snapshot()["tree"])
    record(store, parent, current)
    return current
