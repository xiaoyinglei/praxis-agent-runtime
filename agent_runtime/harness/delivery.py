"""Versioned delivery observations over canonical Items, separate from stopping.

The host checks identity, provenance and freshness. Scope and review conclusions
are declarations by the caller, never an automatic proof of semantic correctness.
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_runtime.harness import artifact_ops, facts, turn_ops
from agent_runtime.harness.command_evidence import command_succeeded, trusted_command_results
from agent_runtime.harness.protocol import item_text, restore_model_response
from agent_runtime.harness.runtime_store import RuntimeStore, owner_operation
from agent_runtime.workspace_snapshot import (
    archive_tree,
    capture_snapshot,
    git_bytes,
    observe_snapshot,
    runtime_excluded_paths,
    tree_from_entries,
)
from agent_runtime.worktrees import WorktreeManager

REVISION = "delivery-v1"
_TOOL_REVISION = "builtin-record-delivery-v2"


def _append(store: RuntimeStore, turn_id: str, kind: str, payload: dict[str, Any]) -> str:
    identity = f"item_{uuid4().hex}"
    thread = turn_ops._running_turn_thread_id(store, turn_id)
    facts.append_fact(
        store,
        thread_id=thread,
        turn_id=turn_id,
        record_type="item_started",
        producer="runtime",
        payload={"item_id": identity, "kind": kind},
    )
    facts.append_fact(
        store,
        thread_id=thread,
        turn_id=turn_id,
        record_type="item_completed",
        producer="runtime",
        payload={"item_id": identity, "payload": payload},
    )
    return identity


def _latest(store: RuntimeStore, turn_id: str, kind: str) -> dict[str, Any] | None:
    return next(
        (
            dict(i.payload)
            for i in reversed(store.list_items(turn_id))
            if i.kind == kind and i.producer == "runtime" and i.status == "completed"
        ),
        None,
    )


def snapshot(workspace: Path, *, excluded: tuple[Path, ...] = (), authorize: Any = None) -> dict[str, Any]:
    # Share D2's raw object capture and ordinary-directory repository. No new
    # worktree allocator, index mutation or alternate versioning algorithm.
    repository = WorktreeManager(workspace)._child_repository(create=True)
    value = capture_snapshot(workspace, object_root=repository, excluded=excluded, authorize=authorize)
    value["repository"] = str(repository)
    return value


def _excluded(store: RuntimeStore) -> tuple[Path, ...]:
    return runtime_excluded_paths(store.persistence._path)


def _effective_permissions(store: RuntimeStore, turn_id: str) -> Any:
    from agent_runtime.harness.child_workspace_ops import ancestor_context
    from agent_runtime.tools.permissions import ToolExecutionContext, restore_tool_policy

    turn = store.read_turn(turn_id)
    thread = store.read_thread(turn.thread_id)
    raw = thread.settings.get("tool_execution_policy", turn.binding_manifest.get("tool_execution_policy", {}))
    context = ToolExecutionContext(workspace_root=Path(thread.workspace), cwd=Path(thread.workspace))
    context = restore_tool_policy(context, turn.binding_manifest.get("tool_execution_policy", {}))
    context = restore_tool_policy(context, raw)
    return ancestor_context(store, turn_id, context).effective_permissions


def _capture(store: RuntimeStore, turn_id: str, *, retain_objects: bool = True) -> dict[str, Any]:
    permissions = _effective_permissions(store, turn_id)

    def authorize(path: Path) -> None:
        if permissions is not None and not permissions.path_access(path):
            raise PermissionError(f"delivery source path denied: {path}")

    workspace = Path(store.read_thread(store.read_turn(turn_id).thread_id).workspace)
    if retain_objects:
        return snapshot(workspace, excluded=_excluded(store), authorize=authorize)
    repository = WorktreeManager(workspace)._child_repository()
    return observe_snapshot(workspace, object_root=repository, excluded=_excluded(store), authorize=authorize)


def _retain(store: RuntimeStore, turn_id: str, value: dict[str, Any], name: str) -> dict[str, Any]:
    manifest = artifact_ops.commit_artifact(
        store, turn_id=turn_id, content=json.dumps(value).encode(), name=f"{name}.json", media_type="application/json"
    )
    archive = artifact_ops.commit_artifact(
        store,
        turn_id=turn_id,
        content=archive_tree(Path(value["repository"]), value["tree"]),
        name=f"{name}.tar",
        media_type="application/x-tar",
    )
    # Entries are required for cheap current-version checks. Exact index bytes
    # and raw source bytes live in immutable artifacts, not model context.
    return {k: v for k, v in value.items() if k not in {"source_index_hex", "source_index_patch_hex"}} | {
        "manifest_artifact_id": manifest.artifact_id,
        "archive_artifact_id": archive.artifact_id,
    }


def _restore_objects(store: RuntimeStore, version: Mapping[str, Any], repository: Path) -> None:
    """Git GC cannot erase canonical provenance: restore raw objects from blobs."""
    identity = version["archive_artifact_id"]
    errors = store.tx.artifact_integrity_errors(artifact_ids={identity})
    if errors:
        raise RuntimeError(errors[0])
    with tarfile.open(store.artifact_blob_path(identity), "r:") as archive:
        for path, entry in version["entries"].items():
            member = archive.getmember(path)
            if entry["mode"] == "120000":
                content = os.fsencode(member.linkname)
            else:
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"missing archived input: {path}")
                with stream:
                    content = stream.read()
            oid = git_bytes(repository, "hash-object", "-w", "--stdin", input=content).decode().strip()
            if oid != entry["oid"]:
                raise RuntimeError(f"archived input identity mismatch: {path}")
    if tree_from_entries(repository, dict(version["entries"])) != version["tree"]:
        raise RuntimeError("archived tree identity mismatch")


@owner_operation
def begin(store: RuntimeStore, turn_id: str) -> dict[str, Any]:
    with store.tx.transaction():
        existing = _latest(store, turn_id, "delivery_baseline")
        if existing is not None:
            return existing
        try:
            value = _retain(store, turn_id, _capture(store, turn_id), "delivery-baseline")
        except (OSError, ValueError, RuntimeError) as error:
            value = {"error": str(error)}
        value["revision"] = REVISION
        _append(store, turn_id, "delivery_baseline", value)
    return value


@owner_operation
def plan(
    store: RuntimeStore, turn_id: str, *, scope: Sequence[str], commands: Sequence[str], omissions: Sequence[str] = ()
) -> dict[str, Any]:
    if not scope or any(not isinstance(s, str) or not s.strip() for s in (*scope, *commands, *omissions)):
        raise ValueError("delivery scope, commands and omission reasons must be non-empty strings")
    with store.tx.transaction():
        begin(store, turn_id)
        value = {
            "revision": REVISION,
            "input_revision": store.user_input_revision(turn_id),
            "scope": list(scope),
            "commands": list(commands),
            "omissions": list(omissions),
        }
        if _latest(store, turn_id, "delivery_plan") != value:
            _append(store, turn_id, "delivery_plan", value)
    return value


def _command_evidence(store: RuntimeStore, turn_id: str, ids: Sequence[str]) -> list[dict[str, Any]]:
    trusted = {
        operation.tool_call_id: (item, operation, arguments)
        for item, operation, arguments in trusted_command_results(store, turn_id)
    }
    selected = list(ids)
    declared = _latest(store, turn_id, "delivery_plan") or {}
    commands = set(declared.get("commands", ()))
    policy_command = store.read_turn(turn_id).binding_manifest.get("completion_policy", {}).get("acceptance_command")
    if policy_command:
        commands.add(policy_command)
    commands.update(trusted[identity][2].get("command") for identity in ids if identity in trusted)
    latest: dict[tuple[Any, Any], str] = {}
    for identity, (item, _operation, args) in trusted.items():
        actual_command = args.get("command")
        matches_explicit = (
            isinstance(actual_command, str) and policy_command and actual_command.strip() == policy_command.strip()
        )
        if actual_command in commands or matches_explicit:
            key = (args.get("command"), args.get("working_dir", "."))
            if key not in latest or item.sequence > trusted[latest[key]][0].sequence:
                latest[key] = identity
    for index, identity in enumerate(selected):
        if identity in trusted:
            args = trusted[identity][2]
            selected[index] = latest.get((args.get("command"), args.get("working_dir", ".")), identity)
    selected = list(dict.fromkeys([*selected, *latest.values()]))
    result = []
    verifications = {
        i.payload.get("operation_id"): i.item_id
        for i in store.list_items(turn_id)
        if i.kind == "verification" and i.producer == "runtime" and i.status == "completed"
    }
    for identity in selected:
        if identity not in trusted:
            raise ValueError(f"unknown trusted command source: {identity}")
        item, operation, args = trusted[identity]
        raw_output = item.payload.get("structured_content")
        output = raw_output if isinstance(raw_output, Mapping) else {}
        receipt = output.get("verification_receipt")
        # Only the product command implementation can attest a frozen input.
        # Foreign tools named run_command retain raw evidence but cannot attest.
        genuine = operation.execution_revision == "builtin-run-command-v13-versioned-verification"
        result.append(
            {
                "tool_call_id": identity,
                "operation_id": operation.operation_id,
                "verification_item_id": verifications.get(operation.operation_id),
                "source_result_item_id": item.item_id,
                "sequence": item.sequence,
                "command": args.get("command"),
                "working_dir": args.get("working_dir", "."),
                "operation_status": operation.status,
                "exit_code": output.get("exit_code"),
                "is_error": item.payload.get("is_error"),
                "status": output.get("status"),
                "timed_out": output.get("timed_out", False),
                "sandbox_error": output.get("sandbox_error"),
                "receipt": dict(receipt) if genuine and isinstance(receipt, Mapping) else None,
            }
        )
    return result


@owner_operation
def record(
    store: RuntimeStore,
    turn_id: str,
    *,
    tool_call_ids: Sequence[str] = (),
    source_item_ids: Sequence[str] = (),
    blockers: Sequence[str] = (),
    review: str = "",
    answer: str | None = None,
) -> dict[str, Any]:
    with store.tx.transaction():
        turn_ops._running_turn_thread_id(store, turn_id)
        baseline = begin(store, turn_id)
        current_plan = _latest(store, turn_id, "delivery_plan")
        items = {i.item_id: i for i in store.list_items(turn_id)}
        sources = []
        for identity in source_item_ids:
            item = items.get(identity) or next(
                (i for i in items.values() if i.kind == "tool_result" and i.payload.get("tool_call_id") == identity),
                None,
            )
            if item is None or item.producer != "tool" or item.kind != "tool_result" or item.payload.get("is_error"):
                raise ValueError(f"invalid explanation source: {identity}")
            if not any(
                op.result_item_id == item.item_id
                and op.status == "succeeded"
                and set(op.effects) & {"read_workspace", "network", "execute_process"}
                and not set(op.effects) & {"write_workspace", "destructive"}
                and op.tool_name not in {"record_delivery", "update_plan"}
                for op in store.list_tool_operations(turn_id)
            ):
                raise ValueError(f"untrusted explanation source: {identity}")
            sources.append(item.item_id)
        checks = _command_evidence(store, turn_id, tool_call_ids)
        errors = list(blockers)
        final = None
        patch = b""
        changed = []
        try:
            captured = _capture(store, turn_id)
            if "tree" in baseline:
                repository = Path(captured["repository"])
                arguments = (
                    "diff",
                    "--binary",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--find-renames",
                    baseline["tree"],
                    captured["tree"],
                )
                try:
                    patch = git_bytes(repository, *arguments)
                except RuntimeError:
                    _restore_objects(store, baseline, repository)
                    patch = git_bytes(repository, *arguments)
                changed = sorted(
                    p
                    for p in baseline["entries"].keys() | captured["entries"].keys()
                    if baseline["entries"].get(p) != captured["entries"].get(p)
                )
            final = captured
        except (OSError, ValueError, RuntimeError) as error:
            errors.append(str(error))
        if baseline.get("error"):
            errors.append("Initial workspace version unavailable: " + baseline["error"])
        if final is not None and current_plan is not None:
            from agent_runtime.harness.tool_ops import record_verification

            for check in checks:
                if (
                    command_succeeded(operation_status=check["operation_status"],
                                      is_error=check["is_error"], output=check)
                    and receipt_issue(check["receipt"], final, permissions=_effective_permissions(store, turn_id))[0]
                    is None
                ):
                    record_verification(
                        store,
                        turn_id=turn_id,
                        operation_id=check["operation_id"],
                        kind="inspection",
                        verifier="declared-delivery-scope",
                        verified_resources=tuple(current_plan["scope"]),
                    )
            checks = _command_evidence(store, turn_id, tool_call_ids)
        children = [dict(i.payload) for i in items.values() if i.kind == "child_workspace"]
        value = {
            "revision": REVISION,
            "input_revision": store.user_input_revision(turn_id),
            "plan": current_plan,
            "baseline": baseline,
            "final": final,
            "checks": checks,
            "source_item_ids": sources,
            "blockers": errors,
            "review": review,
            "answer_sha256": None if answer is None else hashlib.sha256(answer.encode()).hexdigest(),
            "changed_paths": changed,
            "children": children,
        }
        key = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        existing = _latest(store, turn_id, "delivery_record")
        if existing is not None and existing.get("identity_digest") == key:
            return inspect(store, turn_id)
        if final is not None:
            value["final"] = _retain(store, turn_id, final, "delivery-final")
        diff = artifact_ops.commit_artifact(
            store, turn_id=turn_id, content=patch, name="delivery.diff", media_type="text/x-diff"
        )
        value.update(identity_digest=key, diff_artifact_id=diff.artifact_id)
        value["delivery_id"] = _append(store, turn_id, "delivery_record", value)
        # The Item identity itself is the canonical delivery identity.
    return _assess(store, turn_id, value)


def receipt_issue(
    receipt: Mapping[str, Any] | None,
    final: Mapping[str, Any] | None,
    *,
    permissions: Any = None,
) -> tuple[str | None, bool]:
    """One freshness check shared by explicit stopping and delivery projection."""
    import platform

    from agent_runtime.workspace_snapshot import dependency_manifest

    if (
        not receipt
        or receipt.get("revision") != "versioned-command-v1"
        or receipt.get("input_tree") != (final or {}).get("tree")
        or receipt.get("source_head") != (final or {}).get("source_head")
        or receipt.get("source_index_sha256") != (final or {}).get("source_index_sha256")
        or receipt.get("inputs_unchanged") is not True
    ):
        return "Command lacks current stable input evidence; rerun verification", False
    if receipt.get("dependencies_declared") is not True:
        return "Declare the external dependency boundary and rerun", False
    if receipt.get("executable_covered") is not True:
        return "Declare the selected external executable as a dependency and rerun", False

    def authorize_dependency(path: Path) -> None:
        if permissions is not None and (
            not permissions.path_access(path) or not permissions.path_access(path.resolve())
        ):
            raise PermissionError(f"verification dependency authority was revoked: {path}")

    try:
        for path in receipt.get("dependency_paths", ()):
            if permissions is not None and (
                not permissions.path_access(path) or not permissions.path_access(Path(path).resolve())
            ):
                return "Verification dependency authority was revoked", True
        if (
            receipt.get("dependencies_unchanged") is not True
            or dependency_manifest(
                tuple(Path(p) for p in receipt.get("dependency_paths", ())), authorize=authorize_dependency
            )
            != receipt.get("dependencies")
            or platform.platform() != receipt.get("platform")
            or receipt.get("toolchain_unchanged") is not True
            or dependency_manifest(tuple(Path(p) for p in receipt.get("toolchain_paths", ())))
            != receipt.get("toolchain")
        ):
            return "Verification environment changed; rerun", False
    except (OSError, ValueError, RuntimeError) as error:
        return f"Verification environment unavailable: {error}", True
    return None, False


def _current_workspace_command(
    store: RuntimeStore, turn_id: str, check: Mapping[str, Any], live: Mapping[str, Any] | None,
) -> str | None:
    receipt = check.get("receipt")
    root = Path(store.read_thread(store.read_turn(turn_id).thread_id).workspace).resolve()
    if (
        not receipt
        or receipt.get("mode") != "workspace"
        or Path(receipt.get("cwd", "")).resolve() != root
        or receipt.get("command") != check["command"]
    ):
        return (
            "Run the exact acceptance command with verification_mode=workspace and explicit verification_dependencies."
        )
    if live is None:
        return "Cannot inspect current acceptance inputs."
    return receipt_issue(receipt, live, permissions=_effective_permissions(store, turn_id))[0]


def _assess(
    store: RuntimeStore, turn_id: str, value: dict[str, Any], *, include_live_evidence: bool = False,
) -> dict[str, Any]:
    problems = list(value.get("blockers", ()))
    reasons = []
    final = value.get("final")
    current_plan = value.get("plan")
    stale = False
    live = None
    try:
        live = _capture(store, turn_id, retain_objects=False)
        stale = final is None or live["tree"] != final["tree"]
    except (OSError, ValueError, RuntimeError) as error:
        problems.append(str(error))
    if value["input_revision"] != store.user_input_revision(turn_id):
        stale = True
        reasons.append("Requirements changed; record a new scope and verification plan.")
    if stale:
        reasons.append("Workspace version changed; rerun verification and record the new version.")
    if current_plan != _latest(store, turn_id, "delivery_plan"):
        stale = True
        reasons.append("Verification scope or plan changed; record evidence for the current plan.")
    version_keys = ("tree", "source_head", "source_index_sha256")
    live_version = None if live is None else {key: live.get(key) for key in version_keys}
    same_live_version = live_version is not None and live_version == {
        key: (final or {}).get(key) for key in version_keys
    }
    referenced_artifacts = {value["diff_artifact_id"]}
    for version in (value.get("baseline"), final):
        if version:
            referenced_artifacts.update(
                version[k] for k in ("archive_artifact_id", "manifest_artifact_id") if k in version
            )
    problems.extend(store.tx.artifact_integrity_errors(artifact_ids=referenced_artifacts))
    if not current_plan or current_plan["input_revision"] != value["input_revision"]:
        reasons.append("Record scope, exact verification commands and reasons for omissions.")
    if not value.get("review"):
        reasons.append("Record the final review conclusion and its coverage.")
    checks = _command_evidence(store, turn_id, [c["tool_call_id"] for c in value.get("checks", ())])
    valid_commands = set()
    receipt_evidence = []
    for check in checks:
        if command_succeeded(operation_status=check["operation_status"], is_error=check["is_error"], output=check):
            issue, blocked = receipt_issue(
                check.get("receipt"), final, permissions=_effective_permissions(store, turn_id),
            )
            if issue:
                (problems if blocked else reasons).append(f"{issue}: {check['command']}")
                receipt_status = "unavailable" if blocked else "unverified"
            else:
                valid_commands.add(check["command"])
                if live_version is None:
                    receipt_status = "unavailable"
                else:
                    receipt_status = "current" if not stale and same_live_version else "stale"
        else:
            problems.append(f"Command failed or unresolved: {check['command']}")
            receipt_status = "command_failed"
        if include_live_evidence:
            receipt_evidence.append({
                **{key: check[key] for key in (
                    "command", "working_dir", "source_result_item_id", "verification_item_id",
                )},
                "status": receipt_status,
            })
    explicit = store.read_turn(turn_id).binding_manifest.get("completion_policy", {}).get("acceptance_command")
    if explicit:
        candidates = [c for c in checks if isinstance(c["command"], str) and c["command"].strip() == explicit.strip()]
        if not candidates:
            reasons.append("Run and record the exact configured acceptance command in the current workspace.")
        else:
            candidate = max(candidates, key=lambda c: c["sequence"])
            issue = _current_workspace_command(store, turn_id, candidate, live)
            if issue:
                reasons.append(issue)
    if current_plan:
        missing = set(current_plan["commands"]) - valid_commands
        if missing:
            reasons.append("Run and record exact planned commands: " + "; ".join(sorted(missing)))
        if not current_plan["commands"] and not (
            value.get("source_item_ids")
            and value.get("review")
            and value.get("answer_sha256")
            and not value.get("changed_paths")
            and current_plan["omissions"]
        ):
            reasons.append("Record source evidence and answer coverage, or run applicable verification.")
    if value.get("answer_sha256"):
        answer_item = (
            store.read_turn_answer(turn_id)
            if store.read_turn(turn_id).status == "completed"
            else next((i for i in reversed(store.list_items(turn_id)) if i.kind == "final_proposal"), None)
        )
        if answer_item is not None and (
            hashlib.sha256(item_text(answer_item).encode()).hexdigest() != value["answer_sha256"]
        ):
            reasons.append("Final answer differs from the reviewed answer coverage; record a fresh review.")
    unresolved = [
        op.operation_id
        for op in store.list_tool_operations(turn_id)
        if op.status == "unknown" or op.status == "running" and op.tool_name != "record_delivery"
    ]
    unresolved += [op.operation_id for op in store.list_model_operations(turn_id) if op.status == "unknown"]
    if unresolved:
        problems.append("Reconcile unresolved operations: " + ", ".join(unresolved))
    from agent_runtime.budget import ResourceUsage

    budget = store.read_budget_state(turn_id)
    if budget.uncertain != ResourceUsage() or budget.child_reserved != ResourceUsage():
        problems.append("Reconcile uncertain usage and settle attached child budgets.")
    from agent_runtime.harness.child_workspace_ops import unresolved as unresolved_children

    try:
        if unresolved_children(store, turn_id):
            problems.append("Integrate and verify, settle, or explicitly discard child artifacts.")
    except (OSError, ValueError, RuntimeError) as error:
        problems.append(str(error))
    status = "blocked" if problems else "unverified" if reasons else "verified"
    return {
        **value,
        "checks": checks,
        "status": status,
        "freshness": "stale" if stale else "current",
        "verified_scope": current_plan["scope"] if status == "verified" and current_plan else [],
        "next_actions": list(dict.fromkeys([*problems, *reasons])),
        "unknown_operations": unresolved,
        **({
            "live_workspace": live_version,
            "receipt_evidence": receipt_evidence,
        } if include_live_evidence else {}),
        "boundary": (
            "Evidence covers declared scope and dependencies in the recorded environment; "
            "semantic correctness and dependency completeness remain review judgments."
        ),
    }


@owner_operation
def inspect(store: RuntimeStore, turn_id: str, *, include_live_evidence: bool = False) -> dict[str, Any]:
    value = _latest(store, turn_id, "delivery_record")
    if value is None:
        unknown = [op.operation_id for op in store.list_tool_operations(turn_id) if op.status == "unknown"]
        unknown += [op.operation_id for op in store.list_model_operations(turn_id) if op.status == "unknown"]
        pending = [i.request_id for i in store.list_interactions(turn_id) if i.status == "pending"]
        actions = []
        if unknown:
            actions.append("Reconcile unknown operations: " + ", ".join(unknown))
        if pending:
            actions.append("Resolve pending interactions: " + ", ".join(pending))
        return {
            "status": "blocked" if unknown or pending else "unverified",
            "verified_scope": [],
            "unknown_operations": unknown,
            "next_actions": actions or ["Record delivery scope and evidence."],
        }
    value["delivery_id"] = next(i.item_id for i in reversed(store.list_items(turn_id)) if i.kind == "delivery_record")
    return _assess(store, turn_id, value, include_live_evidence=include_live_evidence)


def submitted_answer(store: RuntimeStore, turn_id: str, response_id: str) -> tuple[str, str] | None:
    """Resolve an explicit submission from a settled batch and its canonical evidence."""
    response = store.read_item(response_id)
    if response.turn_id != turn_id or response.kind != "model_response" or response.status != "completed":
        return None
    submissions = [
        call
        for call in restore_model_response(response.payload).tool_calls
        if call.name == "record_delivery" and call.arguments.get("action") == "submit"
    ]
    if len(submissions) != 1:
        return None
    call = submissions[0]
    answer = call.arguments.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None
    operation = next((op for op in store.list_tool_operations(turn_id) if op.tool_call_id == call.id), None)
    if (
        operation is None
        or operation.execution_revision != _TOOL_REVISION
        or operation.status != "succeeded"
        or operation.result_item_id is None
    ):
        return None
    result = store.read_item(operation.result_item_id)
    output = result.payload.get("structured_content")
    record = next((i for i in reversed(store.list_items(turn_id)) if i.kind == "delivery_record"), None)
    if (
        result.kind != "tool_result"
        or result.producer != "tool"
        or result.payload.get("is_error") is not False
        or not isinstance(output, Mapping)
        or record is None
        or output.get("delivery_id") != record.item_id
        or record.payload.get("input_revision") != store.user_input_revision(turn_id)
        or record.payload.get("answer_sha256") != hashlib.sha256(answer.encode()).hexdigest()
    ):
        return None
    return answer, record.item_id


class DeliveryManager:
    """CLI/model and SDK share these owner-serialized delivery operations."""

    def __init__(self, store: RuntimeStore, thread_id: Callable[[], str]) -> None:
        self._store = store
        self._thread_id = thread_id

    async def _check(self, turn_id: str) -> None:
        turn = await self._store.run_async(self._store.read_turn, turn_id)
        if turn.thread_id != self._thread_id():
            raise ValueError("delivery belongs to a different Session Thread")

    async def plan(self, turn_id: str, **values: Any) -> dict[str, Any]:
        await self._check(turn_id)
        return await self._store.run_async(plan, self._store, turn_id, **values)

    async def record(self, turn_id: str, **values: Any) -> dict[str, Any]:
        await self._check(turn_id)
        return await self._store.run_async(record, self._store, turn_id, **values)

    async def inspect(self, turn_id: str) -> dict[str, Any]:
        await self._check(turn_id)
        return await self._store.run_async(inspect, self._store, turn_id)


def create_delivery_tool(manager: DeliveryManager) -> Any:
    from typing import Literal

    from pydantic import BaseModel, ConfigDict, Field

    from agent_runtime.harness.tool_orchestrator import current_tool_turn_id
    from agent_runtime.tools.tool import (
        CancellationMode,
        InterruptBehavior,
        NormalizedToolOutput,
        ResolvedToolUse,
        Tool,
        ToolDefinition,
        ToolEffect,
        pydantic_input,
    )

    class DeliveryInput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        action: Literal["plan", "record", "submit", "inspect"]
        scope: list[str] = Field(default_factory=list)
        commands: list[str] = Field(default_factory=list)
        omissions: list[str] = Field(default_factory=list)
        tool_call_ids: list[str] = Field(default_factory=list)
        source_item_ids: list[str] = Field(default_factory=list)
        blockers: list[str] = Field(default_factory=list)
        review: str = ""
        answer: str | None = None

    schema, validate = pydantic_input(DeliveryInput)

    async def run(arguments: Mapping[str, Any]) -> dict[str, Any]:
        payload = DeliveryInput.model_validate(arguments)
        turn_id = current_tool_turn_id()
        if payload.action == "plan":
            return await manager.plan(
                turn_id, scope=payload.scope, commands=payload.commands, omissions=payload.omissions
            )
        if payload.action in {"record", "submit"}:
            if payload.action == "submit" and (not payload.answer or not payload.answer.strip()):
                raise ValueError("submit requires the exact non-empty reviewed final answer")
            return await manager.record(
                turn_id,
                tool_call_ids=payload.tool_call_ids,
                source_item_ids=payload.source_item_ids,
                blockers=payload.blockers,
                review=payload.review,
                answer=payload.answer,
            )
        return await manager.inspect(turn_id)

    def normalize(raw: Any) -> Any:
        # Full manifests stay inspectable in canonical facts/artifacts. Model
        # feedback carries the concrete IDs, scope and reasons without blobs.
        summary = {k: v for k, v in raw.items() if k not in {"baseline", "final", "children", "checks"}}
        return NormalizedToolOutput(structured_content=summary)

    return Tool(
        definition=ToolDefinition(
            name="record_delivery",
            input_schema=schema,
            description=(
                "Record versioned delivery evidence separately from ending a Turn. plan declares scope, exact "
                "commands and omission reasons; run those via run_command verification_mode=snapshot. When ready "
                "to finish, call submit once with the exact final answer in answer, tool_call_ids and review. "
                "The runtime uses those exact words for the final proposal and applies normal completion checks. "
                "Use record to save evidence while work continues. For unchanged explanation tasks, include trusted "
                "source tool result Item IDs (or call IDs), review answer coverage, and explain omitted commands. "
                "If you later write a different final answer, its review becomes invalid. "
                "Missing evidence is unverified; failures/unknown are blocked. inspect rechecks freshness. "
                "This records declared coverage, not automatic semantic correctness or permission."
            ),
        ),
        validate_input=validate,
        run=run,
        normalize_output=normalize,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda _: ResolvedToolUse(effects=frozenset({ToolEffect.READ_WORKSPACE}), targets=()),
        execution_revision=_TOOL_REVISION,
        idempotent=True,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.FINISH_CURRENT,
        timeout_seconds=120,
        max_model_output_bytes=50_000,
        output_schema={"type": "object"},
    )
