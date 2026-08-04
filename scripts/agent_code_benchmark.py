from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fnmatch import fnmatch
from pathlib import Path
from typing import TextIO
from uuid import uuid4

_PRIMARY_MODEL = "qwen3_5_9b_mlx_4bit"
_CONTROL_MODEL = "groq_gpt_oss_120b"
_DIAGNOSTIC_MODEL = "kimi_cloud"
_CLOUD_BENCHMARK_MODELS = frozenset({_CONTROL_MODEL, _DIAGNOSTIC_MODEL})
_IMPLEMENTATION_INSTRUCTION_PREFIX = (
    "This is an implementation task in the current repository. "
    "Modify the code and run focused tests. "
)
_ARCHITECTURE_LAYERS = frozenset(
    {
        "public_api_cli",
        "service",
        "loop",
        "tool",
        "turn_checkpoint",
        "result_events",
    }
)
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_PROCESS_CLEANUP_GRACE_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    task_id: str
    split: str
    category: str
    layers: tuple[str, ...]
    control: bool
    source_commit: str
    target_commit: str
    instruction: str
    acceptance_command: tuple[str, ...]
    acceptance_files: tuple[str, ...]
    setup_command: tuple[str, ...]
    allowed_paths: tuple[str, ...]
    permissions: TaskPermissions
    budget: TaskBudget


@dataclass(frozen=True, slots=True)
class TaskPermissions:
    write_workspace: bool
    execute_process: bool
    network: bool


@dataclass(frozen=True, slots=True)
class TaskBudget:
    max_turns: int
    timeout_seconds: int


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    schema_version: int
    benchmark_version: str
    primary_model: str
    control_model: str
    diagnostic_model: str
    tasks: tuple[BenchmarkTask, ...]
    fingerprint: str
    content_fingerprint: str


class RunOutcome(StrEnum):
    PASSED = "passed"
    VALIDATION_FAILED = "validation_failed"
    FALSE_COMPLETION = "false_completion"
    BOUNDED_FAILURE = "bounded_failure"
    RUNTIME_FAILED = "runtime_failed"
    SAFETY_VIOLATION = "safety_violation"
    PROVIDER_LIMITED = "provider_limited"
    BENCHMARK_INVALID = "benchmark_invalid"


class DiagnosisCause(StrEnum):
    MODEL_CAPABILITY_GAP = "model_capability_gap"
    TOOL_CONTRACT = "tool_contract"
    CONTEXT_ASSEMBLY = "context_assembly"
    EXECUTION_CLOSURE = "execution_closure"
    PROVIDER_ADAPTER = "provider_adapter"
    PROVIDER_LIMIT = "provider_limit"
    SAFETY_BOUNDARY = "safety_boundary"
    BENCHMARK_ENVIRONMENT = "benchmark_environment"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Diagnosis:
    primary: DiagnosisCause
    secondary: tuple[DiagnosisCause, ...]
    confidence: float
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.primary, DiagnosisCause):
            raise TypeError("diagnosis primary must be a DiagnosisCause")
        if any(not isinstance(cause, DiagnosisCause) for cause in self.secondary):
            raise TypeError("diagnosis secondary values must be DiagnosisCause values")
        if self.primary in self.secondary or len(self.secondary) != len(set(self.secondary)):
            raise ValueError("diagnosis causes must be unique")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("diagnosis confidence must be between 0 and 1")
        if any(not isinstance(item, str) or not item.strip() for item in self.evidence):
            raise ValueError("diagnosis evidence must contain non-empty strings")


@dataclass(frozen=True, slots=True)
class RunFacts:
    model_alias: str
    turn_status: str
    valid_diff: bool
    hidden_acceptance_passed: bool
    stop_reason: str | None = None
    safety_violations: tuple[str, ...] = ()
    provider_error_code: str | None = None
    runtime_error: str | None = None
    benchmark_invalid_reason: str | None = None


@dataclass(frozen=True, slots=True)
class TaskRunRecord:
    task_id: str
    model_alias: str
    max_tokens_total: int | None
    runtime_fingerprint: str
    outcome: RunOutcome
    turn_id: str | None
    turn_status: str
    hidden_acceptance_passed: bool
    diff: str
    safety_violations: tuple[str, ...]
    diagnosis: Diagnosis
    artifact_dir: Path


@dataclass(frozen=True, slots=True)
class _CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    leftover_processes: bool = False


def load_manifest(path: Path) -> BenchmarkManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark manifest must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError("benchmark schema_version must be 1")
    benchmark_version = _non_empty_string(
        payload.get("benchmark_version"),
        field="benchmark_version",
    )
    models = _mapping(payload.get("models"), field="models")
    primary_model = _non_empty_string(models.get("primary"), field="models.primary")
    control_model = _non_empty_string(models.get("control"), field="models.control")
    diagnostic_model = _non_empty_string(
        models.get("diagnostic"),
        field="models.diagnostic",
    )
    if (
        primary_model != _PRIMARY_MODEL
        or control_model != _CONTROL_MODEL
        or diagnostic_model != _DIAGNOSTIC_MODEL
    ):
        raise ValueError(
            "benchmark model policy requires local Qwen primary, Groq control, "
            "and Kimi diagnostic"
        )

    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, (str, bytes)):
        raise ValueError("tasks must be a JSON array")
    tasks = tuple(_parse_task(item) for item in raw_tasks)
    if not tasks:
        raise ValueError("benchmark manifest must contain at least one task")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("benchmark task IDs must be unique")

    canonical = _canonical_json(payload)
    content_payload = {
        "schema_version": schema_version,
        "models": models,
        "tasks": raw_tasks,
    }
    return BenchmarkManifest(
        schema_version=schema_version,
        benchmark_version=benchmark_version,
        primary_model=primary_model,
        control_model=control_model,
        diagnostic_model=diagnostic_model,
        tasks=tasks,
        fingerprint=_sha256(canonical),
        content_fingerprint=_sha256(_canonical_json(content_payload)),
    )


def validate_manifest_evolution(
    previous: BenchmarkManifest,
    current: BenchmarkManifest,
) -> None:
    if (
        previous.benchmark_version == current.benchmark_version
        and previous.content_fingerprint != current.content_fingerprint
    ):
        raise ValueError(
            "benchmark content changed without a new benchmark_version"
        )


def validate_release_shape(manifest: BenchmarkManifest) -> None:
    regression = tuple(task for task in manifest.tasks if task.split == "regression")
    holdout = tuple(task for task in manifest.tasks if task.split == "holdout")
    if len(regression) < 25:
        raise ValueError("release benchmark requires at least 25 regression tasks")
    if len(holdout) < 5:
        raise ValueError("release benchmark requires at least 5 holdout tasks")
    cross_layer = tuple(
        task for task in manifest.tasks if task.category == "cross_layer"
    )
    if len(cross_layer) < 10:
        raise ValueError("release benchmark requires at least 10 cross-layer tasks")
    if sum(task.category == "cross_layer" for task in holdout) < 2:
        raise ValueError("release benchmark requires at least 2 cross-layer holdout tasks")
    if any(task.control for task in holdout):
        raise ValueError("holdout tasks cannot be controls")
    if any(
        not task.instruction.startswith(_IMPLEMENTATION_INSTRUCTION_PREFIX)
        for task in manifest.tasks
    ):
        raise ValueError(
            "release benchmark instructions must identify an implementation task"
        )

    controls = tuple(task for task in regression if task.control)
    if len(controls) != 6:
        raise ValueError("release benchmark requires exactly 6 regression controls")
    control_categories = {
        category: sum(task.category == category for task in controls)
        for category in ("local", "medium", "cross_layer")
    }
    if set(control_categories.values()) != {2}:
        raise ValueError(
            "release benchmark controls require 2 local, 2 medium, and 2 cross-layer tasks"
        )


def evaluate_release_results(
    manifest: BenchmarkManifest,
    results: Sequence[Mapping[str, object]],
    *,
    expected_runtime_fingerprint: str | None = None,
) -> dict[str, object]:
    validate_release_shape(manifest)
    if expected_runtime_fingerprint is not None:
        expected_runtime_fingerprint = _sha256_fingerprint(
            expected_runtime_fingerprint,
            field="expected_runtime_fingerprint",
        )
    tasks_by_id = {task.task_id: task for task in manifest.tasks}
    expected_primary = tuple(
        (task.task_id, manifest.primary_model)
        for task in manifest.tasks
    )
    expected_control = tuple(
        (task.task_id, manifest.control_model)
        for task in manifest.tasks
        if task.control
    )
    expected = {*expected_primary, *expected_control}
    indexed: dict[
        tuple[str, str],
        tuple[RunOutcome, Mapping[str, object]],
    ] = {}
    runtime_fingerprints: set[str] = set()
    for raw in results:
        payload = _mapping(raw, field="result")
        if payload.get("schema_version") != 1:
            raise ValueError("result schema_version must be 1")
        if payload.get("benchmark_version") != manifest.benchmark_version:
            raise ValueError("result benchmark_version does not match manifest")
        if payload.get("manifest_fingerprint") != manifest.fingerprint:
            raise ValueError("result manifest_fingerprint does not match manifest")
        task_id = _non_empty_string(
            payload.get("task_id"),
            field="result.task_id",
        )
        task = tasks_by_id.get(task_id)
        if task is None:
            raise ValueError(f"result references unknown task: {task_id}")
        if (
            payload.get("task_source_commit") != task.source_commit
            or payload.get("task_target_commit") != task.target_commit
        ):
            raise ValueError(f"result task commits do not match manifest: {task_id}")
        runtime_fingerprints.add(
            _sha256_fingerprint(
                payload.get("runtime_fingerprint"),
                field="result.runtime_fingerprint",
            )
        )
        model_alias = _non_empty_string(
            payload.get("model_alias"),
            field="result.model_alias",
        )
        key = (task_id, model_alias)
        if key not in expected:
            raise ValueError(
                f"result is outside required release lanes: {task_id}:{model_alias}"
            )
        if key in indexed:
            raise ValueError(
                f"duplicate result for {task_id}:{model_alias}"
            )
        try:
            outcome = RunOutcome(
                _non_empty_string(
                    payload.get("outcome"),
                    field="result.outcome",
                )
            )
        except ValueError as exc:
            raise ValueError(
                f"result outcome is unsupported: {payload.get('outcome')}"
            ) from exc
        indexed[key] = (outcome, payload)

    reasons: list[str] = []
    if len(runtime_fingerprints) > 1:
        reasons.append("mixed_runtime_fingerprints")
    if (
        expected_runtime_fingerprint is not None
        and runtime_fingerprints != {expected_runtime_fingerprint}
    ):
        reasons.append("runtime_fingerprint_mismatch")
    for task_id, model_alias in (*expected_primary, *expected_control):
        if (task_id, model_alias) not in indexed:
            reasons.append(f"missing_result:{task_id}:{model_alias}")

    false_completions = 0
    safety_violations = 0
    for (task_id, model_alias), (outcome, payload) in indexed.items():
        if outcome is RunOutcome.FALSE_COMPLETION:
            false_completions += 1
        raw_safety = payload.get("safety_violations", ())
        has_safety_evidence = (
            isinstance(raw_safety, Sequence)
            and not isinstance(raw_safety, (str, bytes))
            and bool(raw_safety)
        )
        if outcome is RunOutcome.SAFETY_VIOLATION or has_safety_evidence:
            safety_violations += 1
        if outcome is RunOutcome.PASSED:
            continue
        diagnosis = payload.get("diagnosis")
        diagnosis_payload = (
            diagnosis if isinstance(diagnosis, Mapping) else {}
        )
        primary = diagnosis_payload.get("primary")
        evidence = diagnosis_payload.get("evidence")
        has_evidence = (
            isinstance(evidence, Sequence)
            and not isinstance(evidence, (str, bytes))
            and bool(evidence)
            and all(
                isinstance(item, str) and bool(item.strip())
                for item in evidence
            )
        )
        if primary in {None, "", DiagnosisCause.UNKNOWN.value} or not has_evidence:
            reasons.append(f"unknown_diagnosis:{task_id}:{model_alias}")
        if (
            model_alias == manifest.control_model
            and outcome is RunOutcome.PROVIDER_LIMITED
        ):
            reasons.append(f"control_provider_limited:{task_id}")

    primary_outcomes = {
        task_id: indexed.get((task_id, manifest.primary_model))
        for task_id, _model_alias in expected_primary
    }
    primary_passed = sum(
        value is not None and value[0] is RunOutcome.PASSED
        for value in primary_outcomes.values()
    )
    cross_tasks = tuple(
        task for task in manifest.tasks if task.category == "cross_layer"
    )
    cross_passed = sum(
        (
            value := primary_outcomes.get(task.task_id)
        ) is not None
        and value[0] is RunOutcome.PASSED
        for task in cross_tasks
    )
    overall_rate = primary_passed / len(manifest.tasks)
    cross_rate = cross_passed / len(cross_tasks)
    if overall_rate < 0.70:
        reasons.append("overall_first_pass_rate_below_0.70")
    if cross_rate < 0.50:
        reasons.append("cross_layer_first_pass_rate_below_0.50")
    if false_completions:
        reasons.append("false_completion_blocker")
    if safety_violations:
        reasons.append("safety_violation_blocker")

    reasons = list(dict.fromkeys(reasons))
    return {
        "benchmark_version": manifest.benchmark_version,
        "manifest_fingerprint": manifest.fingerprint,
        "runtime_fingerprint": (
            next(iter(runtime_fingerprints))
            if len(runtime_fingerprints) == 1
            else None
        ),
        "expected_runtime_fingerprint": expected_runtime_fingerprint,
        "release_ready": not reasons,
        "overall_first_pass_rate": overall_rate,
        "cross_layer_first_pass_rate": cross_rate,
        "primary_result_count": sum(
            key in indexed for key in expected_primary
        ),
        "control_result_count": sum(
            key in indexed for key in expected_control
        ),
        "false_completion_count": false_completions,
        "safety_violation_count": safety_violations,
        "reasons": reasons,
    }


def load_result_record(
    path: Path,
    *,
    manifest: BenchmarkManifest,
) -> Mapping[str, object]:
    result_path = path.expanduser().resolve()
    payload = _mapping(
        json.loads(result_path.read_text(encoding="utf-8")),
        field="result",
    )
    if payload.get("benchmark_version") != manifest.benchmark_version:
        raise ValueError(
            f"result benchmark_version does not match manifest: {result_path}"
        )
    if payload.get("manifest_fingerprint") != manifest.fingerprint:
        raise ValueError(
            f"result manifest_fingerprint does not match manifest: {result_path}"
        )
    evidence = _mapping(payload.get("evidence"), field="result.evidence")
    for field in (
        "agent_stdout",
        "agent_stderr",
        "agent_diff",
        "acceptance_stdout",
        "acceptance_stderr",
    ):
        relative = _non_empty_string(
            evidence.get(field),
            field=f"result.evidence.{field}",
        )
        evidence_path = _result_evidence_path(
            result_path.parent,
            relative,
        )
        value = evidence_path.read_text(encoding="utf-8")
        expected = _non_empty_string(
            evidence.get(f"{field}_sha256"),
            field=f"result.evidence.{field}_sha256",
        )
        if _sha256(value) != expected:
            raise ValueError(
                f"evidence hash mismatch for {field}: {result_path}"
            )
    return payload


def validate_repository_bindings(
    manifest: BenchmarkManifest,
    *,
    repository: Path,
) -> None:
    repository = repository.expanduser().resolve()
    _git_text(repository, "rev-parse", "--show-toplevel")
    for task in manifest.tasks:
        parent_line = _git_text(
            repository,
            "rev-list",
            "--parents",
            "-n",
            "1",
            task.target_commit,
        ).split()
        if not parent_line or parent_line[0] != task.target_commit:
            raise ValueError(f"{task.task_id} target commit does not exist")
        if len(parent_line) < 2 or parent_line[1] != task.source_commit:
            raise ValueError(
                f"{task.task_id} source commit must be the target commit's first parent"
            )
        _git_text(repository, "cat-file", "-e", f"{task.source_commit}^{{commit}}")

        changed_paths = tuple(
            line
            for line in _git_text(
                repository,
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                task.target_commit,
            ).splitlines()
            if line
        )
        if task.category == "cross_layer":
            observed_layers = _infer_architecture_layers(changed_paths)
            missing = set(task.layers) - observed_layers
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise ValueError(
                    f"{task.task_id} target commit does not touch declared layers: "
                    f"{missing_text}"
                )

        for relative in task.acceptance_files:
            if not relative.startswith("tests/"):
                raise ValueError(
                    f"{task.task_id} acceptance files must stay under tests/"
                )
            _git_text(
                repository,
                "cat-file",
                "-e",
                f"{task.target_commit}:{relative}",
            )


def classify_outcome(facts: RunFacts) -> RunOutcome:
    if facts.safety_violations:
        return RunOutcome.SAFETY_VIOLATION
    if facts.benchmark_invalid_reason is not None:
        return RunOutcome.BENCHMARK_INVALID
    if facts.provider_error_code is not None:
        if (
            facts.model_alias in _CLOUD_BENCHMARK_MODELS
            and facts.provider_error_code in {"rate_limit", "quota_exceeded", "http_429"}
        ):
            return RunOutcome.PROVIDER_LIMITED
        return RunOutcome.RUNTIME_FAILED
    if facts.runtime_error is not None:
        return RunOutcome.RUNTIME_FAILED
    if facts.turn_status == "done":
        if facts.valid_diff and facts.hidden_acceptance_passed:
            return RunOutcome.PASSED
        return RunOutcome.FALSE_COMPLETION
    if facts.turn_status in {"failed", "paused", "interrupted"}:
        return RunOutcome.BOUNDED_FAILURE
    return RunOutcome.VALIDATION_FAILED


def diagnose_run(
    facts: RunFacts,
    outcome: RunOutcome,
    *,
    agent_stdout: str,
    acceptance_stdout: str,
    acceptance_stderr: str,
) -> Diagnosis:
    combined = f"{agent_stdout}\n{acceptance_stdout}\n{acceptance_stderr}".lower()
    if outcome is RunOutcome.PASSED:
        return Diagnosis(
            primary=DiagnosisCause.UNKNOWN,
            secondary=(),
            confidence=0.0,
            evidence=(),
        )
    if outcome is RunOutcome.SAFETY_VIOLATION:
        return Diagnosis(
            primary=DiagnosisCause.SAFETY_BOUNDARY,
            secondary=(),
            confidence=1.0,
            evidence=tuple(
                f"safety_violation:{violation}"
                for violation in facts.safety_violations
            ),
        )
    if outcome is RunOutcome.BENCHMARK_INVALID:
        return Diagnosis(
            primary=DiagnosisCause.BENCHMARK_ENVIRONMENT,
            secondary=(),
            confidence=1.0,
            evidence=(
                f"benchmark_invalid:{facts.benchmark_invalid_reason or 'unknown'}",
            ),
        )
    if outcome is RunOutcome.PROVIDER_LIMITED:
        secondary, prior_evidence = _pre_provider_limit_diagnosis(
            facts,
            combined,
        )
        return Diagnosis(
            primary=DiagnosisCause.PROVIDER_LIMIT,
            secondary=secondary,
            confidence=1.0,
            evidence=(
                f"provider_error:{facts.provider_error_code or 'unknown'}",
                *prior_evidence,
            ),
        )
    if any(
        marker in combined
        for marker in (
            "context overflow",
            "context_overflow",
            "input uses ",
        )
    ):
        return Diagnosis(
            primary=DiagnosisCause.CONTEXT_ASSEMBLY,
            secondary=(),
            confidence=0.95,
            evidence=("runtime_error:context_overflow",),
        )
    if facts.stop_reason == "delivery_stalled" or "delivery_stalled" in combined:
        evidence = ["stop_reason:delivery_stalled"]
        inspection_calls = _inspection_call_count(agent_stdout)
        if inspection_calls is not None:
            evidence.append(f"inspection_calls:{inspection_calls}")
        secondary = (
            (DiagnosisCause.PROVIDER_ADAPTER,)
            if "request timed out" in combined
            else ()
        )
        if "request timed out" in combined:
            evidence.append("provider_error:request_timeout")
        return Diagnosis(
            primary=DiagnosisCause.MODEL_CAPABILITY_GAP,
            secondary=secondary,
            confidence=0.9,
            evidence=tuple(evidence),
        )
    if outcome is RunOutcome.FALSE_COMPLETION:
        evidence = []
        if not facts.valid_diff:
            evidence.append("workspace_diff:empty")
        if not facts.hidden_acceptance_passed:
            evidence.append("hidden_acceptance:failed")
        return Diagnosis(
            primary=DiagnosisCause.EXECUTION_CLOSURE,
            secondary=(),
            confidence=0.95,
            evidence=tuple(evidence or ("completion_evidence:invalid",)),
        )
    if facts.valid_diff and not facts.hidden_acceptance_passed:
        return Diagnosis(
            primary=DiagnosisCause.EXECUTION_CLOSURE,
            secondary=(),
            confidence=0.85,
            evidence=("hidden_acceptance:failed",),
        )
    if facts.provider_error_code is not None or "model_provider_failed" in combined:
        return Diagnosis(
            primary=DiagnosisCause.PROVIDER_ADAPTER,
            secondary=(),
            confidence=0.8,
            evidence=(
                f"provider_error:{facts.provider_error_code or 'model_provider_failed'}",
            ),
        )
    if facts.runtime_error is not None:
        return Diagnosis(
            primary=DiagnosisCause.EXECUTION_CLOSURE,
            secondary=(),
            confidence=0.75,
            evidence=(f"runtime_error:{facts.runtime_error}",),
        )
    return Diagnosis(
        primary=DiagnosisCause.EXECUTION_CLOSURE,
        secondary=(),
        confidence=0.5,
        evidence=(
            f"turn_status:{facts.turn_status}",
            f"hidden_acceptance:{'passed' if facts.hidden_acceptance_passed else 'failed'}",
        ),
    )


def _pre_provider_limit_diagnosis(
    facts: RunFacts,
    combined_output: str,
) -> tuple[tuple[DiagnosisCause, ...], tuple[str, ...]]:
    causes: list[DiagnosisCause] = []
    evidence: list[str] = []

    for marker in (
        "repeated_inspection",
        "repeated_tool_failure",
        "delivery_stalled",
    ):
        if marker in combined_output:
            causes.append(DiagnosisCause.MODEL_CAPABILITY_GAP)
            evidence.append(f"pre_limit:{marker}")
            break

    for marker in (
        "model_tool_call_rejected",
        "invalid_arguments",
        "normalization_failed",
        "output_validation_failed",
    ):
        if marker in combined_output:
            causes.append(DiagnosisCause.TOOL_CONTRACT)
            evidence.append(f"pre_limit:{marker}")
            break

    if any(
        marker in combined_output
        for marker in (
            "context overflow",
            "context_overflow",
            "input uses ",
        )
    ):
        causes.append(DiagnosisCause.CONTEXT_ASSEMBLY)
        evidence.append("pre_limit:context_overflow")

    if facts.runtime_error is not None:
        causes.append(DiagnosisCause.EXECUTION_CLOSURE)
        evidence.append(f"pre_limit:runtime_error:{facts.runtime_error}")

    return tuple(dict.fromkeys(causes)), tuple(evidence)


def run_task(
    *,
    repository: Path,
    manifest: BenchmarkManifest,
    task: BenchmarkTask,
    model_alias: str,
    agent_command: Sequence[str],
    artifacts_root: Path,
) -> TaskRunRecord:
    if model_alias not in {
        manifest.primary_model,
        manifest.control_model,
        manifest.diagnostic_model,
    }:
        raise ValueError("model alias is outside the benchmark model policy")
    if task.permissions.network:
        raise ValueError("phase-one benchmark tasks cannot authorize network tools")
    command_prefix = tuple(agent_command)
    if not command_prefix:
        raise ValueError("agent_command must not be empty")

    repository = repository.expanduser().resolve()
    runtime_fingerprint = repository_state_fingerprint(repository)
    artifacts_root = artifacts_root.expanduser().resolve()
    run_id = uuid4().hex
    artifact_dir = (
        artifacts_root
        / manifest.benchmark_version
        / task.task_id
        / model_alias
        / run_id
    )
    artifact_dir.mkdir(parents=True, exist_ok=False)

    with tempfile.TemporaryDirectory(prefix="workspace-", dir=artifact_dir) as raw_workspace:
        workspace = Path(raw_workspace)
        _export_snapshot(repository, task.source_commit, workspace)
        _initialize_snapshot_git(workspace)

        setup = _run_command(
            task.setup_command,
            cwd=workspace,
            timeout_seconds=task.budget.timeout_seconds,
            live_log_paths=(
                artifact_dir / "setup.stdout",
                artifact_dir / "setup.stderr",
            ),
            progress_label="setup",
        )
        _write_log(artifact_dir / "setup.stdout", setup.stdout)
        _write_log(artifact_dir / "setup.stderr", setup.stderr)

        baseline_runtime_files = _runtime_files(workspace)
        checkpoint_path = artifact_dir / "checkpoints.sqlite"
        model_session_path = artifact_dir / "model-session.json"
        command = [
            *command_prefix,
            "run",
            task.instruction,
            "--model",
            model_alias,
            "--checkpoint-db",
            str(checkpoint_path),
            "--model-session-path",
            str(model_session_path),
            "--max-turns",
            str(task.budget.max_turns),
            "--require-workspace-change",
            "--non-interactive",
            "--verbose",
        ]
        max_tokens_total = None
        if task.permissions.write_workspace:
            command.append("--allow-write-tools")
        if task.permissions.execute_process:
            command.append("--allow-execute-tools")

        if setup.returncode == 0 and not setup.timed_out:
            agent = _run_command(
                command,
                cwd=workspace,
                timeout_seconds=task.budget.timeout_seconds,
                live_log_paths=(
                    artifact_dir / "agent.stdout",
                    artifact_dir / "agent.stderr",
                ),
                progress_label="agent",
            )
        else:
            agent = _CommandResult(
                returncode=setup.returncode,
                stdout="",
                stderr="benchmark setup failed",
                timed_out=setup.timed_out,
            )
        _write_log(artifact_dir / "agent.stdout", agent.stdout)
        _write_log(artifact_dir / "agent.stderr", agent.stderr)

        changed_paths = _changed_paths(workspace)
        diff = _workspace_diff(workspace, changed_paths)
        archived_diff = _redact_secrets(diff)
        (artifact_dir / "agent.diff").write_text(
            archived_diff,
            encoding="utf-8",
        )
        safety_violations = [
            f"path_outside_allowlist:{path}"
            for path in changed_paths
            if not _path_allowed(path, task.allowed_paths)
        ]
        if archived_diff != diff:
            safety_violations.append("secret_leak:agent.diff")
        new_runtime_files = _runtime_files(workspace) - baseline_runtime_files
        safety_violations.extend(
            f"runtime_residue:{path}" for path in sorted(new_runtime_files)
        )
        if agent.leftover_processes:
            safety_violations.append("leftover_process")

        acceptance_error: str | None = None
        try:
            _materialize_acceptance_files(repository, workspace, task)
            acceptance = _run_command(
                task.acceptance_command,
                cwd=workspace,
                timeout_seconds=task.budget.timeout_seconds,
                live_log_paths=(
                    artifact_dir / "acceptance.stdout",
                    artifact_dir / "acceptance.stderr",
                ),
                progress_label="acceptance",
            )
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            acceptance_error = f"{type(exc).__name__}: {exc}"
            acceptance = _CommandResult(
                returncode=1,
                stdout="",
                stderr=acceptance_error,
            )
        _write_log(artifact_dir / "acceptance.stdout", acceptance.stdout)
        _write_log(artifact_dir / "acceptance.stderr", acceptance.stderr)

        turn_id = _output_field(agent.stdout, "Turn")
        turn_status = _output_field(agent.stdout, "状态") or "failed"
        runtime_error = None
        benchmark_invalid_reason = None
        if setup.returncode != 0 or setup.timed_out:
            benchmark_invalid_reason = "benchmark_setup_failed"
        elif agent.timed_out:
            runtime_error = "agent_timeout"
        elif turn_id is None or _output_field(agent.stdout, "状态") is None:
            runtime_error = "missing_public_result"
        elif acceptance_error is not None:
            benchmark_invalid_reason = "acceptance_materialization_failed"

        hidden_acceptance_passed = (
            acceptance.returncode == 0 and not acceptance.timed_out
        )
        provider_error_code = _provider_error_code(agent.stdout, agent.stderr)
        stop_reason = _output_field(agent.stdout, "停止原因")
        safety = tuple(sorted(set(safety_violations)))
        facts = RunFacts(
            model_alias=model_alias,
            turn_status=turn_status,
            valid_diff=bool(diff.strip()) and not safety,
            hidden_acceptance_passed=hidden_acceptance_passed,
            stop_reason=stop_reason,
            safety_violations=safety,
            provider_error_code=provider_error_code,
            runtime_error=runtime_error,
            benchmark_invalid_reason=benchmark_invalid_reason,
        )
        outcome = classify_outcome(facts)
        diagnosis = diagnose_run(
            facts,
            outcome,
            agent_stdout=agent.stdout,
            acceptance_stdout=acceptance.stdout,
            acceptance_stderr=acceptance.stderr,
        )
        record = TaskRunRecord(
            task_id=task.task_id,
            model_alias=model_alias,
            max_tokens_total=max_tokens_total,
            runtime_fingerprint=runtime_fingerprint,
            outcome=outcome,
            turn_id=turn_id,
            turn_status=turn_status,
            hidden_acceptance_passed=hidden_acceptance_passed,
            diff=archived_diff,
            safety_violations=safety,
            diagnosis=diagnosis,
            artifact_dir=artifact_dir,
        )
        _write_result(
            record,
            manifest=manifest,
            task=task,
            agent=agent,
            acceptance=acceptance,
        )
        return record


def _export_snapshot(repository: Path, commit: str, workspace: Path) -> None:
    archive = workspace.parent / "source.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "-o", str(archive), commit],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    with tarfile.open(archive) as handle:
        handle.extractall(workspace, filter="data")
    archive.unlink()


def _initialize_snapshot_git(workspace: Path) -> None:
    commands = (
        ("init", "-q"),
        ("config", "user.email", "benchmark@example.invalid"),
        ("config", "user.name", "Code Agent Benchmark"),
        ("add", "-A"),
        ("commit", "-qm", "benchmark source snapshot"),
    )
    for args in commands:
        subprocess.run(
            ["git", *args],
            cwd=workspace,
            check=True,
            capture_output=True,
        )


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    live_log_paths: tuple[Path, Path] | None = None,
    progress_label: str = "command",
    heartbeat_seconds: float = 30.0,
) -> _CommandResult:
    if not command:
        return _CommandResult(returncode=0, stdout="", stderr="")
    if live_log_paths is not None and heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    try:
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
    except OSError as exc:
        return _CommandResult(
            returncode=127,
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}",
        )
    if live_log_paths is not None:
        return _stream_command(
            process,
            timeout_seconds=timeout_seconds,
            live_log_paths=live_log_paths,
            progress_label=progress_label,
            heartbeat_seconds=heartbeat_seconds,
        )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        leftover = _terminate_process_group(process.pid)
        return _CommandResult(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            leftover_processes=leftover,
        )
    except subprocess.TimeoutExpired:
        leftover = _kill_process_group(process.pid)
        stdout, stderr = process.communicate()
        return _CommandResult(
            returncode=124,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
            leftover_processes=leftover,
        )
    except KeyboardInterrupt:
        _kill_process_group(process.pid)
        process.communicate()
        raise


def _stream_command(
    process: subprocess.Popen[str],
    *,
    timeout_seconds: int,
    live_log_paths: tuple[Path, Path],
    progress_label: str,
    heartbeat_seconds: float,
) -> _CommandResult:
    stdout_pipe = process.stdout
    stderr_pipe = process.stderr
    if stdout_pipe is None or stderr_pipe is None:
        raise RuntimeError("streamed command requires stdout and stderr pipes")

    stdout_path, stderr_path = live_log_paths
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    output_lock = threading.Lock()
    try:
        stdout_log = stdout_path.open("w", encoding="utf-8")
        stderr_log = stderr_path.open("w", encoding="utf-8")
    except BaseException:
        if "stdout_log" in locals():
            stdout_log.close()
        _kill_process_group(process.pid)
        process.communicate()
        raise

    def emit_status(message: str) -> None:
        with output_lock:
            sys.stderr.write(f"[benchmark:{progress_label}] {message}\n")
            sys.stderr.flush()

    def drain(
        pipe: TextIO,
        *,
        stream_name: str,
        parts: list[str],
        log: TextIO,
    ) -> None:
        while True:
            line = pipe.readline()
            if line == "":
                break
            parts.append(line)
            redacted = _redact_secrets(line)
            with output_lock:
                log.write(redacted)
                log.flush()
                terminal_line = (
                    f"[benchmark:{progress_label}:{stream_name}] {redacted}"
                )
                if not terminal_line.endswith("\n"):
                    terminal_line += "\n"
                sys.stderr.write(terminal_line)
                sys.stderr.flush()
        pipe.close()

    readers = (
        threading.Thread(
            target=drain,
            kwargs={
                "pipe": stdout_pipe,
                "stream_name": "stdout",
                "parts": stdout_parts,
                "log": stdout_log,
            },
            daemon=True,
        ),
        threading.Thread(
            target=drain,
            kwargs={
                "pipe": stderr_pipe,
                "stream_name": "stderr",
                "parts": stderr_parts,
                "log": stderr_log,
            },
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()

    started = time.monotonic()
    deadline = started + timeout_seconds
    next_heartbeat = started + heartbeat_seconds
    timed_out = False
    leftover = False
    emit_status("started")
    try:
        while process.poll() is None:
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                emit_status(f"timed out after {timeout_seconds}s; terminating")
                leftover = _kill_process_group(process.pid)
                process.wait()
                break
            wait_until = min(deadline, next_heartbeat)
            try:
                process.wait(timeout=max(0.01, wait_until - now))
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                if now >= next_heartbeat:
                    emit_status(
                        f"still running elapsed={int(now - started)}s"
                    )
                    while next_heartbeat <= now:
                        next_heartbeat += heartbeat_seconds
        if not timed_out:
            leftover = _terminate_process_group(process.pid)
    except KeyboardInterrupt:
        emit_status("interrupted; terminating process group")
        _kill_process_group(process.pid)
        process.wait()
        raise
    finally:
        for reader in readers:
            reader.join()
        stdout_log.close()
        stderr_log.close()

    returncode = 124 if timed_out else int(process.returncode or 0)
    emit_status(
        f"finished returncode={returncode} elapsed={int(time.monotonic() - started)}s"
    )
    return _CommandResult(
        returncode=returncode,
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
        timed_out=timed_out,
        leftover_processes=leftover,
    )


def _terminate_process_group(process_group: int) -> bool:
    if not _process_group_exists(process_group):
        return False
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return False
    if not _wait_for_process_group(
        process_group,
        timeout_seconds=_PROCESS_CLEANUP_GRACE_SECONDS,
    ):
        return False
    return _kill_process_group(process_group)


def _kill_process_group(process_group: int) -> bool:
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return False
    return _wait_for_process_group(
        process_group,
        timeout_seconds=_PROCESS_CLEANUP_GRACE_SECONDS,
    )


def _wait_for_process_group(
    process_group: int,
    *,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group):
            return False
        time.sleep(0.02)
    return _process_group_exists(process_group)


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _changed_paths(workspace: Path) -> tuple[str, ...]:
    tracked = subprocess.run(
        ["git", "diff", "--name-only", "-z", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=workspace,
        check=True,
        capture_output=True,
    ).stdout
    paths = {
        item.decode("utf-8")
        for payload in (tracked, untracked)
        for item in payload.split(b"\0")
        if item
    }
    return tuple(sorted(paths))


def _workspace_diff(workspace: Path, changed_paths: Sequence[str]) -> str:
    tracked = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    chunks = [tracked]
    untracked = set(
        subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=workspace,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
    )
    for path in changed_paths:
        if path.encode("utf-8") not in untracked:
            continue
        result = subprocess.run(
            ["git", "diff", "--binary", "--no-index", "--", "/dev/null", path],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode not in {0, 1}:
            raise RuntimeError(f"cannot diff untracked path: {path}")
        chunks.append(result.stdout)
    return "".join(chunks)


def _materialize_acceptance_files(
    repository: Path,
    workspace: Path,
    task: BenchmarkTask,
) -> None:
    for relative in task.acceptance_files:
        target = _workspace_target(workspace, relative)
        result = subprocess.run(
            ["git", "show", f"{task.target_commit}:{relative}"],
            cwd=repository,
            check=True,
            capture_output=True,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result.stdout)


def _workspace_target(workspace: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("benchmark paths must be relative")
    target = (workspace / candidate).resolve()
    try:
        target.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("benchmark path escapes workspace") from exc
    return target


def _result_evidence_path(result_dir: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("result evidence paths must be relative")
    target = (result_dir / candidate).resolve()
    try:
        target.relative_to(result_dir)
    except ValueError as exc:
        raise ValueError("result evidence path escapes artifact directory") from exc
    if not target.is_file():
        raise ValueError(f"result evidence file is missing: {relative}")
    return target


def _path_allowed(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch(path, pattern) for pattern in patterns)


def _runtime_files(workspace: Path) -> set[str]:
    runtime_root = workspace / ".rag"
    if not runtime_root.exists():
        return set()
    return {
        path.relative_to(workspace).as_posix()
        for path in runtime_root.rglob("*")
        if path.is_file()
    }


def _output_field(output: str, label: str) -> str | None:
    match = re.search(rf"^{re.escape(label)}:\s*(.+?)\s*$", output, re.MULTILINE)
    return None if match is None else match.group(1)


def _inspection_call_count(output: str) -> int | None:
    matches = re.findall(
        r"(?:after|inspection_calls[\"']?\s*[:=])\s*(\d+)",
        output,
        flags=re.IGNORECASE,
    )
    return max((int(value) for value in matches), default=None)


def _provider_error_code(stdout: str, stderr: str) -> str | None:
    text = f"{stdout}\n{stderr}".lower()
    if any(
        marker in text
        for marker in (
            "429",
            "rate limit",
            "rate_limit_exceeded",
            "quota exceeded",
            "tokens per minute",
        )
    ):
        return "rate_limit"
    if any(
        marker in text
        for marker in (
            "api_key is not set",
            "api key is not set",
            "missing api key",
            "missing credentials",
        )
    ):
        return "missing_credentials"
    if any(
        marker in text
        for marker in (
            "request timed out",
            "request_timeout",
            "read timeout",
            "connect timeout",
        )
    ):
        return "request_timeout"
    if any(
        marker in text
        for marker in (
            "certificate_verify_failed",
            "connecterror",
            "connection error",
            "connection refused",
            "name or service not known",
            "temporary failure in name resolution",
        )
    ):
        return "transport_error"
    return None


def _write_log(path: Path, value: str) -> None:
    path.write_text(_redact_secrets(value), encoding="utf-8")


def _redact_secrets(value: str) -> str:
    redacted = value
    for name, secret in os.environ.items():
        upper = name.upper()
        if (
            secret
            and len(secret) >= 4
            and any(marker in upper for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
        ):
            redacted = redacted.replace(secret, "[REDACTED]")
    redacted = re.sub(
        r"(?<![A-Za-z0-9])(?:gsk_|sk-|ak[-_])[A-Za-z0-9_-]{8,}",
        "[REDACTED]",
        redacted,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}",
        "Bearer [REDACTED]",
        redacted,
    )
    return redacted


def _write_result(
    record: TaskRunRecord,
    *,
    manifest: BenchmarkManifest,
    task: BenchmarkTask,
    agent: _CommandResult,
    acceptance: _CommandResult,
) -> None:
    payload = {
        "schema_version": 1,
        "benchmark_version": manifest.benchmark_version,
        "manifest_fingerprint": manifest.fingerprint,
        "task_id": record.task_id,
        "task_source_commit": task.source_commit,
        "task_target_commit": task.target_commit,
        "runtime_fingerprint": record.runtime_fingerprint,
        "model_alias": record.model_alias,
        "max_tokens_total": record.max_tokens_total,
        "outcome": record.outcome.value,
        "turn_id": record.turn_id,
        "turn_status": record.turn_status,
        "hidden_acceptance_passed": record.hidden_acceptance_passed,
        "safety_violations": list(record.safety_violations),
        "diagnosis": {
            "primary": record.diagnosis.primary.value,
            "secondary": [
                cause.value for cause in record.diagnosis.secondary
            ],
            "confidence": record.diagnosis.confidence,
            "evidence": list(record.diagnosis.evidence),
        },
        "evidence": {
            "agent_stdout": "agent.stdout",
            "agent_stderr": "agent.stderr",
            "agent_diff": "agent.diff",
            "acceptance_stdout": "acceptance.stdout",
            "acceptance_stderr": "acceptance.stderr",
            "agent_diff_sha256": _sha256(
                _redact_secrets(record.diff)
            ),
            "agent_stdout_sha256": _sha256(_redact_secrets(agent.stdout)),
            "agent_stderr_sha256": _sha256(_redact_secrets(agent.stderr)),
            "acceptance_stdout_sha256": _sha256(
                _redact_secrets(acceptance.stdout)
            ),
            "acceptance_stderr_sha256": _sha256(
                _redact_secrets(acceptance.stderr)
            ),
        },
    }
    (record.artifact_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_task(raw: object) -> BenchmarkTask:
    payload = _mapping(raw, field="task")
    task_id = _non_empty_string(payload.get("id"), field="task.id")
    split = _non_empty_string(payload.get("split"), field=f"{task_id}.split")
    if split not in {"regression", "holdout"}:
        raise ValueError(f"{task_id}.split must be regression or holdout")
    category = _non_empty_string(
        payload.get("category"),
        field=f"{task_id}.category",
    )
    if category not in {"local", "medium", "cross_layer"}:
        raise ValueError(f"{task_id}.category is unsupported")
    layers = _string_tuple(payload.get("layers"), field=f"{task_id}.layers")
    unknown_layers = set(layers) - _ARCHITECTURE_LAYERS
    if unknown_layers:
        raise ValueError(f"{task_id}.layers contains unsupported architecture layers")
    if category == "cross_layer" and len(set(layers)) < 2:
        raise ValueError(
            f"{task_id} cross-layer task requires at least two architecture layers"
        )

    source_commit = _commit(payload.get("source_commit"), field=f"{task_id}.source_commit")
    target_commit = _commit(payload.get("target_commit"), field=f"{task_id}.target_commit")
    if source_commit == target_commit:
        raise ValueError(f"{task_id} source and target commits must differ")
    permissions_payload = _mapping(
        payload.get("permissions"),
        field=f"{task_id}.permissions",
    )
    budget_payload = _mapping(payload.get("budget"), field=f"{task_id}.budget")
    return BenchmarkTask(
        task_id=task_id,
        split=split,
        category=category,
        layers=layers,
        control=_strict_bool(payload.get("control", False), field=f"{task_id}.control"),
        source_commit=source_commit,
        target_commit=target_commit,
        instruction=_non_empty_string(
            payload.get("instruction"),
            field=f"{task_id}.instruction",
        ),
        acceptance_command=_string_tuple(
            payload.get("acceptance_command"),
            field=f"{task_id}.acceptance_command",
        ),
        acceptance_files=_string_tuple(
            payload.get("acceptance_files"),
            field=f"{task_id}.acceptance_files",
        ),
        setup_command=_optional_string_tuple(
            payload.get("setup_command", ()),
            field=f"{task_id}.setup_command",
        ),
        allowed_paths=_string_tuple(
            payload.get("allowed_paths"),
            field=f"{task_id}.allowed_paths",
        ),
        permissions=TaskPermissions(
            write_workspace=_strict_bool(
                permissions_payload.get("write_workspace"),
                field=f"{task_id}.permissions.write_workspace",
            ),
            execute_process=_strict_bool(
                permissions_payload.get("execute_process"),
                field=f"{task_id}.permissions.execute_process",
            ),
            network=_strict_bool(
                permissions_payload.get("network"),
                field=f"{task_id}.permissions.network",
            ),
        ),
        budget=TaskBudget(
            max_turns=_positive_int(
                budget_payload.get("max_turns"),
                field=f"{task_id}.budget.max_turns",
            ),
            timeout_seconds=_positive_int(
                budget_payload.get("timeout_seconds"),
                field=f"{task_id}.budget.timeout_seconds",
            ),
        ),
    )


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} keys must be strings")
    return value


def _non_empty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array of strings")
    items = tuple(_non_empty_string(item, field=field) for item in value)
    if not items:
        raise ValueError(f"{field} must not be empty")
    return items


def _optional_string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array of strings")
    return tuple(_non_empty_string(item, field=field) for item in value)


def _commit(value: object, *, field: str) -> str:
    commit = _non_empty_string(value, field=field)
    if _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError(f"{field} must be a full lowercase Git commit SHA")
    return commit


def _sha256_fingerprint(value: object, *, field: str) -> str:
    fingerprint = _non_empty_string(value, field=field)
    if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 fingerprint")
    return fingerprint


def _strict_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_text(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
        raise ValueError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def repository_state_fingerprint(repository: Path) -> str:
    repository = repository.expanduser().resolve()
    root = Path(_git_text(repository, "rev-parse", "--show-toplevel"))
    head = _git_text(root, "rev-parse", "HEAD")
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    listed = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    digest = hashlib.sha256()
    digest.update(b"code-agent-runtime-v1\0")
    digest.update(head.encode("ascii"))
    digest.update(b"\0status\0")
    digest.update(status)
    for raw_relative in sorted(
        value for value in listed.split(b"\0") if value
    ):
        relative = os.fsdecode(raw_relative)
        path = root / relative
        digest.update(b"\0path\0")
        digest.update(raw_relative)
        if path.is_symlink():
            digest.update(b"\0symlink\0")
            digest.update(os.fsencode(os.readlink(path)))
        elif path.is_file():
            digest.update(b"\0file\0")
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        else:
            digest.update(b"\0missing\0")
    return digest.hexdigest()


def _infer_architecture_layers(paths: Sequence[str]) -> set[str]:
    runtime_roots = ("agent_runtime", "rag/" + "agent")
    layers: set[str] = set()
    for path in paths:
        if path in {"agent_runtime/__init__.py", "agent_runtime/agent.py"} or any(
            path == f"{root}/cli.py" for root in runtime_roots
        ):
            layers.add("public_api_cli")
        if (
            any(path == f"{root}/service.py" for root in runtime_roots)
            or path.startswith("agent_runtime/runtime/")
            or path == "rag/models/catalog.py"
            or any(
                path
                in {
                    f"{root}/core/llm_registry.py",
                    f"{root}/core/llm_providers.py",
                    f"{root}/core/model_provider_runtime.py",
                }
                for root in runtime_roots
            )
        ):
            layers.add("service")
        if any(path.startswith(f"{root}/loop/") for root in runtime_roots):
            layers.add("loop")
        if (
            any(
                path.startswith(
                    (
                        f"{root}/tools/",
                        f"{root}/tooling/",
                        f"{root}/builtin/",
                    )
                )
                or path == f"{root}/planning.py"
                for root in runtime_roots
            )
        ):
            layers.add("tool")
        if any(
            path
            in {
                f"{root}/core/checkpointing.py",
                f"{root}/core/context.py",
                f"{root}/sessions.py",
            }
            for root in runtime_roots
        ):
            layers.add("turn_checkpoint")
        if any(
            path == f"{root}/result.py"
            or path.startswith(f"{root}/streaming/")
            for root in runtime_roots
        ):
            layers.add("result_events")
    return layers


def _task_by_id(manifest: BenchmarkManifest, task_id: str) -> BenchmarkTask:
    for task in manifest.tasks:
        if task.task_id == task_id:
            return task
    raise ValueError(f"unknown benchmark task: {task_id}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-code-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--repository", type=Path, default=Path.cwd())
    validate.add_argument("--release", action="store_true")

    run = subparsers.add_parser("run-task")
    run.add_argument("manifest", type=Path)
    run.add_argument("task_id")
    run.add_argument("--repository", type=Path, default=Path.cwd())
    run.add_argument("--artifacts-root", type=Path, required=True)
    run.add_argument(
        "--model",
        required=True,
        help="explicit benchmark lane model; no implicit local-model fallback",
    )
    run.add_argument(
        "--agent-command",
        default="agent",
        help="trusted local command prefix, parsed without a shell",
    )

    gate = subparsers.add_parser("gate")
    gate.add_argument("manifest", type=Path)
    gate.add_argument("results", nargs="+", type=Path)
    gate.add_argument("--repository", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    exit_code = 0
    try:
        manifest = load_manifest(args.manifest)
        validate_repository_bindings(manifest, repository=args.repository)
        if args.command == "validate":
            if args.release:
                validate_release_shape(manifest)
            payload = {
                "benchmark_version": manifest.benchmark_version,
                "manifest_fingerprint": manifest.fingerprint,
                "task_count": len(manifest.tasks),
            }
        elif args.command == "run-task":
            task = _task_by_id(manifest, args.task_id)
            model_alias = args.model
            command = tuple(shlex.split(args.agent_command))
            record = run_task(
                repository=args.repository,
                manifest=manifest,
                task=task,
                model_alias=model_alias,
                agent_command=command,
                artifacts_root=args.artifacts_root,
            )
            payload = {
                "task_id": record.task_id,
                "model_alias": record.model_alias,
                "outcome": record.outcome.value,
                "turn_id": record.turn_id,
                "artifact_dir": str(record.artifact_dir),
            }
        else:
            result_payloads = [
                load_result_record(path, manifest=manifest)
                for path in args.results
            ]
            payload = evaluate_release_results(
                manifest,
                result_payloads,
                expected_runtime_fingerprint=repository_state_fingerprint(
                    args.repository
                ),
            )
            if not payload["release_ready"]:
                exit_code = 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("benchmark interrupted", file=sys.stderr)
        return 130
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
