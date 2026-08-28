#!/usr/bin/env python3
"""Run repeatable real-model public-path scenarios against the Praxis Agent."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

_SCHEMA_VERSION = "praxis-public-path-scenarios-v1"
_RESULT_SCHEMA = "praxis-public-path-result-v1"
_EVIDENCE_SCHEMA = "praxis-public-path-release-evidence-v1"
_MODES = frozenset({"direct", "approval", "crash_after_model_tool_call"})
_SCENARIO_APPROVAL_EFFECTS = frozenset(
    {"read_workspace", "write_workspace", "execute_process", "destructive"}
)


@dataclass(frozen=True, slots=True)
class PublicPathScenario:
    scenario_id: str
    mode: str
    prompt: str
    files: Mapping[str, str]
    acceptance_command: tuple[str, ...]
    required_changes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PublicPathManifest:
    model: str
    repetitions: int
    scenarios: tuple[PublicPathScenario, ...]
    fingerprint: str


def _non_empty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object with string keys")
    return value


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(path: Path) -> PublicPathManifest:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    payload = _mapping(raw, field="manifest")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported public-path manifest schema_version")
    model = _non_empty(payload.get("model"), field="model")
    repetitions = payload.get("repetitions")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions != 3:
        raise ValueError("public-path repetitions must be exactly 3")
    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, Sequence) or isinstance(raw_scenarios, (str, bytes)):
        raise ValueError("scenarios must be a sequence")
    scenarios: list[PublicPathScenario] = []
    identifiers: set[str] = set()
    for index, raw_scenario in enumerate(raw_scenarios):
        scenario = _mapping(raw_scenario, field=f"scenarios[{index}]")
        scenario_id = _non_empty(scenario.get("id"), field=f"scenarios[{index}].id")
        if scenario_id in identifiers:
            raise ValueError(f"duplicate scenario id: {scenario_id}")
        identifiers.add(scenario_id)
        mode = _non_empty(scenario.get("mode"), field=f"{scenario_id}.mode")
        if mode not in _MODES:
            raise ValueError(f"{scenario_id}.mode is unsupported: {mode}")
        raw_files = _mapping(scenario.get("files"), field=f"{scenario_id}.files")
        files = {
            relative: _non_empty(content, field=f"{scenario_id}.files[{relative}]")
            for relative, content in raw_files.items()
        }
        raw_command = scenario.get("acceptance_command")
        if (
            not isinstance(raw_command, Sequence)
            or isinstance(raw_command, (str, bytes))
            or not raw_command
        ):
            raise ValueError(f"{scenario_id}.acceptance_command must be an argv sequence")
        command = tuple(
            _non_empty(part, field=f"{scenario_id}.acceptance_command")
            for part in raw_command
        )
        raw_changes = scenario.get("required_changes")
        if (
            not isinstance(raw_changes, Sequence)
            or isinstance(raw_changes, (str, bytes))
            or not raw_changes
        ):
            raise ValueError(f"{scenario_id}.required_changes must be non-empty")
        required_changes = tuple(
            _non_empty(change, field=f"{scenario_id}.required_changes")
            for change in raw_changes
        )
        scenarios.append(
            PublicPathScenario(
                scenario_id=scenario_id,
                mode=mode,
                prompt=_non_empty(scenario.get("prompt"), field=f"{scenario_id}.prompt"),
                files=files,
                acceptance_command=command,
                required_changes=required_changes,
            )
        )
    if len(scenarios) != 5:
        raise ValueError("public-path manifest must contain exactly five scenarios")
    return PublicPathManifest(
        model=model,
        repetitions=repetitions,
        scenarios=tuple(scenarios),
        fingerprint=_canonical_hash(raw),
    )


def evaluate_results(
    manifest: PublicPathManifest,
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    expected = {
        (scenario.scenario_id, repetition)
        for scenario in manifest.scenarios
        for repetition in range(1, manifest.repetitions + 1)
    }
    indexed: dict[tuple[str, int], Mapping[str, object]] = {}
    reasons: list[str] = []
    for result in results:
        scenario_id = _non_empty(result.get("scenario_id"), field="result.scenario_id")
        repetition = result.get("repetition")
        if isinstance(repetition, bool) or not isinstance(repetition, int):
            raise ValueError("result.repetition must be an integer")
        key = (scenario_id, repetition)
        if key not in expected:
            raise ValueError(f"unexpected public-path result: {scenario_id}:{repetition}")
        if key in indexed:
            raise ValueError(f"duplicate public-path result: {scenario_id}:{repetition}")
        if result.get("model") != manifest.model:
            raise ValueError(f"result model mismatch: {scenario_id}:{repetition}")
        indexed[key] = result
        if result.get("status") != "passed":
            reasons.append(f"failed:{scenario_id}:{repetition}")
        side_effect_count = result.get("side_effect_count")
        if scenario_id == "crash_recovery" and side_effect_count != 1:
            reasons.append(
                f"not_exactly_once:{scenario_id}:{repetition}:{side_effect_count}"
            )
        if scenario_id == "crash_recovery" and result.get("crash_observed") is not True:
            reasons.append(f"crash_not_observed:{scenario_id}:{repetition}")
    for scenario_id, repetition in sorted(expected - indexed.keys()):
        reasons.append(f"missing:{scenario_id}:{repetition}")
    passed = sum(result.get("status") == "passed" for result in indexed.values())
    return {
        "schema_version": "praxis-public-path-gate-v1",
        "manifest_fingerprint": manifest.fingerprint,
        "model": manifest.model,
        "expected": len(expected),
        "passed": passed,
        "release_ready": not reasons,
        "reasons": reasons,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _seed_workspace(workspace: Path, files: Mapping[str, str]) -> None:
    workspace.mkdir(parents=True, exist_ok=False)
    for relative, content in files.items():
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"scenario file escapes workspace: {relative}")
        target = workspace / candidate
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "benchmark@example.invalid"),
        ("git", "config", "user.name", "Public Path Benchmark"),
        ("git", "add", "-A"),
        ("git", "commit", "-qm", "scenario seed"),
    ):
        subprocess.run(command, cwd=workspace, check=True, capture_output=True)


def _output_field(output: str, label: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(label)}:\s*(.+?)\s*$", output)
    return None if match is None else match.group(1).strip()


def _run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int = 900,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _changed_paths(workspace: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--modified",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    return tuple(
        sorted(
            value.decode("utf-8", errors="surrogateescape")
            for value in completed.stdout.split(b"\0")
            if value
        )
    )


def _resume_until_terminal(
    *,
    initial: subprocess.CompletedProcess[str],
    agent_command: Sequence[str],
    workspace: Path,
    checkpoint: Path,
    allowed_approval_effects: frozenset[str],
) -> tuple[subprocess.CompletedProcess[str], tuple[subprocess.CompletedProcess[str], ...]]:
    attempts: list[subprocess.CompletedProcess[str]] = [initial]
    current = initial
    for _ in range(8):
        if current.returncode != 2:
            return current, tuple(attempts)
        combined = current.stdout + "\n" + current.stderr
        turn_id = _output_field(combined, "Turn") or _output_field(
            combined,
            "待恢复 Turn",
        )
        if turn_id is None:
            return current, tuple(attempts)
        if not _approval_is_allowed(
            checkpoint=checkpoint,
            turn_id=turn_id,
            allowed_effects=allowed_approval_effects,
        ):
            return current, tuple(attempts)
        current = _run_process(
            [
                *agent_command,
                "resume",
                turn_id,
                "--checkpoint-db",
                str(checkpoint),
                "--action",
                "allow_once",
                "--verbose",
            ],
            cwd=workspace,
        )
        attempts.append(current)
    return current, tuple(attempts)


def _approval_is_allowed(
    *,
    checkpoint: Path,
    turn_id: str,
    allowed_effects: frozenset[str],
) -> bool:
    from agent_runtime.harness import RolloutStore

    if not checkpoint.is_file():
        return False
    with RolloutStore(checkpoint) as store:
        pending = tuple(
            interaction
            for interaction in store.list_interactions(turn_id)
            if interaction.kind == "tool_approval"
            and interaction.status == "pending"
            and interaction.operation_id is not None
        )
        if len(pending) != 1:
            return False
        operation_id = pending[0].operation_id
        assert operation_id is not None
        operation = store.read_tool_operation(operation_id)
        effects = frozenset(operation.effects)
        return bool(effects) and effects <= allowed_effects


def _workspace_write_attempt_count(checkpoint: Path) -> int:
    from agent_runtime.harness import RolloutStore

    with RolloutStore(checkpoint) as store:
        return sum(
            operation.attempt_count
            for operation in store.list_tool_operations()
            if "write_workspace" in operation.effects
            and operation.status == "succeeded"
        )


def _run_crash_recovery(
    *,
    scenario: PublicPathScenario,
    model: str,
    agent_command: Sequence[str],
    workspace: Path,
    checkpoint: Path,
    model_session: Path,
    run_directory: Path,
) -> tuple[
    subprocess.CompletedProcess[str],
    tuple[subprocess.CompletedProcess[str], ...],
    int,
    bool,
    str | None,
]:
    turn_file = run_directory / "crashed-turn-id.txt"
    crashed = _run_process(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "crash-child",
            "--workspace",
            str(workspace),
            "--checkpoint-db",
            str(checkpoint),
            "--model-session-path",
            str(model_session),
            "--model",
            model,
            "--prompt",
            scenario.prompt,
            "--turn-file",
            str(turn_file),
        ],
        cwd=workspace,
    )
    if crashed.returncode != 97 or not turn_file.is_file():
        return crashed, (crashed,), 0, False, None
    turn_id = turn_file.read_text(encoding="utf-8").strip()
    recovery = _run_process(
        [
            *agent_command,
            "resume",
            turn_id,
            "--checkpoint-db",
            str(checkpoint),
            "--action",
            "continue",
            "--verbose",
        ],
        cwd=workspace,
    )
    terminal, resumed = _resume_until_terminal(
        initial=recovery,
        agent_command=agent_command,
        workspace=workspace,
        checkpoint=checkpoint,
        allowed_approval_effects=_SCENARIO_APPROVAL_EFFECTS,
    )
    return terminal, (crashed, *resumed), len(resumed) - 1, True, turn_id


def run_scenario(
    scenario: PublicPathScenario,
    *,
    repetition: int,
    model: str,
    agent_command: Sequence[str],
    artifacts_root: Path,
) -> tuple[dict[str, object], Path]:
    if repetition < 1:
        raise ValueError("repetition must be positive")
    if not agent_command:
        raise ValueError("agent_command must not be empty")
    run_directory = (
        artifacts_root.expanduser().resolve()
        / "public-path"
        / scenario.scenario_id
        / f"repeat-{repetition}"
        / uuid4().hex
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    workspace = run_directory / "workspace"
    _seed_workspace(workspace, scenario.files)
    checkpoint = run_directory / "rollout.sqlite3"
    model_session = run_directory / "model-session.json"
    command = [
        *agent_command,
        "run",
        scenario.prompt,
        "--model",
        model,
        "--checkpoint-db",
        str(checkpoint),
        "--model-session-path",
        str(model_session),
        "--max-turns",
        "30",
        "--non-interactive",
        "--require-workspace-change",
        "--verbose",
    ]
    terminal: subprocess.CompletedProcess[str]
    attempts: tuple[subprocess.CompletedProcess[str], ...]
    approval_count: int
    crash_observed: bool
    recovered_turn_id: str | None
    if scenario.mode == "direct":
        command.extend(("--allow-write-tools", "--allow-execute-tools"))
        initial = _run_process(command, cwd=workspace)
        terminal, attempts = _resume_until_terminal(
            initial=initial,
            agent_command=agent_command,
            workspace=workspace,
            checkpoint=checkpoint,
            allowed_approval_effects=_SCENARIO_APPROVAL_EFFECTS,
        )
        approval_count = len(attempts) - 1
        crash_observed = False
        recovered_turn_id = None
    elif scenario.mode == "approval":
        initial = _run_process(command, cwd=workspace)
        terminal, attempts = _resume_until_terminal(
            initial=initial,
            agent_command=agent_command,
            workspace=workspace,
            checkpoint=checkpoint,
            allowed_approval_effects=_SCENARIO_APPROVAL_EFFECTS,
        )
        approval_count = len(attempts) - 1
        crash_observed = False
        recovered_turn_id = None
    else:
        (
            terminal,
            attempts,
            approval_count,
            crash_observed,
            recovered_turn_id,
        ) = _run_crash_recovery(
            scenario=scenario,
            model=model,
            agent_command=agent_command,
            workspace=workspace,
            checkpoint=checkpoint,
            model_session=model_session,
            run_directory=run_directory,
        )
    stdout = "\n\n".join(attempt.stdout for attempt in attempts)
    stderr = "\n\n".join(attempt.stderr for attempt in attempts)
    (run_directory / "agent.stdout").write_text(stdout, encoding="utf-8")
    (run_directory / "agent.stderr").write_text(stderr, encoding="utf-8")
    changed_paths = _changed_paths(workspace)
    acceptance = _run_process(scenario.acceptance_command, cwd=workspace)
    (run_directory / "acceptance.stdout").write_text(
        acceptance.stdout,
        encoding="utf-8",
    )
    (run_directory / "acceptance.stderr").write_text(
        acceptance.stderr,
        encoding="utf-8",
    )
    combined = stdout + "\n" + stderr
    turn_id = recovered_turn_id or _output_field(combined, "Turn")
    expected_changes = set(scenario.required_changes)
    status = (
        "passed"
        if terminal.returncode == 0
        and acceptance.returncode == 0
        and set(changed_paths) == expected_changes
        and (scenario.mode != "approval" or approval_count > 0)
        and (scenario.mode != "crash_after_model_tool_call" or crash_observed)
        else "failed"
    )
    changed_files = [
        {
            "path": relative,
            "sha256": hashlib.sha256((workspace / relative).read_bytes()).hexdigest(),
        }
        for relative in changed_paths
        if (workspace / relative).is_file()
    ]
    result: dict[str, object] = {
        "schema_version": _RESULT_SCHEMA,
        "scenario_id": scenario.scenario_id,
        "repetition": repetition,
        "model": model,
        "status": status,
        "turn_id": turn_id,
        "agent_exit_code": terminal.returncode,
        "acceptance_exit_code": acceptance.returncode,
        "approval_count": approval_count,
        "crash_observed": crash_observed,
        "side_effect_count": (
            _workspace_write_attempt_count(checkpoint)
            if scenario.mode == "crash_after_model_tool_call" and checkpoint.is_file()
            else 1 if status == "passed" else 0
        ),
        "changed_paths": list(changed_paths),
        "changed_files": changed_files,
        "rollout_sha256": (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if checkpoint.is_file()
            else None
        ),
    }
    result_path = run_directory / "result.json"
    _write_json_atomic(result_path, result)
    return result, result_path


def _resolve_agent_command(value: str) -> tuple[str, ...]:
    if value == "current-runtime":
        executable = Path(sys.executable).with_name("agent").resolve()
        if not executable.is_file():
            raise ValueError(f"current runtime agent entrypoint is missing: {executable}")
        return (str(executable),)
    command = tuple(shlex.split(value))
    if not command:
        raise ValueError("agent command must not be empty")
    return command


def run_benchmark(
    manifest: PublicPathManifest,
    *,
    agent_command: Sequence[str],
    artifacts_root: Path,
) -> tuple[dict[str, object], tuple[Path, ...]]:
    results: list[dict[str, object]] = []
    paths: list[Path] = []
    for scenario in manifest.scenarios:
        for repetition in range(1, manifest.repetitions + 1):
            result, path = run_scenario(
                scenario,
                repetition=repetition,
                model=manifest.model,
                agent_command=agent_command,
                artifacts_root=artifacts_root,
            )
            results.append(result)
            paths.append(path)
    return evaluate_results(manifest, results), tuple(paths)


def _write_release_evidence(
    *,
    path: Path,
    artifacts_root: Path,
    manifest: PublicPathManifest,
    summary: Mapping[str, object],
    result_paths: Sequence[Path],
) -> None:
    root = artifacts_root.expanduser().resolve()
    expanded = path.expanduser()
    target = (expanded if expanded.is_absolute() else root / expanded).resolve()
    if not target.is_relative_to(root):
        raise ValueError("public-path evidence escapes artifacts_root")
    records: list[dict[str, str]] = []
    for result_path in result_paths:
        resolved = result_path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("public-path result escapes artifacts_root")
        records.append(
            {
                "path": resolved.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            }
        )
    _write_json_atomic(
        target,
        {
            "schema_version": _EVIDENCE_SCHEMA,
            "manifest_fingerprint": manifest.fingerprint,
            "model": manifest.model,
            "results": records,
            "gate": dict(summary),
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-public-path-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("manifest", type=Path)
    run.add_argument("--artifacts-root", type=Path)
    run.add_argument("--evidence-artifact", type=Path, required=True)
    run.add_argument("--agent-command", default="current-runtime")
    gate = subparsers.add_parser("gate")
    gate.add_argument("manifest", type=Path)
    gate.add_argument("results", nargs="+", type=Path)
    crash = subparsers.add_parser("crash-child")
    crash.add_argument("--workspace", type=Path, required=True)
    crash.add_argument("--checkpoint-db", type=Path, required=True)
    crash.add_argument("--model-session-path", type=Path, required=True)
    crash.add_argument("--model", required=True)
    crash.add_argument("--prompt", required=True)
    crash.add_argument("--turn-file", type=Path, required=True)
    return parser


def _crash_child(args: argparse.Namespace) -> int:
    from agent_runtime.agent import Agent

    def listener(record: object) -> None:
        turn_id = getattr(record, "turn_id", None)
        if isinstance(turn_id, str) and turn_id:
            args.turn_file.parent.mkdir(parents=True, exist_ok=True)
            with args.turn_file.open("w", encoding="utf-8") as stream:
                stream.write(turn_id)
                stream.flush()
                os.fsync(stream.fileno())
        if getattr(record, "record_type", None) != "item_completed":
            return
        payload = getattr(record, "payload", None)
        if not isinstance(payload, Mapping):
            return
        item_payload = payload.get("payload")
        if not isinstance(item_payload, Mapping):
            return
        tool_calls = item_payload.get("tool_calls")
        if (
            isinstance(tool_calls, Sequence)
            and not isinstance(tool_calls, (str, bytes))
            and bool(tool_calls)
        ):
            os._exit(97)

    agent = Agent(
        model=args.model,
        workspace_path=args.workspace,
        checkpoint_db=args.checkpoint_db,
        model_session_path=args.model_session_path,
    )
    asyncio.run(
        agent.arun(
            args.prompt,
            max_turns=30,
            require_workspace_change=True,
            _record_listener=listener,
        )
    )
    return 3


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "crash-child":
        return _crash_child(args)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "validate":
            payload: Mapping[str, object] = {
                "schema_version": _SCHEMA_VERSION,
                "manifest_fingerprint": manifest.fingerprint,
                "scenario_count": len(manifest.scenarios),
                "repetitions": manifest.repetitions,
            }
            exit_code = 0
        elif args.command == "run":
            artifacts_root = args.artifacts_root
            if artifacts_root is None:
                configured = os.environ.get("PRAXIS_ACCEPTANCE_ARTIFACT_ROOT")
                if not configured:
                    raise ValueError(
                        "run requires --artifacts-root or "
                        "PRAXIS_ACCEPTANCE_ARTIFACT_ROOT"
                    )
                artifacts_root = Path(configured)
            summary, result_paths = run_benchmark(
                manifest,
                agent_command=_resolve_agent_command(args.agent_command),
                artifacts_root=artifacts_root,
            )
            _write_release_evidence(
                path=args.evidence_artifact,
                artifacts_root=artifacts_root,
                manifest=manifest,
                summary=summary,
                result_paths=result_paths,
            )
            payload = summary
            exit_code = 0 if summary["release_ready"] else 1
        else:
            results = tuple(
                _mapping(
                    json.loads(path.read_text(encoding="utf-8")),
                    field=f"result:{path}",
                )
                for path in args.results
            )
            payload = evaluate_results(manifest, results)
            exit_code = 0 if payload["release_ready"] else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"public-path benchmark error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
