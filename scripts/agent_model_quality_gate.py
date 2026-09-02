#!/usr/bin/env python
"""Calibrate and gate live-model tool-use quality for the product Agent.

This suite deliberately does not replace deterministic fake/stub tests. It
observes only model-dependent behavior: tool choice, valid arguments, recovery,
approval continuation, duplicate-failure control, and call efficiency.

Examples:
    uv run python scripts/agent_model_quality_gate.py calibrate \
        --env-file /path/to/.env --trials 3
    uv run python scripts/agent_model_quality_gate.py gate \
        --env-file /path/to/.env
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from agent_runtime.result import AgentResult, AgentToolCall

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_PATH = ROOT / "tests" / "agent" / "fixtures" / "model_quality_cases.json"
DEFAULT_BASELINE_PATH = ROOT / "evals" / "model_quality" / "baseline_v1.json"
MIN_CALIBRATION_TRIALS = 3
THRESHOLD_METHOD = "empirical_worst_trial_v1"
EVALUATOR_VERSION = "agent_model_quality_gate_v3"
_RUNTIME_INPUT_FILE_PREFIX = ".praxis/runtime/input_files/"
_MAX_FINAL_PAUSE_REASON_CHARS = 2000
_MAX_FINAL_PAUSE_TOOL_NAMES = 32
_ABSOLUTE_PATH_PATTERN = re.compile(r"(?<![\w.])/(?:[^\s`'\"<>]+)")
_WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![\w.])(?:[A-Za-z]:[\\/]|\\\\)[^\s`'\"<>;,]+"
)
_SENSITIVE_ARTIFACT_KEY_TOKENS = frozenset(
    {
        "apikey",
        "auth",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "env",
        "environment",
        "header",
        "headers",
        "key",
        "password",
        "secret",
        "token",
    }
)
_MODEL_QUALITY_FAILURE_REASONS = frozenset(
    {
        "invalid_model_turn",
        "max_iterations",
        "max_turns",
        "repeated_tool_failure",
    }
)

GATED_METRIC_DIRECTIONS: dict[str, Literal["min", "max"]] = {
    "task_success_rate": "min",
    "file_tool_selection_rate": "min",
    "failure_recovery_rate": "min",
    "approval_continuation_rate": "min",
    "repeated_failure_control_rate": "min",
    "argument_validity_rate": "min",
    "redundant_tool_call_rate": "max",
    "mean_tool_calls_per_case": "max",
    "mean_model_calls_per_case": "max",
}

_VALIDATION_ERROR_CODES = frozenset(
    {
        "invalid_arguments",
        "schema_validation_failed",
        "tool_not_found",
        "unknown_tool",
        "validation_failed",
    }
)


class InfrastructureUnavailableError(RuntimeError):
    """The live sample could not measure model tool-use quality."""


class DirtyRepositoryError(RuntimeError):
    """Live evidence cannot be attributed to a repository with changes."""


class SourceRepositoryMismatchError(RuntimeError):
    """The requested evidence repository is not the running source tree."""


class SourceRepositoryChangedError(RuntimeError):
    """The evidence source changed while live model trials were running."""


@dataclass(frozen=True)
class RepositoryFingerprint:
    source_commit: str
    source_tree: str
    dirty: bool


@dataclass(frozen=True)
class ToolCallEvidence:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, object]
    is_error: bool
    error_code: str | None
    structured_output: object | None = None
    error_message: str | None = None
    retryable: bool = False
    truncated: bool = False
    latency_ms: float | None = None

    @property
    def key(self) -> str:
        arguments = json.dumps(
            self.arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"{self.tool_name}:{arguments}"


@dataclass(frozen=True)
class CaseObservation:
    case_id: str
    capability: str
    status: str
    answer: str | None
    tool_calls: tuple[ToolCallEvidence, ...]
    model_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    tool_schema_bytes: int
    approval_pause_observed: bool
    approval_kind: str | None
    approval_resumes: int
    workspace_assertions_passed: bool
    runtime_input_namespace: str | None = None
    stop_reason: str | None = None
    diagnostic_codes: tuple[str, ...] = ()
    diagnostic_error_types: tuple[str, ...] = ()
    infrastructure_failure: bool = False
    error: str = ""
    final_pause_request_kind: str | None = None
    final_pause_reason: str | None = None
    final_pause_tool_names: tuple[str, ...] = ()

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["tool_calls"] = [asdict(call) for call in self.tool_calls]
        return payload


def load_suite(path: Path = DEFAULT_FIXTURE_PATH) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("model quality suite schema_version must be 1")
    suite_id = payload.get("suite_id")
    models = payload.get("models")
    cases = payload.get("cases")
    if not isinstance(suite_id, str) or not suite_id:
        raise ValueError("model quality suite_id must be non-empty")
    if not isinstance(models, list) or not models or not all(isinstance(item, str) and item for item in models):
        raise ValueError("model quality suite models must be non-empty names")
    if not isinstance(cases, list) or not cases:
        raise ValueError("model quality suite cases must be non-empty")
    ids: set[str] = set()
    for raw in cases:
        if not isinstance(raw, dict):
            raise ValueError("model quality cases must be objects")
        case_id = raw.get("id")
        capability = raw.get("capability")
        task = raw.get("task")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("model quality case id must be non-empty")
        if case_id in ids:
            raise ValueError(f"duplicate model quality case id: {case_id}")
        ids.add(case_id)
        if capability not in {
            "file_tool_selection",
            "failure_recovery",
            "approval_continuation",
            "repeated_failure_control",
        }:
            raise ValueError(f"unsupported model quality capability: {capability}")
        if not isinstance(task, str) or not task:
            raise ValueError(f"case {case_id} task must be non-empty")
        if "thresholds" in raw or "threshold" in raw:
            raise ValueError("quality thresholds belong only in measured baselines")
    return cast(dict[str, object], payload)


def suite_revision(suite: Mapping[str, object]) -> str:
    encoded = json.dumps(
        suite,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "suite_" + hashlib.sha256(encoded).hexdigest()[:20]


def derive_thresholds(
    trial_metrics: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, float | str]]:
    """Build an empirical envelope from repeated live trials.

    Floors are the lowest measured trial and ceilings are the highest measured
    trial. No guessed tolerance is introduced. A later gate uses the same
    number of trials and compares its own worst trial with this envelope.
    """

    if len(trial_metrics) < MIN_CALIBRATION_TRIALS:
        raise ValueError(f"calibration requires at least {MIN_CALIBRATION_TRIALS} real trials")
    required = set(GATED_METRIC_DIRECTIONS)
    for index, metrics in enumerate(trial_metrics, start=1):
        if set(metrics) != required:
            missing = sorted(required - set(metrics))
            extra = sorted(set(metrics) - required)
            raise ValueError(f"trial {index} metric mismatch; missing={missing}, extra={extra}")
        for name, value in metrics.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"trial {index} metric {name} must be finite")

    thresholds: dict[str, dict[str, float | str]] = {}
    for name, direction in GATED_METRIC_DIRECTIONS.items():
        values = [float(metrics[name]) for metrics in trial_metrics]
        thresholds[name] = {
            "direction": direction,
            "value": min(values) if direction == "min" else max(values),
        }
    return thresholds


def validate_baseline(
    baseline: Mapping[str, object],
    *,
    suite: Mapping[str, object] | None = None,
) -> None:
    if baseline.get("schema_version") != 1:
        raise ValueError("model quality baseline schema_version must be 1")
    if baseline.get("threshold_method") != THRESHOLD_METHOD:
        raise ValueError(f"model quality baseline must use {THRESHOLD_METHOD}")
    models = baseline.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ValueError("model quality baseline has no models")
    suite_cases: Sequence[Mapping[str, object]] | None = None
    if suite is not None:
        if baseline.get("suite_revision") != suite_revision(suite):
            raise ValueError("model quality baseline suite revision does not match the current fixture")
        declared_models = set(_string_sequence(suite.get("models"), label="suite models"))
        if set(models) != declared_models:
            raise ValueError("model quality baseline model set does not match suite")
        raw_cases = suite.get("cases")
        if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
            raise ValueError("model quality suite cases are invalid")
        suite_cases = [_mapping(item) for item in raw_cases]
    for alias, raw_entry in models.items():
        if not isinstance(alias, str) or not isinstance(raw_entry, Mapping):
            raise ValueError("model quality baseline model entries are invalid")
        provider_model = raw_entry.get("provider_model")
        trial_count = raw_entry.get("trial_count")
        raw_trials = raw_entry.get("trial_metrics")
        thresholds = raw_entry.get("thresholds")
        if not isinstance(provider_model, str) or not provider_model:
            raise ValueError(f"baseline model {alias} has no provider_model")
        if not isinstance(trial_count, int) or trial_count < MIN_CALIBRATION_TRIALS:
            raise ValueError(f"baseline model {alias} has invalid trial_count")
        if not isinstance(raw_trials, Sequence) or isinstance(raw_trials, (str, bytes)):
            raise ValueError(f"baseline model {alias} has no trial_metrics")
        if len(raw_trials) != trial_count:
            raise ValueError(f"baseline model {alias} trial_count does not match trials")
        trial_metrics = [_float_metric_mapping(item, label=f"baseline model {alias} trial") for item in raw_trials]
        expected = derive_thresholds(trial_metrics)
        if thresholds != expected:
            raise ValueError(f"baseline model {alias} thresholds do not match measured trials")
        if suite_cases is not None:
            _validate_raw_model_trials(
                alias=alias,
                model_entry=raw_entry,
                suite_cases=suite_cases,
                trial_metrics=trial_metrics,
            )


def evaluate_model_gate(
    *,
    model_alias: str,
    provider_model: str,
    trial_metrics: Sequence[Mapping[str, float]],
    baseline: Mapping[str, object],
) -> dict[str, object]:
    expected_provider_model = baseline.get("provider_model")
    if provider_model != expected_provider_model:
        raise ValueError(
            f"model {model_alias} provider identity changed: "
            f"baseline={expected_provider_model!r}, current={provider_model!r}"
        )
    trial_count = baseline.get("trial_count")
    if not isinstance(trial_count, int):
        raise ValueError(f"baseline model {model_alias} has invalid trial_count")
    if len(trial_metrics) != trial_count:
        raise ValueError(f"model {model_alias} gate requires {trial_count} trials, got {len(trial_metrics)}")
    thresholds = baseline.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError(f"baseline model {model_alias} has no thresholds")

    observed: dict[str, float] = {}
    failures: list[str] = []
    for name, direction in GATED_METRIC_DIRECTIONS.items():
        values = [float(metrics[name]) for metrics in trial_metrics]
        value = min(values) if direction == "min" else max(values)
        observed[name] = value
        raw_threshold = thresholds.get(name)
        if not isinstance(raw_threshold, Mapping):
            raise ValueError(f"baseline threshold missing: {model_alias}.{name}")
        threshold = float(raw_threshold["value"])
        if direction == "min" and value + 1e-12 < threshold:
            failures.append(f"{name}: observed {value} < baseline floor {threshold}")
        elif direction == "max" and value - 1e-12 > threshold:
            failures.append(f"{name}: observed {value} > baseline ceiling {threshold}")
    return {
        "model_alias": model_alias,
        "provider_model": provider_model,
        "passed": not failures,
        "observed": observed,
        "thresholds": thresholds,
        "failures": failures,
    }


def score_trial(
    cases: Sequence[Mapping[str, object]],
    observations: Sequence[CaseObservation],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    if len(cases) != len(observations):
        raise ValueError("quality case and observation counts differ")
    _raise_for_infrastructure(observations)
    scored: list[dict[str, object]] = []
    for case, observation in zip(cases, observations, strict=True):
        if observation.case_id != case.get("id"):
            raise ValueError("quality observations are out of case order")
        scored.append(_score_case(case, observation))

    total_cases = len(scored)
    total_calls = sum(cast(int, item["tool_call_count"]) for item in scored)
    valid_calls = sum(cast(int, item["valid_tool_calls"]) for item in scored)
    redundant_calls = sum(cast(int, item["redundant_tool_calls"]) for item in scored)

    def capability_rate(capability: str) -> float:
        selected = [item for item in scored if item["capability"] == capability]
        if not selected:
            raise ValueError(f"quality suite has no {capability} cases")
        return sum(bool(item["capability_passed"]) for item in selected) / len(selected)

    metrics = {
        "task_success_rate": sum(bool(item["passed"]) for item in scored) / total_cases,
        "file_tool_selection_rate": capability_rate("file_tool_selection"),
        "failure_recovery_rate": capability_rate("failure_recovery"),
        "approval_continuation_rate": capability_rate("approval_continuation"),
        "repeated_failure_control_rate": capability_rate("repeated_failure_control"),
        "argument_validity_rate": valid_calls / total_calls if total_calls else 0.0,
        "redundant_tool_call_rate": redundant_calls / total_calls if total_calls else 0.0,
        "mean_tool_calls_per_case": total_calls / total_cases,
        "mean_model_calls_per_case": sum(observation.model_calls for observation in observations) / total_cases,
    }
    return metrics, scored


def informational_metrics(
    observations: Sequence[CaseObservation],
) -> dict[str, float]:
    count = len(observations)
    if not count:
        raise ValueError("quality observations must not be empty")
    return {
        "mean_input_tokens_per_case": sum(item.input_tokens for item in observations) / count,
        "mean_output_tokens_per_case": sum(item.output_tokens for item in observations) / count,
        "mean_latency_ms_per_case": sum(item.latency_ms for item in observations) / count,
        "mean_tool_schema_bytes_per_case": sum(item.tool_schema_bytes for item in observations) / count,
    }


async def run_model_trials(
    *,
    model_alias: str,
    cases: Sequence[Mapping[str, object]],
    trials: int,
    env_file: Path,
) -> dict[str, object]:
    if trials < 1:
        raise ValueError("trials must be positive")

    from agent_runtime.models import ModelControlPlane

    try:
        control_plane = ModelControlPlane.from_env(
            env_path=str(env_file),
            initial_model_id=model_alias,
        )
        spec = control_plane.current_model()
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        return _initialization_inconclusive_report(
            model_alias=model_alias,
            trials=trials,
            error_type=type(exc).__name__,
        )
    trial_payloads: list[dict[str, object]] = []
    trial_metrics: list[dict[str, float]] = []
    for trial_index in range(1, trials + 1):
        observations: list[CaseObservation] = []
        for case_index, case in enumerate(cases, start=1):
            case_id = str(case["id"])
            print(
                f"[{model_alias}] trial {trial_index}/{trials} case {case_index}/{len(cases)} {case_id}",
                file=sys.stderr,
                flush=True,
            )
            observation = await run_live_case(
                model_alias=model_alias,
                control_plane=control_plane,
                case=case,
            )
            observations.append(observation)
            if observation.infrastructure_failure:
                return {
                    "status": "inconclusive",
                    "model_alias": model_alias,
                    "provider": spec.provider,
                    "provider_model": spec.provider_model,
                    "trial_count": trials,
                    "trial_metrics": trial_metrics,
                    "trials": [
                        *trial_payloads,
                        {
                            "trial": trial_index,
                            "metrics": None,
                            "informational_metrics": None,
                            "cases": _partial_case_payloads(
                                cases[: len(observations)],
                                observations,
                            ),
                        },
                    ],
                    "infrastructure_failure": _infrastructure_payload(observation),
                }
        metrics, scored = score_trial(cases, observations)
        trial_metrics.append(metrics)
        trial_payloads.append(
            {
                "trial": trial_index,
                "metrics": metrics,
                "informational_metrics": informational_metrics(observations),
                "cases": [
                    {
                        "observation": observation.payload(),
                        "score": score,
                    }
                    for observation, score in zip(
                        observations,
                        scored,
                        strict=True,
                    )
                ],
            }
        )
    return {
        "status": "completed",
        "model_alias": model_alias,
        "provider": spec.provider,
        "provider_model": spec.provider_model,
        "trial_count": trials,
        "trial_metrics": trial_metrics,
        "trials": trial_payloads,
    }


async def run_live_case(
    *,
    model_alias: str,
    control_plane: object,
    case: Mapping[str, object],
) -> CaseObservation:
    from agent_runtime import Agent

    case_id = str(case["id"])
    capability = str(case["capability"])
    agent: Agent | None = None
    with tempfile.TemporaryDirectory(prefix=f"model_gate_{case_id}_") as raw_root:
        root = Path(raw_root)
        source = root / "source"
        workspace = root / "workspace"
        source.mkdir()
        workspace.mkdir()
        files = _write_source_files(source, _mapping(case.get("workspace_files", {})))
        workspace_assertions = _mapping(case.get("workspace_assertions", {}))
        agent = Agent(
            model=model_alias,
            checkpoint_db=root / "checkpoint.sqlite3",
            workspace_path=workspace,
        )
        agent._model_control_plane = control_plane  # type: ignore[assignment]
        pause_observed = False
        approval_kind: str | None = None
        approval_resumes = 0
        result: AgentResult | None = None
        try:
            result = await agent.run(
                str(case["task"]),
                files=files,
                require_workspace_change=False,
            )
            if bool(case.get("auto_approve", False)) and result.status == "paused":
                pause_observed = True
                current_kind = None if result.pause is None else result.pause.kind
                approval_kind = current_kind
                if _is_declared_apply_patch_approval(
                    case,
                    result.pause,
                    workspace=workspace,
                    turn_id=result.turn_id,
                ):
                    result = await agent.resume(result.turn_id, "allow_once")
                    approval_resumes = 1

            evidence = tuple(_tool_call_evidence(call) for call in result.tool_calls)
            final_pause_kind, final_pause_reason, final_pause_tool_names = (
                _final_pause_evidence(result)
            )
            workspace_ok = _workspace_assertions_pass(
                workspace,
                workspace_assertions,
                turn_id=result.turn_id,
            )
            return CaseObservation(
                case_id=case_id,
                capability=capability,
                status=result.status,
                answer=result.answer,
                tool_calls=evidence,
                model_calls=result.usage.model_calls,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                latency_ms=result.usage.latency_ms,
                tool_schema_bytes=result.usage.tool_schema_bytes,
                approval_pause_observed=pause_observed,
                approval_kind=approval_kind,
                approval_resumes=approval_resumes,
                workspace_assertions_passed=workspace_ok,
                runtime_input_namespace=result.turn_id,
                stop_reason=result.stop_reason,
                diagnostic_codes=tuple(item.code for item in result.diagnostics),
                diagnostic_error_types=tuple(
                    item.error_type for item in result.diagnostics if item.error_type is not None
                ),
                infrastructure_failure=is_infrastructure_failure(
                    result.status,
                    result.stop_reason,
                ),
                final_pause_request_kind=final_pause_kind,
                final_pause_reason=final_pause_reason,
                final_pause_tool_names=final_pause_tool_names,
            )
        except Exception as exc:
            return CaseObservation(
                case_id=case_id,
                capability=capability,
                status="error",
                answer=None,
                tool_calls=(),
                model_calls=0,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0.0,
                tool_schema_bytes=0,
                approval_pause_observed=pause_observed,
                approval_kind=approval_kind,
                approval_resumes=approval_resumes,
                workspace_assertions_passed=False,
                runtime_input_namespace=(
                    None if result is None else result.turn_id
                ),
                diagnostic_codes=(type(exc).__name__,),
                diagnostic_error_types=(type(exc).__name__,),
                infrastructure_failure=True,
                error=_safe_error(exc),
            )
        finally:
            store = getattr(agent, "_turn_store", None)
            if store is not None:
                store.close()


def _final_pause_evidence(
    result: AgentResult,
) -> tuple[str | None, str | None, tuple[str, ...]]:
    if result.status != "paused":
        return None, None, ()
    reason = result.needs_user_input
    pause = result.pause
    return (
        None if pause is None else pause.kind,
        None if reason is None else reason[:_MAX_FINAL_PAUSE_REASON_CHARS],
        (
            ()
            if pause is None
            else tuple(
                item.tool_name
                for item in pause.tool_calls[:_MAX_FINAL_PAUSE_TOOL_NAMES]
            )
        ),
    )


def _is_declared_apply_patch_approval(
    case: Mapping[str, object],
    pause: object | None,
    *,
    workspace: Path,
    turn_id: str,
) -> bool:
    """Authorize one declared patch in the disposable benchmark workspace."""

    if pause is None or getattr(pause, "kind", None) != "tool_approval":
        return False
    if str(case.get("capability", "")) != "approval_continuation":
        return False
    expected_calls = case.get("expected_tool_calls", ())
    if (
        not isinstance(expected_calls, Sequence)
        or isinstance(expected_calls, (str, bytes))
        or len(expected_calls) != 1
    ):
        return False
    expected_call = _mapping(expected_calls[0])
    if expected_call.get("tool_name") != "apply_patch":
        return False
    arguments = _mapping(expected_call.get("arguments"))
    expected_path = arguments.get("file_path")
    if not isinstance(expected_path, str) or not expected_path.startswith("input_files/"):
        return False
    relative_parts = expected_path.removeprefix("input_files/").split("/")
    if not relative_parts or any(part in {"", ".", ".."} for part in relative_parts):
        return False
    tool_calls = tuple(getattr(pause, "tool_calls", ()))
    if len(tool_calls) != 1 or getattr(tool_calls[0], "tool_name", None) != "apply_patch":
        return False
    context = getattr(pause, "context", {})
    if not isinstance(context, Mapping):
        return False
    workspace_target = context.get("workspace_path")
    if not isinstance(workspace_target, str):
        return False
    expected_target = (
        workspace
        / _RUNTIME_INPUT_FILE_PREFIX
        / turn_id
        / Path(*relative_parts)
    ).resolve()
    return bool(
        context.get("approval_scope") == "tool"
        and context.get("network_requested") is False
        and context.get("workspace_write") is True
        and Path(workspace_target).resolve() == expected_target
    )


def build_baseline(
    *,
    suite: Mapping[str, object],
    model_reports: Sequence[Mapping[str, object]],
    fingerprint: RepositoryFingerprint,
) -> dict[str, object]:
    models: dict[str, object] = {}
    for report in model_reports:
        alias = str(report["model_alias"])
        trial_metrics = [
            _float_metric_mapping(item, label=f"model {alias} trial")
            for item in cast(Sequence[object], report["trial_metrics"])
        ]
        models[alias] = {
            "provider": report["provider"],
            "provider_model": report["provider_model"],
            "trial_count": report["trial_count"],
            "trial_metrics": trial_metrics,
            "thresholds": derive_thresholds(trial_metrics),
            "trials": report["trials"],
        }
    baseline = {
        "schema_version": 1,
        "suite_id": suite["suite_id"],
        "suite_revision": suite_revision(suite),
        "measured_at": datetime.now(UTC).isoformat(),
        "git_commit": fingerprint.source_commit,
        "source_commit": fingerprint.source_commit,
        "source_tree": fingerprint.source_tree,
        "dirty": fingerprint.dirty,
        "evaluator_version": EVALUATOR_VERSION,
        "threshold_method": THRESHOLD_METHOD,
        "models": models,
    }
    validate_baseline(baseline, suite=suite)
    return baseline


async def _calibrate(args: argparse.Namespace) -> int:
    if args.trials < MIN_CALIBRATION_TRIALS:
        raise ValueError(f"calibration requires at least {MIN_CALIBRATION_TRIALS} real trials")
    fingerprint = source_repository_fingerprint(args.repository)
    suite = load_suite(args.fixture)
    models = _selected_models(suite, None)
    cases = cast(list[Mapping[str, object]], suite["cases"])
    reports: list[dict[str, object]] = []
    infrastructure_failure: Mapping[str, object] | None = None
    for model in models:
        report = await run_model_trials(
            model_alias=model,
            cases=cases,
            trials=args.trials,
            env_file=args.env_file,
        )
        reports.append(report)
        if report.get("status") == "inconclusive":
            infrastructure_failure = report
            break
    _verify_source_repository_unchanged(args.repository, fingerprint)
    if infrastructure_failure is not None:
        raise InfrastructureUnavailableError(
            _infrastructure_message(infrastructure_failure)
        )
    baseline = build_baseline(
        suite=suite,
        model_reports=reports,
        fingerprint=fingerprint,
    )
    args.baseline.parent.mkdir(parents=True, exist_ok=True)
    args.baseline.write_text(_pretty_json(baseline), encoding="utf-8")
    print(f"wrote measured baseline: {args.baseline}")
    for alias, entry in cast(Mapping[str, Mapping[str, object]], baseline["models"]).items():
        print(f"{alias}: {json.dumps(entry['thresholds'], sort_keys=True)}")
    return 0


async def _gate(args: argparse.Namespace) -> int:
    fingerprint = source_repository_fingerprint(args.repository)
    suite = load_suite(args.fixture)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    validate_baseline(baseline, suite=suite)
    models = _selected_models(suite, args.models)
    baseline_models = baseline["models"]
    if not isinstance(baseline_models, Mapping):
        raise ValueError("model quality baseline has no models")
    cases = cast(list[Mapping[str, object]], suite["cases"])
    reports: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    inconclusive = False
    for model in models:
        raw_model_baseline = baseline_models.get(model)
        if not isinstance(raw_model_baseline, Mapping):
            raise KeyError(f"model quality baseline missing model: {model}")
        trial_count = int(raw_model_baseline["trial_count"])
        report = await run_model_trials(
            model_alias=model,
            cases=cases,
            trials=trial_count,
            env_file=args.env_file,
        )
        reports.append(report)
        if report.get("status") == "inconclusive":
            results.append(
                {
                    "model_alias": model,
                    "provider_model": report["provider_model"],
                    "status": "inconclusive",
                    "passed": None,
                    "observed": {},
                    "thresholds": {},
                    "failures": [],
                    "infrastructure_failure": report["infrastructure_failure"],
                }
            )
            inconclusive = True
            break
        current_metrics = [
            _float_metric_mapping(item, label=f"model {model} trial")
            for item in cast(Sequence[object], report["trial_metrics"])
        ]
        results.append(
            evaluate_model_gate(
                model_alias=model,
                provider_model=str(report["provider_model"]),
                trial_metrics=current_metrics,
                baseline=raw_model_baseline,
            )
        )
    _verify_source_repository_unchanged(args.repository, fingerprint)
    passed = None if inconclusive else all(bool(item["passed"]) for item in results)
    status = "inconclusive" if inconclusive else ("passed" if passed else "failed")
    gate_report = {
        "schema_version": 1,
        "status": status,
        "source_commit": fingerprint.source_commit,
        "source_tree": fingerprint.source_tree,
        "source_unchanged": True,
        "dirty": fingerprint.dirty,
        "runtime_platform": runtime_platform_metadata(),
        "suite_id": suite["suite_id"],
        "suite_revision": suite_revision(suite),
        "evaluator_version": EVALUATOR_VERSION,
        "case_metadata": _case_metadata(suite),
        "baseline_path": _redacted_artifact_path(
            args.baseline,
            repository=args.repository,
        ),
        "measured_at": datetime.now(UTC).isoformat(),
        "passed": passed,
        "models": results,
        "runs": reports,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            _pretty_json(_sanitize_artifact(gate_report)),
            encoding="utf-8",
        )
        print(f"wrote gate report: {args.report}")
    for result in results:
        marker = (
            "INCONCLUSIVE"
            if result["passed"] is None
            else ("PASS" if result["passed"] else "FAIL")
        )
        print(f"{marker} {result['model_alias']}")
        for failure in cast(Sequence[str], result["failures"]):
            print(f"  {failure}")
    if inconclusive:
        return 2
    return 0 if passed else 1


def _score_case(
    case: Mapping[str, object],
    observation: CaseObservation,
) -> dict[str, object]:
    calls = observation.tool_calls
    names = tuple(call.tool_name for call in calls)
    expected_answer = case.get("expected_answer_contains")
    answer_ok = (
        True
        if not isinstance(expected_answer, str)
        else expected_answer.casefold() in (observation.answer or "").casefold()
    )
    core_success = (
        observation.status == "done" and answer_ok and observation.workspace_assertions_passed and not observation.error
    )
    expected_first = case.get("expected_first_tool")
    first_ok = True if not isinstance(expected_first, str) else bool(names) and names[0] == expected_first
    expected_sequence = _string_sequence(
        case.get("expected_tool_sequence", ()),
        label="expected_tool_sequence",
    )
    expected_calls = _call_spec_sequence(
        case.get("expected_tool_calls", ()),
        label="expected_tool_calls",
    )
    sequence_ok = (
        _contains_ordered_calls(
            calls,
            expected_calls,
            runtime_input_namespace=observation.runtime_input_namespace,
        )
        if expected_calls
        else _contains_ordered_sequence(names, expected_sequence)
    )
    invalid_calls = sum(call.error_code in _VALIDATION_ERROR_CODES for call in calls)
    failed_duplicates = Counter(call.key for call in calls if call.is_error)
    redundant = sum(max(count - 1, 0) for count in failed_duplicates.values())
    capability = str(case["capability"])

    if capability == "file_tool_selection":
        require_first = case.get("require_first_tool", False)
        if not isinstance(require_first, bool):
            raise ValueError("require_first_tool must be a boolean")
        capability_passed = (first_ok or not require_first) and sequence_ok and invalid_calls == 0
    elif capability == "failure_recovery":
        capability_passed = _failure_recovered(
            case,
            calls,
            runtime_input_namespace=observation.runtime_input_namespace,
        )
    elif capability == "approval_continuation":
        if expected_calls:
            approval_spec = expected_calls[0]
            approved_calls = [
                call
                for call in calls
                if _call_matches(
                    call,
                    approval_spec,
                    runtime_input_namespace=observation.runtime_input_namespace,
                )
            ]
        else:
            approval_tool = expected_sequence[0] if expected_sequence else str(expected_first or "")
            approved_calls = [call for call in calls if call.tool_name == approval_tool]
        capability_passed = (
            observation.approval_pause_observed
            and observation.approval_kind == "tool_approval"
            and observation.approval_resumes >= 1
            and len(approved_calls) == 1
            and not approved_calls[0].is_error
        )
    elif capability == "repeated_failure_control":
        failed_counts = Counter(call.key for call in calls if call.is_error)
        max_failed = max(failed_counts.values(), default=0)
        allowed = _integer(
            case.get("max_identical_failed_calls", 1),
            label="max_identical_failed_calls",
        )
        expected_failed = _optional_call_spec(
            case.get("expected_failed_call"),
            label="expected_failed_call",
        )
        failure_matched = (
            bool(calls)
            and calls[0].is_error
            and (
                expected_failed is None
                or _call_matches(
                    calls[0],
                    expected_failed,
                    runtime_input_namespace=observation.runtime_input_namespace,
                )
            )
        )
        capability_passed = first_ok and failure_matched and max_failed <= allowed and len(calls) == 1
    else:  # pragma: no cover - load_suite rejects this first
        raise ValueError(f"unsupported model quality capability: {capability}")

    return {
        "case_id": observation.case_id,
        "capability": capability,
        "passed": core_success and capability_passed,
        "core_success": core_success,
        "capability_passed": capability_passed,
        "first_tool_matched": first_ok,
        "expected_sequence_matched": sequence_ok,
        "valid_tool_calls": len(calls) - invalid_calls,
        "invalid_tool_calls": invalid_calls,
        "redundant_tool_calls": redundant,
        "tool_call_count": len(calls),
        "model_call_count": observation.model_calls,
    }


def is_infrastructure_failure(
    status: str,
    stop_reason: str | None,
) -> bool:
    if status == "error":
        return True
    if status != "failed":
        return False
    return stop_reason not in _MODEL_QUALITY_FAILURE_REASONS


def _raise_for_infrastructure(
    observations: Sequence[CaseObservation],
    *,
    model_alias: str = "model",
) -> None:
    unavailable = [item for item in observations if item.infrastructure_failure]
    if not unavailable:
        return
    details = []
    for item in unavailable:
        error_types = ",".join(item.diagnostic_error_types) or "unknown"
        details.append(f"{item.case_id}(stop_reason={item.stop_reason or 'unknown'}, error_types={error_types})")
    raise InfrastructureUnavailableError(
        f"{model_alias} live quality sample is infrastructure-inconclusive: " + "; ".join(details)
    )


def _partial_case_payloads(
    cases: Sequence[Mapping[str, object]],
    observations: Sequence[CaseObservation],
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for case, observation in zip(cases, observations, strict=True):
        score = (
            {
                "case_id": observation.case_id,
                "capability": observation.capability,
                "passed": None,
                "core_success": None,
                "capability_passed": None,
                "inconclusive": True,
            }
            if observation.infrastructure_failure
            else _score_case(case, observation)
        )
        payloads.append(
            {
                "observation": observation.payload(),
                "score": score,
            }
        )
    return payloads


def _initialization_inconclusive_report(
    *,
    model_alias: str,
    trials: int,
    error_type: str,
) -> dict[str, object]:
    failure = {
        "case_id": None,
        "stage": "model_control_plane_initialization",
        "stop_reason": "model_control_plane_initialization_failed",
        "diagnostic_error_types": [error_type],
    }
    return {
        "status": "inconclusive",
        "model_alias": model_alias,
        "provider": "unavailable",
        "provider_model": "unavailable",
        "trial_count": trials,
        "trial_metrics": [],
        "trials": [],
        "infrastructure_failure": failure,
    }


def _infrastructure_payload(observation: CaseObservation) -> dict[str, object]:
    return {
        "case_id": observation.case_id,
        "stop_reason": observation.stop_reason,
        "diagnostic_error_types": list(observation.diagnostic_error_types),
    }


def _infrastructure_message(report: Mapping[str, object]) -> str:
    failure = _mapping(report.get("infrastructure_failure"))
    error_types = ",".join(
        _string_sequence(
            failure.get("diagnostic_error_types", ()),
            label="diagnostic_error_types",
        )
    ) or "unknown"
    location = failure.get("case_id") or failure.get("stage") or "unknown"
    return (
        f"{report.get('model_alias', 'model')} live quality sample is infrastructure-inconclusive: "
        f"{location}"
        f"(stop_reason={failure.get('stop_reason') or 'unknown'}, error_types={error_types})"
    )


def _failure_recovered(
    case: Mapping[str, object],
    calls: Sequence[ToolCallEvidence],
    *,
    runtime_input_namespace: str | None = None,
) -> bool:
    expected_failed = _optional_call_spec(
        case.get("expected_failed_call"),
        label="expected_failed_call",
    )
    expected_recovery = _optional_call_spec(
        case.get("expected_recovery_call"),
        label="expected_recovery_call",
    )
    if expected_failed is not None and expected_recovery is not None:
        failure_index = next(
            (
                index
                for index, call in enumerate(calls)
                if call.is_error
                and _call_matches(
                    call,
                    expected_failed,
                    runtime_input_namespace=runtime_input_namespace,
                )
            ),
            None,
        )
        if failure_index is None:
            return False
        return any(
            not call.is_error
            and _call_matches(
                call,
                expected_recovery,
                runtime_input_namespace=runtime_input_namespace,
            )
            for call in calls[failure_index + 1 :]
        )

    intentional_tool = case.get("intentional_error_tool")
    if not isinstance(intentional_tool, str) or not calls:
        return False
    failure_index = next(
        (index for index, call in enumerate(calls) if call.tool_name == intentional_tool and call.is_error),
        None,
    )
    if failure_index is None:
        return False
    failed = calls[failure_index]
    recovery_tools = set(_string_sequence(case.get("recovery_tools", ()), label="recovery_tools"))
    return any(
        call.key != failed.key and call.tool_name in recovery_tools and not call.is_error
        for call in calls[failure_index + 1 :]
    )


def _write_source_files(
    root: Path,
    files: Mapping[str, object],
) -> list[str]:
    paths: list[str] = []
    for relative, raw_content in files.items():
        if not isinstance(raw_content, str):
            raise ValueError(f"workspace file {relative} content must be text")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw_content, encoding="utf-8")
        paths.append(str(path))
    return paths


def _mutable_json_mapping(
    value: Mapping[str, object],
) -> dict[str, object]:
    return {key: _mutable_json_value(item) for key, item in value.items()}


def _mutable_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _mutable_json_mapping({str(key): item for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_mutable_json_value(item) for item in value]
    return value


def _tool_call_evidence(call: AgentToolCall) -> ToolCallEvidence:
    structured_output = (
        None
        if call.structured_output is None
        else _mutable_json_value(call.structured_output)
    )
    return ToolCallEvidence(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        arguments=_mutable_json_mapping(call.arguments or {}),
        is_error=call.is_error,
        error_code=call.error_code,
        structured_output=structured_output,
        error_message=call.error_message,
        retryable=call.retryable,
        truncated=call.truncated,
        latency_ms=call.latency_ms,
    )


def _workspace_assertions_pass(
    workspace: Path,
    assertions: Mapping[str, object],
    *,
    turn_id: str | None = None,
) -> bool:
    for relative, expected in assertions.items():
        if not isinstance(expected, str):
            return False
        path = workspace / relative
        if turn_id is not None and relative.startswith("input_files/"):
            path = (
                workspace
                / _RUNTIME_INPUT_FILE_PREFIX
                / turn_id
                / relative.removeprefix("input_files/")
            )
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            return False
    return True


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("model quality fixture mapping is invalid")
    return cast(Mapping[str, object], value)


def _string_sequence(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return tuple(str(item) for item in value)


def _integer(value: object, *, label: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _contains_ordered_sequence(
    values: Sequence[str],
    expected: Sequence[str],
) -> bool:
    if not expected:
        return True
    expected_index = 0
    for value in values:
        if value != expected[expected_index]:
            continue
        expected_index += 1
        if expected_index == len(expected):
            return True
    return False


def _call_spec_sequence(
    value: object,
    *,
    label: str,
) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return tuple(_call_spec(item, label=label) for item in value)


def _optional_call_spec(
    value: object,
    *,
    label: str,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    return _call_spec(value, label=label)


def _call_spec(value: object, *, label: str) -> Mapping[str, object]:
    spec = _mapping(value)
    tool_name = spec.get("tool_name")
    arguments = spec.get("arguments")
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError(f"{label} tool_name must be non-empty")
    if not isinstance(arguments, Mapping):
        raise ValueError(f"{label} arguments must be an object")
    return spec


def _contains_ordered_calls(
    calls: Sequence[ToolCallEvidence],
    expected: Sequence[Mapping[str, object]],
    *,
    runtime_input_namespace: str | None = None,
) -> bool:
    if not expected:
        return True
    expected_index = 0
    for call in calls:
        if call.is_error or not _call_matches(
            call,
            expected[expected_index],
            runtime_input_namespace=runtime_input_namespace,
        ):
            continue
        expected_index += 1
        if expected_index == len(expected):
            return True
    return False


def _call_matches(
    call: ToolCallEvidence,
    spec: Mapping[str, object],
    *,
    runtime_input_namespace: str | None = None,
) -> bool:
    if call.tool_name != spec.get("tool_name"):
        return False
    arguments = _mapping(spec.get("arguments"))
    return all(
        _argument_matches(
            name=name,
            actual=call.arguments.get(name),
            expected=value,
            runtime_input_namespace=runtime_input_namespace,
        )
        for name, value in arguments.items()
    )


def _argument_matches(
    *,
    name: str,
    actual: object,
    expected: object,
    runtime_input_namespace: str | None,
) -> bool:
    if actual == expected:
        return True
    if (
        name not in {"path", "file_path"}
        or not isinstance(actual, str)
        or not isinstance(expected, str)
        or not expected.startswith("input_files/")
        or not actual.startswith(_RUNTIME_INPUT_FILE_PREFIX)
        or runtime_input_namespace is None
    ):
        return False
    runtime_parts = actual.removeprefix(_RUNTIME_INPUT_FILE_PREFIX).split("/")
    if len(runtime_parts) < 2 or any(
        part in {"", ".", ".."} for part in runtime_parts
    ):
        return False
    if runtime_parts[0] != runtime_input_namespace:
        return False
    logical_path = "input_files/" + "/".join(runtime_parts[1:])
    return logical_path == expected


def _float_metric_mapping(value: object, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} metrics must be an object")
    return {str(key): float(item) for key, item in value.items()}


def _validate_raw_model_trials(
    *,
    alias: str,
    model_entry: Mapping[str, object],
    suite_cases: Sequence[Mapping[str, object]],
    trial_metrics: Sequence[Mapping[str, float]],
) -> None:
    raw_trials = model_entry.get("trials")
    if not isinstance(raw_trials, Sequence) or isinstance(raw_trials, (str, bytes)):
        raise ValueError(f"baseline model {alias} has no raw trials")
    if len(raw_trials) != len(trial_metrics):
        raise ValueError(f"baseline model {alias} raw trial count does not match")
    for index, (raw_trial, stored_metrics) in enumerate(
        zip(raw_trials, trial_metrics, strict=True),
        start=1,
    ):
        trial = _mapping(raw_trial)
        raw_cases = trial.get("cases")
        if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
            raise ValueError(f"baseline model {alias} trial {index} has no raw cases")
        observations: list[CaseObservation] = []
        stored_scores: list[Mapping[str, object]] = []
        for raw_case in raw_cases:
            case_payload = _mapping(raw_case)
            observations.append(_observation_from_payload(case_payload.get("observation")))
            stored_scores.append(_mapping(case_payload.get("score")))
        recomputed_metrics, recomputed_scores = score_trial(
            suite_cases,
            observations,
        )
        trial_payload_metrics = _float_metric_mapping(
            trial.get("metrics"),
            label=f"baseline model {alias} trial {index}",
        )
        if (
            recomputed_metrics != stored_metrics
            or trial_payload_metrics != stored_metrics
            or recomputed_scores != stored_scores
        ):
            raise ValueError(f"baseline model {alias} trial {index} metrics does not match raw observations")


def _observation_from_payload(value: object) -> CaseObservation:
    payload = _mapping(value)
    raw_calls = payload.get("tool_calls")
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
        raise ValueError("raw observation tool_calls must be a sequence")
    calls: list[ToolCallEvidence] = []
    for raw_call in raw_calls:
        call = _mapping(raw_call)
        arguments = _mapping(call.get("arguments"))
        is_error = call.get("is_error")
        if not isinstance(is_error, bool):
            raise ValueError("raw observation tool call is_error must be boolean")
        error_code = call.get("error_code")
        if error_code is not None and not isinstance(error_code, str):
            raise ValueError("raw observation tool call error_code must be text")
        error_message = call.get("error_message")
        if error_message is not None and not isinstance(error_message, str):
            raise ValueError("raw observation tool call error_message must be text")
        retryable = call.get("retryable", False)
        if not isinstance(retryable, bool):
            raise ValueError("raw observation tool call retryable must be boolean")
        truncated = call.get("truncated", False)
        if not isinstance(truncated, bool):
            raise ValueError("raw observation tool call truncated must be boolean")
        latency_ms = call.get("latency_ms")
        if latency_ms is not None and (
            isinstance(latency_ms, bool)
            or not isinstance(latency_ms, (int, float))
            or latency_ms < 0
        ):
            raise ValueError(
                "raw observation tool call latency_ms must be null or non-negative numeric"
            )
        calls.append(
            ToolCallEvidence(
                tool_call_id=str(call.get("tool_call_id", "")),
                tool_name=str(call.get("tool_name", "")),
                arguments=dict(arguments),
                is_error=is_error,
                error_code=error_code,
                structured_output=_mutable_json_value(
                    call.get("structured_output")
                ),
                error_message=error_message,
                retryable=retryable,
                truncated=truncated,
                latency_ms=(
                    None if latency_ms is None else float(latency_ms)
                ),
            )
        )

    def text_or_none(name: str) -> str | None:
        item = payload.get(name)
        if item is None or isinstance(item, str):
            return item
        raise ValueError(f"raw observation {name} must be text or null")

    def boolean(name: str) -> bool:
        item = payload.get(name)
        if not isinstance(item, bool):
            raise ValueError(f"raw observation {name} must be boolean")
        return item

    def number(name: str) -> float:
        item = payload.get(name)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"raw observation {name} must be numeric")
        return float(item)

    return CaseObservation(
        case_id=str(payload.get("case_id", "")),
        capability=str(payload.get("capability", "")),
        status=str(payload.get("status", "")),
        answer=text_or_none("answer"),
        tool_calls=tuple(calls),
        model_calls=_integer(payload.get("model_calls"), label="model_calls"),
        input_tokens=_integer(payload.get("input_tokens"), label="input_tokens"),
        output_tokens=_integer(payload.get("output_tokens"), label="output_tokens"),
        latency_ms=number("latency_ms"),
        tool_schema_bytes=_integer(
            payload.get("tool_schema_bytes"),
            label="tool_schema_bytes",
        ),
        approval_pause_observed=boolean("approval_pause_observed"),
        approval_kind=text_or_none("approval_kind"),
        approval_resumes=_integer(
            payload.get("approval_resumes"),
            label="approval_resumes",
        ),
        workspace_assertions_passed=boolean("workspace_assertions_passed"),
        runtime_input_namespace=text_or_none("runtime_input_namespace"),
        stop_reason=text_or_none("stop_reason"),
        diagnostic_codes=_string_sequence(
            payload.get("diagnostic_codes", ()),
            label="diagnostic_codes",
        ),
        diagnostic_error_types=_string_sequence(
            payload.get("diagnostic_error_types", ()),
            label="diagnostic_error_types",
        ),
        infrastructure_failure=(boolean("infrastructure_failure") if "infrastructure_failure" in payload else False),
        error=str(payload.get("error", "")),
        final_pause_request_kind=text_or_none("final_pause_request_kind"),
        final_pause_reason=text_or_none("final_pause_reason"),
        final_pause_tool_names=_string_sequence(
            payload.get("final_pause_tool_names", ()),
            label="final_pause_tool_names",
        ),
    )


def _selected_models(
    suite: Mapping[str, object],
    selected: Sequence[str] | None,
) -> tuple[str, ...]:
    declared = tuple(str(item) for item in cast(Sequence[object], suite["models"]))
    models = tuple(selected) if selected else declared
    unknown = set(models) - set(declared)
    if unknown:
        raise ValueError(f"models are not declared by the quality suite: {sorted(unknown)}")
    return models


def _safe_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"[:2000]
    for name, value in os.environ.items():
        if not _is_sensitive_artifact_key(name):
            continue
        if len(value) >= 8:
            text = text.replace(value, "[REDACTED]")
    return text


def source_repository_fingerprint(repository: Path) -> RepositoryFingerprint:
    requested = repository.resolve()
    source_root = ROOT.resolve()
    git_root = Path(
        _git_text(repository, "rev-parse", "--show-toplevel")
    ).resolve()
    runtime_source = (source_root / "agent_runtime").resolve()
    runtime_spec = importlib.util.find_spec("agent_runtime")
    runtime_origin = (
        None
        if runtime_spec is None or runtime_spec.origin is None
        else Path(runtime_spec.origin).resolve()
    )
    if (
        requested != source_root
        or git_root != source_root
        or not runtime_source.is_dir()
        or not runtime_source.is_relative_to(source_root)
        or runtime_origin is None
        or not runtime_origin.is_relative_to(runtime_source)
    ):
        raise SourceRepositoryMismatchError(
            "model quality evidence repository must be the running source tree"
        )
    return repository_fingerprint(source_root)


def _verify_source_repository_unchanged(
    repository: Path,
    initial: RepositoryFingerprint,
) -> None:
    current = source_repository_fingerprint(repository)
    if (
        current.dirty
        or current.source_commit != initial.source_commit
        or current.source_tree != initial.source_tree
    ):
        raise SourceRepositoryChangedError(
            "model quality evidence source changed while trials were running"
        )


def repository_fingerprint(repository: Path) -> RepositoryFingerprint:
    source_commit = _git_text(repository, "rev-parse", "HEAD")
    source_tree = _git_text(repository, "rev-parse", "HEAD^{tree}")
    status = _git_text(
        repository,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if status:
        raise DirtyRepositoryError("model quality evidence requires a clean repository")
    return RepositoryFingerprint(
        source_commit=source_commit,
        source_tree=source_tree,
        dirty=False,
    )


def runtime_platform_metadata() -> dict[str, str]:
    """Return the safe, path-free runtime identity committed with gate evidence."""

    return {
        "os": platform.system() or "unknown",
        "os_release": platform.release() or "unknown",
        "architecture": platform.machine() or "unknown",
        "python_version": platform.python_version() or "unknown",
        "python_implementation": platform.python_implementation() or "unknown",
    }


def _git_text(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown git error"
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _case_metadata(suite: Mapping[str, object]) -> list[dict[str, object]]:
    raw_cases = suite.get("cases")
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
        raise ValueError("model quality suite cases are invalid")
    metadata: list[dict[str, object]] = []
    for raw_case in raw_cases:
        case = _mapping(raw_case)
        before = {
            _input_file_path(str(path)): content
            for path, content in _mapping(case.get("workspace_files", {})).items()
        }
        after = dict(_mapping(case.get("workspace_assertions", {})))
        metadata.append(
            {
                "case_id": str(case["id"]),
                "capability": str(case["capability"]),
                "task": str(case["task"]),
                "workspace_before": before,
                "workspace_after": after,
            }
        )
    return metadata


def _input_file_path(path: str) -> str:
    return path if path.startswith("input_files/") else f"input_files/{path}"


def _redacted_artifact_path(path: Path, *, repository: Path) -> str:
    try:
        return path.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError:
        return path.name


def _sanitize_artifact(value: object) -> object:
    secrets = _artifact_secret_values()
    return _sanitize_artifact_value(value, secrets=secrets)


def _sanitize_artifact_value(
    value: object,
    *,
    secrets: Sequence[str],
) -> object:
    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if _is_sensitive_artifact_key(key):
                continue
            sanitized[key] = _sanitize_artifact_value(item, secrets=secrets)
        return sanitized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_sanitize_artifact_value(item, secrets=secrets) for item in value]
    if isinstance(value, str):
        sanitized_text = value
        for secret in secrets:
            sanitized_text = sanitized_text.replace(secret, "[REDACTED]")
        sanitized_text = _WINDOWS_ABSOLUTE_PATH_PATTERN.sub(
            "[REDACTED_ABSOLUTE_PATH]",
            sanitized_text,
        )
        return _ABSOLUTE_PATH_PATTERN.sub(
            "[REDACTED_ABSOLUTE_PATH]",
            sanitized_text,
        )
    return value


def _is_sensitive_artifact_key(key: str) -> bool:
    camel_split = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", camel_split)
    tokens = {
        token.casefold()
        for token in re.split(r"[^A-Za-z0-9]+", normalized)
        if token
    }
    return bool(tokens & _SENSITIVE_ARTIFACT_KEY_TOKENS)


def _artifact_secret_values() -> tuple[str, ...]:
    values: list[str] = []
    for name, value in os.environ.items():
        if not _is_sensitive_artifact_key(name):
            continue
        if len(value) >= 8:
            values.append(value)
    return tuple(values)


def _pretty_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="Run repeated live trials and write a measured baseline.",
    )
    gate = subparsers.add_parser(
        "gate",
        help="Run the baseline trial count and fail on model-quality regression.",
    )
    for command in (calibrate, gate):
        command.add_argument(
            "--repository",
            type=Path,
            default=ROOT,
            help="Clean Git repository whose committed source is being evaluated.",
        )
        command.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)
        command.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
        command.add_argument(
            "--env-file",
            type=Path,
            default=Path(".env"),
            help="Load provider credentials without copying them into artifacts.",
        )
    calibrate.add_argument("--trials", type=int, default=MIN_CALIBRATION_TRIALS)
    gate.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Run one declared model alias; repeat for multiple models.",
    )
    gate.add_argument(
        "--report",
        type=Path,
        help="Optional path for the current raw gate report.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "calibrate":
            return asyncio.run(_calibrate(args))
        return asyncio.run(_gate(args))
    except InfrastructureUnavailableError as exc:
        print(f"INCONCLUSIVE infrastructure: {_safe_error(exc)}", file=sys.stderr)
        return 2
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"model quality gate error: {_safe_error(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
