#!/usr/bin/env python
"""Render redacted live-model gate evidence as human-readable Markdown."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_PATH = ROOT / "docs" / "benchmark.md"
DEFAULT_RUN_RECORD_PATH = ROOT / "docs" / "runs" / "groq-gpt-oss-120b.md"
_SENSITIVE_KEY_TOKENS = frozenset(
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


def render_model_quality_report(
    report_path: Path,
    *,
    benchmark_path: Path,
    run_record_path: Path,
) -> None:
    report = _mapping(
        json.loads(report_path.read_text(encoding="utf-8")),
        label="gate report",
    )
    _validate_report(report)
    benchmark = _render_benchmark(
        report,
        raw_report_reference=_report_reference(report_path, benchmark_path),
    )
    run_record = _render_approval_record(
        report,
        raw_report_reference=_report_reference(report_path, run_record_path),
    )
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    run_record_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(benchmark, encoding="utf-8")
    run_record_path.write_text(run_record, encoding="utf-8")


def _validate_report(report: Mapping[str, object]) -> None:
    if report.get("schema_version") != 1:
        raise ValueError("model quality gate report schema_version must be 1")
    if report.get("dirty") is not False:
        raise ValueError("model quality gate report dirty must be false")
    _required_text(report, "source_commit")
    _required_text(report, "source_tree")
    _required_text(report, "suite_id")
    _required_text(report, "suite_revision")
    _required_text(report, "evaluator_version")
    status = _required_text(report, "status")
    expected_passed = {
        "passed": True,
        "failed": False,
        "inconclusive": None,
    }
    if status not in expected_passed:
        raise ValueError(
            "model quality report status must be passed, failed, or inconclusive"
        )
    if report.get("passed") is not expected_passed[status]:
        raise ValueError("model quality report status and passed are inconsistent")
    raw_models = _sequence(report.get("models"), label="models")
    raw_runs = _sequence(report.get("runs"), label="runs")
    if not raw_models or not raw_runs:
        raise ValueError("model quality report models and runs must be non-empty")
    models = _items_by_alias(raw_models, label="model result")
    runs = _items_by_alias(raw_runs, label="model run")
    if set(models) != set(runs):
        raise ValueError("model quality report model and run aliases are inconsistent")
    metadata = _sequence(report.get("case_metadata"), label="case_metadata")
    metadata_by_id: dict[str, Mapping[str, object]] = {}
    for raw_item in metadata:
        item = _mapping(raw_item, label="case metadata")
        case_id = _required_text(item, "case_id")
        _required_text(item, "capability")
        if case_id in metadata_by_id:
            raise ValueError(f"model quality report duplicate case metadata id: {case_id}")
        metadata_by_id[case_id] = item
    approval_cases = [
        item
        for item in metadata_by_id.values()
        if item.get("capability") == "approval_continuation"
    ]
    if not approval_cases:
        raise ValueError("model quality report has no approval-continuation metadata")
    has_inconclusive = False
    model_passes: list[bool] = []
    for alias, model in models.items():
        run = runs[alias]
        run_status = run.get("status")
        model_passed = model.get("passed")
        model_infrastructure = model.get("infrastructure_failure")
        run_infrastructure = run.get("infrastructure_failure")
        if run_status == "completed":
            if (
                not isinstance(model_passed, bool)
                or isinstance(model_infrastructure, Mapping)
                or isinstance(run_infrastructure, Mapping)
                or model.get("status") == "inconclusive"
            ):
                raise ValueError(
                    f"model quality report completed result for {alias} is inconsistent"
                )
            _validate_run_cases(
                run,
                alias=alias,
                run_status=run_status,
                metadata_by_id=metadata_by_id,
            )
            model_passes.append(model_passed)
        elif run_status == "inconclusive":
            if (
                model_passed is not None
                or model.get("status") != "inconclusive"
                or not isinstance(model_infrastructure, Mapping)
                or not isinstance(run_infrastructure, Mapping)
                or model_infrastructure != run_infrastructure
            ):
                raise ValueError(
                    f"model quality report inconclusive result for {alias} is inconsistent"
                )
            _validate_run_cases(
                run,
                alias=alias,
                run_status=run_status,
                metadata_by_id=metadata_by_id,
            )
            has_inconclusive = True
        else:
            raise ValueError(
                f"model quality report run status for {alias} is inconsistent"
            )
    derived_status = (
        "inconclusive"
        if has_inconclusive
        else ("passed" if all(model_passes) else "failed")
    )
    if status != derived_status:
        raise ValueError("model quality report overall verdict is inconsistent")


def _render_benchmark(
    report: Mapping[str, object],
    *,
    raw_report_reference: str,
) -> str:
    verdict = _verdict(report)
    lines = [
        "# Model quality benchmark",
        "",
        f"Overall verdict: **{verdict}**",
        "",
        "## Evidence provenance",
        "",
        f"- source_commit: `{_required_text(report, 'source_commit')}`",
        f"- source_tree: `{_required_text(report, 'source_tree')}`",
        "- dirty: `false`",
        f"- suite_id: `{_required_text(report, 'suite_id')}`",
        f"- suite_revision: `{_required_text(report, 'suite_revision')}`",
        f"- evaluator_version: `{_required_text(report, 'evaluator_version')}`",
        f"- measured_at: `{_required_text(report, 'measured_at')}`",
        f"- Redacted raw report: [{Path(raw_report_reference).name}]({raw_report_reference})",
        "",
        "## Model results",
        "",
    ]
    for raw_model in _sequence(report.get("models"), label="models"):
        model = _mapping(raw_model, label="model result")
        model_alias = _required_text(model, "model_alias")
        model_verdict = _boolean_verdict(model.get("passed"))
        lines.extend(
            [
                f"### `{model_alias}`",
                "",
                f"- Provider model: `{_required_text(model, 'provider_model')}`",
                f"- Evaluator verdict: **{model_verdict}**",
                "",
            ]
        )
        infrastructure = model.get("infrastructure_failure")
        if isinstance(infrastructure, Mapping):
            lines.extend(
                [
                    "- Infrastructure status: **INCONCLUSIVE**",
                    f"- Infrastructure stage: `{_scalar(infrastructure.get('stage'))}`",
                    f"- Stop reason: `{_scalar(infrastructure.get('stop_reason'))}`",
                    "- Diagnostic error types: "
                    f"`{_scalar(infrastructure.get('diagnostic_error_types'))}`",
                    "",
                ]
            )
        observed = _mapping(model.get("observed"), label=f"{model_alias} observed metrics")
        thresholds = _mapping(model.get("thresholds"), label=f"{model_alias} thresholds")
        if observed:
            lines.extend(["| Metric | Observed | Threshold |", "| --- | ---: | --- |"])
            for metric, value in observed.items():
                raw_threshold = thresholds.get(metric)
                threshold = _threshold_text(raw_threshold)
                lines.append(f"| `{_cell(metric)}` | `{_scalar(value)}` | {threshold} |")
            lines.append("")
        failures = _string_sequence(model.get("failures", ()), label=f"{model_alias} failures")
        if failures:
            lines.append("Reported failures:")
            lines.append("")
            lines.extend(f"- {_safe_text(failure)}" for failure in failures)
            lines.append("")

    lines.extend(
        [
            "## Case results",
            "",
            "| Model | Trial | Case | Capability | Verdict |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    for case in _iter_cases(report):
        lines.append(
            "| "
            f"`{_cell(case.model_alias)}` | `{case.trial}` | `{_cell(case.case_id)}` | "
            f"`{_cell(case.capability)}` | **{case.verdict}** |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_approval_record(
    report: Mapping[str, object],
    *,
    raw_report_reference: str,
) -> str:
    metadata = _approval_metadata(report)
    case_id = _required_text(metadata, "case_id")
    task = _required_text(metadata, "task")
    lines = [
        "# Approval-continuation model run",
        "",
        f"Overall verdict: **{_verdict(report)}**",
        "",
        f"- source_commit: `{_required_text(report, 'source_commit')}`",
        f"- source_tree: `{_required_text(report, 'source_tree')}`",
        "- dirty: `false`",
        f"- suite_revision: `{_required_text(report, 'suite_revision')}`",
        f"- evaluator_version: `{_required_text(report, 'evaluator_version')}`",
        f"- Redacted raw report: [{Path(raw_report_reference).name}]({raw_report_reference})",
        "",
        "## Task",
        "",
        _safe_text(task),
        "",
    ]
    rendered_cases = _iter_cases(report)
    approval_runs = [case for case in rendered_cases if case.case_id == case_id]
    if not approval_runs:
        if _verdict(report) != "INCONCLUSIVE":
            raise ValueError(f"model quality report has no observation for {case_id}")
        earlier_failure = next(
            (case for case in rendered_cases if case.verdict == "INCONCLUSIVE"),
            None,
        )
        lines.extend(
            [
                "## Inconclusive",
                "",
                "Approval-continuation case was not reached.",
                "",
                "No tool trace, approval/resume event, workspace diff assertion, "
                "or final answer was reported for this case.",
                "",
            ]
        )
        if earlier_failure is not None:
            lines.extend(
                [
                    f"- Earlier infrastructure-failed case: `{earlier_failure.case_id}`",
                    f"- Stop reason: `{_scalar(earlier_failure.observation.get('stop_reason'))}`",
                    "- Diagnostic error types: "
                    f"`{_scalar(earlier_failure.observation.get('diagnostic_error_types'))}`",
                    "",
                ]
            )
        else:
            infrastructure = _report_infrastructure_failure(report)
            if infrastructure is not None:
                lines.extend(
                    [
                        f"- Infrastructure stage: `{_scalar(infrastructure.get('stage'))}`",
                        f"- Stop reason: `{_scalar(infrastructure.get('stop_reason'))}`",
                        "- Diagnostic error types: "
                        f"`{_scalar(infrastructure.get('diagnostic_error_types'))}`",
                        "",
                    ]
                )
        lines.extend(
            [
                "### Evaluator verdict",
                "",
                "Evaluator verdict: **INCONCLUSIVE**",
                "",
            ]
        )
        return "\n".join(lines)
    before = _text_mapping(metadata.get("workspace_before"), label="workspace_before")
    after = _text_mapping(metadata.get("workspace_after"), label="workspace_after")
    for case in approval_runs:
        observation = case.observation
        score = case.score
        lines.extend(
            [
                f"## `{case.model_alias}` trial {case.trial}",
                "",
                "### Tool trace",
                "",
            ]
        )
        raw_calls = _sequence(observation.get("tool_calls", ()), label="tool_calls")
        if not raw_calls:
            lines.append("No tool calls were reported.")
        for index, raw_call in enumerate(raw_calls, start=1):
            call = _mapping(raw_call, label="tool call")
            tool_name = _required_text(call, "tool_name")
            arguments = _mapping(call.get("arguments"), label="tool arguments")
            error = " error" if call.get("is_error") is True else ""
            lines.append(f"{index}. `{tool_name}`{error}: `{_safe_json(arguments)}`")
        lines.extend(
            [
                "",
                "### Approval and resume",
                "",
                f"- Approval pause observed: `{_scalar(observation.get('approval_pause_observed'))}`",
                f"- Approval kind: `{_scalar(observation.get('approval_kind'))}`",
                f"- Approval resumes: `{_scalar(observation.get('approval_resumes'))}`",
                "",
            ]
        )
        approval_pause_observed = (
            observation.get("approval_pause_observed") is True
            and observation.get("approval_kind") == "tool_approval"
        )
        resumes = observation.get("approval_resumes")
        approval_resumed = (
            isinstance(resumes, int)
            and not isinstance(resumes, bool)
            and resumes >= 1
        )
        infrastructure_failure = observation.get("infrastructure_failure") is True
        if not approval_pause_observed:
            lines.extend(
                [
                    "### Approval evidence",
                    "",
                    "Approval was not reached.",
                    "",
                    "No workspace diff assertion or final answer evidence was reported.",
                    "",
                    f"- Stop reason: `{_scalar(observation.get('stop_reason'))}`",
                    "- Diagnostic error types: "
                    f"`{_scalar(observation.get('diagnostic_error_types'))}`",
                    "",
                    "### Evaluator verdict",
                    "",
                    f"Evaluator verdict: **{case.verdict}**",
                    "",
                ]
            )
            continue
        if not approval_resumed:
            lines.extend(
                [
                    "### Approval evidence",
                    "",
                    "Approval pause was observed, but approval resume was not observed.",
                    "",
                    "No workspace diff assertion or final answer evidence was reported.",
                    "",
                    f"- Stop reason: `{_scalar(observation.get('stop_reason'))}`",
                    "- Diagnostic error types: "
                    f"`{_scalar(observation.get('diagnostic_error_types'))}`",
                    "",
                    "### Evaluator verdict",
                    "",
                    f"Evaluator verdict: **{case.verdict}**",
                    "",
                ]
            )
            continue
        if infrastructure_failure:
            lines.extend(
                [
                    "### Approval evidence",
                    "",
                    "Approval pause and resume were observed before infrastructure failure.",
                    "",
                ]
            )
            workspace_assertions_passed = observation.get(
                "workspace_assertions_passed"
            )
            if workspace_assertions_passed is True:
                lines.extend(
                    [
                        "### Fixture workspace assertion contract",
                        "",
                        "This is the fixture's expected before/after contract, not a captured filesystem diff.",
                        "",
                        "```diff",
                        *_workspace_diff(before, after),
                        "```",
                        "",
                        "Workspace assertions passed: `true`",
                        "",
                    ]
                )
            elif workspace_assertions_passed is False:
                lines.extend(
                    [
                        "### Workspace assertion evidence",
                        "",
                        "Workspace assertions passed: `false`",
                        "",
                        "The fixture assertion contract was not satisfied, so its "
                        "expected diff is not shown as observed evidence.",
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        "Workspace assertion evidence was not reported.",
                        "",
                    ]
                )
            answer = observation.get("answer")
            if isinstance(answer, str) and answer:
                lines.extend(
                    [
                        "### Observed answer before infrastructure failure",
                        "",
                        _safe_text(answer),
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        "No answer was reported before infrastructure failure.",
                        "",
                    ]
                )
            lines.extend(
                [
                    f"- Stop reason: `{_scalar(observation.get('stop_reason'))}`",
                    "- Diagnostic error types: "
                    f"`{_scalar(observation.get('diagnostic_error_types'))}`",
                    "",
                    "### Evaluator verdict",
                    "",
                    f"Evaluator verdict: **{case.verdict}**",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                "### Fixture workspace assertion contract",
                "",
                "This is the fixture's expected before/after contract, not a captured filesystem diff.",
                "",
                "```diff",
                *_workspace_diff(before, after),
                "```",
                "",
                f"Workspace assertions passed: `{_scalar(observation.get('workspace_assertions_passed'))}`",
                "",
                "### Final answer",
                "",
                _safe_text(str(observation.get("answer") or "<none>")),
                "",
                "### Evaluator verdict",
                "",
                f"Evaluator verdict: **{case.verdict}**",
                "",
                f"- Core success: `{_scalar(score.get('core_success'))}`",
                f"- Capability passed: `{_scalar(score.get('capability_passed'))}`",
                "",
            ]
        )
    return "\n".join(lines)


def _report_infrastructure_failure(
    report: Mapping[str, object],
) -> Mapping[str, object] | None:
    for field in ("runs", "models"):
        for raw_item in _sequence(report.get(field), label=field):
            item = _mapping(raw_item, label=field[:-1])
            failure = item.get("infrastructure_failure")
            if isinstance(failure, Mapping):
                return cast(Mapping[str, object], failure)
    return None


class _RenderedCase:
    def __init__(
        self,
        *,
        model_alias: str,
        trial: int,
        case_id: str,
        capability: str,
        verdict: str,
        observation: Mapping[str, object],
        score: Mapping[str, object],
    ) -> None:
        self.model_alias = model_alias
        self.trial = trial
        self.case_id = case_id
        self.capability = capability
        self.verdict = verdict
        self.observation = observation
        self.score = score


def _iter_cases(report: Mapping[str, object]) -> list[_RenderedCase]:
    rendered: list[_RenderedCase] = []
    for raw_run in _sequence(report.get("runs"), label="runs"):
        run = _mapping(raw_run, label="model run")
        model_alias = _required_text(run, "model_alias")
        for raw_trial in _sequence(run.get("trials"), label=f"{model_alias} trials"):
            trial = _mapping(raw_trial, label="trial")
            trial_number = _integer(trial.get("trial"), label="trial")
            for raw_case in _sequence(trial.get("cases"), label="trial cases"):
                case = _mapping(raw_case, label="case")
                observation = _mapping(case.get("observation"), label="case observation")
                score = _mapping(case.get("score"), label="case score")
                rendered.append(
                    _RenderedCase(
                        model_alias=model_alias,
                        trial=trial_number,
                        case_id=_required_text(observation, "case_id"),
                        capability=_required_text(observation, "capability"),
                        verdict=(
                            "INCONCLUSIVE"
                            if observation.get("infrastructure_failure") is True
                            else _boolean_verdict(score.get("passed"))
                        ),
                        observation=observation,
                        score=score,
                    )
                )
    return rendered


def _approval_metadata(report: Mapping[str, object]) -> Mapping[str, object]:
    for raw_item in _sequence(report.get("case_metadata"), label="case_metadata"):
        item = _mapping(raw_item, label="case metadata")
        if item.get("capability") == "approval_continuation":
            return item
    raise ValueError("model quality report has no approval-continuation metadata")


def _items_by_alias(
    items: Sequence[object],
    *,
    label: str,
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for raw_item in items:
        item = _mapping(raw_item, label=label)
        alias = _required_text(item, "model_alias")
        if alias in result:
            raise ValueError(f"model quality report duplicate {label} alias: {alias}")
        result[alias] = item
    return result


def _validate_run_cases(
    run: Mapping[str, object],
    *,
    alias: str,
    run_status: str,
    metadata_by_id: Mapping[str, Mapping[str, object]],
) -> None:
    trials = _sequence(run.get("trials"), label=f"{alias} trials")
    infrastructure_observed = False
    for raw_trial in trials:
        trial = _mapping(raw_trial, label=f"{alias} trial")
        for raw_case in _sequence(trial.get("cases"), label=f"{alias} trial cases"):
            case = _mapping(raw_case, label=f"{alias} case")
            observation = _mapping(
                case.get("observation"),
                label=f"{alias} case observation",
            )
            score = _mapping(case.get("score"), label=f"{alias} case score")
            case_id = _required_text(observation, "case_id")
            capability = _required_text(observation, "capability")
            metadata = metadata_by_id.get(case_id)
            if (
                metadata is None
                or metadata.get("capability") != capability
                or score.get("case_id") != case_id
                or score.get("capability") != capability
            ):
                raise ValueError(
                    f"model quality report case evidence for {alias}.{case_id} is inconsistent"
                )
            observation_infrastructure = (
                observation.get("infrastructure_failure") is True
            )
            score_passed = score.get("passed")
            if run_status == "completed":
                if observation_infrastructure or not isinstance(score_passed, bool):
                    raise ValueError(
                        f"model quality report completed case for {alias}.{case_id} is inconsistent"
                    )
            elif observation_infrastructure:
                if score_passed is not None:
                    raise ValueError(
                        f"model quality report infrastructure case for {alias}.{case_id} is inconsistent"
                    )
                infrastructure_observed = True
            elif not isinstance(score_passed, bool):
                raise ValueError(
                    f"model quality report non-infrastructure case for {alias}.{case_id} is inconsistent"
                )
    if run_status == "inconclusive" and trials and not infrastructure_observed:
        raise ValueError(
            f"model quality report inconclusive run for {alias} has inconsistent case evidence"
        )


def _workspace_diff(before: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
    lines: list[str] = []
    for path in sorted(set(before) | set(after)):
        lines.extend(
            difflib.unified_diff(
                before.get(path, "").splitlines(),
                after.get(path, "").splitlines(),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
        )
    return lines or ["# No workspace content assertion was reported."]


def _verdict(report: Mapping[str, object]) -> str:
    status = report.get("status")
    if status == "passed":
        return "PASSED"
    if status == "failed":
        return "FAILED"
    if status == "inconclusive":
        return "INCONCLUSIVE"
    raise ValueError("model quality report status must be passed, failed, or inconclusive")


def _boolean_verdict(value: object) -> str:
    if value is True:
        return "PASSED"
    if value is False:
        return "FAILED"
    return "INCONCLUSIVE"


def _threshold_text(value: object) -> str:
    if not isinstance(value, Mapping):
        return "not reported"
    direction = value.get("direction")
    threshold = value.get("value")
    if direction not in {"min", "max"}:
        return "not reported"
    return f"`{direction} {_scalar(threshold)}`"


def _report_reference(report_path: Path, output_path: Path) -> str:
    try:
        report_path.resolve().relative_to(ROOT)
        output_path.resolve().relative_to(ROOT)
    except ValueError:
        return report_path.name
    return Path(os.path.relpath(report_path.resolve(), output_path.resolve().parent)).as_posix()


def _safe_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        _redacted_mapping(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _redacted_mapping(value: Mapping[str, object]) -> dict[str, object]:
    redacted: dict[str, object] = {}
    for key, item in value.items():
        if _is_sensitive_key(key):
            redacted[key] = "[REDACTED]"
        elif isinstance(item, Mapping):
            redacted[key] = _redacted_mapping(cast(Mapping[str, object], item))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            redacted[key] = [_redacted_value(entry) for entry in item]
        else:
            redacted[key] = _redacted_value(item)
    return redacted


def _is_sensitive_key(key: str) -> bool:
    camel_split = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", camel_split)
    tokens = {
        token.casefold()
        for token in re.split(r"[^A-Za-z0-9]+", normalized)
        if token
    }
    return bool(tokens & _SENSITIVE_KEY_TOKENS)


def _redacted_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _redacted_mapping(cast(Mapping[str, object], value))
    if isinstance(value, str) and Path(value).is_absolute():
        return "[REDACTED_ABSOLUTE_PATH]"
    return value


def _safe_text(value: str) -> str:
    words = value.split()
    return " ".join("[REDACTED_ABSOLUTE_PATH]" if Path(word).is_absolute() else word for word in words)


def _scalar(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"model quality report {label} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"model quality report {label} must be a sequence")
    return cast(Sequence[object], value)


def _string_sequence(value: object, *, label: str) -> tuple[str, ...]:
    return tuple(str(item) for item in _sequence(value, label=label))


def _text_mapping(value: object, *, label: str) -> dict[str, str]:
    mapping = _mapping(value, label=label)
    result: dict[str, str] = {}
    for key, item in mapping.items():
        if not isinstance(item, str):
            raise ValueError(f"model quality report {label}.{key} must be text")
        result[str(key)] = item
    return result


def _required_text(value: Mapping[str, object], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise ValueError(f"model quality report {name} must be non-empty text")
    return item


def _integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"model quality report {label} must be an integer")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Redacted gate report JSON.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--run-record", type=Path, default=DEFAULT_RUN_RECORD_PATH)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        render_model_quality_report(
            args.report,
            benchmark_path=args.benchmark,
            run_record_path=args.run_record,
        )
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"model quality report error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
