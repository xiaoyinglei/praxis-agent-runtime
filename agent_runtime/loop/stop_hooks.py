from __future__ import annotations

import hashlib
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.core.definition import AgentRuntimePolicy
from agent_runtime.core.goal_contract import GoalContractEvaluator, GoalSpec
from agent_runtime.core.observations import (
    ComputationResult,
    ContextBinding,
    EvidenceRef,
    runtime_workspace_change,
    runtime_workspace_file_changes,
    runtime_workspace_snapshot,
)
from agent_runtime.core.output_finalizer import (
    OutputValidationExhaustedError,
    StructuredOutputFinalizer,
    validated_final_output,
)
from agent_runtime.core.output_models import ValidatedFinalOutput
from agent_runtime.loop.state import (
    LoopState,
    StopHookFeedback,
    append_stop_hook_feedback,
    append_stop_hook_warning,
)
from agent_runtime.tools.tool import ToolCall, ToolResult
from agent_runtime.workspace import workspace_tree_sha256


class StopVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Literal["accept", "warn", "block", "halt"]
    code: str = Field(min_length=1, max_length=120)
    message: str | None = Field(default=None, max_length=1000)
    detail: dict[str, object] = Field(default_factory=dict)
    final_output: ValidatedFinalOutput | None = None


class StopHookOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Literal["accept", "warn", "block", "halt"]
    code: str
    message: str | None = None
    detail: dict[str, object] = Field(default_factory=dict)
    verdicts: tuple[StopVerdict, ...] = ()
    final_output: ValidatedFinalOutput | None = None

    @property
    def accepted(self) -> bool:
        return self.action in {"accept", "warn"}

    @property
    def blocked(self) -> bool:
        return self.action == "block"

    @property
    def halted(self) -> bool:
        return self.action == "halt"


@dataclass(frozen=True, slots=True)
class RuntimeVerificationEvidence:
    attempt_tool_call_ids: tuple[str, ...]
    successful_tool_call_ids: tuple[str, ...]
    satisfied: bool


class StopHook(Protocol):
    async def evaluate(
        self,
        *,
        state: LoopState,
        candidate: str,
    ) -> StopVerdict: ...


@dataclass(frozen=True)
class StopHookBinding:
    name: str
    hook: StopHook
    critical: bool


class StopHookRunner:
    def __init__(
        self,
        *,
        hooks: list[StopHookBinding] | tuple[StopHookBinding, ...],
        max_blocks: int,
    ) -> None:
        if max_blocks < 1:
            raise ValueError("max_blocks must be positive")
        self._hooks = tuple(hooks)
        self._max_blocks = max_blocks

    async def evaluate(
        self,
        *,
        state: LoopState,
        candidate: str,
    ) -> StopHookOutcome:
        verdicts: list[StopVerdict] = []
        final_output: ValidatedFinalOutput | None = None
        warned = False
        for binding in self._hooks:
            try:
                verdict = await binding.hook.evaluate(
                    state=state,
                    candidate=candidate,
                )
            except Exception as exc:
                verdict = StopVerdict(
                    action="halt" if binding.critical else "warn",
                    code=f"{binding.name}_failed",
                    message=str(exc) or type(exc).__name__,
                    detail={"error_type": type(exc).__name__},
                )
            verdicts.append(verdict)
            if verdict.final_output is not None:
                final_output = verdict.final_output

            if verdict.action == "warn":
                warned = True
                append_stop_hook_warning(
                    state,
                    StopHookFeedback(
                        code=verdict.code,
                        message=verdict.message or verdict.code,
                    ),
                )
                continue
            if verdict.action == "block":
                feedback = append_stop_hook_feedback(
                    state,
                    StopHookFeedback(
                        code=verdict.code,
                        message=verdict.message or verdict.code,
                    ),
                )
                if feedback.occurrences >= self._max_blocks:
                    return StopHookOutcome(
                        action="halt",
                        code="stop_hook_block_limit",
                        message=("Equivalent stop-hook feedback reached the configured block limit."),
                        detail={
                            "blocked_code": verdict.code,
                            "occurrences": feedback.occurrences,
                        },
                        verdicts=tuple(verdicts),
                        final_output=final_output,
                    )
                return StopHookOutcome(
                    action="block",
                    code=verdict.code,
                    message=verdict.message,
                    detail=verdict.detail,
                    verdicts=tuple(verdicts),
                    final_output=final_output,
                )
            if verdict.action == "halt":
                return StopHookOutcome(
                    action="halt",
                    code=verdict.code,
                    message=verdict.message,
                    detail=verdict.detail,
                    verdicts=tuple(verdicts),
                    final_output=final_output,
                )

        return StopHookOutcome(
            action="warn" if warned else "accept",
            code="accepted_with_warnings" if warned else "accepted",
            verdicts=tuple(verdicts),
            final_output=final_output,
        )


class StructuredOutputStopHook:
    def __init__(
        self,
        *,
        definition: AgentRuntimePolicy,
        finalizer: StructuredOutputFinalizer | None,
    ) -> None:
        self._definition = definition
        self._finalizer = finalizer

    async def evaluate(
        self,
        *,
        state: LoopState,
        candidate: str,
    ) -> StopVerdict:
        if self._finalizer is None:
            raise RuntimeError("structured output is configured without a finalizer")
        try:
            output = await _await_output(
                self._finalizer.finalize(
                    definition=self._definition,
                    state=state,
                    candidate_text=candidate,
                )
            )
        except OutputValidationExhaustedError as exc:
            return StopVerdict(
                action="halt",
                code="structured_output_invalid",
                message=str(exc),
                detail={
                    "attempts": exc.attempts,
                    "validation_errors": exc.validation_errors,
                },
            )
        return StopVerdict(
            action="accept",
            code="structured_output_valid",
            final_output=validated_final_output(output),
        )


class GoalContractStopHook:
    def __init__(
        self,
        *,
        goal_spec: GoalSpec,
        workspace_root: Path | None = None,
    ) -> None:
        self._goal_spec = goal_spec
        self._workspace_root = (
            None if workspace_root is None else workspace_root.resolve()
        )

    @staticmethod
    def _collect_evidence_refs(
        tool_results: list[ToolResult],
    ) -> list[EvidenceRef]:
        """Derive evidence_refs from tool_results instead of deprecated state field."""
        refs: list[EvidenceRef] = []
        for result in tool_results:
            values = _structured_items(result, "evidence_refs")
            refs.extend(EvidenceRef.model_validate(item) for item in values)
        return refs

    @staticmethod
    def _collect_computation_results(
        tool_results: list[ToolResult],
    ) -> list[ComputationResult]:
        """Derive computation_results from tool_results instead of deprecated state field."""
        results: list[ComputationResult] = []
        for result in tool_results:
            values = _structured_items(result, "computation_results")
            results.extend(
                ComputationResult.model_validate(item) for item in values
            )
        return results

    @staticmethod
    def _collect_context_bindings(
        tool_results: list[ToolResult],
    ) -> list[ContextBinding]:
        """Derive context_bindings from tool_results instead of deprecated state field."""
        bindings: list[ContextBinding] = []
        for result in tool_results:
            values = _structured_items(result, "context_bindings")
            bindings.extend(ContextBinding.model_validate(item) for item in values)
        return bindings

    async def evaluate(
        self,
        *,
        state: LoopState,
        candidate: str,
    ) -> StopVerdict:
        tool_results = list(state.get("tool_results", []))
        runtime_owned_constraint_ids = {
            constraint.constraint_id
            for constraint in self._goal_spec.constraints
            if constraint.constraint_type
            in {"workspace_change", "verification_after_change"}
        }
        context_bindings = [
            binding
            for binding in self._collect_context_bindings(tool_results)
            if binding.constraint_id not in runtime_owned_constraint_ids
        ]
        workspace_change_constraints = tuple(
            constraint
            for constraint in self._goal_spec.constraints
            if (
                constraint.required
                and constraint.constraint_type == "workspace_change"
                and constraint.expected_value is True
            )
        )
        workspace_changed = _has_net_workspace_change(
            tool_results,
            workspace_root=self._workspace_root,
        )
        if workspace_changed:
            context_bindings.extend(
                ContextBinding(
                    binding_id=f"runtime:workspace_change:{constraint.constraint_id}",
                    constraint_id=constraint.constraint_id,
                    status="satisfied",
                    rationale="A runtime write tool reported a real workspace change.",
                )
                for constraint in workspace_change_constraints
            )
        verification_constraints = tuple(
            constraint
            for constraint in self._goal_spec.constraints
            if (
                constraint.required
                and constraint.constraint_type == "verification_after_change"
                and constraint.expected_value is True
            )
        )
        if runtime_verification_after_latest_change(state).satisfied:
            context_bindings.extend(
                ContextBinding(
                    binding_id=(
                        "runtime:verification_after_change:"
                        f"{constraint.constraint_id}"
                    ),
                    constraint_id=constraint.constraint_id,
                    status="satisfied",
                    rationale=(
                        "Every recognized behavior check or exact artifact "
                        "inspection after the latest workspace change completed "
                        "successfully."
                    ),
                )
                for constraint in verification_constraints
            )
        evaluation = GoalContractEvaluator().evaluate(
            goal_spec=self._goal_spec,
            candidate=candidate,
            evidence_refs=self._collect_evidence_refs(tool_results),
            computation_results=self._collect_computation_results(tool_results),
            context_bindings=context_bindings,
        )
        if evaluation.satisfied:
            return StopVerdict(
                action="accept",
                code="goal_contract_satisfied",
            )
        return StopVerdict(
            action="block",
            code="goal_contract_unsatisfied",
            message="; ".join(issue.description for issue in evaluation.issues)
            or "Explicit goal contract is not satisfied.",
            detail={
                "unsatisfied_issue_ids": evaluation.issue_ids,
            },
        )


_DIRECT_VERIFICATION_EXECUTABLES = frozenset(
    {
        "biome",
        "eslint",
        "jest",
        "mypy",
        "nox",
        "pyright",
        "pytest",
        "tsc",
        "tox",
        "vitest",
    }
)
_VERIFICATION_SUBCOMMANDS = frozenset(
    {"build", "check", "clippy", "lint", "test", "typecheck", "verify", "vet"}
)
_MUTATING_VERIFICATION_FLAGS = frozenset(
    {"--apply", "--fix", "--update-snapshots", "--write"}
)
_NON_EXECUTING_VERIFICATION_ARGUMENTS = frozenset(
    {
        "--collect-only",
        "--co",
        "--dry-run",
        "--exclude-task",
        "--fixtures",
        "--fixtures-per-test",
        "--help",
        "--if-present",
        "--just-print",
        "--list",
        "--list-sessions",
        "--list-tests",
        "--listenvs",
        "--listenvs-all",
        "--listtests",
        "--markers",
        "--no-run",
        "--passwithnotests",
        "--print-config",
        "--question",
        "--recon",
        "--setup-only",
        "--setup-plan",
        "--show-config",
        "--show-files",
        "--show-settings",
        "--showconfig",
        "--trace-config",
        "--touch",
        "--version",
        "-list",
        "list",
    }
)
_VERIFICATION_CONFIG_OVERRIDE_FLAGS: Mapping[
    str,
    frozenset[str],
] = {
    "eslint": frozenset({"--config", "-c"}),
    "jest": frozenset({"--config", "-c"}),
    "mypy": frozenset({"--config-file"}),
    "pytest": frozenset({"--override-ini", "-c", "-o"}),
    "ruff": frozenset({"--config"}),
    "tox": frozenset({"--conf", "-c"}),
    "vitest": frozenset({"--config", "-c"}),
}
_MAKE_SHORT_OPTIONS_WITH_VALUES = frozenset(
    {"C", "I", "O", "W", "f", "j", "l"}
)
_MAKE_LONG_OPTIONS_WITH_VALUES = frozenset(
    {
        "--assume-new",
        "--assume-old",
        "--directory",
        "--eval",
        "--file",
        "--include-dir",
        "--jobs",
        "--load-average",
        "--new-file",
        "--old-file",
        "--output-sync",
        "--what-if",
    }
)
_SHELL_FAILURE_MASK = re.compile(r"\|\||(?<!\|)\|(?!\|)|;|\n")
_UNSAFE_VERIFICATION_SHELL_SYNTAX = re.compile(
    r"\$\(|`|[<>]|(?<!&)&(?!&)"
)


def runtime_verification_after_latest_change(
    state: LoopState,
) -> RuntimeVerificationEvidence:
    tool_results = list(state.get("tool_results", ()))
    calls = state.get("canonical_tool_calls", {})
    latest_change_index = max(
        (
            index
            for index, result in enumerate(tool_results)
            if _is_runtime_workspace_write(result, calls=calls)
        ),
        default=-1,
    )
    if latest_change_index < 0:
        return RuntimeVerificationEvidence((), (), False)

    changed_files = {
        path: after_sha256
        for path, _before_sha256, after_sha256 in runtime_workspace_file_changes(
            tool_results[latest_change_index]
        )
    }
    attempts: list[tuple[str, bool]] = []
    for result in tool_results[latest_change_index + 1 :]:
        if result.tool_name == "inspect_data_file":
            call = calls.get(result.tool_call_id)
            if call is None:
                continue
            requested_path = call.arguments.get("path")
            normalized_path = _normalized_workspace_relative_path(requested_path)
            expected_sha256 = (
                changed_files.get(normalized_path)
                if normalized_path is not None
                else None
            )
            if expected_sha256 is None or normalized_path is None:
                continue
            attempts.append(
                (
                    result.tool_call_id,
                    _data_inspection_result_succeeded(
                        result,
                        path=normalized_path,
                        expected_sha256=expected_sha256,
                    ),
                )
            )
            continue
        if result.tool_name != "run_command":
            continue
        call = calls.get(result.tool_call_id)
        if call is None:
            continue
        if call.arguments.get("workspace_write") is True:
            continue
        command = call.arguments.get("command")
        if not isinstance(command, str) or not _is_verification_command(command):
            continue
        attempts.append(
            (result.tool_call_id, _command_result_succeeded(result))
        )
    return RuntimeVerificationEvidence(
        attempt_tool_call_ids=tuple(tool_call_id for tool_call_id, _success in attempts),
        successful_tool_call_ids=tuple(
            tool_call_id
            for tool_call_id, success in attempts
            if success
        ),
        satisfied=bool(attempts) and all(success for _tool_call_id, success in attempts),
    )


def _is_runtime_workspace_write(
    result: ToolResult,
    *,
    calls: Mapping[str, ToolCall],
) -> bool:
    if result.metadata.get("runtime_workspace_write") is True:
        return True
    if runtime_workspace_change(result) is not None:
        return True
    call = calls.get(result.tool_call_id)
    return bool(
        result.tool_name in {"run_command", "execute_python"}
        and call is not None
        and call.arguments.get("workspace_write") is True
        and isinstance(result.metadata.get("operation_id"), str)
    )


def _has_net_workspace_change(
    tool_results: Sequence[ToolResult],
    *,
    workspace_root: Path | None = None,
) -> bool:
    snapshot_results = [
        (index, runtime_workspace_snapshot(result))
        for index, result in enumerate(tool_results)
        if result.metadata.get("runtime_workspace_write") is True
    ]
    if snapshot_results:
        first_index, first_snapshot = snapshot_results[0]
        if any(snapshot is None for _index, snapshot in snapshot_results):
            return False
        assert first_snapshot is not None
        if any(
            runtime_workspace_snapshot(result) is None
            and runtime_workspace_change(result) is not None
            for result in tool_results[:first_index]
        ):
            return _has_legacy_net_workspace_change(
                tool_results,
                workspace_root=workspace_root,
            )
        valid_snapshots = [
            snapshot
            for _index, snapshot in snapshot_results
            if snapshot is not None
        ]
        baseline_sha256 = first_snapshot[0]
        reachable: dict[str, bool] = {baseline_sha256: False}
        # Parallel writes may share one baseline. A later state is attributable
        # only when an executor receipt connects it to that baseline; an
        # unrelated external mutation creates a gap and is rejected.
        for before_sha256, after_sha256 in valid_snapshots:
            changed_on_path = reachable.get(before_sha256)
            if changed_on_path is None:
                continue
            reachable[after_sha256] = bool(
                reachable.get(after_sha256, False)
                or changed_on_path
                or before_sha256 != after_sha256
            )
        if workspace_root is not None:
            final_sha256 = workspace_tree_sha256(workspace_root)
        else:
            final_sha256 = valid_snapshots[-1][1]
        return bool(
            final_sha256 is not None
            and final_sha256 != baseline_sha256
            and reachable.get(final_sha256, False)
        )
    return _has_legacy_net_workspace_change(
        tool_results,
        workspace_root=workspace_root,
    )


def _has_legacy_net_workspace_change(
    tool_results: Sequence[ToolResult],
    *,
    workspace_root: Path | None,
) -> bool:
    hashes_by_path: dict[str, tuple[str, str]] = {}
    for result in tool_results:
        if runtime_workspace_snapshot(result) is not None:
            continue
        change = runtime_workspace_change(result)
        if change is None:
            continue
        path, before_sha256, after_sha256 = change
        previous = hashes_by_path.get(path)
        if previous is not None and before_sha256 != previous[1]:
            return False
        original_sha256 = (
            before_sha256 if previous is None else previous[0]
        )
        hashes_by_path[path] = (original_sha256, after_sha256)
    if not any(
        before_sha256 != after_sha256
        for before_sha256, after_sha256 in hashes_by_path.values()
    ):
        return False
    if workspace_root is None:
        return True
    return all(
        _workspace_file_sha256(workspace_root, path) == after_sha256
        for path, (_before_sha256, after_sha256) in hashes_by_path.items()
    )


def _workspace_file_sha256(workspace_root: Path, path: str) -> str | None:
    try:
        target = (workspace_root / path).resolve()
        target.relative_to(workspace_root)
        if not target.is_file():
            return None
        return hashlib.sha256(target.read_bytes()).hexdigest()
    except (OSError, ValueError):
        return None


def _command_result_succeeded(result: ToolResult) -> bool:
    output = result.structured_content
    return bool(
        not result.is_error
        and isinstance(output, Mapping)
        and output.get("exit_code") == 0
        and output.get("timed_out") is False
        and output.get("sandbox_error") in (None, "")
    )


def _data_inspection_result_succeeded(
    result: ToolResult,
    *,
    path: str,
    expected_sha256: str,
) -> bool:
    output = result.structured_content
    return bool(
        not result.is_error
        and isinstance(output, Mapping)
        and output.get("valid") is True
        and _normalized_workspace_relative_path(output.get("path")) == path
        and output.get("sha256") == expected_sha256
    )


def _normalized_workspace_relative_path(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = PurePath(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    normalized = path.as_posix()
    return None if normalized in {"", "."} else normalized


def _is_verification_command(command: str) -> bool:
    """Recognize check commands without trusting a model-supplied purpose label."""

    if (
        _SHELL_FAILURE_MASK.search(command)
        or _UNSAFE_VERIFICATION_SHELL_SYNTAX.search(command)
    ):
        return False
    segments = [segment.strip() for segment in command.split("&&")]
    if any(_segment_uses_mutating_verification_flag(segment) for segment in segments):
        return False
    verified_segments = [
        _segment_runs_verification(segment)
        for segment in segments
    ]
    return bool(verified_segments) and all(verified_segments)


def _segment_uses_mutating_verification_flag(segment: str) -> bool:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return True
    return any(_is_mutating_verification_flag(token.lower()) for token in tokens)


def _segment_runs_verification(segment: str) -> bool:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return False
    if tokens and _is_environment_assignment(tokens[0]):
        return False
    if (
        tokens
        and (
            PurePath(tokens[0]).name != tokens[0]
            or "\\" in tokens[0]
        )
    ):
        return False
    if tokens and PurePath(tokens[0]).name == "env":
        return False
    if len(tokens) >= 2 and PurePath(tokens[0]).name in {
        "pipenv",
        "poetry",
        "uv",
    }:
        if tokens[1] != "run":
            return False
        tokens = tokens[2:]
    if not tokens:
        return False
    if PurePath(tokens[0]).name != tokens[0] or "\\" in tokens[0]:
        return False

    executable = PurePath(tokens[0]).name.lower()
    raw_arguments = tokens[1:]
    arguments = [value.lower() for value in tokens[1:]]
    if any(_is_mutating_verification_flag(argument) for argument in arguments):
        return False
    if any(
        _is_verification_config_override(executable, argument)
        for argument in arguments
    ):
        return False
    if any(
        _is_non_executing_verification_argument(argument)
        for argument in arguments
    ):
        return False
    if _runner_uses_non_executing_mode(
        executable,
        arguments,
        raw_arguments=raw_arguments,
    ):
        return False
    if executable in _DIRECT_VERIFICATION_EXECUTABLES:
        return True
    if executable == "ruff":
        return bool(arguments and arguments[0] == "check")
    if executable in {"npm", "pnpm", "yarn", "bun"}:
        package_args = (
            arguments[1:]
            if arguments and arguments[0] == "run"
            else arguments
        )
        return bool(
            package_args
            and package_args[0] in _VERIFICATION_SUBCOMMANDS
        )
    if executable in {"cargo", "dotnet", "go"}:
        return bool(
            arguments and arguments[0] in _VERIFICATION_SUBCOMMANDS
        )
    if executable == "make":
        return any(
            target in _VERIFICATION_SUBCOMMANDS
            for target in _make_targets(raw_arguments)
        )
    if executable in {"gradle", "gradlew", "mvn", "mvnw"}:
        return any(
            value.lstrip("-") in _VERIFICATION_SUBCOMMANDS
            for value in arguments
        )
    return False


def _is_mutating_verification_flag(value: str) -> bool:
    return bool(
        value in _MUTATING_VERIFICATION_FLAGS
        or any(
            value.startswith(f"{flag}=")
            for flag in _MUTATING_VERIFICATION_FLAGS
        )
    )


def _is_non_executing_verification_argument(value: str) -> bool:
    return bool(
        value in _NON_EXECUTING_VERIFICATION_ARGUMENTS
        or any(
            value.startswith(f"{argument}=")
            for argument in _NON_EXECUTING_VERIFICATION_ARGUMENTS
            if argument.startswith("-")
        )
    )


def _is_environment_assignment(value: str) -> bool:
    name, separator, _assigned = value.partition("=")
    return bool(
        separator
        and name
        and (name[0].isalpha() or name[0] == "_")
        and all(character.isalnum() or character == "_" for character in name)
    )


def _is_verification_config_override(
    executable: str,
    value: str,
) -> bool:
    flags = _VERIFICATION_CONFIG_OVERRIDE_FLAGS.get(
        executable,
        frozenset(),
    )
    return any(
        value == flag
        or (
            flag.startswith("--")
            and value.startswith(f"{flag}=")
        )
        or (
            flag.startswith("-")
            and not flag.startswith("--")
            and value.startswith(flag)
        )
        for flag in flags
    )


def _runner_uses_non_executing_mode(
    executable: str,
    arguments: Sequence[str],
    *,
    raw_arguments: Sequence[str],
) -> bool:
    if executable == "make":
        return any(
            _make_short_option_disables_execution(argument)
            for argument in raw_arguments
        )
    if executable in {"gradle", "gradlew"}:
        return any(argument in {"-m", "-x"} for argument in arguments)
    if executable == "nox":
        return "-l" in arguments
    if executable == "tox":
        return any(argument in {"-a", "-l"} for argument in arguments)
    if executable in {"mvn", "mvnw"}:
        return any(
            _maven_property_is_enabled(argument, "-dskiptests")
            or _maven_property_is_enabled(
                argument,
                "-dmaven.test.skip",
            )
            for argument in arguments
        )
    return False


def _make_short_option_disables_execution(argument: str) -> bool:
    if (
        not argument.startswith("-")
        or argument.startswith("--")
    ):
        return False
    for option in argument[1:]:
        if option in {"n", "q", "t"}:
            return True
        if option in _MAKE_SHORT_OPTIONS_WITH_VALUES:
            return False
    return False


def _make_targets(arguments: Sequence[str]) -> tuple[str, ...]:
    targets: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            targets.extend(
                value.lower()
                for value in arguments[index + 1 :]
                if not _is_environment_assignment(value)
            )
            break
        if _is_environment_assignment(argument):
            index += 1
            continue
        if argument.startswith("--"):
            option, separator, _value = argument.partition("=")
            index += (
                2
                if (
                    not separator
                    and option in _MAKE_LONG_OPTIONS_WITH_VALUES
                )
                else 1
            )
            continue
        if argument.startswith("-") and argument != "-":
            options = argument[1:]
            consumes_next = False
            for option_index, option in enumerate(options):
                if option not in _MAKE_SHORT_OPTIONS_WITH_VALUES:
                    continue
                consumes_next = option_index == len(options) - 1
                break
            index += 2 if consumes_next else 1
            continue
        targets.append(argument.lower())
        index += 1
    return tuple(targets)


def _maven_property_is_enabled(
    argument: str,
    property_name: str,
) -> bool:
    if argument == property_name:
        return True
    prefix = f"{property_name}="
    if not argument.startswith(prefix):
        return False
    return argument.removeprefix(prefix) not in {
        "0",
        "false",
        "no",
        "off",
    }


def build_stop_hooks(
    *,
    definition: AgentRuntimePolicy,
    output_finalizer: StructuredOutputFinalizer | None = None,
    goal_spec: GoalSpec | None = None,
    workspace_root: Path | None = None,
) -> tuple[StopHookBinding, ...]:
    hooks: list[StopHookBinding] = []
    if goal_spec is not None:
        hooks.append(
            StopHookBinding(
                name="goal_contract",
                hook=GoalContractStopHook(
                    goal_spec=goal_spec,
                    workspace_root=workspace_root,
                ),
                critical=True,
            )
        )
    if definition.output_model is not None:
        hooks.append(
            StopHookBinding(
                name="structured_output",
                hook=StructuredOutputStopHook(
                    definition=definition,
                    finalizer=output_finalizer,
                ),
                critical=True,
            )
        )
    return tuple(hooks)


async def _await_output(value: object) -> BaseModel:
    from inspect import isawaitable

    if isawaitable(value):
        value = await value
    if not isinstance(value, BaseModel):
        raise TypeError("structured output finalizer returned a non-model value")
    return value


def _structured_items(
    result: ToolResult,
    key: str,
) -> tuple[Mapping[str, object], ...]:
    if result.is_error or not isinstance(result.structured_content, Mapping):
        return ()
    raw = result.structured_content.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


__all__ = [
    "GoalContractStopHook",
    "RuntimeVerificationEvidence",
    "StopHook",
    "StopHookBinding",
    "StopHookOutcome",
    "StopHookRunner",
    "StopVerdict",
    "StructuredOutputStopHook",
    "build_stop_hooks",
    "runtime_verification_after_latest_change",
]
