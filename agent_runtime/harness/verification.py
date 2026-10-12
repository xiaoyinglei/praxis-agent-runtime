from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Literal

VerificationKind = Literal["test", "static_analysis", "assertion", "inspection"]


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    kind: VerificationKind
    verifier: str
    verified_resources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VerificationAssessment:
    reason_code: str
    message: str
    evidence: VerificationEvidence | None = None
    suggested_command: str | None = None


def classify_tool_verification(
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    structured_content: object,
    resources: Sequence[Mapping[str, Any]] = (),
) -> VerificationEvidence | None:
    return assess_tool_verification(
        tool_name=tool_name,
        arguments=arguments,
        structured_content=structured_content,
        resources=resources,
    ).evidence


def assess_tool_verification(
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    structured_content: object,
    resources: Sequence[Mapping[str, Any]] = (),
) -> VerificationAssessment:
    if not isinstance(structured_content, Mapping):
        return VerificationAssessment("invalid_result", "The tool did not return a structured verification result.")
    if tool_name in {"read_file", "inspect_data_file"}:
        if tool_name == "inspect_data_file" and structured_content.get("valid") is not True:
            return VerificationAssessment("invalid_document", "Artifact inspection did not report a valid document.")
        verified_resources = tuple(
            str(resource["identity"])
            for resource in resources
            if resource.get("kind") == "filesystem"
            and resource.get("access") == "read"
            and isinstance(resource.get("identity"), str)
        )
        if not verified_resources:
            return VerificationAssessment("missing_read_resources", "The inspection has no attested file targets.")
        return VerificationAssessment(
            "verified",
            "The artifact was inspected.",
            VerificationEvidence(kind="inspection", verifier=tool_name, verified_resources=verified_resources),
        )

    if structured_content.get("timed_out") is True or structured_content.get("sandbox_error"):
        return VerificationAssessment(
            "execution_incomplete",
            "The execution timed out or was blocked by the sandbox; it is not successful verification.",
        )
    exit_code = structured_content.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        return VerificationAssessment("invalid_exit_status", "The execution has no integer process exit status.")
    if exit_code != 0:
        return VerificationAssessment(
            "execution_failed",
            f"The execution exited with status {exit_code}. "
            "Read its stdout/stderr, resolve the failure, and rerun verification.",
        )
    if tool_name == "execute_python":
        observation = structured_content.get("verification")
        if isinstance(observation, Mapping):
            passed = observation.get("assertions_passed")
            failed = observation.get("assertions_failed")
        else:
            passed, failed = None, None
        if (
            isinstance(passed, int)
            and not isinstance(passed, bool)
            and passed > 0
            and isinstance(failed, int)
            and failed == 0
            and not isinstance(failed, bool)
        ):
            return VerificationAssessment(
                "verified",
                "The managed interpreter observed successful assertions. Their coverage of changed files is unknown.",
                VerificationEvidence(kind="assertion", verifier="python"),
            )
        return VerificationAssessment(
            "assertion_not_executed",
            "No successful assertion execution was observed. Define and run the check with execute_python; "
            "unreached assertions, optimized-out assertions and process exit status alone are not evidence.",
        )
    if tool_name != "run_command":
        return VerificationAssessment("unsupported_tool", "This tool result is not verification evidence.")
    command = arguments.get("command")
    if not isinstance(command, str):
        return VerificationAssessment("invalid_command", "No command was recorded for this execution.")
    return _assess_command(command)


def _assess_command(command: str) -> VerificationAssessment:
    try:
        # stdlib shell lexer separates operators even without surrounding spaces.
        # This is deliberately not a shell parser: only a direct invocation is attested.
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        # shlex's comment rules differ from shell comments inside an argument.
        # Never let comment stripping hide a following fallback or command.
        lexer.commenters = ""
        words = list(lexer)
    except ValueError:
        return VerificationAssessment("invalid_command", "The command could not be parsed as a direct invocation.")
    operators = [i for i, word in enumerate(words) if word and all(c in "();<>|&" for c in word)]
    if operators or any(c in command.strip() for c in "\n\r") or any("$(" in word or "`" in word for word in words):
        suggested = None
        if operators:
            prefix = words[: operators[0]]
            if prefix and prefix[-1].isdigit() and any(c in "<>" for c in words[operators[0]]):
                prefix = prefix[:-1]
            if (
                prefix
                and not any("$" in word or "`" in word or any(c in word for c in "*?[") for word in prefix)
                and _classify_invocation(prefix) is not None
            ):
                suggested = shlex.join(prefix)
        return VerificationAssessment(
            "shell_composition",
            "The shell exit status cannot attest the test process in this command form. "
            "Run the project's actual test runner directly in a separate run_command call, "
            "preserving its interpreter and test arguments. Remove pipes, redirects, conditional branches, "
            "fallbacks, and extra commands. run_command already returns bounded stdout/stderr.",
            suggested_command=suggested,
        )
    evidence = _classify_invocation(words)
    if evidence is not None:
        return VerificationAssessment("verified", "The direct verifier completed successfully.", evidence)
    if _classify_invocation(words, allow_non_execution_flags=True) is not None:
        if _verifier_mode(words) == "unattested_verifier_mode":
            return VerificationAssessment(
                "unattested_verifier_mode",
                "This verifier mode can suppress check failures or replace the test process. "
                "Its successful exit does not attest successful checks. Run the project's actual checks "
                "in their ordinary checking mode, preserving failure exit status.",
            )
        return VerificationAssessment(
            "no_tests_executed",
            "Help, listing, collection, setup-only, or build-only modes do not execute verification. "
            "Run the actual checks instead.",
        )
    return VerificationAssessment(
        "unsupported_command",
        "This invocation is not a recognized test runner, static check, or assertion. "
        "Use the project's documented verification command directly. Executing a Python file alone "
        "does not attest that its tests ran; invoke its test runner explicitly.",
    )


def _classify_invocation(
    words: Sequence[str],
    *,
    allow_non_execution_flags: bool = False,
) -> VerificationEvidence | None:
    if not words:
        return None
    if not allow_non_execution_flags and _verifier_mode(words) is not None:
        return None

    words = _unwrap_uv(words)
    if not words:
        return None
    executable = PurePath(words[0]).name.lower()
    arguments = words[1:]

    if executable in {"python", "python3"}:
        if arguments[:2] == ["-m", "pytest"]:
            return VerificationEvidence(kind="test", verifier="pytest")
        if arguments[:2] == ["-m", "unittest"]:
            return VerificationEvidence(kind="test", verifier="unittest")
        # Shell inline source has no managed execution observations. Even an
        # assert AST node can be unreachable, optimized out or caught.
        return None

    if executable == "pytest":
        return VerificationEvidence(kind="test", verifier="pytest")
    if executable == "mypy":
        return VerificationEvidence(kind="static_analysis", verifier="mypy")
    if executable == "ruff" and arguments[:1] == ["check"]:
        return VerificationEvidence(kind="static_analysis", verifier="ruff")
    if executable == "cargo" and arguments[:1] == ["test"]:
        return VerificationEvidence(kind="test", verifier="cargo")
    if executable == "go" and arguments[:1] == ["test"]:
        return VerificationEvidence(kind="test", verifier="go")
    if executable in {"npm", "pnpm", "yarn"}:
        is_test = arguments[:1] == ["test"] or arguments[:2] == ["run", "test"]
        if is_test:
            return VerificationEvidence(kind="test", verifier=executable)
    return None


def _verifier_mode(words: Sequence[str]) -> str | None:
    """Recognize documented verifier modes that break the exit-status contract.

    This describes CLI semantics, not task relevance or human output text.
    Project configuration, plugins and package test scripts remain trusted;
    successful checks never imply coverage or independent task acceptance.
    """
    words = _unwrap_uv(words)
    if not words:
        return None
    runner, arguments = PurePath(words[0]).name.lower(), words[1:]
    if runner in {"python", "python3"} and arguments[:1] == ["-m"] and len(arguments) > 1:
        runner, arguments = arguments[1], arguments[2:]
    # Cargo forwards libtest listing options; Go forwards test binary options.
    # Other runners stop interpreting their own options at '--'.
    if runner not in {"cargo", "go"} and "--" in arguments:
        arguments = arguments[:arguments.index("--")]
    options = {argument.split("=", 1)[0] for argument in arguments if argument.startswith("--")}
    short = {
        char for argument in arguments if argument.startswith("-") and not argument.startswith("--")
        and runner != "go" for char in argument[1:]
    }
    if options & {"--help", "--version", "--collect-only", "--co", "--no-run"} or short & {"h", "V"}:
        return "no_tests_executed"
    if runner == "ruff":
        if options & {"--exit-zero", "--fix-only", "--diff", "--add-noqa", "--watch"} or short & {"e", "w"}:
            return "unattested_verifier_mode"
        if options & {"--show-files", "--show-settings"}:
            return "no_tests_executed"
    elif runner == "pytest":
        if options & {"--setup-plan", "--setup-only", "--fixtures", "--fixtures-per-test", "--cache-show"}:
            return "no_tests_executed"
    elif runner == "cargo":
        if "--list" in options:
            return "no_tests_executed"
    elif runner == "go":
        names = {argument.split("=", 1)[0] for argument in arguments}
        if names & {"-h", "-n", "-c", "-list", "-test.list"}:
            return "no_tests_executed"
        if "-exec" in names:
            return "unattested_verifier_mode"
    elif runner in {"npm", "pnpm", "yarn"}:
        if options & {"--if-present", "--script-shell", "--dry-run", "--ignore-scripts"}:
            return "unattested_verifier_mode"
    return None


def _unwrap_uv(words: Sequence[str]) -> Sequence[str]:
    """Separate documented wrapper options from its program; unknown forms fail closed."""
    if not words or PurePath(words[0]).name.lower() != "uv":
        return words
    switches = {
        "--offline",
        "--no-sync",
        "--locked",
        "--frozen",
        "--active",
        "--isolated",
        "--no-project",
        "--no-dev",
        "--no-default-groups",
        "--all-extras",
        "--no-cache",
    }
    values = {
        "--project",
        "--directory",
        "--python",
        "--extra",
        "--group",
        "--package",
        "--env-file",
        "--cache-dir",
        "--index",
        "--default-index",
    }
    index, running = 1, False
    while index < len(words):
        word = words[index]
        if word == "run" and not running:
            running = True
            index += 1
        elif word == "--" and running:
            return words[index + 1 :]
        elif word in switches:
            index += 1
        elif word.split("=", 1)[0] in values:
            index += 1 if "=" in word else 2
        elif word.startswith("-") or not running:
            return ()
        else:
            return words[index:]
    return ()


__all__ = [
    "VerificationAssessment",
    "VerificationEvidence",
    "VerificationKind",
    "assess_tool_verification",
    "classify_tool_verification",
]
