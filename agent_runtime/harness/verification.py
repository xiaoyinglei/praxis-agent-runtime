from __future__ import annotations

import ast
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


def classify_tool_verification(
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    structured_content: object,
    resources: Sequence[Mapping[str, Any]] = (),
) -> VerificationEvidence | None:
    if not isinstance(structured_content, Mapping):
        return None
    if tool_name in {"read_file", "inspect_data_file"}:
        if tool_name == "inspect_data_file" and structured_content.get("valid") is not True:
            return None
        verified_resources = tuple(
            str(resource["identity"])
            for resource in resources
            if resource.get("kind") == "filesystem"
            and resource.get("access") == "read"
            and isinstance(resource.get("identity"), str)
        )
        if not verified_resources:
            return None
        return VerificationEvidence(
            kind="inspection",
            verifier=tool_name,
            verified_resources=verified_resources,
        )

    exit_code = structured_content.get("exit_code")
    if isinstance(exit_code, bool) or exit_code != 0:
        return None
    if tool_name == "execute_python":
        code = arguments.get("code")
        if isinstance(code, str) and _contains_assertion(code):
            return VerificationEvidence(kind="assertion", verifier="python")
        return None
    if tool_name != "run_command":
        return None
    command = arguments.get("command")
    if not isinstance(command, str):
        return None
    return _classify_command(command)


def _classify_command(command: str) -> VerificationEvidence | None:
    try:
        words = shlex.split(command)
    except ValueError:
        return None
    if not words or any(word in {"|", "||", "&&", ";", "&"} for word in words):
        return None

    executable = PurePath(words[0]).name.lower()
    arguments = words[1:]
    if executable == "uv" and arguments[:1] == ["run"]:
        if len(arguments) < 2:
            return None
        executable = PurePath(arguments[1]).name.lower()
        arguments = arguments[2:]

    if executable in {"python", "python3"}:
        if arguments[:2] == ["-m", "pytest"]:
            return _runner_evidence("pytest", arguments[2:])
        if arguments[:2] == ["-m", "unittest"]:
            return _runner_evidence("unittest", arguments[2:])
        if arguments[:1] == ["-c"] and len(arguments) == 2:
            if _contains_assertion(arguments[1]):
                return VerificationEvidence(kind="assertion", verifier="python")
        return None
    if executable == "pytest":
        return _runner_evidence("pytest", arguments)
    if executable == "mypy":
        return _runner_evidence("mypy", arguments, kind="static_analysis")
    if executable == "ruff" and arguments[:1] == ["check"]:
        return _runner_evidence("ruff", arguments[1:], kind="static_analysis")
    if executable == "cargo" and arguments[:1] == ["test"]:
        return _runner_evidence("cargo", arguments[1:])
    if executable == "go" and arguments[:1] == ["test"]:
        return _runner_evidence("go", arguments[1:])
    if executable in {"npm", "pnpm", "yarn"}:
        is_test = arguments[:1] == ["test"] or arguments[:2] == ["run", "test"]
        if is_test:
            return _runner_evidence(executable, arguments[1:])
    return None


def _runner_evidence(
    verifier: str,
    arguments: Sequence[str],
    *,
    kind: VerificationKind = "test",
) -> VerificationEvidence | None:
    disallowed = {"-h", "--help", "--version", "--collect-only"}
    if any(argument.lower() in disallowed for argument in arguments):
        return None
    return VerificationEvidence(kind=kind, verifier=verifier)


def _contains_assertion(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Assert) for node in ast.walk(tree))


__all__ = [
    "VerificationEvidence",
    "VerificationKind",
    "classify_tool_verification",
]
