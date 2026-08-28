#!/usr/bin/env python3
"""Run one exact acceptance case and emit evidence only after success."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run-harness-acceptance-case")
    parser.add_argument("--requirement", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("an exact command is required after --")
    bound_requirement = os.environ.get("PRAXIS_ACCEPTANCE_REQUIREMENT_ID")
    if bound_requirement != args.requirement:
        parser.error(
            "--requirement must match PRAXIS_ACCEPTANCE_REQUIREMENT_ID"
        )
    root_value = os.environ.get("PRAXIS_ACCEPTANCE_ARTIFACT_ROOT")
    if not root_value:
        parser.error("PRAXIS_ACCEPTANCE_ARTIFACT_ROOT is required")
    root = Path(root_value).expanduser().resolve()
    artifact = (root / args.artifact).resolve()
    if not artifact.is_relative_to(root):
        parser.error("artifact path escapes PRAXIS_ACCEPTANCE_ARTIFACT_ROOT")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
    )
    sys.stdout.buffer.write(completed.stdout)
    sys.stderr.buffer.write(completed.stderr)
    if completed.returncode != 0:
        return completed.returncode
    encoded_command = json.dumps(
        command,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    _write_json_atomic(
        artifact,
        {
            "schema_version": "praxis-harness-case-evidence-v1",
            "requirement_id": args.requirement,
            "command": command,
            "command_sha256": _sha256(encoded_command),
            "exit_code": 0,
            "stdout_sha256": _sha256(completed.stdout),
            "stderr_sha256": _sha256(completed.stderr),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
