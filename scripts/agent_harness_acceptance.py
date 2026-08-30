#!/usr/bin/env python3
"""Validate and audit the Praxis Harness acceptance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

REQUIREMENT_PATTERN = re.compile(r"\*\*([A-Z]+-[0-9]+)\*\*")
ALLOWED_STATES = frozenset({"planned", "pending", "achieved", "missing", "contradicted", "inconclusive"})


class ManifestError(ValueError):
    """The acceptance manifest cannot support a trustworthy audit."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError("manifest root must be an object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract_ids(contract_path: Path) -> tuple[str, ...]:
    try:
        text = contract_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"cannot read contract: {contract_path}: {exc}") from exc
    identifiers = tuple(REQUIREMENT_PATTERN.findall(text))
    duplicates = sorted(identifier for identifier, count in Counter(identifiers).items() if count > 1)
    if duplicates:
        raise ManifestError(f"duplicate requirement IDs in contract: {duplicates}")
    if not identifiers:
        raise ManifestError("contract has no requirement IDs")
    return identifiers


def validate_schema(manifest_path: Path, *, contract_path: Path) -> dict[str, Any]:
    """Validate planned evidence contracts without claiming final completion."""
    payload = _load_object(manifest_path)
    if payload.get("schema_version") != "praxis-harness-acceptance-v1":
        raise ManifestError("unsupported schema_version")

    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ManifestError("contract must be an object")
    expected_contract_hash = contract.get("sha256")
    if expected_contract_hash != _sha256(contract_path):
        raise ManifestError("contract sha256 does not match current contract")

    bindings = payload.get("evidence_bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ManifestError("evidence_bindings must be a non-empty object")

    requirements = payload.get("requirements")
    if not isinstance(requirements, list):
        raise ManifestError("requirements must be a list")
    identifiers = [requirement.get("id") for requirement in requirements if isinstance(requirement, dict)]
    if len(identifiers) != len(requirements) or not all(isinstance(identifier, str) for identifier in identifiers):
        raise ManifestError("every requirement must have a string id")
    identifier_strings = [str(identifier) for identifier in identifiers]

    duplicate_ids = sorted(identifier for identifier, count in Counter(identifier_strings).items() if count > 1)
    if duplicate_ids:
        raise ManifestError(f"duplicate requirement IDs: {duplicate_ids}")
    contract_ids = set(_contract_ids(contract_path))
    manifest_ids = set(identifier_strings)
    missing_ids = sorted(contract_ids - manifest_ids)
    if missing_ids:
        raise ManifestError(f"missing requirement IDs: {missing_ids}")
    unknown_ids = sorted(manifest_ids - contract_ids)
    if unknown_ids:
        raise ManifestError(f"unknown requirement IDs: {unknown_ids}")

    state_counts: Counter[str] = Counter()
    for requirement in requirements:
        identifier = requirement["id"]
        state = requirement.get("state")
        if state not in ALLOWED_STATES:
            raise ManifestError(f"{identifier}: invalid state: {state!r}")
        state_counts[state] += 1
        command = requirement.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(part, str) and part for part in command):
            raise ManifestError(f"{identifier}: command must be a non-empty argv list")
        evidence = requirement.get("expected_evidence")
        if not isinstance(evidence, dict):
            raise ManifestError(f"{identifier}: expected_evidence must be an object")
        if not isinstance(evidence.get("kind"), str) or not evidence["kind"]:
            raise ManifestError(f"{identifier}: expected_evidence.kind is required")
        artifacts = evidence.get("artifacts")
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or not all(isinstance(artifact, str) and artifact for artifact in artifacts)
        ):
            raise ManifestError(f"{identifier}: expected_evidence.artifacts must be non-empty")
        profile = evidence.get("bindings_profile")
        if profile not in bindings:
            raise ManifestError(f"{identifier}: unknown expected_evidence bindings_profile: {profile!r}")

    return {
        "mode": "schema",
        "schema_valid": True,
        "final_ready": False,
        "requirement_count": len(requirements),
        "state_counts": dict(sorted(state_counts.items())),
        "contract_sha256": expected_contract_hash,
    }


def _run_git(repository: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode(errors="replace").strip()
        raise ManifestError(f"git {' '.join(args)} failed: {message}")
    return completed.stdout


def repository_identity(repository: Path) -> dict[str, str]:
    """Bind evidence to the exact HEAD plus tracked and untracked source state."""
    repository = repository.resolve()
    source_head = _run_git(repository, "rev-parse", "HEAD").decode().strip()
    digest = hashlib.sha256()
    digest.update(source_head.encode())
    digest.update(b"\0tracked-diff\0")
    digest.update(_run_git(repository, "diff", "--binary", "HEAD", "--"))
    untracked = _run_git(
        repository,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    ).split(b"\0")
    for encoded_path in sorted(path for path in untracked if path):
        relative_path = encoded_path.decode("utf-8", errors="surrogateescape")
        path = repository / relative_path
        digest.update(b"\0untracked\0")
        digest.update(encoded_path)
        digest.update(b"\0")
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        else:
            digest.update(b"non-file")
    return {"source_head": source_head, "source_tree": digest.hexdigest()}


def _command_sha256(command: list[str]) -> str:
    encoded = json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _audit_achieved_requirement(
    requirement: dict[str, Any],
    *,
    bindings: dict[str, Any],
    identity: dict[str, str],
    artifact_root: Path,
) -> list[str]:
    identifier = requirement["id"]
    evidence = requirement.get("evidence")
    if not isinstance(evidence, dict):
        return ["missing evidence record"]
    expected = requirement["expected_evidence"]
    profile = expected["bindings_profile"]
    reasons: list[str] = []
    for binding in bindings[profile]:
        if binding == "source_head":
            if evidence.get(binding) != identity[binding]:
                reasons.append("source_head mismatch")
        elif binding == "source_tree":
            if evidence.get(binding) != identity[binding]:
                reasons.append("source_tree mismatch")
        elif binding == "command_sha256":
            if evidence.get(binding) != _command_sha256(requirement["command"]):
                reasons.append("command_sha256 mismatch")
        elif binding == "artifact_sha256":
            continue
        elif not isinstance(evidence.get(binding), str) or not evidence[binding]:
            reasons.append(f"missing binding: {binding}")

    artifact_records = evidence.get("artifacts")
    if not isinstance(artifact_records, list):
        return [*reasons, "missing artifact records"]
    expected_paths = set(expected["artifacts"])
    actual_paths = {
        artifact.get("path")
        for artifact in artifact_records
        if isinstance(artifact, dict) and isinstance(artifact.get("path"), str)
    }
    if actual_paths != expected_paths:
        reasons.append("artifact path set mismatch")
    root = artifact_root.resolve()
    for artifact in artifact_records:
        if not isinstance(artifact, dict):
            reasons.append("invalid artifact record")
            continue
        relative = artifact.get("path")
        expected_hash = artifact.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            reasons.append("invalid artifact record")
            continue
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root):
            reasons.append(f"artifact escapes root: {relative}")
        elif not candidate.is_file():
            reasons.append(f"missing artifact: {relative}")
        elif _sha256(candidate) != expected_hash:
            reasons.append(f"artifact sha256 mismatch: {relative}")
    if not artifact_records:
        reasons.append(f"{identifier}: no artifact records")
    receipt = evidence.get("runner_receipt")
    if not isinstance(receipt, dict):
        reasons.append("missing runner receipt")
        return reasons
    receipt_path = receipt.get("path")
    receipt_hash = receipt.get("sha256")
    if not isinstance(receipt_path, str) or not isinstance(receipt_hash, str):
        reasons.append("invalid runner receipt record")
        return reasons
    candidate = (root / receipt_path).resolve()
    if not candidate.is_relative_to(root):
        reasons.append("runner receipt escapes artifact root")
        return reasons
    if not candidate.is_file():
        reasons.append("missing runner receipt artifact")
        return reasons
    if _sha256(candidate) != receipt_hash:
        reasons.append("runner receipt sha256 mismatch")
        return reasons
    try:
        receipt_payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        reasons.append("invalid runner receipt payload")
        return reasons
    expected_receipt = {
        "schema_version": "praxis-harness-run-receipt-v1",
        "requirement_id": identifier,
        "source_head": evidence.get("source_head"),
        "source_tree": evidence.get("source_tree"),
        "command_sha256": evidence.get("command_sha256"),
        "exit_code": 0,
        "artifacts": artifact_records,
    }
    if receipt_payload != expected_receipt:
        reasons.append("runner receipt payload mismatch")
    return reasons


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_requirement(
    manifest_path: Path,
    *,
    requirement_id: str,
    contract_path: Path,
    repository: Path,
    artifact_root: Path,
    extra_bindings: dict[str, str],
) -> dict[str, Any]:
    """Execute one exact acceptance command and record machine-auditable evidence."""
    validate_schema(manifest_path, contract_path=contract_path)
    payload = _load_object(manifest_path)
    requirement = next(
        (candidate for candidate in payload["requirements"] if candidate["id"] == requirement_id),
        None,
    )
    if requirement is None:
        raise ManifestError(f"unknown requirement ID: {requirement_id}")
    profile = requirement["expected_evidence"]["bindings_profile"]
    required_extras = tuple(
        binding
        for binding in payload["evidence_bindings"][profile]
        if binding not in {"source_head", "source_tree", "command_sha256", "artifact_sha256"}
    )
    missing_extras = [binding for binding in required_extras if not extra_bindings.get(binding)]
    if missing_extras:
        raise ManifestError(f"{requirement_id}: missing run bindings: {missing_extras}")
    unknown_extras = sorted(set(extra_bindings) - set(required_extras))
    if unknown_extras:
        raise ManifestError(f"{requirement_id}: unknown run bindings: {unknown_extras}")

    repository = repository.resolve()
    artifact_root = artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    identity = repository_identity(repository)
    command = requirement["command"]
    environment = os.environ.copy()
    environment.update(
        {
            "PRAXIS_ACCEPTANCE_ARTIFACT_ROOT": str(artifact_root),
            "PRAXIS_ACCEPTANCE_REQUIREMENT_ID": requirement_id,
        }
    )
    completed = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ManifestError(
            f"{requirement_id}: command failed with exit code {completed.returncode}: {completed.stderr[-2000:]}"
        )
    if repository_identity(repository) != identity:
        raise ManifestError(f"{requirement_id}: command changed the bound repository source tree")

    artifact_records: list[dict[str, str]] = []
    root = artifact_root.resolve()
    for relative in requirement["expected_evidence"]["artifacts"]:
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root):
            raise ManifestError(f"{requirement_id}: artifact escapes root: {relative}")
        if not candidate.is_file():
            raise ManifestError(f"{requirement_id}: command did not produce: {relative}")
        artifact_records.append({"path": relative, "sha256": _sha256(candidate)})

    evidence: dict[str, Any] = {
        **identity,
        "command_sha256": _command_sha256(command),
        "artifacts": artifact_records,
        **extra_bindings,
    }
    receipt_payload = {
        "schema_version": "praxis-harness-run-receipt-v1",
        "requirement_id": requirement_id,
        "source_head": evidence["source_head"],
        "source_tree": evidence["source_tree"],
        "command_sha256": evidence["command_sha256"],
        "exit_code": 0,
        "artifacts": artifact_records,
    }
    receipt_relative = f".acceptance_receipts/{requirement_id}/{uuid4().hex}.json"
    receipt_path = artifact_root / receipt_relative
    _write_json_atomic(receipt_path, receipt_payload)
    evidence["runner_receipt"] = {
        "path": receipt_relative,
        "sha256": _sha256(receipt_path),
    }
    requirement["state"] = "achieved"
    requirement["evidence"] = evidence
    _write_json_atomic(manifest_path, payload)
    return {
        "mode": "run",
        "requirement_id": requirement_id,
        "exit_code": 0,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "runner_receipt": evidence["runner_receipt"],
    }


def audit_final(
    manifest_path: Path,
    *,
    contract_path: Path,
    repository: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    """Verify current evidence; planned or unverifiable entries block completion."""
    validate_schema(manifest_path, contract_path=contract_path)
    payload = _load_object(manifest_path)
    identity = repository_identity(repository)
    bindings = payload["evidence_bindings"]
    results: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for requirement in payload["requirements"]:
        declared_state = requirement["state"]
        reasons: list[str] = []
        if declared_state == "achieved":
            reasons = _audit_achieved_requirement(
                requirement,
                bindings=bindings,
                identity=identity,
                artifact_root=artifact_root,
            )
            status = "contradicted" if reasons else "achieved"
        elif declared_state in {"contradicted", "inconclusive"}:
            status = declared_state
            reasons = [f"declared {declared_state}"]
        else:
            status = "missing"
            reasons = [f"declared {declared_state}"]
        status_counts[status] += 1
        results.append({"id": requirement["id"], "status": status, "reasons": reasons})
    return {
        "mode": "final",
        "schema_valid": True,
        "final_ready": status_counts == {"achieved": len(results)},
        "requirement_count": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "source_head": identity["source_head"],
        "source_tree": identity["source_tree"],
        "results": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-harness-acceptance")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--schema", action="store_true", required=True)
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--contract", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("manifest", type=Path)
    run.add_argument("--requirement", required=True)
    run.add_argument("--contract", type=Path, required=True)
    run.add_argument("--repository", type=Path, required=True)
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument("--binding", action="append", default=[])
    audit = subparsers.add_parser("audit")
    audit.add_argument("--final", action="store_true", required=True)
    audit.add_argument("manifest", type=Path)
    audit.add_argument("--contract", type=Path, required=True)
    audit.add_argument("--repository", type=Path, required=True)
    audit.add_argument("--artifact-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_schema(args.manifest, contract_path=args.contract)
        elif args.command == "run":
            bindings: dict[str, str] = {}
            for raw_binding in args.binding:
                key, separator, value = raw_binding.partition("=")
                if not separator or not key or not value:
                    raise ManifestError("--binding must use non-empty key=value")
                if key in bindings:
                    raise ManifestError(f"duplicate --binding: {key}")
                bindings[key] = value
            report = run_requirement(
                args.manifest,
                requirement_id=args.requirement,
                contract_path=args.contract,
                repository=args.repository,
                artifact_root=args.artifact_root,
                extra_bindings=bindings,
            )
        else:
            report = audit_final(
                args.manifest,
                contract_path=args.contract,
                repository=args.repository,
                artifact_root=args.artifact_root,
            )
    except ManifestError as exc:
        print(
            json.dumps(
                {
                    "mode": ("schema" if args.command == "validate" else "final" if args.command == "audit" else "run"),
                    "schema_valid": False,
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if args.command in {"validate", "run"} or report["final_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
