from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT_PATH = ROOT / "scripts" / "agent_harness_acceptance.py"
CASE_RUNNER_PATH = ROOT / "scripts" / "run_harness_acceptance_case.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("agent_harness_acceptance", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


@dataclass(frozen=True)
class AcceptanceAssets:
    contract: Path
    manifest: Path
    repository: Path


@pytest.fixture
def acceptance_assets(tmp_path: Path) -> AcceptanceAssets:
    """Three synthetic requirements test the validator, not product readiness."""
    contract = tmp_path / "contract.md"
    contract.write_text(
        "# Synthetic validator contract\n\n"
        "**ARCH-01** Code evidence.\n"
        "**RUN-01** Runner evidence.\n"
        "**BENCH-01** Benchmark evidence.\n",
        encoding="utf-8",
    )
    code_bindings = ["source_head", "source_tree", "command_sha256", "artifact_sha256"]
    payload = {
        "schema_version": "praxis-harness-acceptance-v1",
        "contract": {"path": contract.name, "sha256": hashlib.sha256(contract.read_bytes()).hexdigest()},
        "evidence_bindings": {
            "code": code_bindings,
            "benchmark": [*code_bindings, "model", "task_revision", "evaluator_revision"],
        },
        "requirements": [
            {
                "id": identifier,
                "state": "planned",
                "command": [sys.executable, "-c", "print('synthetic validator case')"],
                "expected_evidence": {
                    "kind": "synthetic_validator_evidence",
                    "artifacts": [f"artifacts/harness/{identifier}/evidence.json"],
                    "bindings_profile": profile,
                },
            }
            for identifier, profile in (("ARCH-01", "code"), ("RUN-01", "code"), ("BENCH-01", "benchmark"))
        ],
    }
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, payload)
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    for arguments in (
        ("init", "-q"),
        ("add", "source.py"),
        ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"),
    ):
        subprocess.run(["git", *arguments], cwd=repository, check=True, capture_output=True)
    return AcceptanceAssets(contract, manifest, repository)


def test_harness_acceptance_entrypoints_exist() -> None:
    assert SCRIPT_PATH.is_file()
    assert CASE_RUNNER_PATH.is_file()


def test_case_runner_writes_evidence_only_after_exact_command_succeeds(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    environment = {
        **os.environ,
        "PRAXIS_ACCEPTANCE_ARTIFACT_ROOT": str(artifact_root),
        "PRAXIS_ACCEPTANCE_REQUIREMENT_ID": "MIG-01",
    }
    relative = "artifacts/harness/MIG-01/evidence.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CASE_RUNNER_PATH),
            "--requirement",
            "MIG-01",
            "--artifact",
            relative,
            "--",
            sys.executable,
            "-c",
            "print('verified migration case')",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    evidence = json.loads((artifact_root / relative).read_text(encoding="utf-8"))
    assert evidence["schema_version"] == "praxis-harness-case-evidence-v1"
    assert evidence["requirement_id"] == "MIG-01"
    assert evidence["exit_code"] == 0
    assert evidence["stdout_sha256"] == hashlib.sha256(
        b"verified migration case\n"
    ).hexdigest()


def test_case_runner_does_not_write_evidence_when_command_fails(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    environment = {
        **os.environ,
        "PRAXIS_ACCEPTANCE_ARTIFACT_ROOT": str(artifact_root),
        "PRAXIS_ACCEPTANCE_REQUIREMENT_ID": "MIG-01",
    }
    relative = "artifacts/harness/MIG-01/evidence.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CASE_RUNNER_PATH),
            "--requirement",
            "MIG-01",
            "--artifact",
            relative,
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 7
    assert not (artifact_root / relative).exists()


def test_schema_validation_accepts_planned_manifest_with_exact_contract_coverage(
    acceptance_assets: AcceptanceAssets,
) -> None:
    module = _load_module()

    report = module.validate_schema(acceptance_assets.manifest, contract_path=acceptance_assets.contract)

    assert report["mode"] == "schema"
    assert report["schema_valid"] is True
    assert report["final_ready"] is False
    assert report["requirement_count"] == 3
    assert report["state_counts"] == {"planned": 3}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown"])
def test_schema_validation_rejects_requirement_coverage_drift(
    tmp_path: Path,
    acceptance_assets: AcceptanceAssets,
    mutation: str,
) -> None:
    module = _load_module()
    payload = _manifest_payload(acceptance_assets.manifest)
    requirements = payload["requirements"]
    assert isinstance(requirements, list)
    if mutation == "missing":
        requirements.pop()
    elif mutation == "duplicate":
        requirements.append(deepcopy(requirements[0]))
    else:
        requirement = deepcopy(requirements[0])
        requirement["id"] = "UNKNOWN-99"
        requirements.append(requirement)
    path = tmp_path / "acceptance.json"
    _write_manifest(path, payload)

    with pytest.raises(module.ManifestError, match=mutation):
        module.validate_schema(path, contract_path=acceptance_assets.contract)


@pytest.mark.parametrize("field", ["command", "expected_evidence"])
def test_schema_validation_requires_executable_evidence_contract(
    tmp_path: Path,
    acceptance_assets: AcceptanceAssets,
    field: str,
) -> None:
    module = _load_module()
    payload = _manifest_payload(acceptance_assets.manifest)
    requirements = payload["requirements"]
    assert isinstance(requirements, list)
    requirements[0].pop(field)
    path = tmp_path / "acceptance.json"
    _write_manifest(path, payload)

    with pytest.raises(module.ManifestError, match=field):
        module.validate_schema(path, contract_path=acceptance_assets.contract)


def test_schema_validation_rejects_contract_hash_drift(
    tmp_path: Path, acceptance_assets: AcceptanceAssets,
) -> None:
    module = _load_module()
    payload = _manifest_payload(acceptance_assets.manifest)
    contract = payload["contract"]
    assert isinstance(contract, dict)
    contract["sha256"] = "0" * 64
    path = tmp_path / "acceptance.json"
    _write_manifest(path, payload)

    with pytest.raises(module.ManifestError, match="contract sha256"):
        module.validate_schema(path, contract_path=acceptance_assets.contract)


def test_validate_schema_cli_labels_nonfinal_mode(acceptance_assets: AcceptanceAssets) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "validate",
            "--schema",
            str(acceptance_assets.manifest),
            "--contract",
            str(acceptance_assets.contract),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["mode"] == "schema"
    assert report["schema_valid"] is True
    assert report["final_ready"] is False


def _command_sha256(command: list[str]) -> str:
    encoded = json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _achieved_manifest(module, tmp_path: Path, acceptance_assets: AcceptanceAssets) -> tuple[Path, Path]:
    payload = _manifest_payload(acceptance_assets.manifest)
    identity = module.repository_identity(acceptance_assets.repository)
    artifact_root = tmp_path / "artifacts"
    requirements = payload["requirements"]
    assert isinstance(requirements, list)
    for requirement in requirements:
        identifier = requirement["id"]
        relative_artifact = requirement["expected_evidence"]["artifacts"][0]
        artifact = artifact_root / relative_artifact
        artifact.parent.mkdir(parents=True)
        artifact.write_text(f"evidence for {identifier}\n", encoding="utf-8")
        evidence = {
            "source_head": identity["source_head"],
            "source_tree": identity["source_tree"],
            "command_sha256": _command_sha256(requirement["command"]),
            "artifacts": [
                {
                    "path": relative_artifact,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ],
        }
        if requirement["expected_evidence"]["bindings_profile"] == "benchmark":
            evidence.update(
                {
                    "model": "frozen-real-model",
                    "task_revision": "task-v1",
                    "evaluator_revision": "evaluator-v1",
                }
            )
        receipt_payload = {
            "schema_version": "praxis-harness-run-receipt-v1",
            "requirement_id": identifier,
            "source_head": evidence["source_head"],
            "source_tree": evidence["source_tree"],
            "command_sha256": evidence["command_sha256"],
            "exit_code": 0,
            "artifacts": evidence["artifacts"],
        }
        receipt_relative = f".acceptance_receipts/{identifier}.json"
        receipt = artifact_root / receipt_relative
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
        evidence["runner_receipt"] = {
            "path": receipt_relative,
            "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        }
        requirement["state"] = "achieved"
        requirement["evidence"] = evidence
    path = tmp_path / "acceptance.json"
    _write_manifest(path, payload)
    return path, artifact_root


def test_final_audit_rejects_planned_manifest(acceptance_assets: AcceptanceAssets) -> None:
    module = _load_module()

    report = module.audit_final(
        acceptance_assets.manifest,
        contract_path=acceptance_assets.contract,
        repository=acceptance_assets.repository,
        artifact_root=acceptance_assets.manifest.parent,
    )

    assert report["mode"] == "final"
    assert report["final_ready"] is False
    assert report["status_counts"] == {"missing": 3}


def test_final_audit_cli_returns_nonzero_for_planned_manifest(acceptance_assets: AcceptanceAssets) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "audit",
            "--final",
            str(acceptance_assets.manifest),
            "--contract",
            str(acceptance_assets.contract),
            "--repository",
            str(acceptance_assets.repository),
            "--artifact-root",
            str(acceptance_assets.manifest.parent),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["mode"] == "final"
    assert report["final_ready"] is False
    assert report["status_counts"] == {"missing": 3}


def test_final_audit_accepts_only_current_hashed_evidence(
    tmp_path: Path, acceptance_assets: AcceptanceAssets,
) -> None:
    module = _load_module()
    manifest_path, artifact_root = _achieved_manifest(module, tmp_path, acceptance_assets)

    report = module.audit_final(
        manifest_path,
        contract_path=acceptance_assets.contract,
        repository=acceptance_assets.repository,
        artifact_root=artifact_root,
    )

    assert report["mode"] == "final"
    assert report["final_ready"] is True
    assert report["status_counts"] == {"achieved": 3}


def test_final_audit_rejects_achieved_state_without_runner_receipt(
    tmp_path: Path,
    acceptance_assets: AcceptanceAssets,
) -> None:
    module = _load_module()
    manifest_path, artifact_root = _achieved_manifest(module, tmp_path, acceptance_assets)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for requirement in payload["requirements"]:
        requirement["evidence"].pop("runner_receipt")
    _write_manifest(manifest_path, payload)

    report = module.audit_final(
        manifest_path,
        contract_path=acceptance_assets.contract,
        repository=acceptance_assets.repository,
        artifact_root=artifact_root,
    )

    assert report["final_ready"] is False
    assert report["status_counts"] == {"contradicted": 3}
    assert all(
        "missing runner receipt" in result["reasons"]
        for result in report["results"]
    )


def test_run_requirement_executes_exact_command_and_records_receipt(
    tmp_path: Path,
    acceptance_assets: AcceptanceAssets,
) -> None:
    module = _load_module()
    payload = _manifest_payload(acceptance_assets.manifest)
    requirement = payload["requirements"][0]
    relative_artifact = requirement["expected_evidence"]["artifacts"][0]
    requirement["command"] = [
        sys.executable,
        "-c",
        (
            "import os; from pathlib import Path; "
            f"p=Path(os.environ['PRAXIS_ACCEPTANCE_ARTIFACT_ROOT'])/{relative_artifact!r}; "
            "p.parent.mkdir(parents=True, exist_ok=True); "
            "p.write_text('executed by acceptance runner\\n', encoding='utf-8')"
        ),
    ]
    manifest_path = tmp_path / "acceptance.json"
    artifact_root = tmp_path / "evidence"
    _write_manifest(manifest_path, payload)

    report = module.run_requirement(
        manifest_path,
        requirement_id="ARCH-01",
        contract_path=acceptance_assets.contract,
        repository=acceptance_assets.repository,
        artifact_root=artifact_root,
        extra_bindings={},
    )

    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))["requirements"][0]
    assert report["requirement_id"] == "ARCH-01"
    assert report["exit_code"] == 0
    assert recorded["state"] == "achieved"
    assert recorded["evidence"]["command_sha256"] == _command_sha256(
        recorded["command"]
    )
    receipt = recorded["evidence"]["runner_receipt"]
    assert (artifact_root / receipt["path"]).is_file()

    audit = module.audit_final(
        manifest_path,
        contract_path=acceptance_assets.contract,
        repository=acceptance_assets.repository,
        artifact_root=artifact_root,
    )
    assert audit["results"][0] == {
        "id": "ARCH-01",
        "status": "achieved",
        "reasons": [],
    }


def test_run_cli_records_evidence_in_the_requested_manifest(
    tmp_path: Path, acceptance_assets: AcceptanceAssets,
) -> None:
    payload = _manifest_payload(acceptance_assets.manifest)
    requirement = payload["requirements"][0]
    relative_artifact = requirement["expected_evidence"]["artifacts"][0]
    requirement["command"] = [
        sys.executable,
        "-c",
        (
            "import os; from pathlib import Path; "
            f"p=Path(os.environ['PRAXIS_ACCEPTANCE_ARTIFACT_ROOT'])/{relative_artifact!r}; "
            "p.parent.mkdir(parents=True, exist_ok=True); p.write_text('ok')"
        ),
    ]
    manifest_path = tmp_path / "acceptance.json"
    artifact_root = tmp_path / "evidence"
    _write_manifest(manifest_path, payload)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "run",
            str(manifest_path),
            "--requirement",
            "ARCH-01",
            "--contract",
            str(acceptance_assets.contract),
            "--repository",
            str(acceptance_assets.repository),
            "--artifact-root",
            str(artifact_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["mode"] == "run"
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert recorded["requirements"][0]["state"] == "achieved"


@pytest.mark.parametrize("tamper", ["source_head", "command_sha256", "artifact"])
def test_final_audit_rejects_stale_or_tampered_evidence(
    tmp_path: Path,
    acceptance_assets: AcceptanceAssets,
    tamper: str,
) -> None:
    module = _load_module()
    manifest_path, artifact_root = _achieved_manifest(module, tmp_path, acceptance_assets)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tamper == "artifact":
        artifact = artifact_root / "artifacts" / "harness" / "ARCH-01" / "evidence.json"
        artifact.write_text("tampered\n", encoding="utf-8")
    else:
        payload["requirements"][0]["evidence"][tamper] = "0" * 64
        _write_manifest(manifest_path, payload)

    report = module.audit_final(
        manifest_path,
        contract_path=acceptance_assets.contract,
        repository=acceptance_assets.repository,
        artifact_root=artifact_root,
    )

    assert report["final_ready"] is False
    assert report["status_counts"]["contradicted"] == 1
