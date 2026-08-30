from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT_PATH = ROOT / "scripts" / "agent_harness_acceptance.py"
CASE_RUNNER_PATH = ROOT / "scripts" / "run_harness_acceptance_case.py"
MANIFEST_PATH = ROOT / "evals" / "harness" / "acceptance_v1.json"
CONTRACT_PATH = ROOT / "docs" / "design" / "praxis_harness_architecture.md"


def _load_module():
    spec = importlib.util.spec_from_file_location("agent_harness_acceptance", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_payload() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_harness_acceptance_entrypoint_and_manifest_exist() -> None:
    assert SCRIPT_PATH.is_file()
    assert CASE_RUNNER_PATH.is_file()
    assert MANIFEST_PATH.is_file()


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


def test_schema_validation_accepts_planned_manifest_with_exact_contract_coverage() -> None:
    module = _load_module()

    report = module.validate_schema(MANIFEST_PATH, contract_path=CONTRACT_PATH)

    assert report["mode"] == "schema"
    assert report["schema_valid"] is True
    assert report["final_ready"] is False
    assert report["requirement_count"] == 84
    assert report["state_counts"] == {"planned": 84}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown"])
def test_schema_validation_rejects_requirement_coverage_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _load_module()
    payload = _manifest_payload()
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
        module.validate_schema(path, contract_path=CONTRACT_PATH)


@pytest.mark.parametrize("field", ["command", "expected_evidence"])
def test_schema_validation_requires_executable_evidence_contract(
    tmp_path: Path,
    field: str,
) -> None:
    module = _load_module()
    payload = _manifest_payload()
    requirements = payload["requirements"]
    assert isinstance(requirements, list)
    requirements[0].pop(field)
    path = tmp_path / "acceptance.json"
    _write_manifest(path, payload)

    with pytest.raises(module.ManifestError, match=field):
        module.validate_schema(path, contract_path=CONTRACT_PATH)


def test_schema_validation_rejects_contract_hash_drift(tmp_path: Path) -> None:
    module = _load_module()
    payload = _manifest_payload()
    contract = payload["contract"]
    assert isinstance(contract, dict)
    contract["sha256"] = "0" * 64
    path = tmp_path / "acceptance.json"
    _write_manifest(path, payload)

    with pytest.raises(module.ManifestError, match="contract sha256"):
        module.validate_schema(path, contract_path=CONTRACT_PATH)


def test_validate_schema_cli_labels_nonfinal_mode() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "validate",
            "--schema",
            str(MANIFEST_PATH),
            "--contract",
            str(CONTRACT_PATH),
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


def _achieved_manifest(module, tmp_path: Path) -> tuple[Path, Path]:
    payload = _manifest_payload()
    identity = module.repository_identity(ROOT)
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


def test_final_audit_rejects_planned_manifest() -> None:
    module = _load_module()

    report = module.audit_final(
        MANIFEST_PATH,
        contract_path=CONTRACT_PATH,
        repository=ROOT,
        artifact_root=ROOT,
    )

    assert report["mode"] == "final"
    assert report["final_ready"] is False
    assert report["status_counts"] == {"missing": 84}


def test_final_audit_cli_returns_nonzero_for_planned_manifest() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "audit",
            "--final",
            str(MANIFEST_PATH),
            "--contract",
            str(CONTRACT_PATH),
            "--repository",
            str(ROOT),
            "--artifact-root",
            str(ROOT),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["mode"] == "final"
    assert report["final_ready"] is False
    assert report["status_counts"] == {"missing": 84}


def test_final_audit_accepts_only_current_hashed_evidence(tmp_path: Path) -> None:
    module = _load_module()
    manifest_path, artifact_root = _achieved_manifest(module, tmp_path)

    report = module.audit_final(
        manifest_path,
        contract_path=CONTRACT_PATH,
        repository=ROOT,
        artifact_root=artifact_root,
    )

    assert report["mode"] == "final"
    assert report["final_ready"] is True
    assert report["status_counts"] == {"achieved": 84}


def test_final_audit_rejects_achieved_state_without_runner_receipt(
    tmp_path: Path,
) -> None:
    module = _load_module()
    manifest_path, artifact_root = _achieved_manifest(module, tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for requirement in payload["requirements"]:
        requirement["evidence"].pop("runner_receipt")
    _write_manifest(manifest_path, payload)

    report = module.audit_final(
        manifest_path,
        contract_path=CONTRACT_PATH,
        repository=ROOT,
        artifact_root=artifact_root,
    )

    assert report["final_ready"] is False
    assert report["status_counts"] == {"contradicted": 84}
    assert all(
        "missing runner receipt" in result["reasons"]
        for result in report["results"]
    )


def test_run_requirement_executes_exact_command_and_records_receipt(
    tmp_path: Path,
) -> None:
    module = _load_module()
    payload = _manifest_payload()
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
        contract_path=CONTRACT_PATH,
        repository=ROOT,
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
        contract_path=CONTRACT_PATH,
        repository=ROOT,
        artifact_root=artifact_root,
    )
    assert audit["results"][0] == {
        "id": "ARCH-01",
        "status": "achieved",
        "reasons": [],
    }


def test_run_cli_records_evidence_in_the_requested_manifest(tmp_path: Path) -> None:
    payload = _manifest_payload()
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
            str(CONTRACT_PATH),
            "--repository",
            str(ROOT),
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
    tamper: str,
) -> None:
    module = _load_module()
    manifest_path, artifact_root = _achieved_manifest(module, tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tamper == "artifact":
        artifact = artifact_root / "artifacts" / "harness" / "ARCH-01" / "evidence.json"
        artifact.write_text("tampered\n", encoding="utf-8")
    else:
        payload["requirements"][0]["evidence"][tamper] = "0" * 64
        _write_manifest(manifest_path, payload)

    report = module.audit_final(
        manifest_path,
        contract_path=CONTRACT_PATH,
        repository=ROOT,
        artifact_root=artifact_root,
    )

    assert report["final_ready"] is False
    assert report["status_counts"]["contradicted"] == 1
