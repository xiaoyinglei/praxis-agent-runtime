from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "model_quality_cases.json"
BASELINE_PATH = Path(__file__).parents[2] / "evals" / "model_quality" / "baseline_v1.json"
SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "agent_model_quality_gate.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location(
        "agent_model_quality_gate",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _trial_metrics(**overrides: float) -> dict[str, float]:
    metrics = {
        "task_success_rate": 1.0,
        "file_tool_selection_rate": 1.0,
        "failure_recovery_rate": 1.0,
        "approval_continuation_rate": 1.0,
        "repeated_failure_control_rate": 1.0,
        "argument_validity_rate": 1.0,
        "redundant_tool_call_rate": 0.0,
        "mean_tool_calls_per_case": 1.4,
        "mean_model_calls_per_case": 2.4,
    }
    metrics.update(overrides)
    return metrics


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _clean_git_repository(root: Path) -> Path:
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "quality-gate@example.test")
    _git(root, "config", "user.name", "Quality Gate Test")
    (root / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (root / "tracked.txt").write_text("committed\n", encoding="utf-8")
    _git(root, "add", ".gitignore", "tracked.txt")
    _git(root, "commit", "--quiet", "-m", "fixture")
    return root


def _quality_suite() -> dict[str, object]:
    return {
        "schema_version": 1,
        "suite_id": "agent-model-tool-quality-v1",
        "models": ["groq_gpt_oss_120b"],
        "cases": [
            {
                "id": "approval_continue",
                "capability": "approval_continuation",
                "task": "Change before_gate to after_gate.",
                "workspace_files": {"approval.txt": "before_gate\n"},
                "workspace_assertions": {
                    "input_files/approval.txt": "after_gate\n"
                },
            }
        ],
    }


def _passing_model_report() -> dict[str, object]:
    metrics = _trial_metrics()
    return {
        "status": "completed",
        "model_alias": "groq_gpt_oss_120b",
        "provider": "groq",
        "provider_model": "openai/gpt-oss-120b",
        "trial_count": 1,
        "trial_metrics": [metrics],
        "trials": [],
    }


def _inconclusive_model_report(
    module: object,
    *,
    case_id: str = "approval_continue",
    capability: str = "approval_continuation",
    model_alias: str = "groq_gpt_oss_120b",
    provider: str = "groq",
    provider_model: str = "openai/gpt-oss-120b",
) -> dict[str, object]:
    observation = module.CaseObservation(
        case_id=case_id,
        capability=capability,
        status="failed",
        answer=None,
        tool_calls=(),
        model_calls=0,
        input_tokens=0,
        output_tokens=0,
        latency_ms=0.0,
        tool_schema_bytes=0,
        approval_pause_observed=False,
        approval_kind=None,
        approval_resumes=0,
        workspace_assertions_passed=False,
        stop_reason="model_provider_failed",
        diagnostic_codes=("model_provider_failed",),
        diagnostic_error_types=("RateLimitError",),
        infrastructure_failure=True,
    )
    return {
        "status": "inconclusive",
        "model_alias": model_alias,
        "provider": provider,
        "provider_model": provider_model,
        "trial_count": 3,
        "trial_metrics": [],
        "trials": [
            {
                "trial": 1,
                "metrics": None,
                "informational_metrics": None,
                "cases": [
                    {
                        "observation": observation.payload(),
                        "score": {
                            "case_id": case_id,
                            "capability": capability,
                            "passed": None,
                            "core_success": None,
                            "capability_passed": None,
                            "inconclusive": True,
                        },
                    }
                ],
            }
        ],
        "infrastructure_failure": {
            "case_id": case_id,
            "stop_reason": "model_provider_failed",
            "diagnostic_error_types": ["RateLimitError"],
        },
    }


def test_evaluator_version_tracks_namespaced_least_authority_contract() -> None:
    module = _load_gate_module()

    assert module.EVALUATOR_VERSION == "agent_model_quality_gate_v3"


def test_repository_fingerprint_binds_clean_commit_and_tree(tmp_path: Path) -> None:
    module = _load_gate_module()
    repository = _clean_git_repository(tmp_path / "repository")

    fingerprint = module.repository_fingerprint(repository)

    assert fingerprint.dirty is False
    assert fingerprint.source_commit == _git(repository, "rev-parse", "HEAD")
    assert fingerprint.source_tree == _git(repository, "rev-parse", "HEAD^{tree}")


@pytest.mark.parametrize("change", ["tracked", "untracked"])
def test_repository_fingerprint_rejects_non_ignored_changes(
    tmp_path: Path,
    change: str,
) -> None:
    module = _load_gate_module()
    repository = _clean_git_repository(tmp_path / "repository")
    if change == "tracked":
        (repository / "tracked.txt").write_text("modified\n", encoding="utf-8")
    else:
        (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(module.DirtyRepositoryError):
        module.repository_fingerprint(repository)


def test_repository_fingerprint_allows_ignored_untracked_files(tmp_path: Path) -> None:
    module = _load_gate_module()
    repository = _clean_git_repository(tmp_path / "repository")
    (repository / "ignored.txt").write_text("ignored\n", encoding="utf-8")

    fingerprint = module.repository_fingerprint(repository)

    assert fingerprint.dirty is False


def test_repository_fingerprint_invokes_git_with_argv_and_without_shell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_gate_module()
    repository = _clean_git_repository(tmp_path / "repository")
    real_run = module.subprocess.run
    calls: list[tuple[object, dict[str, Any]]] = []

    def recording_run(command: object, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return real_run(command, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", recording_run)

    module.repository_fingerprint(repository)

    assert calls
    assert all(isinstance(command, list) for command, _kwargs in calls)
    assert all(kwargs.get("shell") is not True for _command, kwargs in calls)


@pytest.mark.anyio
async def test_gate_preflight_runs_before_model_calls_and_shapes_redacted_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_gate_module()
    suite = _quality_suite()
    secret = "provider-secret-sentinel"
    auth_secret = "provider-auth-sentinel"
    cookie_secret = "provider-cookie-sentinel"
    credential_secret = "provider-credential-sentinel"
    absolute_path = str(tmp_path / "private" / "approval.txt")
    windows_path = r"C:\private\approval.txt"
    unc_path = r"\\private-server\workspace\approval.txt"
    monkeypatch.setenv("GROQ_API_KEY", secret)
    monkeypatch.setenv("GROQ_SESSION_AUTH", auth_secret)
    monkeypatch.setenv("GROQ_SESSION_COOKIE", cookie_secret)
    monkeypatch.setenv("GROQ_CREDENTIAL", credential_secret)
    raw_case = suite["cases"][0]
    assert isinstance(raw_case, dict)
    raw_case["task"] = (
        f"Change before_gate to after_gate using {absolute_path} token {secret} "
        f"auth {auth_secret} cookie {cookie_secret} credential {credential_secret}."
    )
    raw_case["workspace_files"] = {
        "approval.txt": f"before_gate\n{secret}\n"
    }
    fingerprint = SimpleNamespace(
        source_commit="a" * 40,
        source_tree="b" * 40,
        dirty=False,
    )
    events: list[str] = []

    def preflight(repository: Path) -> object:
        assert repository == tmp_path / "repository"
        events.append("preflight")
        return fingerprint

    async def run_trials(**_kwargs: object) -> dict[str, object]:
        events.append("model")
        model_report = _passing_model_report()
        model_report["trials"] = [
            {
                "trial": 1,
                "metrics": _trial_metrics(),
                "informational_metrics": {},
                "cases": [
                    {
                        "observation": {
                            "case_id": "approval_continue",
                            "capability": "approval_continuation",
                            "status": "done",
                            "answer": f"Updated {absolute_path} with {secret}.",
                            "tool_calls": [
                                {
                                    "tool_call_id": "call-1",
                                    "tool_name": "apply_patch",
                                    "arguments": {
                                        "file_path": absolute_path,
                                        "Authorization": f"Bearer {secret}",
                                        "nested": [
                                            {
                                                "access_token": "access-token-value",
                                                "client-secret": "client-secret-value",
                                                "x_api_key": "x-api-key-value",
                                                "privateKey": "private-key-value",
                                                "cookie": "cookie-value",
                                                "auth": "auth-value",
                                                "apiKey": "api-key-value",
                                                "monkey": "banana",
                                            }
                                        ],
                                    },
                                    "structured_output": {
                                        "summary": "VISIBLE_TOOL_RESULT",
                                        "message": f"result contains {secret}",
                                        "paths": {
                                            "posix": absolute_path,
                                            "windows": windows_path,
                                            "unc": unc_path,
                                        },
                                        "nested": {
                                            "client_secret": "nested-secret-value",
                                            "monkey": "banana",
                                        },
                                    },
                                    "is_error": True,
                                    "error_code": "runner_failed",
                                    "error_message": (
                                        f"failed at {absolute_path}, {windows_path}, {unc_path}; credential {secret}"
                                    ),
                                    "retryable": True,
                                    "truncated": True,
                                    "latency_ms": None,
                                }
                            ],
                            "model_calls": 2,
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "latency_ms": 12.5,
                            "tool_schema_bytes": 400,
                            "approval_pause_observed": True,
                            "approval_kind": "tool_approval",
                                "approval_resumes": 1,
                                "workspace_assertions_passed": True,
                                "runtime_input_namespace": "turn-123",
                                "stop_reason": "completed",
                            "diagnostic_error_types": [],
                            "error": f"diagnostic {secret}",
                            "infrastructure_failure": False,
                        },
                        "score": {
                            "case_id": "approval_continue",
                            "capability": "approval_continuation",
                            "passed": True,
                            "core_success": True,
                            "capability_passed": True,
                        },
                    }
                ],
            }
        ]
        return model_report

    monkeypatch.setattr(module, "source_repository_fingerprint", preflight)
    monkeypatch.setattr(module, "load_suite", lambda _path: suite)
    monkeypatch.setattr(module, "validate_baseline", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "run_model_trials", run_trials)
    monkeypatch.setattr(
        module,
        "evaluate_model_gate",
        lambda **_kwargs: {
            "model_alias": "groq_gpt_oss_120b",
            "provider_model": "openai/gpt-oss-120b",
            "passed": True,
            "observed": _trial_metrics(),
            "thresholds": {},
            "failures": [],
        },
    )
    baseline = tmp_path / "private" / "baseline.json"
    baseline.parent.mkdir()
    baseline.write_text(
        json.dumps(
            {
                "models": {
                    "groq_gpt_oss_120b": {
                        "trial_count": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    args = SimpleNamespace(
        repository=tmp_path / "repository",
        fixture=tmp_path / "private" / "cases.json",
        baseline=baseline,
        env_file=tmp_path / "private" / ".env",
        models=["groq_gpt_oss_120b"],
        report=report,
    )

    exit_code = await module._gate(args)

    payload = json.loads(report.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert exit_code == 0
    assert events == ["preflight", "model", "preflight"]
    assert payload["source_commit"] == "a" * 40
    assert payload["source_tree"] == "b" * 40
    assert payload["source_unchanged"] is True
    assert payload["dirty"] is False
    assert set(payload["runtime_platform"]) == {
        "os",
        "os_release",
        "architecture",
        "python_version",
        "python_implementation",
    }
    assert all(
        isinstance(value, str) and value
        for value in payload["runtime_platform"].values()
    )
    assert payload["suite_revision"] == module.suite_revision(suite)
    assert payload["evaluator_version"] == module.EVALUATOR_VERSION
    assert payload["case_metadata"] == [
        {
            "case_id": "approval_continue",
            "capability": "approval_continuation",
            "task": (
                "Change before_gate to after_gate using "
                "[REDACTED_ABSOLUTE_PATH] token [REDACTED] auth [REDACTED] "
                "cookie [REDACTED] credential [REDACTED]."
            ),
            "workspace_before": {
                "input_files/approval.txt": "before_gate\n[REDACTED]\n"
            },
            "workspace_after": {
                "input_files/approval.txt": "after_gate\n"
            },
        }
    ]
    assert str(tmp_path) not in serialized
    assert secret not in serialized
    assert auth_secret not in serialized
    assert cookie_secret not in serialized
    assert credential_secret not in serialized
    assert windows_path not in serialized
    assert unc_path not in serialized
    assert "Authorization" not in serialized
    assert "access_token" not in serialized
    assert "client-secret" not in serialized
    assert "x_api_key" not in serialized
    assert "privateKey" not in serialized
    assert '"cookie"' not in serialized
    assert '"auth"' not in serialized
    assert "apiKey" not in serialized
    assert '"monkey": "banana"' in serialized
    tool_call = payload["runs"][0]["trials"][0]["cases"][0]["observation"]["tool_calls"][0]
    assert tool_call["structured_output"]["summary"] == "VISIBLE_TOOL_RESULT"
    assert tool_call["structured_output"]["message"] == "result contains [REDACTED]"
    assert tool_call["structured_output"]["paths"] == {
        "posix": "[REDACTED_ABSOLUTE_PATH]",
        "windows": "[REDACTED_ABSOLUTE_PATH]",
        "unc": "[REDACTED_ABSOLUTE_PATH]",
    }
    assert tool_call["structured_output"]["nested"] == {"monkey": "banana"}
    assert tool_call["error_code"] == "runner_failed"
    assert tool_call["retryable"] is True
    assert tool_call["truncated"] is True
    assert tool_call["latency_ms"] is None
    assert absolute_path not in tool_call["error_message"]
    assert windows_path not in tool_call["error_message"]
    assert unc_path not in tool_call["error_message"]
    assert secret not in tool_call["error_message"]
    assert ".env" not in serialized

    renderer_path = SCRIPT_PATH.with_name("render_model_quality_report.py")
    renderer_spec = importlib.util.spec_from_file_location(
        "render_model_quality_report_from_sanitized_gate",
        renderer_path,
    )
    assert renderer_spec is not None
    assert renderer_spec.loader is not None
    renderer = importlib.util.module_from_spec(renderer_spec)
    sys.modules[renderer_spec.name] = renderer
    renderer_spec.loader.exec_module(renderer)
    run_record_path = tmp_path / "run.md"
    renderer.render_model_quality_report(
        report,
        benchmark_path=tmp_path / "benchmark.md",
        run_record_path=run_record_path,
    )
    rendered = run_record_path.read_text(encoding="utf-8")
    assert "VISIBLE_TOOL_RESULT" in rendered
    assert "runner_failed" in rendered
    assert "Retryable: `true`" in rendered
    assert "Truncated: `true`" in rendered
    assert "Tool latency ms: `null`" in rendered
    assert "call-1" not in rendered
    assert absolute_path not in rendered
    assert windows_path not in rendered
    assert unc_path not in rendered
    assert secret not in rendered


def test_tool_call_evidence_projects_the_public_bounded_result_contract() -> None:
    module = _load_gate_module()
    from agent_runtime.result import AgentToolCall

    public_call = AgentToolCall(
        tool_call_id="call-1",
        tool_name="read_file",
        arguments={"path": "input_files/data.json"},
        structured_output={
            "items": ({"name": "visible"},),
            "metadata": {"count": 1},
        },
        is_error=True,
        error_code="runner_failed",
        error_message="bounded failure",
        retryable=True,
        truncated=True,
        latency_ms=None,
    )

    evidence = module._tool_call_evidence(public_call)

    assert evidence.tool_call_id == "call-1"
    assert evidence.tool_name == "read_file"
    assert evidence.arguments == {"path": "input_files/data.json"}
    assert evidence.structured_output == {
        "items": [{"name": "visible"}],
        "metadata": {"count": 1},
    }
    assert evidence.is_error is True
    assert evidence.error_code == "runner_failed"
    assert evidence.error_message == "bounded failure"
    assert evidence.retryable is True
    assert evidence.truncated is True
    assert evidence.latency_ms is None


@pytest.mark.parametrize(
    "workspace_assertions",
    [
        {},
        {"input_files/output.txt": "after\n"},
    ],
    ids=["read-only", "workspace-mutation"],
)
@pytest.mark.anyio
async def test_run_live_case_leaves_workspace_change_checks_to_the_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    workspace_assertions: dict[str, str],
) -> None:
    module = _load_gate_module()
    captured: list[bool] = []

    class FakeAgent:
        def __init__(self, **_kwargs: object) -> None:
            self._turn_store = None

        async def arun(
            self,
            _task: str,
            *,
            files: list[str],
            require_workspace_change: bool = True,
        ) -> object:
            del files
            captured.append(require_workspace_change)
            return SimpleNamespace(
                status="done",
                answer="complete",
                turn_id="turn-123",
                tool_calls=(),
                usage=SimpleNamespace(
                    model_calls=1,
                    input_tokens=1,
                    output_tokens=1,
                    latency_ms=1.0,
                    tool_schema_bytes=1,
                ),
                stop_reason=None,
                diagnostics=(),
                pause=None,
            )

    import agent_runtime

    monkeypatch.setattr(agent_runtime, "Agent", FakeAgent)
    await module.run_live_case(
        model_alias="groq_gpt_oss_120b",
        control_plane=object(),
        case={
            "id": "workspace-contract",
            "capability": "file_tool_selection",
            "task": "Inspect the input file.",
            "workspace_files": {"input.txt": "before\n"},
            "workspace_assertions": workspace_assertions,
        },
    )

    assert captured == [False]


@pytest.mark.anyio
async def test_run_live_case_approves_only_the_declared_write_and_refuses_follow_up_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_gate_module()
    resume_actions: list[str] = []

    from agent_runtime.result import AgentToolCall

    usage = SimpleNamespace(
        model_calls=4,
        input_tokens=100,
        output_tokens=20,
        latency_ms=125.0,
        tool_schema_bytes=400,
    )

    class FakeAgent:
        def __init__(self, *, workspace_path: Path, **_kwargs: object) -> None:
            self._turn_store = None
            self.workspace = workspace_path
            self.runtime_file = (
                workspace_path
                / ".praxis/runtime/input_files/turn-123/approval.txt"
            )
            self.runtime_file.parent.mkdir(parents=True)
            self.runtime_file.write_text("before_gate\n", encoding="utf-8")
            self.calls: tuple[AgentToolCall, ...] = (
                AgentToolCall(
                    "read",
                    "read_file",
                    {"path": ".praxis/runtime/input_files/turn-123/approval.txt"},
                ),
            )

        def result(
            self,
            *,
            status: str,
            answer: str | None = None,
            pause_tool_name: str = "apply_patch",
            workspace_write: bool = True,
            workspace_path: str | None = None,
        ) -> object:
            return SimpleNamespace(
                status=status,
                answer=answer,
                tool_calls=self.calls,
                usage=usage,
                turn_id="turn-123",
                stop_reason=None,
                diagnostics=(),
                pause=(
                    SimpleNamespace(
                        kind="tool_approval",
                        tool_calls=(SimpleNamespace(tool_name=pause_tool_name),),
                        context={
                            "approval_scope": "tool",
                            "network_requested": False,
                            "workspace_write": workspace_write,
                            "workspace_path": (
                                str(self.runtime_file)
                                if workspace_path is None
                                else workspace_path
                            ),
                        },
                    )
                    if status == "paused"
                    else None
                ),
            )

        async def arun(self, _task: str, **_kwargs: object) -> object:
            return self.result(status="paused")

        async def aresume(self, _turn_id: str, action: str) -> object:
            resume_actions.append(action)
            if len(resume_actions) == 1:
                self.runtime_file.write_text("after_gate\n", encoding="utf-8")
                self.calls = (
                    *self.calls,
                    AgentToolCall(
                        "patch",
                        "apply_patch",
                        {
                            "file_path": (
                                ".praxis/runtime/input_files/turn-123/approval.txt"
                            ),
                            "old_string": "before_gate",
                            "new_string": "after_gate",
                        },
                    ),
                )
                return self.result(
                    status="paused",
                    pause_tool_name="run_command",
                    workspace_write=True,
                    workspace_path=str(self.workspace),
                )
            self.calls = (
                *self.calls,
                AgentToolCall(
                    "verify",
                    "run_command",
                    {"command": "cat approval.txt"},
                ),
            )
            return self.result(status="done", answer="complete")

    import agent_runtime

    monkeypatch.setattr(agent_runtime, "Agent", FakeAgent)
    observation = await module.run_live_case(
        model_alias="groq_gpt_oss_120b",
        control_plane=object(),
        case={
            "id": "approval_continue",
            "capability": "approval_continuation",
            "task": "Change before_gate to after_gate and verify it.",
            "workspace_files": {"approval.txt": "before_gate\n"},
            "workspace_assertions": {
                "input_files/approval.txt": "after_gate\n"
            },
            "expected_tool_calls": [
                {
                    "tool_name": "apply_patch",
                    "arguments": {
                        "file_path": "input_files/approval.txt",
                        "old_string": "before_gate",
                        "new_string": "after_gate",
                    },
                }
            ],
            "auto_approve": True,
        },
    )

    assert resume_actions == ["allow_once"]
    assert observation.status == "paused"
    assert observation.approval_resumes == 1
    assert observation.workspace_assertions_passed is True
    assert observation.runtime_input_namespace == "turn-123"


@pytest.mark.anyio
async def test_declared_apply_patch_approval_accepts_the_real_public_pause(
    tmp_path: Path,
) -> None:
    module = _load_gate_module()
    from agent_runtime.loop.runtime import _approval_request
    from agent_runtime.result import _project_pause
    from agent_runtime.tools.builtins.filesystem import create_apply_patch_tool
    from agent_runtime.tools.executor import ToolExecutor
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.tools.tool import ToolCall, ToolCallOrigin
    from agent_runtime.workspace import open_workspace

    workspace_root = tmp_path / "workspace"
    workspace = open_workspace(workspace_root, create=True)
    turn_id = "turn-123"
    runtime_path = f".praxis/runtime/input_files/{turn_id}/approval.txt"
    target = workspace_root / runtime_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("before_gate\n", encoding="utf-8")
    call = ToolCall(
        tool_call_id="call-patch",
        tool_name="apply_patch",
        arguments={
            "file_path": runtime_path,
            "old_string": "before_gate",
            "new_string": "after_gate",
        },
        origin=ToolCallOrigin(
            request_id="request-1",
            toolset_revision="tools-v1",
            exposed_tool_names=("apply_patch",),
        ),
    )
    execution = await ToolExecutor(
        {"apply_patch": create_apply_patch_tool(workspace)}
    ).execute(
        call,
        context=ToolExecutionContext(
            workspace_root=workspace_root,
            cwd=workspace_root,
        ),
    )
    assert execution.result.error_code == "approval_required"
    pause = _project_pause(_approval_request(execution.result, call))
    assert pause is not None
    case = {
        "capability": "approval_continuation",
        "expected_tool_calls": [
            {
                "tool_name": "apply_patch",
                "arguments": {
                    "file_path": "input_files/approval.txt",
                },
            }
        ],
    }

    assert module._is_declared_apply_patch_approval(
        case,
        pause,
        workspace=workspace_root,
        turn_id=turn_id,
    )


def test_declared_apply_patch_approval_rejects_a_different_workspace_target(
    tmp_path: Path,
) -> None:
    module = _load_gate_module()
    case = {
        "capability": "approval_continuation",
        "expected_tool_calls": [
            {
                "tool_name": "apply_patch",
                "arguments": {
                    "file_path": "input_files/approval.txt",
                },
            }
        ],
    }
    pause = SimpleNamespace(
        kind="tool_approval",
        tool_calls=(SimpleNamespace(tool_name="apply_patch"),),
        context={
            "approval_scope": "tool",
            "network_requested": False,
            "workspace_write": True,
            "workspace_path": str(tmp_path / "unrelated.txt"),
        },
    )

    assert (
        module._is_declared_apply_patch_approval(
            case,
            pause,
            workspace=tmp_path,
            turn_id="turn-123",
        )
        is False
    )


def test_workspace_assertions_never_fall_back_from_the_current_turn_namespace(
    tmp_path: Path,
) -> None:
    module = _load_gate_module()
    stale_file = tmp_path / "input_files/approval.txt"
    stale_file.parent.mkdir()
    stale_file.write_text("after_gate\n", encoding="utf-8")

    passed = module._workspace_assertions_pass(
        tmp_path,
        {"input_files/approval.txt": "after_gate\n"},
        turn_id="turn-123",
    )

    assert passed is False


def test_artifact_secret_values_match_sensitive_name_tokens_without_substrings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_gate_module()
    expected_secrets = {
        "GROQ_API_KEY": "groq-api-key-sentinel",
        "AUTH_TOKEN": "auth-token-sentinel",
        "CLIENT_SECRET": "client-secret-sentinel",
        "SESSION_COOKIE": "session-cookie-sentinel",
        "CREDENTIAL": "credential-sentinel",
    }
    for name, value in expected_secrets.items():
        monkeypatch.setenv(name, value)
    monkey_value = "monkey-value-sentinel"
    monkeypatch.setenv("MONKEY_VALUE", monkey_value)

    collected = module._artifact_secret_values()

    assert set(expected_secrets.values()) <= set(collected)
    assert monkey_value not in collected


@pytest.mark.anyio
async def test_calibration_preflight_runs_before_model_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_gate_module()
    suite = _quality_suite()
    events: list[str] = []

    def preflight(repository: Path) -> object:
        assert repository == tmp_path / "repository"
        events.append("preflight")
        return SimpleNamespace(
            source_commit="a" * 40,
            source_tree="b" * 40,
            dirty=False,
        )

    async def run_trials(**_kwargs: object) -> dict[str, object]:
        events.append("model")
        return _passing_model_report()

    monkeypatch.setattr(module, "source_repository_fingerprint", preflight)
    monkeypatch.setattr(module, "load_suite", lambda _path: suite)
    monkeypatch.setattr(module, "run_model_trials", run_trials)
    monkeypatch.setattr(module, "build_baseline", lambda **_kwargs: {"models": {}})
    args = SimpleNamespace(
        repository=tmp_path / "repository",
        fixture=tmp_path / "private" / "cases.json",
        baseline=tmp_path / "baseline.json",
        env_file=tmp_path / "private" / ".env",
        trials=3,
    )

    exit_code = await module._calibrate(args)

    assert exit_code == 0
    assert events == ["preflight", "model", "preflight"]


@pytest.mark.parametrize("command", ["gate", "calibrate"])
@pytest.mark.parametrize("repository_change", ["dirty", "changed_tree"])
def test_cli_rechecks_source_after_model_trials_before_writing_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
    repository_change: str,
) -> None:
    module = _load_gate_module()
    suite = _quality_suite()
    repository = tmp_path / "repository"
    output_path = tmp_path / ("report.json" if command == "gate" else "baseline.json")
    baseline_path = tmp_path / "input-baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "models": {
                    "groq_gpt_oss_120b": {
                        "trial_count": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    initial = module.RepositoryFingerprint(
        source_commit="a" * 40,
        source_tree="b" * 40,
        dirty=False,
    )
    events: list[str] = []

    def preflight(actual_repository: Path) -> object:
        assert actual_repository == repository
        events.append("preflight")
        if events.count("preflight") == 1:
            return initial
        if repository_change == "dirty":
            raise module.DirtyRepositoryError("repository became dirty")
        return module.RepositoryFingerprint(
            source_commit="a" * 40,
            source_tree="c" * 40,
            dirty=False,
        )

    async def run_trials(**_kwargs: object) -> dict[str, object]:
        events.append("model")
        return _passing_model_report()

    monkeypatch.setattr(module, "source_repository_fingerprint", preflight)
    monkeypatch.setattr(module, "load_suite", lambda _path: suite)
    monkeypatch.setattr(module, "run_model_trials", run_trials)
    monkeypatch.setattr(module, "validate_baseline", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "evaluate_model_gate",
        lambda **_kwargs: {
            "model_alias": "groq_gpt_oss_120b",
            "provider_model": "openai/gpt-oss-120b",
            "passed": True,
            "observed": _trial_metrics(),
            "thresholds": {},
            "failures": [],
        },
    )
    monkeypatch.setattr(
        module,
        "build_baseline",
        lambda **_kwargs: pytest.fail("baseline must not be built after source drift"),
    )
    argv = [
        str(SCRIPT_PATH),
        command,
        "--repository",
        str(repository),
        "--fixture",
        str(tmp_path / "cases.json"),
        "--env-file",
        str(tmp_path / ".env"),
    ]
    if command == "gate":
        argv.extend(
            [
                "--baseline",
                str(baseline_path),
                "--model",
                "groq_gpt_oss_120b",
                "--report",
                str(output_path),
            ]
        )
    else:
        argv.extend(["--baseline", str(output_path), "--trials", "3"])
    monkeypatch.setattr(sys, "argv", argv)

    exit_code = module.main()

    assert exit_code == 2
    assert events == ["preflight", "model", "preflight"]
    assert not output_path.exists()


@pytest.mark.anyio
async def test_gate_rejects_a_clean_repository_that_is_not_the_running_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_gate_module()
    other_repository = _clean_git_repository(tmp_path / "other-repository")
    model_called = False

    async def should_not_run(**_kwargs: object) -> dict[str, object]:
        nonlocal model_called
        model_called = True
        return {}

    monkeypatch.setattr(module, "run_model_trials", should_not_run)
    args = SimpleNamespace(
        repository=other_repository,
        fixture=FIXTURE_PATH,
        baseline=BASELINE_PATH,
        env_file=tmp_path / ".env",
        models=["groq_gpt_oss_120b"],
        report=tmp_path / "report.json",
    )

    with pytest.raises(module.SourceRepositoryMismatchError):
        await module._gate(args)

    assert model_called is False


@pytest.mark.anyio
async def test_gate_rejects_runtime_import_resolving_outside_source_repository(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_gate_module()
    model_called = False

    async def should_not_run(**_kwargs: object) -> dict[str, object]:
        nonlocal model_called
        model_called = True
        return {}

    outside_runtime = tmp_path / "installed" / "agent_runtime" / "__init__.py"
    monkeypatch.setattr(module, "repository_fingerprint", lambda _root: object())
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(outside_runtime)),
    )
    monkeypatch.setattr(module, "run_model_trials", should_not_run)
    args = SimpleNamespace(
        repository=module.ROOT,
        fixture=FIXTURE_PATH,
        baseline=BASELINE_PATH,
        env_file=tmp_path / ".env",
        models=["groq_gpt_oss_120b"],
        report=tmp_path / "report.json",
    )

    with pytest.raises(module.SourceRepositoryMismatchError):
        await module._gate(args)

    assert model_called is False


@pytest.mark.anyio
async def test_run_model_trials_returns_inconclusive_partial_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_gate_module()
    suite = module.load_suite(FIXTURE_PATH)
    first_case = suite["cases"][0]
    inconclusive = _inconclusive_model_report(
        module,
        case_id=str(first_case["id"]),
        capability=str(first_case["capability"]),
    )
    observation = inconclusive["trials"][0]["cases"][0]["observation"]
    live_observation = module._observation_from_payload(observation)

    class FakeControlPlane:
        @classmethod
        def from_env(cls, **_kwargs: object) -> FakeControlPlane:
            return cls()

        def current_model(self) -> object:
            return SimpleNamespace(
                provider="groq",
                provider_model="openai/gpt-oss-120b",
            )

    async def run_case(**_kwargs: object) -> object:
        return live_observation

    from agent_runtime import models as model_module

    monkeypatch.setattr(model_module, "ModelControlPlane", FakeControlPlane)
    monkeypatch.setattr(module, "run_live_case", run_case)

    report = await module.run_model_trials(
        model_alias="groq_gpt_oss_120b",
        cases=suite["cases"],
        trials=3,
        env_file=tmp_path / ".env",
    )

    assert report["status"] == "inconclusive"
    assert report["trial_metrics"] == []
    raw_case = report["trials"][0]["cases"][0]
    assert raw_case["observation"]["infrastructure_failure"] is True
    assert raw_case["score"]["passed"] is None


@pytest.mark.parametrize("failure_stage", ["from_env", "current_model"])
@pytest.mark.anyio
async def test_gate_records_control_plane_initialization_failure_before_cases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str,
) -> None:
    module = _load_gate_module()
    case_called = False

    class FailingControlPlane:
        @classmethod
        def from_env(cls, **_kwargs: object) -> FailingControlPlane:
            if failure_stage == "from_env":
                raise RuntimeError("provider bootstrap unavailable")
            return cls()

        def current_model(self) -> object:
            raise RuntimeError("model identity unavailable")

    async def should_not_run_case(**_kwargs: object) -> object:
        nonlocal case_called
        case_called = True
        raise AssertionError("live case must not run")

    from agent_runtime import models as model_module

    monkeypatch.setattr(model_module, "ModelControlPlane", FailingControlPlane)
    monkeypatch.setattr(module, "run_live_case", should_not_run_case)
    monkeypatch.setattr(
        module,
        "source_repository_fingerprint",
        lambda _repository: SimpleNamespace(
            source_commit="e" * 40,
            source_tree="f" * 40,
            dirty=False,
        ),
    )
    report_path = tmp_path / "initialization-report.json"
    args = SimpleNamespace(
        repository=module.ROOT,
        fixture=FIXTURE_PATH,
        baseline=BASELINE_PATH,
        env_file=tmp_path / ".env",
        models=["deepseek_v4_flash"],
        report=report_path,
    )

    exit_code = await module._gate(args)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert case_called is False
    assert payload["status"] == "inconclusive"
    assert payload["passed"] is None
    assert payload["source_commit"] == "e" * 40
    assert payload["source_tree"] == "f" * 40
    assert payload["dirty"] is False
    assert payload["runs"][0]["trials"] == []
    assert payload["runs"][0]["infrastructure_failure"] == {
        "case_id": None,
        "diagnostic_error_types": ["RuntimeError"],
        "stage": "model_control_plane_initialization",
        "stop_reason": "model_control_plane_initialization_failed",
    }

    renderer_path = SCRIPT_PATH.with_name("render_model_quality_report.py")
    spec = importlib.util.spec_from_file_location(
        "render_model_quality_report_from_initialization",
        renderer_path,
    )
    assert spec is not None
    assert spec.loader is not None
    renderer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = renderer
    spec.loader.exec_module(renderer)
    benchmark = tmp_path / "initialization-benchmark.md"
    run_record = tmp_path / "initialization-run.md"

    renderer.render_model_quality_report(
        report_path,
        benchmark_path=benchmark,
        run_record_path=run_record,
    )

    assert "Overall verdict: **INCONCLUSIVE**" in benchmark.read_text(
        encoding="utf-8"
    )
    rendered_run = run_record.read_text(encoding="utf-8")
    assert "Approval-continuation case was not reached." in rendered_run
    assert "model_control_plane_initialization_failed" in rendered_run


def test_cli_returns_two_and_writes_report_for_control_plane_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_gate_module()
    case_called = False

    class FailingControlPlane:
        @classmethod
        def from_env(cls, **_kwargs: object) -> FailingControlPlane:
            raise RuntimeError("provider bootstrap unavailable")

    async def should_not_run_case(**_kwargs: object) -> object:
        nonlocal case_called
        case_called = True
        raise AssertionError("live case must not run")

    from agent_runtime import models as model_module

    monkeypatch.setattr(model_module, "ModelControlPlane", FailingControlPlane)
    monkeypatch.setattr(module, "run_live_case", should_not_run_case)
    monkeypatch.setattr(
        module,
        "source_repository_fingerprint",
        lambda _repository: SimpleNamespace(
            source_commit="1" * 40,
            source_tree="2" * 40,
            dirty=False,
        ),
    )
    report_path = tmp_path / "cli-initialization-report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "gate",
            "--repository",
            str(module.ROOT),
            "--fixture",
            str(FIXTURE_PATH),
            "--baseline",
            str(BASELINE_PATH),
            "--env-file",
            str(tmp_path / ".env"),
            "--model",
            "deepseek_v4_flash",
            "--report",
            str(report_path),
        ],
    )

    exit_code = module.main()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert case_called is False
    assert payload["status"] == "inconclusive"
    assert payload["passed"] is None


@pytest.mark.anyio
async def test_model_trial_initialization_does_not_swallow_programming_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_gate_module()

    class BuggyControlPlane:
        @classmethod
        def from_env(cls, **_kwargs: object) -> BuggyControlPlane:
            raise TypeError("programming defect")

    from agent_runtime import models as model_module

    monkeypatch.setattr(model_module, "ModelControlPlane", BuggyControlPlane)

    with pytest.raises(TypeError, match="programming defect"):
        await module.run_model_trials(
            model_alias="groq_gpt_oss_120b",
            cases=module.load_suite(FIXTURE_PATH)["cases"],
            trials=3,
            env_file=tmp_path / ".env",
        )


def test_calibrate_cli_returns_two_without_partial_initialization_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_gate_module()

    class FailingControlPlane:
        @classmethod
        def from_env(cls, **_kwargs: object) -> FailingControlPlane:
            raise RuntimeError("provider bootstrap unavailable")

    from agent_runtime import models as model_module

    monkeypatch.setattr(model_module, "ModelControlPlane", FailingControlPlane)
    monkeypatch.setattr(
        module,
        "source_repository_fingerprint",
        lambda _repository: SimpleNamespace(
            source_commit="3" * 40,
            source_tree="4" * 40,
            dirty=False,
        ),
    )
    baseline_path = tmp_path / "partial-baseline.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "calibrate",
            "--repository",
            str(module.ROOT),
            "--fixture",
            str(FIXTURE_PATH),
            "--baseline",
            str(baseline_path),
            "--env-file",
            str(tmp_path / ".env"),
        ],
    )

    exit_code = module.main()

    assert exit_code == 2
    assert not baseline_path.exists()


@pytest.mark.anyio
async def test_gate_writes_provenance_complete_inconclusive_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_gate_module()
    suite = module.load_suite(FIXTURE_PATH)

    monkeypatch.setattr(
        module,
        "source_repository_fingerprint",
        lambda _repository: SimpleNamespace(
            source_commit="c" * 40,
            source_tree="d" * 40,
            dirty=False,
        ),
    )
    monkeypatch.setattr(module, "load_suite", lambda _path: suite)
    monkeypatch.setattr(module, "validate_baseline", lambda *_args, **_kwargs: None)

    async def run_trials(**_kwargs: object) -> dict[str, object]:
        return _inconclusive_model_report(
            module,
            case_id="exact_file_read",
            capability="file_tool_selection",
            model_alias="deepseek_v4_flash",
            provider="deepseek",
            provider_model="deepseek-v4-flash",
        )

    monkeypatch.setattr(module, "run_model_trials", run_trials)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "models": {
                    "deepseek_v4_flash": {
                        "trial_count": 3,
                        "thresholds": {},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    args = SimpleNamespace(
        repository=module.ROOT,
        fixture=tmp_path / "cases.json",
        baseline=baseline,
        env_file=tmp_path / ".env",
        models=["deepseek_v4_flash"],
        report=report_path,
    )

    exit_code = await module._gate(args)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["status"] == "inconclusive"
    assert payload["passed"] is None
    assert payload["source_commit"] == "c" * 40
    assert payload["source_tree"] == "d" * 40
    assert payload["dirty"] is False
    assert payload["models"][0]["passed"] is None
    assert payload["runs"][0]["status"] == "inconclusive"

    renderer_path = SCRIPT_PATH.with_name("render_model_quality_report.py")
    spec = importlib.util.spec_from_file_location(
        "render_model_quality_report_from_gate",
        renderer_path,
    )
    assert spec is not None
    assert spec.loader is not None
    renderer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = renderer
    spec.loader.exec_module(renderer)
    benchmark = tmp_path / "benchmark.md"
    run_record = tmp_path / "run.md"

    renderer.render_model_quality_report(
        report_path,
        benchmark_path=benchmark,
        run_record_path=run_record,
    )

    assert "Overall verdict: **INCONCLUSIVE**" in benchmark.read_text(
        encoding="utf-8"
    )
    assert "Evaluator verdict: **INCONCLUSIVE**" in run_record.read_text(
        encoding="utf-8"
    )
    assert "Approval-continuation case was not reached." in run_record.read_text(
        encoding="utf-8"
    )


@pytest.mark.anyio
async def test_gate_approval_infrastructure_report_renders_only_partial_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_gate_module()
    suite = _quality_suite()
    observation = module.CaseObservation(
        case_id="approval_continue",
        capability="approval_continuation",
        status="failed",
        answer="UNTRUSTED_FINAL_SENTINEL",
        tool_calls=(
            module.ToolCallEvidence(
                "partial-read",
                "read_file",
                {"path": "input_files/approval.txt"},
                False,
                None,
            ),
        ),
        model_calls=1,
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
        tool_schema_bytes=1,
        approval_pause_observed=False,
        approval_kind=None,
        approval_resumes=0,
        workspace_assertions_passed=False,
        stop_reason="model_provider_failed",
        diagnostic_codes=("model_provider_failed",),
        diagnostic_error_types=("RateLimitError",),
        infrastructure_failure=True,
    )

    class FakeControlPlane:
        @classmethod
        def from_env(cls, **_kwargs: object) -> FakeControlPlane:
            return cls()

        def current_model(self) -> object:
            return SimpleNamespace(
                provider="groq",
                provider_model="openai/gpt-oss-120b",
            )

    async def run_case(**_kwargs: object) -> object:
        return observation

    from agent_runtime import models as model_module

    monkeypatch.setattr(model_module, "ModelControlPlane", FakeControlPlane)
    monkeypatch.setattr(module, "run_live_case", run_case)
    monkeypatch.setattr(module, "load_suite", lambda _path: suite)
    monkeypatch.setattr(module, "validate_baseline", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "source_repository_fingerprint",
        lambda _repository: SimpleNamespace(
            source_commit="5" * 40,
            source_tree="6" * 40,
            dirty=False,
        ),
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "models": {
                    "groq_gpt_oss_120b": {
                        "trial_count": 3,
                        "thresholds": {},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "approval-infrastructure.json"
    exit_code = await module._gate(
        SimpleNamespace(
            repository=module.ROOT,
            fixture=tmp_path / "cases.json",
            baseline=baseline,
            env_file=tmp_path / ".env",
            models=["groq_gpt_oss_120b"],
            report=report_path,
        )
    )
    renderer_path = SCRIPT_PATH.with_name("render_model_quality_report.py")
    spec = importlib.util.spec_from_file_location(
        "render_model_quality_report_from_partial_approval",
        renderer_path,
    )
    assert spec is not None
    assert spec.loader is not None
    renderer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = renderer
    spec.loader.exec_module(renderer)
    run_record = tmp_path / "approval-run.md"

    renderer.render_model_quality_report(
        report_path,
        benchmark_path=tmp_path / "approval-benchmark.md",
        run_record_path=run_record,
    )

    rendered = run_record.read_text(encoding="utf-8")
    assert exit_code == 2
    assert "read_file" in rendered
    assert "Approval was not reached." in rendered
    assert "Approval pause observed: `false`" in rendered
    assert "Approval kind: `null`" in rendered
    assert "Approval resumes: `0`" in rendered
    assert "model_provider_failed" in rendered
    assert "RateLimitError" in rendered
    assert "Evaluator verdict: **INCONCLUSIVE**" in rendered
    assert "-before_gate" not in rendered
    assert "+after_gate" not in rendered
    assert "UNTRUSTED_FINAL_SENTINEL" not in rendered


def test_fixture_locks_models_and_quality_capabilities() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["models"] == ["deepseek_v4_flash"]
    assert {case["capability"] for case in payload["cases"]} == {
        "file_tool_selection",
        "failure_recovery",
        "approval_continuation",
        "repeated_failure_control",
    }
    assert len({case["id"] for case in payload["cases"]}) == 5
    assert "thresholds" not in payload


def test_thresholds_are_the_empirical_worst_trial_not_literals() -> None:
    module = _load_gate_module()
    trials = [
        _trial_metrics(),
        _trial_metrics(
            task_success_rate=0.8,
            redundant_tool_call_rate=0.2,
            mean_tool_calls_per_case=1.8,
        ),
        _trial_metrics(
            failure_recovery_rate=0.0,
            mean_model_calls_per_case=2.8,
        ),
    ]

    thresholds = module.derive_thresholds(trials)

    assert thresholds["task_success_rate"] == {
        "direction": "min",
        "value": 0.8,
    }
    assert thresholds["failure_recovery_rate"] == {
        "direction": "min",
        "value": 0.0,
    }
    assert thresholds["redundant_tool_call_rate"] == {
        "direction": "max",
        "value": 0.2,
    }
    assert thresholds["mean_tool_calls_per_case"] == {
        "direction": "max",
        "value": 1.8,
    }
    assert thresholds["mean_model_calls_per_case"] == {
        "direction": "max",
        "value": 2.8,
    }


def test_calibration_requires_repeated_real_trials() -> None:
    module = _load_gate_module()

    with pytest.raises(ValueError, match="at least 3 real trials"):
        module.derive_thresholds([_trial_metrics(), _trial_metrics()])


def test_baseline_validation_recomputes_and_rejects_edited_thresholds() -> None:
    module = _load_gate_module()
    trials = [_trial_metrics() for _ in range(3)]
    baseline = {
        "schema_version": 1,
        "suite_revision": "suite_123",
        "threshold_method": module.THRESHOLD_METHOD,
        "models": {
            "qwen3_5_9b_mlx_4bit": {
                "provider_model": "mlx-community/Qwen3.5-9B-4bit",
                "trial_count": 3,
                "trial_metrics": trials,
                "thresholds": module.derive_thresholds(trials),
            }
        },
    }
    baseline["models"]["qwen3_5_9b_mlx_4bit"]["thresholds"]["task_success_rate"]["value"] = 0.5

    with pytest.raises(ValueError, match="do not match measured trials"):
        module.validate_baseline(baseline)


def test_committed_live_baseline_recomputes_from_raw_observations() -> None:
    module = _load_gate_module()
    suite = module.load_suite(FIXTURE_PATH)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    module.validate_baseline(baseline, suite=suite)

    assert set(baseline["models"]) == set(suite["models"])
    assert all(entry["trial_count"] == 3 for entry in baseline["models"].values())


def test_raw_baseline_tampering_is_detected_before_thresholds() -> None:
    module = _load_gate_module()
    suite = module.load_suite(FIXTURE_PATH)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(baseline)
    first_model = tampered["models"][suite["models"][0]]
    first_model["trials"][0]["cases"][0]["observation"]["tool_calls"] = []

    with pytest.raises(ValueError, match="does not match raw observations"):
        module.validate_baseline(tampered, suite=suite)


def test_gate_uses_worst_current_trial_and_reports_regressions() -> None:
    module = _load_gate_module()
    baseline_trials = [_trial_metrics() for _ in range(3)]
    model_baseline = {
        "provider_model": "openai/gpt-oss-120b",
        "trial_count": 3,
        "trial_metrics": baseline_trials,
        "thresholds": module.derive_thresholds(baseline_trials),
    }
    current_trials = [
        _trial_metrics(),
        _trial_metrics(repeated_failure_control_rate=0.0),
        _trial_metrics(),
    ]

    result = module.evaluate_model_gate(
        model_alias="groq_gpt_oss_120b",
        provider_model="openai/gpt-oss-120b",
        trial_metrics=current_trials,
        baseline=model_baseline,
    )

    assert result["passed"] is False
    assert result["observed"]["repeated_failure_control_rate"] == 0.0
    assert result["failures"] == ["repeated_failure_control_rate: observed 0.0 < baseline floor 1.0"]


def test_live_gate_excludes_runtime_determinism_from_threshold_metrics() -> None:
    module = _load_gate_module()

    assert "latency_ms" not in module.GATED_METRIC_DIRECTIONS
    assert "tool_schema_bytes" not in module.GATED_METRIC_DIRECTIONS
    assert "checkpoint_count" not in module.GATED_METRIC_DIRECTIONS
    assert set(module.GATED_METRIC_DIRECTIONS) == set(_trial_metrics())


def test_approval_quality_allows_reads_around_one_approved_write() -> None:
    module = _load_gate_module()
    case = {
        "id": "approval_continue",
        "capability": "approval_continuation",
        "expected_first_tool": "apply_patch",
        "expected_tool_sequence": ["apply_patch"],
    }
    calls = (
        module.ToolCallEvidence("read-1", "read_file", {"path": "a.txt"}, False, None),
        module.ToolCallEvidence(
            "patch-1",
            "apply_patch",
            {"file_path": "a.txt", "old_string": "a", "new_string": "b"},
            False,
            None,
        ),
        module.ToolCallEvidence("read-2", "read_file", {"path": "a.txt"}, False, None),
    )
    observation = module.CaseObservation(
        case_id="approval_continue",
        capability="approval_continuation",
        status="done",
        answer="done",
        tool_calls=calls,
        model_calls=4,
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
        tool_schema_bytes=1,
        approval_pause_observed=True,
        approval_kind="tool_approval",
        approval_resumes=1,
        workspace_assertions_passed=True,
    )

    score = module._score_case(case, observation)

    assert score["capability_passed"] is True
    assert score["redundant_tool_calls"] == 0


def test_approval_quality_allows_a_second_resume_for_post_write_verification() -> None:
    module = _load_gate_module()
    case = {
        "id": "approval_continue",
        "capability": "approval_continuation",
        "expected_tool_calls": [
            {
                "tool_name": "apply_patch",
                "arguments": {
                    "file_path": "input_files/approval.txt",
                    "old_string": "before_gate",
                    "new_string": "after_gate",
                },
            }
        ],
    }
    calls = (
        module.ToolCallEvidence(
            "patch",
            "apply_patch",
            {
                "file_path": (
                    ".praxis/runtime/input_files/turn-123/approval.txt"
                ),
                "old_string": "before_gate",
                "new_string": "after_gate",
            },
            False,
            None,
        ),
        module.ToolCallEvidence(
            "verify",
            "run_command",
            {"command": "cat approval.txt"},
            False,
            None,
        ),
    )
    observation = module.CaseObservation(
        case_id="approval_continue",
        capability="approval_continuation",
        status="done",
        answer="done",
        tool_calls=calls,
        model_calls=3,
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
        tool_schema_bytes=1,
        approval_pause_observed=True,
        approval_kind="tool_approval",
        approval_resumes=2,
        workspace_assertions_passed=True,
        runtime_input_namespace="turn-123",
    )

    score = module._score_case(case, observation)

    assert score["capability_passed"] is True


def test_call_matching_accepts_only_equivalent_namespaced_input_file_paths() -> None:
    module = _load_gate_module()
    spec = {
        "tool_name": "read_file",
        "arguments": {"path": "input_files/exact.txt"},
    }

    equivalent = module.ToolCallEvidence(
        "read",
        "read_file",
        {"path": ".praxis/runtime/input_files/turn-123/exact.txt"},
        False,
        None,
    )
    wrong_file = module.ToolCallEvidence(
        "read-wrong",
        "read_file",
        {"path": ".praxis/runtime/input_files/turn-123/other.txt"},
        False,
        None,
    )

    assert (
        module._call_matches(
            equivalent,
            spec,
            runtime_input_namespace="turn-123",
        )
        is True
    )
    assert (
        module._call_matches(
            wrong_file,
            spec,
            runtime_input_namespace="turn-123",
        )
        is False
    )


def test_file_selection_rejects_an_attachment_path_from_another_turn() -> None:
    module = _load_gate_module()
    case = {
        "id": "exact_file_read",
        "capability": "file_tool_selection",
        "expected_answer_contains": "QUALITY_GATE_EXACT",
        "expected_first_tool": "read_file",
        "require_first_tool": True,
        "expected_tool_calls": [
            {
                "tool_name": "read_file",
                "arguments": {"path": "input_files/exact.txt"},
            }
        ],
    }
    observation = SimpleNamespace(
        case_id="exact_file_read",
        capability="file_tool_selection",
        status="done",
        answer="QUALITY_GATE_EXACT",
        tool_calls=(
            module.ToolCallEvidence(
                "read",
                "read_file",
                {
                    "path": (
                        ".praxis/runtime/input_files/wrong-turn/exact.txt"
                    )
                },
                False,
                None,
            ),
        ),
        model_calls=2,
        workspace_assertions_passed=True,
        runtime_input_namespace="turn-123",
        error="",
    )

    score = module._score_case(case, observation)

    assert score["capability_passed"] is False
    assert score["passed"] is False


def test_file_selection_rejects_a_failed_expected_read() -> None:
    module = _load_gate_module()
    case = {
        "id": "exact_file_read",
        "capability": "file_tool_selection",
        "expected_answer_contains": "QUALITY_GATE_EXACT",
        "expected_first_tool": "read_file",
        "require_first_tool": True,
        "expected_tool_calls": [
            {
                "tool_name": "read_file",
                "arguments": {"path": "input_files/exact.txt"},
            }
        ],
    }
    observation = SimpleNamespace(
        case_id="exact_file_read",
        capability="file_tool_selection",
        status="done",
        answer="QUALITY_GATE_EXACT",
        tool_calls=(
            module.ToolCallEvidence(
                "read",
                "read_file",
                {
                    "path": (
                        ".praxis/runtime/input_files/turn-123/exact.txt"
                    )
                },
                True,
                "runner_failed",
            ),
        ),
        model_calls=2,
        workspace_assertions_passed=True,
        runtime_input_namespace="turn-123",
        error="",
    )

    score = module._score_case(case, observation)

    assert score["expected_sequence_matched"] is False
    assert score["capability_passed"] is False
    assert score["passed"] is False


def test_redundancy_counts_only_replayed_identical_failures() -> None:
    module = _load_gate_module()
    case = {
        "id": "single_failure_no_retry",
        "capability": "repeated_failure_control",
        "expected_first_tool": "read_file",
        "intentional_error_tool": "read_file",
        "max_identical_failed_calls": 1,
    }
    call = module.ToolCallEvidence(
        "read-1",
        "read_file",
        {"path": "missing.txt"},
        True,
        "runner_failed",
    )
    observation = module.CaseObservation(
        case_id="single_failure_no_retry",
        capability="repeated_failure_control",
        status="done",
        answer="FILE_UNAVAILABLE",
        tool_calls=(call, call),
        model_calls=3,
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
        tool_schema_bytes=1,
        approval_pause_observed=False,
        approval_kind=None,
        approval_resumes=0,
        workspace_assertions_passed=True,
    )

    score = module._score_case(case, observation)

    assert score["capability_passed"] is False
    assert score["redundant_tool_calls"] == 1


def test_file_selection_allows_preflight_but_keeps_call_cost_visible() -> None:
    module = _load_gate_module()
    case = {
        "id": "symbol_search_then_read",
        "capability": "file_tool_selection",
        "expected_first_tool": "search_text",
        "expected_tool_sequence": ["search_text", "read_file"],
        "expected_tool_calls": [
            {"tool_name": "search_text", "arguments": {"pattern": "target"}},
            {"tool_name": "read_file", "arguments": {"path": "target.py"}},
        ],
    }
    calls = (
        module.ToolCallEvidence("list", "list_files", {"path": "input_files"}, False, None),
        module.ToolCallEvidence("search", "search_text", {"pattern": "target"}, False, None),
        module.ToolCallEvidence("read", "read_file", {"path": "target.py"}, False, None),
    )
    observation = module.CaseObservation(
        case_id="symbol_search_then_read",
        capability="file_tool_selection",
        status="done",
        answer="TARGET_VALUE_731",
        tool_calls=calls,
        model_calls=4,
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
        tool_schema_bytes=1,
        approval_pause_observed=False,
        approval_kind=None,
        approval_resumes=0,
        workspace_assertions_passed=True,
    )

    score = module._score_case(case, observation)

    assert score["capability_passed"] is True
    assert score["tool_call_count"] == 3


def test_file_selection_rejects_the_right_tool_with_the_wrong_path() -> None:
    module = _load_gate_module()
    case = {
        "id": "exact_file_read",
        "capability": "file_tool_selection",
        "expected_first_tool": "read_file",
        "require_first_tool": True,
        "expected_tool_sequence": ["read_file"],
        "expected_tool_calls": [
            {
                "tool_name": "read_file",
                "arguments": {"path": "input_files/exact.txt"},
            }
        ],
    }
    observation = module.CaseObservation(
        case_id="exact_file_read",
        capability="file_tool_selection",
        status="done",
        answer="QUALITY_GATE_EXACT",
        tool_calls=(
            module.ToolCallEvidence(
                "read",
                "read_file",
                {"path": "input_files/wrong.txt"},
                False,
                None,
            ),
        ),
        model_calls=2,
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
        tool_schema_bytes=1,
        approval_pause_observed=False,
        approval_kind=None,
        approval_resumes=0,
        workspace_assertions_passed=True,
    )

    score = module._score_case(case, observation)

    assert score["capability_passed"] is False


def test_recovery_is_measured_after_the_actual_failure_not_call_zero() -> None:
    module = _load_gate_module()
    case = {
        "id": "missing_file_recovery",
        "capability": "failure_recovery",
        "intentional_error_tool": "read_file",
        "expected_failed_call": {
            "tool_name": "read_file",
            "arguments": {"path": "report.txt"},
        },
        "expected_recovery_call": {
            "tool_name": "read_file",
            "arguments": {"path": "report-final.txt"},
        },
        "recovery_tools": ["list_files", "search_text", "read_file"],
    }
    calls = (
        module.ToolCallEvidence("list", "list_files", {"path": "input_files"}, False, None),
        module.ToolCallEvidence(
            "missing",
            "read_file",
            {"path": "report.txt"},
            True,
            "runner_failed",
        ),
        module.ToolCallEvidence(
            "found",
            "read_file",
            {"path": "report-final.txt"},
            False,
            None,
        ),
    )

    assert module._failure_recovered(case, calls) is True


def test_provider_failure_is_inconclusive_not_a_model_quality_score() -> None:
    module = _load_gate_module()
    suite = module.load_suite(FIXTURE_PATH)
    observations = [
        module.CaseObservation(
            case_id=str(case["id"]),
            capability=str(case["capability"]),
            status="failed",
            answer=None,
            tool_calls=(),
            model_calls=0,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            tool_schema_bytes=0,
            approval_pause_observed=False,
            approval_kind=None,
            approval_resumes=0,
            workspace_assertions_passed=False,
            stop_reason="model_provider_failed",
            diagnostic_codes=("model_provider_failed",),
            diagnostic_error_types=("RateLimitError",),
            infrastructure_failure=True,
        )
        for case in suite["cases"]
    ]

    with pytest.raises(module.InfrastructureUnavailableError, match="RateLimitError"):
        module.score_trial(suite["cases"], observations)


@pytest.mark.parametrize(
    ("status", "stop_reason", "expected"),
    [
        ("error", None, True),
        ("failed", "model_provider_failed", True),
        ("failed", "context_overflow", True),
        ("failed", "budget_exhausted", True),
        ("failed", "tool_error", True),
        ("failed", "invalid_model_turn", False),
        ("failed", "max_iterations", False),
        ("failed", "max_turns", False),
        ("failed", "repeated_tool_failure", False),
        ("done", "accepted", False),
    ],
)
def test_infrastructure_classification_excludes_model_behavior(
    status: str,
    stop_reason: str | None,
    expected: bool,
) -> None:
    module = _load_gate_module()

    assert module.is_infrastructure_failure(status, stop_reason) is expected


def test_calibrate_cli_cannot_create_a_partial_model_baseline() -> None:
    module = _load_gate_module()
    parser = module._parser()

    args = parser.parse_args(["calibrate"])

    assert not hasattr(args, "models")
    with pytest.raises(SystemExit):
        parser.parse_args(["calibrate", "--model", "qwen3_5_9b_mlx_4bit"])


@pytest.mark.anyio
async def test_calibration_rejects_too_few_trials_before_live_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_gate_module()
    called = False

    async def should_not_run(**_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(module, "run_model_trials", should_not_run)
    args = SimpleNamespace(
        fixture=FIXTURE_PATH,
        trials=2,
        env_file=tmp_path / ".env",
        baseline=tmp_path / "baseline.json",
    )

    with pytest.raises(ValueError, match="at least 3 real trials"):
        await module._calibrate(args)

    assert called is False
