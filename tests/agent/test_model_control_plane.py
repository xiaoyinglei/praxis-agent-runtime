from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from agent_runtime.local_runtime import EndpointConflictError, LocalRuntimeManager
from agent_runtime.models import (
    ModelCatalog,
    ModelControlPlane,
    ModelPolicy,
    ModelPolicyError,
    ModelRuntimeSpec,
    ModelSessionState,
    ModelSpec,
)
from rag.agent.cli import agent_app
from rag.agent.core.llm_registry import ModelNotAvailableError, ModelRegistry
from rag.schema.llm import DEFAULT_LLM_STAGE_BUDGETS, LLMCallStage
from rag.utils.text import load_env_file


def _write_models_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "local_qwen": {
                        "capability": "chat",
                        "provider": "qwen",
                        "protocol": "openai_compatible",
                        "model": "mlx-community/Qwen3-14B-4bit",
                        "base_url": "http://127.0.0.1:8080/v1",
                        "context_window_tokens": 32768,
                        "tools": True,
                        "structured_output": True,
                        "location": "local",
                        "runtime": {
                            "health_url": "http://127.0.0.1:8080/v1/models",
                            "launch_command": ["uv", "run", "python", "-m", "mlx_lm.server"],
                            "expected_model_contains": "Qwen3-14B",
                            "startup_timeout_seconds": 5,
                        },
                    },
                    "mimo_cloud": {
                        "capability": "chat",
                        "provider": "mimo",
                        "protocol": "openai_compatible",
                        "model": "mimo-v2.5-pro",
                        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
                        "api_key_env": "MIMO_API_KEY",
                        "context_window_tokens": 256000,
                        "tools": True,
                        "structured_output": True,
                        "location": "cloud",
                        "cost": {
                            "input_per_1m": 0.5,
                            "output_per_1m": 2.0,
                        },
                    },
                    "embed": {
                        "capability": "embedding",
                        "provider": "qwen",
                        "model": "embedding-model",
                    },
                },
                "defaults": {"primary_model": "local_qwen"},
            }
        ),
        encoding="utf-8",
    )


def test_model_catalog_loads_runtime_specs_without_embedding_models(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    _write_models_config(config_path)

    catalog = ModelCatalog.from_config_file(config_path)

    assert [spec.id for spec in catalog.list_models()] == ["local_qwen", "mimo_cloud"]
    spec = catalog.get("mimo_cloud")
    assert spec.provider == "mimo"
    assert spec.provider_model == "mimo-v2.5-pro"
    assert spec.context_window == 256000
    assert spec.supports_tools is True
    assert spec.supports_structured_output is True
    assert spec.location == "cloud"
    assert spec.runtime is None
    assert spec.input_cost_per_1m == 0.5
    assert spec.output_cost_per_1m == 2.0
    assert catalog.default_model_id == "local_qwen"
    local = catalog.get("local_qwen")
    assert local.runtime is not None
    assert local.runtime.health_url == "http://127.0.0.1:8080/v1/models"
    assert local.runtime.expected_model_contains == "Qwen3-14B"


def test_bundled_default_chat_model_is_groq_control() -> None:
    catalog = ModelCatalog.from_config_file(Path("configs/models.yaml"))

    spec = catalog.get(catalog.default_model_id)

    assert catalog.default_model_id == "groq_gpt_oss_120b"
    assert spec.provider == "groq"
    assert spec.provider_model == "openai/gpt-oss-120b"
    assert spec.location == "cloud"
    assert spec.api_key_env == "GROQ_API_KEY"


def test_bundled_kimi_k26_cloud_model_is_available_for_diagnostics() -> None:
    catalog = ModelCatalog.from_config_file(Path("configs/models.yaml"))

    spec = catalog.get("kimi_cloud")

    assert spec.provider == "kimi"
    assert spec.provider_model == "kimi-k2.6"
    assert spec.location == "cloud"
    assert spec.api_key_env == "MOONSHOT_API_KEY"
    assert spec.context_window == 262_144


def test_bundled_local_qwen8_runtime_is_available_for_local_testing() -> None:
    catalog = ModelCatalog.from_config_file(Path("configs/models.yaml"))

    spec = catalog.get("qwen3_8b_mlx_4bit")

    assert spec.provider == "local_mlx_chat_8080"
    assert spec.provider_model == "mlx-community/Qwen3-8B-4bit"
    assert spec.location == "local"
    assert spec.runtime is not None
    assert spec.runtime.health_url == "http://127.0.0.1:8080/v1/models"
    assert spec.runtime.expected_model_contains == "Qwen3-8B-4bit"
    assert "{model}" not in spec.runtime.launch_command
    assert "mlx-community/Qwen3-8B-4bit" in spec.runtime.launch_command


def test_bundled_tool_decision_budget_supports_coding_turns() -> None:
    payload = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))

    budget = payload["llm_budgets"]["tool_decision"]
    assert budget["max_input_tokens"] == 32_000
    assert budget["max_output_tokens"] == 4_096
    default = DEFAULT_LLM_STAGE_BUDGETS[LLMCallStage.TOOL_DECISION]
    assert default.max_input_tokens == 32_000
    assert default.max_output_tokens == 4_096


def test_env_loader_uses_shared_env_for_linked_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = tmp_path / "repository"
    common_git = primary / ".git"
    worktree_git = common_git / "worktrees" / "feature"
    worktree = tmp_path / "feature-worktree"
    worktree_git.mkdir(parents=True)
    worktree.mkdir()
    (worktree / ".git").write_text(
        f"gitdir: {worktree_git}\n",
        encoding="utf-8",
    )
    (worktree_git / "commondir").write_text("../..\n", encoding="utf-8")
    shared_env = primary / ".env"
    shared_env.write_text("WORKTREE_SHARED_KEY=available\n", encoding="utf-8")
    monkeypatch.delenv("AGENT_ENV_FILE", raising=False)
    monkeypatch.delenv("WORKTREE_SHARED_KEY", raising=False)

    loaded = load_env_file(worktree / ".env")

    assert loaded == shared_env.resolve()
    assert os.environ["WORKTREE_SHARED_KEY"] == "available"


def test_model_policy_reviews_agent_model_switch_requests(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    _write_models_config(config_path)
    catalog = ModelCatalog.from_config_file(config_path)
    state = ModelSessionState(current_model_id="local_qwen")
    policy = ModelPolicy(allowed_agent_model_ids=frozenset({"local_qwen"}))
    control = ModelControlPlane(catalog=catalog, state=state, policy=policy)

    with pytest.raises(ModelPolicyError, match="not allowed"):
        control.switch_model("mimo_cloud", requested_by="agent")

    assert state.current_model_id == "local_qwen"
    control.switch_model("mimo_cloud", requested_by="user")
    assert state.current_model_id == "mimo_cloud"


def test_control_plane_resolves_provider_from_session_current_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    _write_models_config(config_path)
    monkeypatch.setenv("RAG_AGENT_MODELS_PATH", str(config_path))
    monkeypatch.setenv("MIMO_API_KEY", "sk-test")
    resolved_aliases: list[str] = []

    def fake_resolve(self: ModelRegistry, alias: str):  # type: ignore[no-untyped-def]
        resolved_aliases.append(alias)
        return object()

    monkeypatch.setattr(ModelRegistry, "resolve", fake_resolve)

    control = ModelControlPlane.from_env(initial_model_id="mimo_cloud")
    resolved = control.resolve_for_node(node_model=None, node_name="tool_decision")

    assert resolved is not None
    assert resolved_aliases == ["mimo_cloud"]


def test_control_plane_does_not_fallback_from_explicit_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    _write_models_config(config_path)
    monkeypatch.setenv("RAG_AGENT_MODELS_PATH", str(config_path))
    resolved_aliases: list[str] = []

    def fail_resolve(self: ModelRegistry, alias: str):  # type: ignore[no-untyped-def]
        resolved_aliases.append(alias)
        raise ModelNotAvailableError(f"{alias} failed")

    monkeypatch.setattr(ModelRegistry, "resolve", fail_resolve)
    monkeypatch.setattr(LocalRuntimeManager, "ensure_ready", lambda self, spec: None)

    control = ModelControlPlane.from_env(initial_model_id="local_qwen")

    with pytest.raises(ModelNotAvailableError, match="local_qwen failed"):
        control.resolve_for_node(node_model=None, node_name="tool_decision")

    assert resolved_aliases == ["local_qwen"]


def test_control_plane_ensures_local_runtime_before_resolving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    _write_models_config(config_path)
    monkeypatch.setenv("RAG_AGENT_MODELS_PATH", str(config_path))
    ensured: list[str] = []
    resolved_aliases: list[str] = []

    def ensure_ready(self: LocalRuntimeManager, spec: ModelSpec) -> None:
        del self
        ensured.append(spec.id)

    def fake_resolve(self: ModelRegistry, alias: str):  # type: ignore[no-untyped-def]
        resolved_aliases.append(alias)
        return object()

    monkeypatch.setattr(LocalRuntimeManager, "ensure_ready", ensure_ready)
    monkeypatch.setattr(ModelRegistry, "resolve", fake_resolve)

    control = ModelControlPlane.from_env(initial_model_id="local_qwen")
    result = control.resolve_for_node(node_model=None, node_name="tool_decision")

    assert result is not None
    assert ensured == ["local_qwen"]
    assert resolved_aliases == ["local_qwen"]


def test_control_plane_rejects_cloud_model_without_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    _write_models_config(config_path)
    monkeypatch.delenv("MIMO_API_KEY", raising=False)

    control = ModelControlPlane.from_config_file(config_path, initial_model_id="mimo_cloud")

    with pytest.raises(
        ModelNotAvailableError,
        match="MIMO_API_KEY is not set",
    ):
        control.resolve_for_node(node_model=None, node_name="tool_decision")


def test_model_session_state_persists_without_rewriting_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    session_path = tmp_path / "model-session.json"
    _write_models_config(config_path)
    before = config_path.read_text(encoding="utf-8")

    control = ModelControlPlane.from_config_file(
        config_path,
        session_path=session_path,
    )
    control.switch_model("mimo_cloud", requested_by="user")

    restored = ModelControlPlane.from_config_file(
        config_path,
        session_path=session_path,
    )
    assert restored.current_model().id == "mimo_cloud"
    assert config_path.read_text(encoding="utf-8") == before


def test_agent_model_cli_uses_session_state_not_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    session_path = tmp_path / "model-session.json"
    _write_models_config(config_path)
    before = config_path.read_text(encoding="utf-8")
    monkeypatch.setenv("RAG_AGENT_MODELS_PATH", str(config_path))
    runner = CliRunner()

    listed = runner.invoke(
        agent_app,
        ["model", "list", "--session-path", str(session_path)],
        env={"COLUMNS": "240"},
    )
    current = runner.invoke(
        agent_app,
        ["model", "current", "--session-path", str(session_path)],
        env={"COLUMNS": "240"},
    )
    switched = runner.invoke(
        agent_app,
        ["model", "switch", "mimo_cloud", "--session-path", str(session_path)],
        env={"COLUMNS": "240"},
    )
    after = runner.invoke(
        agent_app,
        ["model", "current", "--session-path", str(session_path)],
        env={"COLUMNS": "240"},
    )

    assert listed.exit_code == 0, listed.output
    assert "local_qwen" in listed.output
    assert "mimo_cloud" in listed.output
    assert current.exit_code == 0, current.output
    assert "local_qwen" in current.output
    assert switched.exit_code == 0, switched.output
    assert "mimo_cloud" in switched.output
    assert after.exit_code == 0, after.output
    assert "mimo_cloud" in after.output
    assert config_path.read_text(encoding="utf-8") == before


def test_local_runtime_manager_launches_and_polls_until_expected_model() -> None:
    requests = [
        OSError("not listening"),
        {"data": [{"id": "models--mlx-community--Qwen3-14B-4bit"}]},
    ]
    launched: list[list[str]] = []

    def request_json(url: str, timeout: float) -> object:
        del url, timeout
        item = requests.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def launch(command: list[str]) -> object:
        launched.append(command)
        return SimpleNamespace(pid=123)

    manager = LocalRuntimeManager(
        request_json=request_json,
        launch_process=launch,
        sleep=lambda _: None,
        monotonic=_counter(),
    )

    manager.ensure_ready(
        ModelSpec(
            id="local_qwen",
            provider="qwen",
            provider_model="models--mlx-community--Qwen3-14B-4bit",
            context_window=32768,
            supports_tools=True,
            supports_structured_output=True,
            location="local",
            runtime=ModelRuntimeSpec(
                health_url="http://127.0.0.1:8080/v1/models",
                launch_command=("uv", "run", "python", "-m", "mlx_lm.server"),
                expected_model_contains="Qwen3-14B",
                startup_timeout_seconds=5,
            ),
        )
    )

    assert launched == [["uv", "run", "python", "-m", "mlx_lm.server"]]


def test_local_runtime_manager_closes_only_the_process_it_launched() -> None:
    requests = [
        OSError("not listening"),
        {"data": [{"id": "models--mlx-community--Qwen3-14B-4bit"}]},
    ]
    process = SimpleNamespace(pid=123)
    stopped: list[object] = []

    def request_json(url: str, timeout: float) -> object:
        del url, timeout
        item = requests.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    manager = LocalRuntimeManager(
        request_json=request_json,
        launch_process=lambda _command: process,
        stop_process=stopped.append,
        sleep=lambda _: None,
        monotonic=_counter(),
    )
    manager.ensure_ready(
        ModelSpec(
            id="local_qwen",
            provider="qwen",
            provider_model="models--mlx-community--Qwen3-14B-4bit",
            context_window=32768,
            supports_tools=True,
            supports_structured_output=True,
            location="local",
            runtime=ModelRuntimeSpec(
                health_url="http://127.0.0.1:8080/v1/models",
                launch_command=("uv", "run", "python", "-m", "mlx_lm.server"),
                expected_model_contains="Qwen3-14B",
                startup_timeout_seconds=5,
            ),
        )
    )

    manager.close()
    manager.close()

    assert stopped == [process]


def test_local_runtime_manager_rejects_endpoint_conflict() -> None:
    manager = LocalRuntimeManager(
        request_json=lambda *_: {"data": [{"id": "other-model"}]},
    )

    with pytest.raises(EndpointConflictError, match="endpoint conflict"):
        manager.ensure_ready(
            ModelSpec(
                id="local_qwen",
                provider="qwen",
                provider_model="models--mlx-community--Qwen3-14B-4bit",
                context_window=32768,
                supports_tools=True,
                supports_structured_output=True,
                location="local",
                runtime=ModelRuntimeSpec(
                    health_url="http://127.0.0.1:8080/v1/models",
                    launch_command=("uv", "run", "python", "-m", "mlx_lm.server"),
                    expected_model_contains="Qwen3-14B",
                ),
            )
        )


def test_bundled_qwen14_runtime_accepts_mlx_canonical_model_id() -> None:
    spec = ModelCatalog.from_config_file(Path("configs/models.yaml")).get("qwen3_14b_mlx_4bit")
    assert spec.runtime is not None

    manager = LocalRuntimeManager(
        request_json=lambda *_: {"data": [{"id": "mlx-community/Qwen3-14B-4bit"}]},
        launch_process=lambda command: pytest.fail(f"unexpected launch: {command}"),
    )

    manager.ensure_ready(spec)


def _counter():
    value = -1.0

    def now() -> float:
        nonlocal value
        value += 1.0
        return value

    return now
