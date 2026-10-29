from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_runtime.cli import agent_app
from agent_runtime.model_probe import ModelProbe, ModelProbeError, ModelProbeEvidence, ProbeLevel
from tests.agent.test_model_probe import provider_server as provider_server


@pytest.fixture
def setup_environment(tmp_path, monkeypatch):
    workspace = tmp_path / "project"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("PRAXIS_DISABLE_DOTENV", "1")
    monkeypatch.delenv("PRAXIS_MODEL_API_KEY", raising=False)
    return workspace


def successful_probe(monkeypatch):
    definitions = []

    async def run(self, definition, *, level):
        definitions.append((definition, level))
        return ModelProbeEvidence(level, True, 1, True, True, True)

    monkeypatch.setattr(ModelProbe, "run", run)
    return definitions


def test_setup_reuses_existing_environment_and_selects_model(setup_environment, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "existing-environment-secret")
    definitions = successful_probe(monkeypatch)
    result = CliRunner().invoke(
        agent_app, ["setup", "--model", "openai/gpt-oss-120b", "--no-mcp", "--no-web"], input="\n",
    )
    assert result.exit_code == 0, result.output
    assert definitions[0][1] is ProbeLevel.FULL
    assert "existing-environment-secret" not in result.output
    current = CliRunner().invoke(agent_app, ["model", "current"])
    assert current.exit_code == 0
    assert "openai/gpt-oss-120b" in current.output


def test_setup_hidden_key_is_persistent_without_mutating_environment(setup_environment, monkeypatch):
    from agent_runtime import model_credentials

    monkeypatch.setattr(model_credentials, "_PLATFORM", "linux")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    successful_probe(monkeypatch)
    secret = "private-setup-secret"
    result = CliRunner().invoke(
        agent_app, ["setup", "--model", "openai/gpt-oss-120b", "--no-mcp", "--no-web"], input=secret + "\n",
    )
    assert result.exit_code == 0, result.output
    assert secret not in result.output
    assert "GROQ_API_KEY" not in os.environ
    from agent_runtime.core.llm_registry import ModelRegistry

    resolved = ModelRegistry.from_env().resolve("openai/gpt-oss-120b")
    assert resolved.generator._client.api_key == secret
    assert secret not in (setup_environment / ".praxis/model_session.json").read_text()


def test_failed_probe_preserves_credential_and_selection(setup_environment, monkeypatch):
    from agent_runtime import model_credentials

    monkeypatch.setattr(model_credentials, "_PLATFORM", "linux")
    monkeypatch.setenv("GROQ_API_KEY", "old-environment-secret")
    session = setup_environment / ".praxis/model_session.json"
    before = session.read_bytes() if session.exists() else None
    entered = "failed-new-secret"

    async def fail(self, definition, *, level):
        assert os.environ["GROQ_API_KEY"] == entered
        raise ModelProbeError(phase="connectivity", detail="provider echoed " + entered)

    monkeypatch.setattr(ModelProbe, "run", fail)
    result = CliRunner().invoke(
        agent_app, ["setup", "--model", "openai/gpt-oss-120b", "--no-mcp", "--no-web"], input="n\n" + entered + "\n"
    )
    assert result.exit_code != 0
    assert entered not in result.output
    assert os.environ["GROQ_API_KEY"] == "old-environment-secret"
    assert (session.read_bytes() if session.exists() else None) == before
    assert model_credentials.load_model_credential("GROQ_API_KEY", "https://api.groq.com/openai/v1") is None


def test_successful_replacement_explains_parent_environment_precedence(setup_environment, monkeypatch):
    from agent_runtime import model_credentials

    monkeypatch.setattr(model_credentials, "_PLATFORM", "linux")
    monkeypatch.setenv("GROQ_API_KEY", "old-environment-key")
    successful_probe(monkeypatch)
    result = CliRunner().invoke(
        agent_app, ["setup", "--model", "openai/gpt-oss-120b", "--no-mcp", "--no-web"], input="n\nnew-stored-key\n"
    )
    assert result.exit_code == 0, result.output
    assert "unset GROQ_API_KEY" in result.output
    assert "new-stored-key" not in result.output
    assert os.environ["GROQ_API_KEY"] == "old-environment-key"
    monkeypatch.delenv("GROQ_API_KEY")
    assert (
        model_credentials.resolve_model_credential("GROQ_API_KEY", "https://api.groq.com/openai/v1") == "new-stored-key"
    )


def test_custom_setup_uses_real_full_probe_and_survives_fresh_registry(
    setup_environment,
    monkeypatch,
    provider_server,
):
    from agent_runtime import model_credentials
    from agent_runtime.core.llm_registry import ModelRegistry

    monkeypatch.setattr(model_credentials, "_PLATFORM", "linux")
    state, endpoint = provider_server
    result = CliRunner().invoke(
        agent_app,
        ["setup", "--model", "probe-model", "--no-mcp", "--no-web"],
        input=f"{endpoint}\n4096\nPRAXIS_MODEL_API_KEY\nprobe-secret\n",
    )
    assert result.exit_code == 0, result.output
    assert state.model_list_requests > 0 and state.requests
    assert any(body.get("tools") for body in state.requests)
    assert "probe-secret" not in result.output
    assert "PRAXIS_MODEL_API_KEY" not in os.environ
    model_file = Path(os.environ["PRAXIS_MODEL_REGISTRY_PATH"])
    assert "probe-secret" not in model_file.read_text()
    registry = ModelRegistry.from_env()
    assert registry.resolve("probe-model").generator._client.api_key == "probe-secret"
    # Subsequent probes use the saved credential without putting it into os.environ.
    import asyncio

    asyncio.run(ModelProbe(registry).run(registry.get_model_definition("probe-model"), level=ProbeLevel.FULL))
    from agent_runtime.local_runtime import LocalProviderProbe
    from agent_runtime.models import ModelCatalog

    asyncio.run(LocalProviderProbe().ensure_ready(ModelCatalog.from_registry(registry).get("probe-model")))
    from agent_runtime.models import ModelControlPlane

    plane = ModelControlPlane.from_env(
        workspace=setup_environment, session_path=setup_environment / ".praxis/model_session.json"
    )
    plane._trust_domain.initialize()
    try:
        binding = plane.freeze_model_binding(thread_id="setup-thread", turn_id="setup-turn")
        first = plane.resolve_frozen_binding(binding, thread_id="setup-thread", turn_id="setup-turn")
        assert first.generator._client.api_key == "probe-secret"
        model_credentials.save_model_credential("PRAXIS_MODEL_API_KEY", endpoint, "rotated-key")
        second = plane.resolve_frozen_binding(binding, thread_id="setup-thread", turn_id="setup-turn")
        assert first is not second
        assert second.generator._client.api_key == "rotated-key"
    finally:
        plane.close()


def test_setup_cancel_does_not_select_model(setup_environment, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = CliRunner().invoke(
        agent_app, ["setup", "--model", "openai/gpt-oss-120b", "--no-mcp", "--no-web"], input="\x03",
    )
    assert result.exit_code != 0
    assert not (setup_environment / ".praxis/model_session.json").exists()


def test_existing_github_connection_is_checked_without_re_registration(setup_environment, monkeypatch):
    from agent_runtime import mcp_admin, setup_cli

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    successful_probe(monkeypatch)
    monkeypatch.setattr(mcp_admin, "mcp_connections", lambda: [{"name": "github", "enabled": True}])
    monkeypatch.setattr(
        mcp_admin, "register_remote_mcp", lambda *_args, **_kwargs: pytest.fail("duplicate registration")
    )
    checked = []

    async def check(name, workspace):
        checked.append(name)
        return {"connected": True, "tools": ["mcp__github__get_file_contents"], "diagnostics": []}

    monkeypatch.setattr(setup_cli, "check_mcp_connection", check)
    result = CliRunner().invoke(agent_app, ["setup", "--model", "openai/gpt-oss-120b", "--no-web"], input="\ny\n")
    assert result.exit_code == 0, result.output
    assert checked == ["github"]


def test_setup_mcp_failure_does_not_claim_full_success(setup_environment, monkeypatch):
    from agent_runtime import mcp_admin, setup_cli

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    successful_probe(monkeypatch)
    monkeypatch.setattr(mcp_admin, "mcp_connections", lambda: [{"name": "github", "enabled": True}])

    async def check(name, workspace):
        return {"connected": False, "tools": [], "diagnostics": []}

    monkeypatch.setattr(setup_cli, "check_mcp_connection", check)
    result = CliRunner().invoke(agent_app, ["setup", "--model", "openai/gpt-oss-120b", "--no-web"], input="\ny\n")
    assert result.exit_code != 0
    current = CliRunner().invoke(agent_app, ["model", "current"])
    assert "openai/gpt-oss-120b" in current.output
    assert "GitHub" in result.output


def test_first_setup_offers_search_and_reuses_the_official_deepseek_account(setup_environment, monkeypatch):
    from agent_runtime import Agent
    from agent_runtime.web_config import load_search_key

    secret = "deepseek-model-and-search-credential"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    successful_probe(monkeypatch)
    result = CliRunner().invoke(
        agent_app, ["setup", "--model", "deepseek-flash", "--no-mcp"], input="\n\n\n",
    )
    assert result.exit_code == 0, result.output
    assert "联网搜索" in result.output
    assert "复用" in result.output and secret not in result.output
    agent = Agent(workspace_path=setup_environment)
    assert agent.web_search_provider == "deepseek"
    assert load_search_key(agent.web_search_key_file, workspace=setup_environment) == secret


def test_setup_can_skip_search_without_repeating_the_question(setup_environment, monkeypatch):
    from agent_runtime.web_config import resolve_web_configuration

    monkeypatch.setenv("GROQ_API_KEY", "existing-environment-secret")
    successful_probe(monkeypatch)
    runner = CliRunner()
    first = runner.invoke(
        agent_app, ["setup", "--model", "openai/gpt-oss-120b", "--no-mcp"], input="\nn\n",
    )
    assert first.exit_code == 0, first.output
    assert "联网搜索" in first.output
    configuration = resolve_web_configuration()
    assert configuration.provider is None and configuration.source == "configuration"
    second = runner.invoke(
        agent_app, ["setup", "--model", "openai/gpt-oss-120b", "--no-mcp"], input="\n",
    )
    assert second.exit_code == 0, second.output
    assert "启用联网搜索" not in second.output


def test_non_deepseek_model_gets_a_separate_hidden_search_key(setup_environment, monkeypatch):
    from agent_runtime.web_config import load_search_key, resolve_web_configuration

    monkeypatch.setenv("GROQ_API_KEY", "existing-groq-credential")
    successful_probe(monkeypatch)
    secret = "separate-deepseek-search-credential"
    result = CliRunner().invoke(
        agent_app, ["setup", "--model", "openai/gpt-oss-120b", "--no-mcp"], input="\ny\n" + secret + "\n",
    )
    assert result.exit_code == 0, result.output
    configuration = resolve_web_configuration()
    assert configuration.provider == "deepseek"
    assert load_search_key(configuration.key_file, workspace=setup_environment) == secret
    assert secret not in result.output and "existing-groq-credential" not in result.output


def test_setup_preserves_existing_search_provider(setup_environment, monkeypatch, tmp_path):
    from agent_runtime.web_config import resolve_web_configuration, save_web_configuration

    key = tmp_path / "tavily.key"
    key.write_text("existing-tavily-search-credential")
    key.chmod(0o600)
    save_web_configuration("tavily", key, workspace=setup_environment)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "existing-deepseek-credential")
    successful_probe(monkeypatch)
    result = CliRunner().invoke(
        agent_app, ["setup", "--model", "deepseek-flash", "--no-mcp"], input="\n",
    )
    assert result.exit_code == 0, result.output
    assert resolve_web_configuration().provider == "tavily"
    assert "启用联网搜索" not in result.output
