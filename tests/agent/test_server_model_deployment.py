"""Operational boundary checks for the server-side model service."""

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

from agent_runtime.agent import Agent
from agent_runtime.models import ModelCatalog, UnknownModelIdError

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "ubuntu"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_server_catalog_switches_existing_aliases_without_restarting(tmp_path, monkeypatch):
    catalog = SCRIPTS / "server-models.yaml"
    models = ModelCatalog.from_config_file(catalog)
    assert {m.id for m in models.list_models()} == {"deepseek-flash", "openai/gpt-oss-120b"}
    assert all(m.base_url == "http://127.0.0.1:18444/v1" for m in models.list_models())
    monkeypatch.setenv("RAG_AGENT_MODELS_PATH", str(catalog))
    monkeypatch.setenv("PRAXIS_GATEWAY_TOKEN", "fake-agent-token-secret")
    agent = Agent(workspace_path=tmp_path, checkpoint_db=None, model_session_path=None)
    agent.switch_model("openai/gpt-oss-120b")
    assert agent.current_model().id == "openai/gpt-oss-120b"
    agent.switch_model("deepseek-flash")
    assert agent.current_model().id == "deepseek-flash"
    with pytest.raises(UnknownModelIdError):
        agent.switch_model("https://evil")
    assert agent.current_model().id == "deepseek-flash"


def test_provisioned_files_and_rotation_are_private(tmp_path):
    setup = load_script("provision-model-secrets.py")
    directory = tmp_path / "credentials"
    token_path = tmp_path / "agent" / "proxy.token"
    token_path.parent.mkdir(mode=0o700)
    prompts = []

    def prompt(label):
        prompts.append(label)
        return "fake-deepseek-provider-key"

    setup.provision(directory, token_path, os.getuid(), os.getgid(), ["deepseek"], prompt=prompt)
    before = (directory / "agent-token").read_text()
    assert token_path.read_text() == before
    assert (directory.stat().st_mode & 0o777) == 0o700
    assert (directory / "deepseek-key").read_text() == "fake-deepseek-provider-key\n"
    assert all((p.stat().st_mode & 0o777) == 0o400 for p in [token_path, *directory.iterdir()])
    assert len(prompts) == 1
    with pytest.raises(FileExistsError):
        setup.provision(directory, token_path, os.getuid(), os.getgid(), ["deepseek"], prompt=prompt)
    setup.rotate_token(directory, token_path, os.getuid(), os.getgid())
    assert token_path.read_text() != before
    assert (directory / "agent-token").read_text() == token_path.read_text()
    assert (directory / "deepseek-key").read_text() == "fake-deepseek-provider-key\n"


def test_provision_validates_all_keys_before_writing_and_refuses_symlinks(tmp_path):
    setup = load_script("provision-model-secrets.py")
    directory = tmp_path / "credentials"
    with pytest.raises(ValueError):
        setup.provision(
            directory, tmp_path / "token", os.getuid(), os.getgid(), ["deepseek"], prompt=lambda _: "bad\nkey"
        )
    assert not directory.exists()
    target = tmp_path / "target"
    target.mkdir()
    directory.symlink_to(target)
    with pytest.raises(FileExistsError):
        setup.provision(
            directory,
            tmp_path / "token",
            os.getuid(),
            os.getgid(),
            ["deepseek"],
            prompt=lambda _: "valid-fake-provider-key",
        )
    assert not list(target.iterdir())


def test_agent_environment_only_receives_proxy_token(tmp_path, monkeypatch):
    launcher = load_script("launch-server-agent.py")
    token = tmp_path / "proxy.token"
    token.write_text("fake-agent-token-secret\n")
    token.chmod(0o400)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "should-never-forward")
    monkeypatch.setenv("GROQ_API_KEY", "should-never-forward")
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    monkeypatch.setenv("LD_PRELOAD", "/untrusted.so")
    env = launcher.agent_environment(token, tmp_path, "dumb")
    assert env["PRAXIS_GATEWAY_TOKEN"] == "fake-agent-token-secret"
    assert env["RAG_AGENT_MODELS_PATH"] == "/opt/praxis/server-models.yaml"
    assert not {"DEEPSEEK_API_KEY", "GROQ_API_KEY", "PYTHONPATH", "LD_PRELOAD", "SSH_AUTH_SOCK"} & env.keys()
    token.chmod(0o644)
    with pytest.raises(ValueError):
        launcher.agent_environment(token, tmp_path, "dumb")


def test_service_venv_rejects_editable_import_paths(tmp_path):
    validator = load_script("validate-model-venv.py")
    site = tmp_path / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    hook = site / "editable.pth"
    hook.write_text("/agent-writable/workspace\n")
    with pytest.raises(ValueError):
        validator.validate_site_packages(site)
    hook.unlink()
    (site / "outward").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError):
        validator.validate_site_packages(site)
    (site / "outward").unlink()
    validator.validate_site_packages(site)


def test_unit_and_launch_scripts_enforce_separate_identity():
    unit = (SCRIPTS / "praxis-model.service").read_text()
    for directive in [
        "User=praxis-model",
        "Group=praxis-model",
        "NoNewPrivileges=yes",
        "ProtectSystem=strict",
        "ProtectHome=yes",
        "StateDirectoryMode=0700",
        "LimitCORE=0",
        "UMask=0077",
        "LoadCredential=agent-token:",
        "python -I -B",
        "server_model_gateway.py",
    ]:
        assert directive in unit
    assert "praxis-agent" not in unit
    for name in ["install-model-service.sh", "start-server-agent.sh"]:
        subprocess.run(["bash", "-n", str(SCRIPTS / name)], check=True)
    result = subprocess.run(["bash", str(SCRIPTS / "start-server-agent.sh")], capture_output=True, text=True)
    assert result.returncode != 0
    assert "praxis-agent" in result.stderr


def test_minimal_service_requirements_are_pinned_to_lock():
    import tomllib

    packages = {p["name"]: p for p in tomllib.loads((ROOT / "uv.lock").read_text())["package"]}
    text = (SCRIPTS / "requirements-model-service.txt").read_text()
    for name in ["httpx", "starlette", "uvicorn"]:
        assert name + "==" + packages[name]["version"] in text
    assert "--hash=sha256:" in text
    assert "-e " not in text


def test_server_agent_does_not_load_workspace_dotenv(tmp_path, monkeypatch):
    from agent_runtime.text import load_env_file

    launcher = load_script("launch-server-agent.py")
    token = tmp_path / "token"
    token.write_text("fake-agent-proxy-token-secret")
    token.chmod(0o400)
    env = launcher.agent_environment(token, tmp_path, "dumb")
    assert env["PRAXIS_DISABLE_DOTENV"] == "1"
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("RAG_AGENT_MODELS_PATH", str(SCRIPTS / "server-models.yaml"))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    (workspace / ".env").write_text("DEEPSEEK_API_KEY=fake-do-not-load\nGROQ_API_KEY=fake-do-not-load\n")
    assert load_env_file() is None
    assert Agent(workspace_path=workspace, checkpoint_db=None).current_model().id == "deepseek-flash"
    assert "DEEPSEEK_API_KEY" not in os.environ and "GROQ_API_KEY" not in os.environ


def test_hidden_prompt_rejects_getpass_echo_fallback(monkeypatch):
    import getpass
    import warnings

    setup = load_script("provision-model-secrets.py")
    read = []

    def fallback(label):
        warnings.warn("echo cannot be controlled", getpass.GetPassWarning, stacklevel=1)
        read.append(True)
        return "would-have-read-secret"

    monkeypatch.setattr(getpass, "getpass", fallback)
    with pytest.raises(getpass.GetPassWarning):
        setup.hidden_prompt("secret")
    assert not read


@pytest.mark.parametrize("key", ["version", "version_info"])
def test_agent_venv_accepts_stdlib_and_uv_config_then_relocates(tmp_path, key):
    validator = load_script("validate-model-venv.py")
    config = tmp_path / "pyvenv.cfg"
    config.write_text(f"home = /home/admin/.local/share/uv/python/bin\n{key} = 3.12.3\n")
    validator.validate_agent_config(config)
    validator.relocate_agent_config(config)
    text = config.read_text()
    assert "home = /usr/bin" in text
    assert "include-system-site-packages = false" in text
    assert "/home/admin" not in text
    config.write_text("version_info = 3.13.0\n")
    with pytest.raises(ValueError):
        validator.validate_agent_config(config)


@pytest.mark.parametrize("args", [[], ["run", "hello world", "--model", "deepseek-flash"]])
def test_praxis_entrypoint_switches_identity_and_preserves_arguments(tmp_path, args):
    import json

    commands = tmp_path / "bin"
    commands.mkdir()
    capture = tmp_path / "arguments.json"
    for name, source in {
        "id": "#!/bin/sh\nprintf '%s\\n' admin\n",
        "systemctl": "#!/bin/sh\nexit 0\n",
        "sudo": (
            "#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\n"
            "Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
        ),
    }.items():
        command = commands / name
        command.write_text(source)
        command.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPTS / "praxis.sh"), *args],
        env={
            **os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "CAPTURE": str(capture),
            "TERM": "xterm-256color",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(capture.read_text()) == [
        "-H",
        "-u",
        "praxis-agent",
        "--",
        "/usr/bin/env",
        "-i",
        "PATH=/usr/bin:/bin",
        "TERM=xterm-256color",
        "/usr/local/bin/praxis-agent",
        *(args or ["chat"]),
    ]


def test_praxis_entrypoint_explains_inactive_service_without_launching(tmp_path):
    commands = tmp_path / "bin"
    commands.mkdir()
    command = commands / "systemctl"
    command.write_text("#!/bin/sh\nexit 3\n")
    command.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPTS / "praxis.sh")],
        env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "sudo systemctl start praxis-model" in result.stderr
