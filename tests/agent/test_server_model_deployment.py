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


def test_core_reinstall_repairs_cached_wheel_hardlinks(tmp_path, monkeypatch):
    """Exercise real uv files; mock only resolver preparation for a tiny wheel."""
    import hashlib
    import sys
    import zipfile

    from agent_runtime.tools.builtins.shell import _workspace_sandbox_preflight_issue

    installer = load_script("install-agent-deps.py")
    repo = tmp_path / "repo"
    (repo / "scripts/ubuntu").mkdir(parents=True)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel = wheels / "demo-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        files = {
            "demo.py": "VALUE = 42\n",
            "demo-1.0.data/headers/demo.h": "/* header fixture */\n",
            "demo-1.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n",
            "demo-1.0.dist-info/WHEEL": (
                "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
            ),
        }
        files["demo-1.0.dist-info/RECORD"] = "".join(f"{name},,\n" for name in [*files, "demo-1.0.dist-info/RECORD"])
        for name, content in files.items():
            archive.writestr(name, content)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    (repo / "uv.lock").write_text(
        '[[package]]\nname="demo"\nversion="1.0"\n'
        '[package.source]\nregistry="https://pypi.org/simple"\n'
        '[[package.wheels]]\nhash="sha256:' + digest + '"\n'
    )
    (repo / "scripts/ubuntu/requirements-core.in").write_text("demo\n")
    environment = repo / ".venv"
    cache = tmp_path / "cache"
    subprocess.run(
        ["uv", "--cache-dir", str(cache / "uv"), "venv", "--python", sys.executable, str(environment)],
        check=True,
    )
    python = environment / "bin/python"
    subprocess.run(
        ["uv", "--cache-dir", str(cache / "uv"), "pip", "install", "--python", str(python),
         "--link-mode", "hardlink", str(wheel)], check=True,
    )
    installed = next(environment.glob("lib/python*/site-packages/demo.py"))
    header = next(environment.glob("include/**/demo.h"))
    assert installed.stat().st_nlink > 1
    assert header.stat().st_nlink > 1
    assert _workspace_sandbox_preflight_issue(repo, allow_workspace_write=False)[0] == "workspace_hardlink_detected"
    real_run = subprocess.run
    installation_calls = []

    def run(args, **kwargs):
        if "--output-file" in args:
            Path(args[args.index("--output-file") + 1]).write_text("demo==1.0\n")
            return subprocess.CompletedProcess(args, 0)
        if "pip" in args:
            installation_calls.append(args)
        if "-e" in args:
            return subprocess.CompletedProcess(args, 0)
        return real_run(args, **kwargs)

    monkeypatch.setattr(installer.subprocess, "run", run)
    installer.install(repo, python, cache, offline=True, wheel_dir=wheels)
    assert installed.stat().st_nlink == 1
    assert header.stat().st_nlink == 1
    assert _workspace_sandbox_preflight_issue(repo, allow_workspace_write=False) is None
    assert all("--link-mode" in args and args[args.index("--link-mode") + 1] == "copy" for args in installation_calls)
    assert all("--reinstall" in args for args in installation_calls)


def test_dependency_file_check_includes_headers_outside_site_packages(tmp_path):
    installer = load_script("install-agent-deps.py")
    environment = tmp_path / "venv"
    header = environment / "include/site/python3.12/greenlet/greenlet.h"
    header.parent.mkdir(parents=True)
    cached = tmp_path / "cached-header.h"
    cached.write_text("/* fixture */\n")
    os.link(cached, header)
    with pytest.raises(ValueError, match="include/site/python3.12/greenlet/greenlet.h"):
        installer.validate_environment_files(environment)
    header.unlink()
    header.write_text(cached.read_text())
    installer.validate_environment_files(environment)


def test_server_launcher_readiness_uses_shell_workspace_containment(tmp_path):
    launcher = load_script("launch-server-agent.py")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".venv/include").mkdir(parents=True)
    cached = tmp_path / "outside.h"
    cached.write_text("fixture")
    linked = workspace / ".venv/include/header.h"
    os.link(cached, linked)
    with pytest.raises(ValueError, match="workspace_hardlink_detected"):
        launcher.validate_workspace_readiness(workspace)
    linked.unlink()
    linked.write_text(cached.read_text())
    launcher.validate_workspace_readiness(workspace)


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_repaired_workspace_runs_real_shell_without_relaxing_hardlink_check(tmp_path):
    from agent_runtime.tools.builtins import shell
    from agent_runtime.workspace import open_workspace

    workspace = open_workspace(tmp_path / "workspace", create=True)
    cached = tmp_path / "wheel-file"
    cached.write_text("fixture")
    linked = workspace.root / ".venv/include/header.h"
    linked.parent.mkdir(parents=True)
    os.link(cached, linked)
    result = await shell._run_command(
        workspace, shell.RunCommandInput(command="printf deployment-ready"), termination_grace_seconds=0.1,
    )
    assert result.sandbox_error == "workspace_hardlink_detected"
    assert not result.stdout
    linked.unlink()
    linked.write_text(cached.read_text())
    result = await shell._run_command(
        workspace, shell.RunCommandInput(command="printf deployment-ready"), termination_grace_seconds=0.1,
    )
    assert result.sandbox_error is None
    assert result.exit_code == 0
    assert result.stdout == "deployment-ready"


@pytest.mark.parametrize(
    "argv", [["chat"], ["run", "task"], ["resume", "id"], ["recover", "id"], [],
             ["model", "list"], ["history"], ["--help"]],
)
def test_server_launcher_checks_only_agent_turn_entrypoints(tmp_path, monkeypatch, argv):
    import sys
    from types import SimpleNamespace

    launcher = load_script("launch-server-agent.py")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(launcher.pwd, "getpwnam", lambda _: SimpleNamespace(pw_uid=os.getuid(), pw_dir=str(tmp_path)))
    monkeypatch.setattr(launcher, "agent_environment", lambda *args: {})
    monkeypatch.setattr(sys, "argv", ["launch-server-agent.py", *argv])
    checks = []
    launched = []
    monkeypatch.setattr(launcher, "validate_workspace_readiness", lambda root: checks.append(root))
    monkeypatch.setattr(launcher.os, "execve", lambda *args: launched.append(args))
    monkeypatch.chdir(tmp_path)
    launcher.main()
    assert checks == ([workspace] if not argv or argv[0] in {"chat", "run", "resume", "recover"} else [])
    assert launched


def _direct_environment(tmp_path):
    import shlex
    import sys

    environment = tmp_path / "runtime"
    (environment / "bin").mkdir(parents=True)
    agent = environment / "bin/agent"
    agent.write_text("#!/bin/sh\nexit 0\n")
    agent.chmod(0o700)
    python = environment / "bin/python"
    python.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} \"$@\"\n")
    python.chmod(0o700)
    return environment


def test_direct_start_refuses_hardlinks_before_loading_credentials(tmp_path, monkeypatch):
    launcher = load_script("start-deepseek-chat.py")
    workspace = tmp_path / "repo"
    (workspace / ".venv/include").mkdir(parents=True)
    cached = tmp_path / "cached-header"
    cached.write_text("fixture")
    os.link(cached, workspace / ".venv/include/header.h")
    environment = _direct_environment(tmp_path)
    monkeypatch.setattr(launcher, "load_key", lambda *args, **kwargs: pytest.fail("credential read before readiness"))
    monkeypatch.setattr(launcher.os, "execve", lambda *args: pytest.fail("must not launch Agent"))
    with pytest.raises(ValueError, match="workspace_hardlink_detected.*header.h"):
        launcher.start(workspace, tmp_path / "unused-key", venv=environment)


@pytest.mark.parametrize("external", [False, True])
def test_direct_start_keeps_exec_path_and_arguments_after_readiness(tmp_path, monkeypatch, external):
    launcher = load_script("start-deepseek-chat.py")
    workspace = tmp_path / "repo"
    workspace.mkdir()
    environment = _direct_environment(tmp_path)
    if not external:
        environment.rename(workspace / ".venv")
        environment = workspace / ".venv"
    configured = []
    monkeypatch.setattr(launcher, "load_key", lambda path, **kwargs: configured.append(kwargs) or "fake-credential")
    calls = []
    monkeypatch.setattr(launcher.os, "execve", lambda *args: calls.append(args))
    monkeypatch.chdir(tmp_path)
    launcher.start(workspace, tmp_path / "unused-key", configure=True, venv=environment if external else None)
    assert configured == [{"configure": True}]
    agent = str(environment / "bin/agent")
    assert calls[0][0:2] == (agent, [agent, "chat", "--model", "deepseek-flash"])
    assert Path.cwd() == workspace
