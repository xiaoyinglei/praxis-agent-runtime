"""One-time credential setup for the direct Ubuntu CLI entrypoint."""
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def launcher():
    spec = importlib.util.spec_from_file_location("deepseek_startup", ROOT / "scripts/ubuntu/start-deepseek-chat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_first_start_saves_once_and_later_start_never_prompts(tmp_path):
    module = launcher()
    path = tmp_path / "private" / "deepseek.key"
    prompts = []
    key = "fake-deepseek-key-for-tests"
    assert module.load_key(path, prompt=lambda label: prompts.append(label) or key) == key
    assert len(prompts) == 1
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700

    def unexpected_prompt(label):
        raise AssertionError("already configured; do not prompt again")
    assert module.load_key(path, prompt=unexpected_prompt) == key


@pytest.mark.parametrize("unsafe", ["symlink", "public", "directory"])
def test_existing_unsafe_credentials_are_refused(tmp_path, unsafe):
    module = launcher()
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    key = directory / "deepseek.key"
    if unsafe == "directory":
        key.mkdir()
    else:
        key.write_text("fake-key-long-enough")
        if unsafe == "symlink":
            target = directory / "actual"
            key.rename(target)
            key.symlink_to(target)
        else:
            key.chmod(0o644)
    with pytest.raises(ValueError):
        module.load_key(key, prompt=lambda _: "do-not-overwrite-key")


def test_rotation_is_explicit_and_invalid_input_preserves_key(tmp_path):
    module = launcher()
    key = tmp_path / "private/deepseek.key"
    before = "fake-first-deepseek-key"
    module.load_key(key, prompt=lambda _: before)
    with pytest.raises(ValueError):
        module.load_key(key, configure=True, prompt=lambda _: "bad\nkey")
    assert module.load_key(key) == before
    assert module.load_key(key, configure=True, prompt=lambda _: "fake-rotated-deepseek-key") != before


def test_explicit_rotation_can_repair_invalid_private_file(tmp_path):
    module = launcher()
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    key = directory / "deepseek.key"
    key.write_text("")
    key.chmod(0o600)
    assert module.load_key(key, configure=True, prompt=lambda _: "fake-repaired-deepseek-key")


def test_hidden_prompt_never_falls_back_to_echo(monkeypatch):
    import getpass
    import warnings
    module = launcher()
    def fallback(label):
        warnings.warn("echo unavailable", getpass.GetPassWarning, stacklevel=1)
        return "fake-key-that-must-not-be-read"
    monkeypatch.setattr(getpass, "getpass", fallback)
    with pytest.raises(getpass.GetPassWarning):
        module.hidden_prompt("Key:")


def test_launcher_passes_key_only_in_environment_and_skips_dotenv(monkeypatch, tmp_path):
    module = launcher()
    repo = tmp_path / "repo"
    (repo / ".venv/bin").mkdir(parents=True)
    agent = repo / ".venv/bin/agent"
    agent.write_text("placeholder")
    agent.chmod(0o700)
    key_path = tmp_path / "private/deepseek.key"
    module.load_key(key_path, prompt=lambda _: "fake-protected-deepseek-key")
    calls = []
    monkeypatch.setattr(os, "execve", lambda path, argv, env: calls.append((path, argv, env)))
    monkeypatch.chdir(tmp_path)
    module.start(repo, key_path)
    path, argv, env = calls[0]
    assert path == str(agent)
    assert argv == [str(agent), "chat", "--model", "deepseek-flash"]
    assert env["DEEPSEEK_API_KEY"] == "fake-protected-deepseek-key"
    assert env["PRAXIS_DISABLE_DOTENV"] == "1"
    assert all("fake-protected" not in arg for arg in argv)
