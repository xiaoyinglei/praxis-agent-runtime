"""Use real repositories to verify update safety; replace only network fetches."""
import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_updater():
    spec = importlib.util.spec_from_file_location("praxis_git_update", ROOT / "scripts/ubuntu/update-repo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def repositories(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-b", "main")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "Test")
    (source / "file").write_text("initial")
    git(source, "add", "file")
    git(source, "commit", "-m", "initial")
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", str(source), str(checkout)], check=True, capture_output=True)
    git(checkout, "remote", "set-url", "origin", "https://github.com/example/project.git")
    (source / "file").write_text("new")
    git(source, "commit", "-am", "new")
    return source, checkout


def update_with_network(monkeypatch, source, checkout, failures=()):
    updater = load_updater()
    original = updater.run_git
    fetches = []
    failures = iter(failures)

    def run(repo, args, deadline, **kwargs):
        if "fetch" in args:
            fetches.append(args)
            failure = next(failures, None)
            if failure:
                raise updater.GitUpdateError(failure)
            args = list(args)
            args[args.index("fetch") + 3] = str(source)
        return original(repo, args, deadline, **kwargs)

    monkeypatch.setattr(updater, "run_git", run)
    updater.update(checkout, "main", timeout=5)
    return fetches


def test_network_retry_then_fast_forward(monkeypatch, repositories):
    source, checkout = repositories
    before_config = (checkout / ".git/config").read_bytes()
    fetches = update_with_network(monkeypatch, source, checkout, ["tls", "timeout"])
    assert len(fetches) == 3
    assert git(checkout, "rev-parse", "HEAD") == git(source, "rev-parse", "HEAD")
    assert (checkout / ".git/config").read_bytes() == before_config
    assert all("http.version=HTTP/1.1" in f and "http.sslVerify=true" in f for f in fetches)
    assert not git(checkout, "for-each-ref", "refs/praxis-update")


@pytest.mark.parametrize("failure", ["authentication", "certificate", "repository", "ref"])
def test_non_network_failures_stop_without_updating(monkeypatch, repositories, failure):
    source, checkout = repositories
    head = git(checkout, "rev-parse", "HEAD")
    with pytest.raises(Exception, match=failure):
        update_with_network(monkeypatch, source, checkout, [failure])
    assert git(checkout, "rev-parse", "HEAD") == head


@pytest.mark.parametrize("dirty", ["tracked", "staged", "untracked"])
def test_dirty_checkout_is_not_stashed_or_overwritten(repositories, dirty):
    _, checkout = repositories
    (checkout / ("extra" if dirty == "untracked" else "file")).write_text("keep me")
    if dirty == "staged":
        git(checkout, "add", "file")
    before = git(checkout, "status", "--porcelain")
    with pytest.raises(Exception, match="dirty"):
        load_updater().update(checkout, "main", timeout=5)
    assert git(checkout, "status", "--porcelain") == before


def test_diverged_checkout_refuses_merge(monkeypatch, repositories):
    source, checkout = repositories
    git(checkout, "config", "user.email", "test@example.com")
    git(checkout, "config", "user.name", "Test")
    (checkout / "local").write_text("keep")
    git(checkout, "add", "local")
    git(checkout, "commit", "-m", "local")
    head = git(checkout, "rev-parse", "HEAD")
    with pytest.raises(Exception, match="fast-forward"):
        update_with_network(monkeypatch, source, checkout)
    assert git(checkout, "rev-parse", "HEAD") == head


def test_git_environment_cannot_disable_tls(monkeypatch, tmp_path):
    updater = load_updater()
    monkeypatch.setenv("GIT_SSL_NO_VERIFY", "true")
    env = updater.git_environment()
    assert env["GIT_SSL_NO_VERIFY"] == "false"
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_inherited_git_directory_cannot_redirect_update(monkeypatch, repositories, tmp_path):
    source, checkout = repositories
    other = tmp_path / "other"
    subprocess.run(["git", "clone", str(checkout), str(other)], check=True, capture_output=True)
    git(other, "remote", "set-url", "origin", "https://github.com/example/project.git")
    other_head = git(other, "rev-parse", "HEAD")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    update_with_network(monkeypatch, source, checkout)
    monkeypatch.delenv("GIT_DIR")
    monkeypatch.delenv("GIT_WORK_TREE")
    assert git(checkout, "rev-parse", "HEAD") == git(source, "rev-parse", "HEAD")
    assert git(other, "rev-parse", "HEAD") == other_head


def test_bundle_keeps_local_recovery_budget_after_network_timeout(monkeypatch, repositories, tmp_path):
    import time
    source, checkout = repositories
    bundle = tmp_path / "approved.bundle"
    commit = git(source, "rev-parse", "HEAD")
    git(source, "bundle", "create", str(bundle), "main", "^" + git(checkout, "rev-parse", "HEAD"))
    updater = load_updater()
    original = updater.run_git

    def blackhole(repo, args, deadline, **kwargs):
        if "fetch" in args and any(a.startswith("https://") for a in args):
            time.sleep(kwargs["attempt_timeout"])
            raise updater.GitUpdateError("timeout")
        return original(repo, args, deadline, **kwargs)

    monkeypatch.setattr(updater, "run_git", blackhole)
    updater.update(checkout, "main", timeout=1.5, bundle=bundle, expected_commit=commit)
    assert git(checkout, "rev-parse", "HEAD") == commit


def test_timeout_kills_fetch_process_group(tmp_path):
    import time
    updater = load_updater()
    command = tmp_path / "git"
    command.write_text("#!/bin/sh\nsleep 60 &\nwait\n")
    command.chmod(0o755)
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{tmp_path}:{old}"
    try:
        started = time.monotonic()
        with pytest.raises(updater.GitUpdateError, match="timeout"):
            updater.run_git(tmp_path, ["fetch"], time.monotonic() + 0.1)
        assert time.monotonic() - started < 2
    finally:
        os.environ["PATH"] = old
