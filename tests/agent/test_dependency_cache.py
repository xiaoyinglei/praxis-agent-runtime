import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def installer():
    spec = importlib.util.spec_from_file_location("dependency_cache", ROOT / "scripts/ubuntu/install-agent-deps.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_uses_only_lock_artifact_hashes():
    module = installer()
    lock = {"package": [{"name": "demo", "version": "1.0", "source": {"registry": "https://pypi.org/simple"},
                         "wheels": [{"hash": "sha256:" + "a" * 64}]}]}
    manifest = module.locked_manifest("demo==1.0 \\\n --hash=sha256:" + "b" * 64 + "\n", lock)
    assert "a" * 64 in manifest
    assert "b" * 64 not in manifest
    with pytest.raises(ValueError):
        module.locked_manifest("demo==2.0\n", lock)


def test_repeated_offline_install_reuses_manifest_and_wheel_cache(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts/ubuntu").mkdir(parents=True)
    (repo / "scripts/ubuntu/requirements-core.in").write_text("demo\n")
    (repo / "uv.lock").write_text('[[package]]\nname="demo"\nversion="1.0"\n'
        '[package.source]\nregistry="https://pypi.org/simple"\n'
        '[[package.wheels]]\nhash="sha256:' + "a" * 64 + '"\n')
    commands = tmp_path / "bin"
    commands.mkdir()
    calls = tmp_path / "calls.jsonl"
    fake_uv = commands / "uv"
    fake_uv.write_text("#!/usr/bin/env python3\nimport sys,os,json\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\nwith open(os.environ['CALLS'],'a') as f: f.write(json.dumps(args)+'\\n')\n"
        "if '--output-file' in args:\n"
        " Path(args[args.index('--output-file')+1]).write_text('demo==1.0\\n')\n")
    fake_uv.chmod(0o755)
    import os
    env = {**os.environ, "PATH": f"{commands}:{os.environ['PATH']}", "CALLS": str(calls)}
    args = [sys.executable, str(ROOT / "scripts/ubuntu/install-agent-deps.py"), str(repo),
            "--python", sys.executable, "--cache-dir", str(tmp_path / "cache")]
    first = subprocess.run(args, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    second = subprocess.run([*args, "--offline", "--wheel-dir", str(wheels)], env=env, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    assert sum("compile" in args for args in records) == 1
    hashed = [args for args in records if "--require-hashes" in args]
    assert len(hashed) == 2
    assert all("--cache-dir" in args and "--only-binary" in args for args in hashed)
    assert "--offline" in hashed[-1]
    assert "--no-index" in hashed[-1] and "--find-links" in hashed[-1]
    assert "--offline" in records[-1] and "--no-deps" in records[-1]
    third = subprocess.run([*args, "--cache-dir", str(tmp_path / "fresh-cache"), "--offline",
                            "--wheel-dir", str(wheels)], env=env, capture_output=True, text=True)
    assert third.returncode == 0, third.stderr
    all_records = [json.loads(line) for line in calls.read_text().splitlines()]
    compiled = [args for args in all_records if "compile" in args][-1]
    assert "--offline" in compiled and "--no-index" in compiled and "--find-links" in compiled


def test_cache_refuses_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    cache = tmp_path / "cache"
    cache.symlink_to(target)
    with pytest.raises(ValueError, match="cache"):
        installer().prepare_cache(cache)


def test_service_installer_preserves_hashed_binary_cache_policy():
    script = (ROOT / "scripts/ubuntu/install-model-service.sh").read_text()
    assert "--no-cache-dir" not in script
    assert '--cache-dir "$model_cache"' in script
    assert "--require-hashes --only-binary=:all:" in script
