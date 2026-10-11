from __future__ import annotations

import hashlib
import io
import runpy
import sys
import tarfile
from pathlib import Path

import pytest


@pytest.mark.parametrize("valid", [False, True])
def test_installer_checks_digest_before_replacing_destination(tmp_path, monkeypatch, valid):
    script = Path(__file__).resolve().parents[2] / "scripts/ubuntu/install-codex-patch.py"
    main = runpy.run_path(str(script))["main"]
    archive = tmp_path / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("codex-x86_64-unknown-linux-musl")
        info.size = 5
        bundle.addfile(info, io.BytesIO(b"codex"))
    destination = tmp_path / "tools with spaces" / "codex"
    destination.parent.mkdir()
    destination.write_text("previous")
    monkeypatch.setattr(main.__globals__["platform"], "system", lambda: "Linux")
    monkeypatch.setattr(main.__globals__["platform"], "machine", lambda: "x86_64")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest() if valid else "0" * 64
    monkeypatch.setitem(main.__globals__, "ARCHIVES", {"x86_64": ("x86_64-unknown-linux-musl", digest)})
    monkeypatch.setattr(sys, "argv", [str(script), "--archive", str(archive), "--destination", str(destination)])
    if valid:
        main()
        assert destination.read_bytes() == b"codex"
        assert destination.stat().st_mode & 0o777 == 0o755
    else:
        with pytest.raises(ValueError, match="digest mismatch"):
            main()
        assert destination.read_text() == "previous"
    assert list(destination.parent.iterdir()) == [destination]


def test_server_launcher_activates_only_fixed_installed_patch_path(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/ubuntu/launch-server-agent.py"
    environment = runpy.run_path(str(script))["agent_environment"]
    token = tmp_path / "proxy.token"
    token.write_text("synthetic-proxy-token")
    token.chmod(0o400)
    monkeypatch.setenv("PRAXIS_CODEX_APPLY_PATCH", "/untrusted/ambient-binary")
    assert "PRAXIS_CODEX_APPLY_PATCH" not in environment(token, tmp_path, "dumb")
    binary = tmp_path / ".local/lib/praxis/codex-patch"
    binary.parent.mkdir(parents=True)
    binary.write_text("synthetic executable")
    binary.chmod(0o755)
    assert environment(token, tmp_path, "dumb")["PRAXIS_CODEX_APPLY_PATCH"] == str(binary)
    binary.chmod(0o777)
    with pytest.raises(ValueError, match="patch"):
        environment(token, tmp_path, "dumb")


def test_server_launcher_rejects_symlink_patch_executor(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/ubuntu/launch-server-agent.py"
    environment = runpy.run_path(str(script))["agent_environment"]
    token = tmp_path / "proxy.token"
    token.write_text("synthetic-proxy-token")
    token.chmod(0o400)
    binary = tmp_path / ".local/lib/praxis/codex-patch"
    binary.parent.mkdir(parents=True)
    binary.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="patch"):
        environment(token, tmp_path, "dumb")
