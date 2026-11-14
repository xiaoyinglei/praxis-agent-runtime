"""凭据卫生(设计 §5.15;D6):env 优先、macOS Keychain、POSIX 0600 文件。

约定:
- 任何测试都不依赖真实 Keychain 内容。darwin 分支用注入的 ``subprocess.run``
  假实现,断言命令形状 + 值只走 stdin、绝不进 argv;
- POSIX 分支用 ``PRAXIS_CONFIG_DIR`` 指向 tmp_path 跑真实文件权限校验;
- 失败文案固定,绝不回显文件 body 或原异常文本。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from praxis.adapters.config import credentials
from praxis.adapters.config.credentials import (
    credential_identity,
    delete_model_credential,
    resolve_model_credential,
    save_model_credential,
)
from praxis.protocol.floor import host_config_dir

_ENDPOINT = "https://api.deepseek.com/v1"
_NAME = "DEEPSEEK_API_KEY"
_FILE_ERROR = "Credential file must be private, bounded, and owned by the current user"
_KEYCHAIN_READ_ERROR = "Unable to read model credential from Keychain"


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path))


@pytest.fixture
def posix(monkeypatch: pytest.MonkeyPatch) -> None:
    """强制走 POSIX 文件分支(本机 darwin 也不碰真钥匙串)。"""
    monkeypatch.setattr(credentials, "_PLATFORM", "linux")


@pytest.fixture
def darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(credentials, "_PLATFORM", "darwin")


def _identity(name: str = _NAME, endpoint: str = _ENDPOINT) -> str:
    return credential_identity(name, endpoint)


def _credential_file() -> Path:
    return host_config_dir() / "credentials" / "models.json"


def _write_keys(keys: dict[str, str], *, mode: int = 0o600) -> Path:
    directory = host_config_dir() / "credentials"
    directory.mkdir(mode=0o700, exist_ok=True)
    os.chmod(directory, 0o700)
    path = directory / "models.json"
    path.write_text(json.dumps({"version": 1, "keys": keys}), encoding="utf-8")
    os.chmod(path, mode)
    return path


def _write_raw(text: str, *, mode: int = 0o600) -> Path:
    directory = host_config_dir() / "credentials"
    directory.mkdir(mode=0o700, exist_ok=True)
    os.chmod(directory, 0o700)
    path = directory / "models.json"
    path.write_text(text, encoding="utf-8")
    os.chmod(path, mode)
    return path


class _FakeRun:
    """注入的 subprocess.run 假实现:按顺序返回预设 (returncode, stdout, stderr)。"""

    def __init__(self, responses: list[tuple[int, str, str]]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(argv), dict(kwargs)))
        returncode, stdout, stderr = self._responses.pop(0)
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


def _inject(monkeypatch: pytest.MonkeyPatch, responses: list[tuple[int, str, str]]) -> _FakeRun:
    fake = _FakeRun(responses)
    monkeypatch.setattr(credentials.subprocess, "run", fake)
    return fake


# ---------------------------------------------------------------------------
# identity 与环境变量优先级
# ---------------------------------------------------------------------------


def test_identity_is_endpoint_bound_and_trailing_slash_stable() -> None:
    assert _identity() == _identity(endpoint=_ENDPOINT + "/")
    assert _identity() != _identity(endpoint="https://other.test/v1")
    assert _identity() != _identity(name="OTHER_KEY")


def test_env_var_takes_precedence(monkeypatch: pytest.MonkeyPatch, darwin: None) -> None:
    monkeypatch.setenv(_NAME, "from-env")
    fake = _inject(monkeypatch, [])  # 任何 Keychain 读取都会因空队列而失败
    assert resolve_model_credential(_NAME, _ENDPOINT) == "from-env"
    assert fake.calls == []


# ---------------------------------------------------------------------------
# POSIX 文件分支:往返、权限链、固定文案
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX hygiene checks")
def test_posix_roundtrip_creates_private_file(posix: None) -> None:
    save_model_credential(_NAME, _ENDPOINT, "  sk-secret  ")
    directory = host_config_dir() / "credentials"
    path = _credential_file()
    assert path.exists()
    assert directory.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"version": 1, "keys": {_identity(): "sk-secret"}}
    assert resolve_model_credential(_NAME, _ENDPOINT) == "sk-secret"

    # merge:保存第二个凭据不丢第一个。
    save_model_credential("OTHER_KEY", _ENDPOINT, "other-secret")
    keys = json.loads(path.read_text(encoding="utf-8"))["keys"]
    assert keys == {
        _identity(): "sk-secret",
        _identity("OTHER_KEY"): "other-secret",
    }

    delete_model_credential(_NAME, _ENDPOINT)
    assert resolve_model_credential(_NAME, _ENDPOINT) is None
    delete_model_credential(_NAME, _ENDPOINT)  # 再删一次是 no-op


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX hygiene checks")
def test_file_mode_and_nlink_enforced(posix: None) -> None:
    path = _write_keys({_identity(): "secret"})
    os.chmod(path, 0o644)
    with pytest.raises(ValueError, match="private, bounded"):
        resolve_model_credential(_NAME, _ENDPOINT)

    os.chmod(path, 0o600)
    assert resolve_model_credential(_NAME, _ENDPOINT) == "secret"
    os.link(path, host_config_dir() / "dup")
    with pytest.raises(ValueError, match="private, bounded"):  # nlink == 1 强制
        resolve_model_credential(_NAME, _ENDPOINT)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX hygiene checks")
def test_payload_shape_enforced(posix: None) -> None:
    for payload in (
        {"version": 2, "keys": {}},
        {"version": 1, "keys": {}, "extra": 1},
        {"version": 1},
        {"keys": {}},
    ):
        _write_raw(json.dumps(payload))
        with pytest.raises(ValueError, match="private, bounded"):
            resolve_model_credential(_NAME, _ENDPOINT)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX hygiene checks")
def test_oversized_file_rejected(posix: None) -> None:
    _write_raw("x" * (1_048_576 + 1))
    with pytest.raises(ValueError, match="private, bounded"):
        resolve_model_credential(_NAME, _ENDPOINT)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX hygiene checks")
def test_failure_never_echoes_body(posix: None) -> None:
    # 恶意/损坏载荷含疑似明文;异常消息必须只有固定文案。
    _write_raw('{"version": 1, "keys": {"x": "sk-super-secret"')
    with pytest.raises(ValueError) as excinfo:
        resolve_model_credential(_NAME, _ENDPOINT)
    message = str(excinfo.value)
    assert message == _FILE_ERROR
    assert "sk-super-secret" not in message
    assert excinfo.value.__cause__ is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX hygiene checks")
def test_insecure_directory_rejected(posix: None) -> None:
    directory = host_config_dir() / "credentials"
    directory.mkdir(mode=0o700)
    os.chmod(directory, 0o755)
    _credential_file().write_text(json.dumps({"version": 1, "keys": {}}), encoding="utf-8")
    os.chmod(_credential_file(), 0o600)
    with pytest.raises(ValueError, match="dir"):
        resolve_model_credential(_NAME, _ENDPOINT)


def test_validate_key_rejects_empty_and_control_characters(posix: None) -> None:
    for bad in ("", "   ", "sk\nsecret", "sk\x00secret"):
        with pytest.raises(ValueError, match="nonempty and contain no control"):
            save_model_credential(_NAME, _ENDPOINT, bad)


# ---------------------------------------------------------------------------
# darwin:值只走 stdin,argv 永不含密钥;回读校验
# ---------------------------------------------------------------------------


def test_keychain_save_uses_stdin_not_argv(monkeypatch: pytest.MonkeyPatch, darwin: None) -> None:
    fake = _inject(monkeypatch, [(0, "", ""), (0, "sk-secret\n", "")])
    save_model_credential(_NAME, _ENDPOINT, "sk-secret")

    save_argv, save_kwargs = fake.calls[0]
    assert save_argv == ["/usr/bin/security", "-i"]
    assert "sk-secret" not in " ".join(save_argv)
    command = save_kwargs["input"]
    assert isinstance(command, str)
    assert "add-generic-password" in command
    assert "sk-secret" in command
    assert _identity() in command

    # 回读校验走 find-generic-password,同样不含密钥。
    read_argv, _ = fake.calls[1]
    assert read_argv[:3] == ["/usr/bin/security", "find-generic-password", "-a"]
    assert "sk-secret" not in " ".join(read_argv)


def test_keychain_missing_returns_none(monkeypatch: pytest.MonkeyPatch, darwin: None) -> None:
    _inject(monkeypatch, [(44, "", "")])
    assert resolve_model_credential(_NAME, _ENDPOINT) is None


def test_keychain_read_failure_is_fixed_text(monkeypatch: pytest.MonkeyPatch, darwin: None) -> None:
    _inject(monkeypatch, [(1, "secret-body", "boom sk-secret")])
    with pytest.raises(ValueError) as excinfo:
        resolve_model_credential(_NAME, _ENDPOINT)
    assert str(excinfo.value) == _KEYCHAIN_READ_ERROR
    assert "sk-secret" not in str(excinfo.value)


def test_keychain_save_verification_failure(monkeypatch: pytest.MonkeyPatch, darwin: None) -> None:
    # 保存成功但回读到不同值 → 明确拒绝,不静默。
    _inject(monkeypatch, [(0, "", ""), (0, "different\n", "")])
    with pytest.raises(ValueError, match="verification failed"):
        save_model_credential(_NAME, _ENDPOINT, "sk-secret")


def test_keychain_delete_tolerates_missing(monkeypatch: pytest.MonkeyPatch, darwin: None) -> None:
    fake = _inject(monkeypatch, [(0, "", ""), (44, "", "")])
    delete_model_credential(_NAME, _ENDPOINT)
    delete_model_credential(_NAME, _ENDPOINT)
    assert fake.calls[0][0][:2] == ["/usr/bin/security", "delete-generic-password"]
    assert "sk-secret" not in " ".join(fake.calls[0][0])


# ---------------------------------------------------------------------------
# win32:明确拒绝
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="only exercise the rejection branch off Windows")
def test_win32_rejects_both_read_and_write(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(credentials, "_PLATFORM", "win32")
    with pytest.raises(ValueError, match="macOS or POSIX"):
        resolve_model_credential(_NAME, _ENDPOINT)
    with pytest.raises(ValueError, match="macOS or POSIX"):
        save_model_credential(_NAME, _ENDPOINT, "sk-secret")
    with pytest.raises(ValueError, match="macOS or POSIX"):
        delete_model_credential(_NAME, _ENDPOINT)


def test_reads_missing_file_as_none(posix: None) -> None:
    assert resolve_model_credential(_NAME, _ENDPOINT) is None


def test_no_credential_returns_none_without_touching_filesystem(
    posix: None,
) -> None:
    # 目录存在但无该 identity 条目 → None(不是错误)。
    _write_keys({_identity("OTHER_KEY"): "other"})
    assert resolve_model_credential(_NAME, _ENDPOINT) is None