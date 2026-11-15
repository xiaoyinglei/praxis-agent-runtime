"""模型凭据的唯一落点(设计 §5.15;D6):env 优先、macOS Keychain、POSIX 0600 文件。

凭据值**绝不**进入任何 SQLite 库、YAML、item payload 或 argv:全局配置库只存
``api_key_env``(变量名),本模块按 ``sha256(config_dir\\nname\\nendpoint)`` 的
identity 绑定读写。macOS 走 ``security -i`` 的 stdin(值不进进程参数表),POSIX
走 ``host_config_dir()/credentials/models.json``(0700 目录 + 0600 文件,排他锁下
read-merge-atomic-write)。

失败一律固定文案且 ``raise ... from None``:损坏文件/畸形 JSON/越权权限都不得把
body 或原异常文本带进错误消息。
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from praxis.protocol.floor import host_config_dir

try:
    import fcntl
except ImportError:  # pragma: no cover - win32 在入口就拒绝,不会走到文件分支
    fcntl = None  # type: ignore[assignment]

_PLATFORM = sys.platform

_KEYCHAIN_ACCOUNT = "praxis"
_SERVICE_PREFIX = "praxis.model."
# Task 13:web 搜索凭据与模型凭据共用同一套机制,仅 Keychain service 前缀不同
# (``praxis.web.<identity>``),identity 仍是 sha256(config_dir\\napi_key_env\\nendpoint)。
_WEB_SERVICE_PREFIX = "praxis.web."
_MAX_FILE_BYTES = 1_048_576

_FILE_ERROR = "Credential file must be private, bounded, and owned by the current user"
_DIRECTORY_ERROR = "Credential directory must be private and owned by the current user"
_KEYCHAIN_READ_ERROR = "Unable to read model credential from Keychain"
_KEYCHAIN_WRITE_ERROR = "Unable to save model credential in Keychain"
_KEYCHAIN_DELETE_ERROR = "Unable to remove model credential from Keychain"
_KEYCHAIN_VERIFY_ERROR = "Model credential storage verification failed"
_WEB_KEYCHAIN_READ_ERROR = "Unable to read web search credential from Keychain"
_WEB_KEYCHAIN_WRITE_ERROR = "Unable to save web search credential in Keychain"
_WEB_KEYCHAIN_DELETE_ERROR = "Unable to remove web search credential from Keychain"
_WEB_KEYCHAIN_VERIFY_ERROR = "Web search credential storage verification failed"
_UNSUPPORTED_PLATFORM = (
    "Saved model credentials require macOS or POSIX; provide the named environment variable"
)


def credential_identity(name: str, endpoint: str) -> str:
    """凭据 identity:目录 + 环境变量名 + 规范化 endpoint 的 sha256。

    endpoint 末尾斜杠被折叠,因此同一端点的两种写法命中同一条凭据。
    """
    value = f"{host_config_dir()}\n{name}\n{endpoint.rstrip('/')}"
    return hashlib.sha256(value.encode()).hexdigest()


def _service(identity: str, prefix: str = _SERVICE_PREFIX) -> str:
    return prefix + identity


def _validate_key(key: object) -> str:
    if (
        not isinstance(key, str)
        or not key.strip()
        or any(ord(char) < 32 for char in key)
    ):
        raise ValueError("API Key must be nonempty and contain no control characters")
    return key.strip()


def _credential_directory() -> Path:
    return host_config_dir() / "credentials"


def _private_directory() -> Path:
    directory = _credential_directory()
    try:
        if directory.is_symlink():
            raise ValueError(_DIRECTORY_ERROR)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        metadata = directory.stat()
    except ValueError:
        raise
    except OSError:
        raise ValueError(_DIRECTORY_ERROR) from None
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise ValueError(_DIRECTORY_ERROR) from None
    return directory


def _read_keys(path: Path) -> dict[str, str]:
    """读取并校验凭据文件;任何失败都收敛到固定文案(不回显 body/异常)。"""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return {}
    except OSError:
        raise ValueError(_FILE_ERROR) from None
    try:
        handle = os.fdopen(fd, "rb")
    except OSError:
        os.close(fd)
        raise ValueError(_FILE_ERROR) from None
    with handle:
        try:
            metadata = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_mode & 0o077
                or metadata.st_nlink != 1
                or metadata.st_size > _MAX_FILE_BYTES
            ):
                raise ValueError
            data = json.loads(handle.read())
            if (
                not isinstance(data, dict)
                or set(data) != {"version", "keys"}
                or data["version"] != 1
            ):
                raise ValueError
            keys = data["keys"]
            if not isinstance(keys, dict):
                raise ValueError
            parsed: dict[str, str] = {}
            for name, value in keys.items():
                if not isinstance(name, str):
                    raise ValueError
                parsed[name] = _validate_key(value)
            return parsed
        except (OSError, TypeError, UnicodeError, ValueError):
            raise ValueError(_FILE_ERROR) from None


@contextmanager
def _exclusive(directory: Path) -> Iterator[None]:
    """排他锁:进程内/跨进程的 read-merge-atomic-write 必须串行。"""
    if fcntl is None:  # pragma: no cover - 平台入口已拒绝
        raise ValueError(_UNSUPPORTED_PLATFORM) from None
    lock_path = directory / "models.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _atomic_write_keys(path: Path, keys: dict[str, str]) -> None:
    directory = path.parent
    fd, temporary = tempfile.mkstemp(prefix=".models.json.", dir=str(directory))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "keys": keys}, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _load_stored(
    api_key_env: str, endpoint: str, *, service_prefix: str, read_error: str
) -> str | None:
    """通用读取(模型/搜索共用):Keychain(darwin)或 0600 文件;未保存返回 None。"""
    identity = credential_identity(api_key_env, endpoint)
    if _PLATFORM == "darwin":
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-a",
                _KEYCHAIN_ACCOUNT,
                "-s",
                _service(identity, service_prefix),
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode == 44:
            return None
        if result.returncode != 0:
            raise ValueError(read_error) from None
        return _validate_key(result.stdout.strip())
    if _PLATFORM == "win32":
        raise ValueError(_UNSUPPORTED_PLATFORM) from None
    directory = _credential_directory()
    if not directory.exists() and not directory.is_symlink():
        return None
    return _read_keys(_private_directory() / "models.json").get(identity)


def _save_stored(
    api_key_env: str,
    endpoint: str,
    value: str,
    *,
    service_prefix: str,
    write_error: str,
    verify_error: str,
    read_error: str,
) -> None:
    """通用保存 + 回读校验;值绝不进入 argv(Keychain 走 stdin,文件走 0600 原子写)。"""
    identity = credential_identity(api_key_env, endpoint)
    key = _validate_key(value)
    if _PLATFORM == "darwin":
        # security 的交互输入把密钥留在 stdin,不进进程参数表。
        command = (
            "add-generic-password -U -a "
            + shlex.quote(_KEYCHAIN_ACCOUNT)
            + " -s "
            + shlex.quote(_service(identity, service_prefix))
            + " -w "
            + shlex.quote(key)
            + "\n"
        )
        result = subprocess.run(
            ["/usr/bin/security", "-i"],
            input=command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0 or "SecKeychain" in result.stderr:
            raise ValueError(write_error) from None
        if (
            _load_stored(api_key_env, endpoint, service_prefix=service_prefix, read_error=read_error)
            != key
        ):
            raise ValueError(verify_error) from None
        return
    if _PLATFORM == "win32":
        raise ValueError(_UNSUPPORTED_PLATFORM) from None
    path = _private_directory() / "models.json"
    with _exclusive(path.parent):
        keys = _read_keys(path)
        keys[identity] = key
        _atomic_write_keys(path, keys)


def _delete_stored(
    api_key_env: str, endpoint: str, *, service_prefix: str, delete_error: str
) -> None:
    """通用删除;不存在是 no-op。"""
    identity = credential_identity(api_key_env, endpoint)
    if _PLATFORM == "darwin":
        result = subprocess.run(
            [
                "/usr/bin/security",
                "delete-generic-password",
                "-a",
                _KEYCHAIN_ACCOUNT,
                "-s",
                _service(identity, service_prefix),
            ],
            capture_output=True,
            timeout=15,
            check=False,
        )
        if result.returncode not in {0, 44}:
            raise ValueError(delete_error) from None
        return
    if _PLATFORM == "win32":
        raise ValueError(_UNSUPPORTED_PLATFORM) from None
    directory = _credential_directory()
    if not directory.exists() and not directory.is_symlink():
        return
    private = _private_directory()
    path = private / "models.json"
    with _exclusive(private):
        keys = _read_keys(path)
        if keys.pop(identity, None) is None:
            return
        _atomic_write_keys(path, keys)


def load_model_credential(api_key_env: str, endpoint: str) -> str | None:
    """读取已保存的凭据值;未保存返回 None。"""
    return _load_stored(
        api_key_env, endpoint, service_prefix=_SERVICE_PREFIX, read_error=_KEYCHAIN_READ_ERROR
    )


def resolve_model_credential(api_key_env: str, endpoint: str) -> str | None:
    """解析运行期凭据:非空环境变量优先,其次已保存的 Keychain/0600 文件。"""
    value = os.environ.get(api_key_env)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return load_model_credential(api_key_env, endpoint)


CredentialSource = Literal["env", "store", "missing"]


def credential_source(api_key_env: str, endpoint: str) -> CredentialSource:
    """凭据来源归因(只判存在性,不返回值):env / store / missing。

    探针用它把"provider 拒绝凭据"归因到正确来源,避免把已保存的
    Keychain/0600 凭据误报成"环境变量没设"(R-T6-8)。
    """
    value = os.environ.get(api_key_env)
    if isinstance(value, str) and value.strip():
        return "env"
    return "store" if load_model_credential(api_key_env, endpoint) is not None else "missing"


def save_model_credential(api_key_env: str, endpoint: str, value: str) -> None:
    """保存凭据值并回读校验;值绝不进入 argv。"""
    _save_stored(
        api_key_env,
        endpoint,
        value,
        service_prefix=_SERVICE_PREFIX,
        write_error=_KEYCHAIN_WRITE_ERROR,
        verify_error=_KEYCHAIN_VERIFY_ERROR,
        read_error=_KEYCHAIN_READ_ERROR,
    )


def delete_model_credential(api_key_env: str, endpoint: str) -> None:
    """删除已保存的凭据;不存在是 no-op。"""
    _delete_stored(
        api_key_env,
        endpoint,
        service_prefix=_SERVICE_PREFIX,
        delete_error=_KEYCHAIN_DELETE_ERROR,
    )


# ---------------------------------------------------------------------------
# Web 搜索凭据(Task 13;留档③):同一套 env→Keychain/0600 机制,service 前缀
# ``praxis.web.``。endpoint 是 provider API base URL(search.SEARCH_ENDPOINTS),
# 使同一 env 名在不同 provider 下互不串用。
# ---------------------------------------------------------------------------


def load_web_search_credential(api_key_env: str, endpoint: str) -> str | None:
    """读取已保存的搜索凭据;未保存返回 None。"""
    return _load_stored(
        api_key_env, endpoint, service_prefix=_WEB_SERVICE_PREFIX, read_error=_WEB_KEYCHAIN_READ_ERROR
    )


def resolve_web_search_credential(api_key_env: str, endpoint: str) -> str | None:
    """解析运行期搜索凭据:非空环境变量优先,其次已保存的 Keychain/0600 文件。"""
    value = os.environ.get(api_key_env)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return load_web_search_credential(api_key_env, endpoint)


def web_search_credential_source(api_key_env: str, endpoint: str) -> CredentialSource:
    """搜索凭据来源归因(只判存在性,不返回值):env / store / missing。"""
    value = os.environ.get(api_key_env)
    if isinstance(value, str) and value.strip():
        return "env"
    return (
        "store"
        if load_web_search_credential(api_key_env, endpoint) is not None
        else "missing"
    )


def save_web_search_credential(api_key_env: str, endpoint: str, value: str) -> None:
    """保存搜索凭据并回读校验;值绝不进入 argv。"""
    _save_stored(
        api_key_env,
        endpoint,
        value,
        service_prefix=_WEB_SERVICE_PREFIX,
        write_error=_WEB_KEYCHAIN_WRITE_ERROR,
        verify_error=_WEB_KEYCHAIN_VERIFY_ERROR,
        read_error=_WEB_KEYCHAIN_READ_ERROR,
    )


def delete_web_search_credential(api_key_env: str, endpoint: str) -> None:
    """删除已保存的搜索凭据;不存在是 no-op。"""
    _delete_stored(
        api_key_env,
        endpoint,
        service_prefix=_WEB_SERVICE_PREFIX,
        delete_error=_WEB_KEYCHAIN_DELETE_ERROR,
    )


__all__ = [
    "CredentialSource",
    "credential_identity",
    "credential_source",
    "delete_model_credential",
    "delete_web_search_credential",
    "load_model_credential",
    "load_web_search_credential",
    "resolve_model_credential",
    "resolve_web_search_credential",
    "save_model_credential",
    "save_web_search_credential",
    "web_search_credential_source",
]
