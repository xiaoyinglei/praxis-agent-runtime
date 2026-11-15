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

import httpx

from praxis.adapters.mcp.names import normalize_mcp_name
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
# Task 18:MCP bearer 以 **server 名**寻址(旧 mcp_admin 的 ``praxis.mcp.<norm>``
# 服务名),载荷自带 endpoint 绑定;不同 server 同名环境变量互不串用。
_MCP_SERVICE_PREFIX = "praxis.mcp."
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
_MCP_KEYCHAIN_READ_ERROR = "Unable to read MCP credential from Keychain"
_MCP_KEYCHAIN_WRITE_ERROR = "Unable to save MCP credential in Keychain"
_MCP_KEYCHAIN_DELETE_ERROR = "Unable to remove MCP credential from Keychain"
_MCP_KEYCHAIN_VERIFY_ERROR = "MCP credential storage verification failed"
_MCP_UNSUPPORTED_PLATFORM = (
    "Saved MCP credentials require macOS or POSIX; provide the bearer token environment variable"
)
_MCP_NAME_ERROR = "MCP credential name must normalize to a non-empty server name"
_MCP_ENDPOINT_ERROR = "MCP credential endpoint must be a valid URL"
_MCP_VALUE_ERROR = "MCP bearer token must be a nonempty single-line value"
_MCP_BINDING_ERROR = "MCP credential is bound to a different endpoint"
_MCP_INVALID_ERROR = "Saved MCP credential is invalid; save it again"


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


def _load_service(
    *,
    service: str,
    file_key: str,
    read_error: str,
    unsupported_error: str = _UNSUPPORTED_PLATFORM,
) -> str | None:
    """按**显式 service/file_key** 读取(Keychain 或 0600 文件);未保存返回 None。

    模型/搜索凭据的 file_key 是 identity,service 是前缀 + identity;MCP bearer
    两者都是 ``praxis.mcp.<normalized-name>``(载荷自身携带 endpoint 绑定)。
    这样两族凭据共用同一套权限/原子写/锁机制,而磁盘键格式互不迁移。
    """
    if _PLATFORM == "darwin":
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-a",
                _KEYCHAIN_ACCOUNT,
                "-s",
                service,
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
        raise ValueError(unsupported_error) from None
    directory = _credential_directory()
    if not directory.exists() and not directory.is_symlink():
        return None
    return _read_keys(_private_directory() / "models.json").get(file_key)


def _save_service(
    *,
    service: str,
    file_key: str,
    value: str,
    write_error: str,
    verify_error: str,
    read_error: str,
    unsupported_error: str = _UNSUPPORTED_PLATFORM,
) -> None:
    """按显式 service/file_key 保存 + 回读校验;值绝不进入 argv(Keychain 走 stdin)。"""
    key = _validate_key(value)
    if _PLATFORM == "darwin":
        # security 的交互输入把密钥留在 stdin,不进进程参数表。
        command = (
            "add-generic-password -U -a "
            + shlex.quote(_KEYCHAIN_ACCOUNT)
            + " -s "
            + shlex.quote(service)
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
            _load_service(
                service=service,
                file_key=file_key,
                read_error=read_error,
                unsupported_error=unsupported_error,
            )
            != key
        ):
            raise ValueError(verify_error) from None
        return
    if _PLATFORM == "win32":
        raise ValueError(unsupported_error) from None
    path = _private_directory() / "models.json"
    with _exclusive(path.parent):
        keys = _read_keys(path)
        keys[file_key] = key
        _atomic_write_keys(path, keys)


def _delete_service(
    *,
    service: str,
    file_key: str,
    delete_error: str,
    unsupported_error: str = _UNSUPPORTED_PLATFORM,
) -> None:
    """按显式 service/file_key 删除;不存在是 no-op。"""
    if _PLATFORM == "darwin":
        result = subprocess.run(
            [
                "/usr/bin/security",
                "delete-generic-password",
                "-a",
                _KEYCHAIN_ACCOUNT,
                "-s",
                service,
            ],
            capture_output=True,
            timeout=15,
            check=False,
        )
        if result.returncode not in {0, 44}:
            raise ValueError(delete_error) from None
        return
    if _PLATFORM == "win32":
        raise ValueError(unsupported_error) from None
    directory = _credential_directory()
    if not directory.exists() and not directory.is_symlink():
        return
    private = _private_directory()
    path = private / "models.json"
    with _exclusive(private):
        keys = _read_keys(path)
        if keys.pop(file_key, None) is None:
            return
        _atomic_write_keys(path, keys)


def _load_stored(
    api_key_env: str, endpoint: str, *, service_prefix: str, read_error: str
) -> str | None:
    """通用读取(模型/搜索共用):Keychain(darwin)或 0600 文件;未保存返回 None。"""
    identity = credential_identity(api_key_env, endpoint)
    return _load_service(
        service=_service(identity, service_prefix), file_key=identity, read_error=read_error
    )


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
    _save_service(
        service=_service(identity, service_prefix),
        file_key=identity,
        value=value,
        write_error=write_error,
        verify_error=verify_error,
        read_error=read_error,
    )


def _delete_stored(
    api_key_env: str, endpoint: str, *, service_prefix: str, delete_error: str
) -> None:
    """通用删除;不存在是 no-op。"""
    identity = credential_identity(api_key_env, endpoint)
    _delete_service(
        service=_service(identity, service_prefix), file_key=identity, delete_error=delete_error
    )


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


# ---------------------------------------------------------------------------
# MCP bearer(Task 18;Q6):只承载手填 bearer(``bearer_token_env_var`` /
# ``keychain: true`` 两条非 OAuth 源)。服务名按 **server 名**寻址
# (``praxis.mcp.<normalize_mcp_name(name)>``),载荷 ``{"url": endpoint,
# "bearer": value}``——回读时 stored url 必须与配置 endpoint **逐字相等**
# (双方都先经 ``str(httpx.URL(...))``,与 T15 配置存储同源),否则有界拒绝。
# OAuth 凭据留阶段 5;本模块不读也绝不写 ``praxis.mcp.oauth.*``。
# ---------------------------------------------------------------------------


def mcp_credential_service(name: str) -> str:
    """MCP 凭据的 service/文件键(``praxis.mcp.<normalized-name>``)。"""
    normalized = normalize_mcp_name(name)
    if not normalized:
        raise ValueError(_MCP_NAME_ERROR)
    return _MCP_SERVICE_PREFIX + normalized


def _mcp_endpoint(endpoint_url: str) -> str:
    """与 T15 配置存储同一规范化形态(host 小写/IDNA、省略 443、百分号编码)。"""
    try:
        url = httpx.URL(endpoint_url)
        if not url.scheme or not url.host:
            raise ValueError
        return str(url)
    except (httpx.InvalidURL, ValueError):
        raise ValueError(_MCP_ENDPOINT_ERROR) from None


def load_mcp_bearer(name: str, endpoint_url: str) -> str | None:
    """读取 server 名的 bearer;未保存返回 None。

    URL 绑定:stored url 与 ``endpoint_url`` 逐字不相等 ⇒
    ``ValueError``("bound to a different endpoint")。错误文案只点名绑定关系,
    绝不回显 token 或 stored 载荷。
    """
    service = mcp_credential_service(name)
    stored = _load_service(
        service=service,
        file_key=service,
        read_error=_MCP_KEYCHAIN_READ_ERROR,
        unsupported_error=_MCP_UNSUPPORTED_PLATFORM,
    )
    if stored is None:
        return None
    try:
        payload = json.loads(stored)
    except (TypeError, ValueError):
        raise ValueError(_MCP_INVALID_ERROR) from None
    if not isinstance(payload, dict):
        raise ValueError(_MCP_INVALID_ERROR) from None
    bound_url = payload.get("url")
    bearer = payload.get("bearer")
    if not isinstance(bound_url, str) or not isinstance(bearer, str):
        raise ValueError(_MCP_INVALID_ERROR) from None
    if bound_url != _mcp_endpoint(endpoint_url):
        raise ValueError(_MCP_BINDING_ERROR) from None
    try:
        return _validate_key(bearer)
    except ValueError:
        raise ValueError(_MCP_INVALID_ERROR) from None


def save_mcp_bearer(name: str, endpoint_url: str, value: str) -> None:
    """保存 server 名 bearer(载荷键 ``bearer``)+ endpoint 绑定并回读校验。

    值只经 stdin(Keychain)或 0600 原子写(POSIX),绝不进入 argv/错误文案。
    """
    service = mcp_credential_service(name)
    try:
        token = _validate_key(value)
    except ValueError:
        raise ValueError(_MCP_VALUE_ERROR) from None
    payload = json.dumps(
        {"url": _mcp_endpoint(endpoint_url), "bearer": token}, sort_keys=True
    )
    _save_service(
        service=service,
        file_key=service,
        value=payload,
        write_error=_MCP_KEYCHAIN_WRITE_ERROR,
        verify_error=_MCP_KEYCHAIN_VERIFY_ERROR,
        read_error=_MCP_KEYCHAIN_READ_ERROR,
        unsupported_error=_MCP_UNSUPPORTED_PLATFORM,
    )


def clear_mcp_bearer(name: str, endpoint_url: str) -> None:
    """删除 server 名的 bearer;不存在是 no-op。

    ``endpoint_url`` 只做与 save/load 相同的规范化校验;删除按 **server 名**
    作用(旧 ``_delete_keychain(name)`` 语义):endpoint 轮换不应让旧凭据变成
    无法清除的孤儿。
    """
    service = mcp_credential_service(name)
    _mcp_endpoint(endpoint_url)
    _delete_service(
        service=service,
        file_key=service,
        delete_error=_MCP_KEYCHAIN_DELETE_ERROR,
        unsupported_error=_MCP_UNSUPPORTED_PLATFORM,
    )


__all__ = [
    "CredentialSource",
    "clear_mcp_bearer",
    "credential_identity",
    "credential_source",
    "delete_model_credential",
    "delete_web_search_credential",
    "load_mcp_bearer",
    "load_model_credential",
    "load_web_search_credential",
    "mcp_credential_service",
    "resolve_model_credential",
    "resolve_web_search_credential",
    "save_mcp_bearer",
    "save_model_credential",
    "save_web_search_credential",
    "web_search_credential_source",
]
