"""Host-owned model credentials, bound to a named key and an exact endpoint."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path

from agent_runtime.core.llm_config import validate_api_key_env_name, validate_http_url
from agent_runtime.model_config_io import atomic_replace_bytes, exclusive_config_lock, file_fingerprint
from agent_runtime.user_config import user_config_directory

_PLATFORM = sys.platform


def credential_directory() -> Path:
    return user_config_directory() / "credentials"


def _identity(name: str, endpoint: str) -> str:
    validate_api_key_env_name(name)
    validate_http_url(endpoint, field_name="credential endpoint")
    # Isolated personal settings must not read the real user's Keychain entries.
    value = f"{user_config_directory()}\n{name}\n{endpoint.rstrip('/')}"
    return hashlib.sha256(value.encode()).hexdigest()


def _validate_key(key: str) -> str:
    if not isinstance(key, str) or not key.strip() or any(ord(char) < 32 for char in key):
        raise ValueError("API Key must be nonempty and contain no control characters")
    return key.strip()


def _private_directory() -> Path:
    directory = credential_directory()
    if directory.is_symlink():
        raise ValueError("Credential directory must be a private, real directory")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = directory.stat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise ValueError("Credential directory must be private and owned by the current user")
    return directory


def _read_file(path: Path) -> dict[str, str]:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return {}
    except OSError:
        raise ValueError("Credential file must be private and regular") from None
    with os.fdopen(fd, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
            or metadata.st_nlink != 1
            or metadata.st_size > 1_048_576
        ):
            raise ValueError("Credential file must be private, bounded, and owned by the current user")
        try:
            data = json.loads(handle.read())
            if not isinstance(data, dict) or set(data) != {"version", "keys"} or data["version"] != 1:
                raise ValueError
            keys = data["keys"]
            if not isinstance(keys, dict) or any(not isinstance(key, str) for key in keys):
                raise ValueError
            return {key: _validate_key(value) for key, value in keys.items()}
        except (ValueError, TypeError, UnicodeError):
            raise ValueError("Credential file is invalid; run agent setup again") from None


def load_model_credential(name: str, endpoint: str) -> str | None:
    identity = _identity(name, endpoint)
    if _PLATFORM == "darwin":
        result = subprocess.run(
            ["/usr/bin/security", "find-generic-password", "-a", "praxis", "-s", "praxis.model." + identity, "-w"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode == 44:
            return None
        if result.returncode != 0:
            raise ValueError("Unable to read model credential from Keychain")
        return _validate_key(result.stdout.strip())
    if _PLATFORM == "win32":
        raise ValueError("Saved model credentials require macOS or POSIX; provide the named environment variable")
    directory = credential_directory()
    if not directory.exists() and not directory.is_symlink():
        return None
    return _read_file(_private_directory() / "models.json").get(identity)


def resolve_model_credential(name: str | None, endpoint: str | None) -> str | None:
    if name is None:
        return None
    value = os.environ.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None if endpoint is None else load_model_credential(name, endpoint)


def save_model_credential(name: str, endpoint: str, key: str) -> None:
    identity = _identity(name, endpoint)
    key = _validate_key(key)
    if _PLATFORM == "darwin":
        # Interactive stdin keeps the key out of the process argument list.
        command = "add-generic-password -U -a praxis -s " + shlex.quote("praxis.model." + identity)
        command += " -w " + shlex.quote(key) + "\n"
        result = subprocess.run(
            ["/usr/bin/security", "-i"], input=command, capture_output=True, text=True, timeout=30, check=False
        )
        if result.returncode != 0 or "SecKeychain" in result.stderr:
            raise ValueError("Unable to save model credential in Keychain")
        if load_model_credential(name, endpoint) != key:
            raise ValueError("Model credential storage verification failed")
        return
    if _PLATFORM == "win32":
        raise ValueError("Saved model credentials require macOS or POSIX; provide the named environment variable")
    path = _private_directory() / "models.json"
    with exclusive_config_lock(path):
        keys = _read_file(path)
        keys[identity] = key
        content = json.dumps({"version": 1, "keys": keys}).encode()
        atomic_replace_bytes(path, content, intended_fingerprint=file_fingerprint(content))
