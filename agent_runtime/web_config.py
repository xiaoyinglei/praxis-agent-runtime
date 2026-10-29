"""Host-owned web backend configuration shared by the CLI and Python SDK."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agent_runtime.model_config_io import (
    atomic_install_bytes,
    atomic_replace_bytes,
    ensure_durable_directory,
    exclusive_config_lock,
    file_fingerprint,
)
from agent_runtime.user_config import user_config_directory

PROVIDERS = frozenset({"deepseek", "brave", "tavily"})


@dataclass(frozen=True, slots=True)
class WebConfiguration:
    provider: str | None = None
    key_file: Path | None = None
    source: str = "default"

    @property
    def acquisition(self) -> str:
        return "provider" if self.provider == "deepseek" else "direct"

    def public_status(self) -> dict[str, str | None]:
        return {"provider": self.provider, "key_file": str(self.key_file) if self.key_file else None,
                "source": self.source, "acquisition": self.acquisition}


def web_configuration_path() -> Path:
    return user_config_directory() / "web.json"


def _configuration(provider: object, key_file: object, *, source: str) -> WebConfiguration:
    if provider is None and key_file is None:
        return WebConfiguration(source=source)
    if provider is not None and (not isinstance(provider, str) or provider not in PROVIDERS):
        raise ValueError("Web search provider must be deepseek, brave or tavily.")
    if not isinstance(key_file, (str, Path)) or not str(key_file).strip():
        raise ValueError("Web search requires a key file for deepseek/brave/tavily.")
    # Preserve the existing explicit-key-only Brave contract. A selected source
    # supplies the whole pair; never borrow another provider's credential.
    return WebConfiguration(provider or "brave", Path(key_file).expanduser().absolute(), source)


def resolve_web_configuration(
    provider: str | None = None, key_file: Path | str | None = None,
    *, environ: Mapping[str, str] | None = None,
) -> WebConfiguration:
    if provider is not None or key_file is not None:
        return _configuration(provider, key_file, source="arguments")
    environment = os.environ if environ is None else environ
    env_provider = environment.get("PRAXIS_WEB_SEARCH_PROVIDER") or None
    env_key = environment.get("PRAXIS_WEB_SEARCH_KEY_FILE") or None
    if env_provider is not None or env_key is not None:
        return _configuration(env_provider, env_key, source="environment")
    path = web_configuration_path()
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return WebConfiguration()
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o022 or info.st_uid not in {0, os.getuid()}:
            raise ValueError("web configuration must be a user-owned, non-writable-by-others regular file")
        raw = stream.read(16_385)
    if len(raw) > 16_384:
        raise ValueError("web configuration exceeds its size limit")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError):
        raise ValueError("web configuration is not valid JSON") from None
    if (not isinstance(data, dict) or set(data) != {"version", "provider", "key_file"}
            or type(data["version"]) is not int or data["version"] != 1):
        raise ValueError("unsupported web configuration; run agent web configure")
    return _configuration(data["provider"], data["key_file"], source="configuration")


def _validate_key(value: str) -> str:
    key = value.strip()
    if not 16 <= len(key) <= 4096 or not key.isascii() or not key.isprintable() or any(c.isspace() for c in key):
        raise ValueError("invalid search credential")
    return key


def load_search_key(path: Path | None, *, workspace: Path) -> str | None:
    """Read only an explicit protected credential outside the task workspace."""
    if path is None:
        return None
    path = path.expanduser().absolute()
    if path.resolve().is_relative_to(workspace.resolve()):
        raise ValueError("search credential must be outside the workspace")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid not in {0, os.getuid()}:
            raise ValueError("search credential must be a protected regular file owned by the user or root")
        raw = stream.read(4097)
    try:
        return _validate_key(raw.decode("ascii"))
    except UnicodeError:
        raise ValueError("invalid search credential") from None


def install_search_key(provider: str, value: str, *, workspace: Path) -> Path:
    configuration = _configuration(provider, user_config_directory() / f"{provider}.key", source="configuration")
    assert configuration.key_file is not None
    path = configuration.key_file
    if path.resolve().is_relative_to(workspace.resolve()):
        raise ValueError("search credential must be outside the workspace")
    payload = (_validate_key(value) + "\n").encode("ascii")
    ensure_durable_directory(path.parent)
    if atomic_install_bytes(path, payload) != "created":
        raise ValueError("credential file already exists; use --key-file to select it without overwriting")
    return path


def save_web_configuration(provider: str | None, key_file: Path | None, *, workspace: Path) -> WebConfiguration:
    configuration = _configuration(provider, key_file, source="configuration")
    load_search_key(configuration.key_file, workspace=workspace)
    path = web_configuration_path()
    if path.resolve().is_relative_to(workspace.resolve()):
        raise ValueError("web configuration must be outside the workspace")
    payload = (json.dumps({"version": 1, "provider": configuration.provider,
                           "key_file": str(configuration.key_file) if configuration.key_file else None},
                          indent=2) + "\n").encode()
    ensure_durable_directory(path.parent)
    with exclusive_config_lock(path):
        atomic_replace_bytes(path, payload, intended_fingerprint=file_fingerprint(payload))
    return configuration
