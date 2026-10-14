"""Host-controlled MCP registration and endpoint-bound macOS credentials."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import httpx
import yaml

from agent_runtime.permission_policy import NetworkRule, PermissionPolicy, resolve_permission_sources
from agent_runtime.runtime.mcp import validate_http_mcp_url
from agent_runtime.user_config import user_config_directory

GITHUB_READONLY_URL = "https://api.githubcopilot.com/mcp/readonly"


def user_mcp_config() -> Path:
    return user_config_directory() / "mcp_servers.yaml"


def _credential_service(name: str) -> str:
    from agent_runtime.tools.integrations.mcp import normalize_mcp_name
    normalized = normalize_mcp_name(name)
    if not normalized or normalized != name:
        raise ValueError("MCP credential name must be a canonical server name")
    return "praxis.mcp." + normalized


def _read_keychain(name: str) -> str:
    if sys.platform != "darwin":
        raise ValueError("Keychain credentials require macOS; use --bearer-token-env-var on other platforms")
    completed = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-a", "praxis", "-s", _credential_service(name), "-w"],
        capture_output=True, text=True, timeout=15, check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"MCP credential missing: run agent mcp login {name}")
    return completed.stdout.strip()


def load_mcp_credential(name: str, url: str) -> str:
    try:
        payload = json.loads(_read_keychain(name))
        if payload["url"] != url:
            raise ValueError("MCP credential is bound to a different endpoint; log in for this endpoint")
        token = payload["token"]
        if not isinstance(token, str) or not token or any(ord(char) < 32 for char in token):
            raise ValueError("MCP credential is invalid; log in again")
        return token
    except (KeyError, TypeError, json.JSONDecodeError):
        raise ValueError("MCP credential is invalid; log in again") from None



def _write_keychain(name: str, value: str) -> None:
    if sys.platform != "darwin":
        raise ValueError("Keychain credentials require macOS; use an environment variable")
    # security's interactive input keeps credentials out of the process argv.
    command = "add-generic-password -U -a praxis -s " + shlex.quote(_credential_service(name))
    command += " -w " + shlex.quote(value) + "\n"
    completed = subprocess.run(["/usr/bin/security", "-i"], input=command, capture_output=True,
                               text=True, timeout=30, check=False)
    if completed.returncode != 0 or "SecKeychain" in completed.stderr:
        raise ValueError("Unable to save MCP credential in macOS Keychain")
    if _read_keychain(name) != value:
        raise ValueError("MCP credential storage verification failed")


def _delete_keychain(name: str) -> None:
    completed = subprocess.run(
        ["/usr/bin/security", "delete-generic-password", "-a", "praxis", "-s", _credential_service(name)],
        capture_output=True, timeout=15, check=False,
    )
    if completed.returncode not in {0, 44}:
        raise ValueError("Unable to remove MCP credential from Keychain")


def save_mcp_credential(name: str, url: str, token: str) -> None:
    validate_http_mcp_url(url)
    if not token or any(ord(char) < 32 for char in token):
        raise ValueError("MCP credential must be a nonempty single-line token")
    _write_keychain(name, json.dumps({"url": url, "token": token}))


@contextmanager
def configuration_lock() -> Iterator[None]:
    try:
        import fcntl
    except ImportError:
        raise ValueError("MCP registration requires POSIX file locking on this platform") from None
    directory = user_config_directory()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = directory.stat()
    if metadata.st_uid != os.getuid() or metadata.st_mode & 0o022:
        raise ValueError("Personal settings directory must be user-owned and not group/world writable")
    with (directory / ".mcp-admin.lock").open("a") as handle:
        os.chmod(handle.name, 0o600)
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def authorize_github_cli_endpoint(url: str) -> None:
    if url not in {GITHUB_READONLY_URL, "https://api.githubcopilot.com/mcp/"}:
        raise ValueError("GitHub CLI credentials can only be used with the official GitHub MCP endpoint")

def github_cli_token() -> str:
    executable = next((Path(path) for path in ("/opt/homebrew/bin/gh", "/usr/local/bin/gh", "/usr/bin/gh")
                       if Path(path).is_file()), None)
    if executable is None:
        raise ValueError("GitHub CLI is not installed; provide a token with agent mcp login github")
    completed = subprocess.run([str(executable), "auth", "token", "--hostname", "github.com"],
                               capture_output=True, text=True, timeout=30, check=False)
    if completed.returncode != 0 or not completed.stdout.strip():
        raise ValueError("GitHub CLI is not logged in; run gh auth login --hostname github.com, then try again")
    return completed.stdout.strip()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".mcp-setup-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def mcp_connections() -> list[dict[str, Any]]:
    config = user_mcp_config()
    if not config.is_file():
        return []
    raw = yaml.safe_load(config.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("servers"), list):
        raise ValueError("Personal MCP configuration must contain a servers list")
    if any(not isinstance(item, dict) or not isinstance(item.get("name"), str) for item in raw["servers"]):
        raise ValueError("MCP server entries require names")
    return cast(list[dict[str, Any]], raw["servers"])



def _policy_toml(policy: PermissionPolicy) -> str:
    data = policy.model_dump(mode="json")
    lines = [f"{key}={json.dumps(data[key])}" for key in ("profile", "approval")]
    for group in ("filesystem", "tools"):
        for rule in data[group]:
            lines.append(f"\n[[{group}]]")
            lines.extend(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in rule.items())
    lines.append("\n[network]")
    lines.extend(f"{key}={json.dumps(value, ensure_ascii=False)}"
                 for key, value in data["network"].items() if key != "rules")
    for rule in data["network"]["rules"]:
        lines.append("\n[[network.rules]]")
        lines.extend(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in rule.items())
    return "\n".join(lines) + "\n"

def _register_remote_mcp(name: str, *, url: str | None = None, bearer_token_env_var: str | None = None,
                        tools: tuple[str, ...] = (), read_only: bool = True,
                        workspace: Path | None = None) -> Path:
    _credential_service(name)
    if name == "github" and url is None:
        url = GITHUB_READONLY_URL if read_only else "https://api.githubcopilot.com/mcp/"
        tools = tools or ("search_repositories", "get_file_contents", "issue_read", "pull_request_read")
    if url is None:
        raise ValueError("A remote MCP service URL is required")
    url = validate_http_mcp_url(url)
    if bearer_token_env_var is not None:
        import re
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", bearer_token_env_var):
            raise ValueError("bearer_token_env_var must name an environment variable")
    scope = Path.cwd() if workspace is None else workspace
    effective = resolve_permission_sources(scope)
    if effective.network_decision(url) == "deny":
        raise ValueError("Current permission policy denies this destination; registration cannot override it")
    config = user_mcp_config()
    entries = mcp_connections()
    if any(entry["name"] == name for entry in entries):
        raise ValueError(f"MCP service already registered: {name}; remove it before changing the connection")
    directory = config.parent
    toml = directory / "permissions.toml"
    permissions = toml if toml.exists() else directory / "permissions.json"
    before = permissions.read_text() if permissions.exists() else None
    data = (tomllib.loads(before) if permissions.suffix == ".toml" else json.loads(before)) if before else {}
    declared = PermissionPolicy.model_validate(data)
    rule = NetworkRule(domain=str(httpx.URL(url).host), decision="allow")
    candidate = declared.model_copy(update={"network": declared.network.model_copy(update={
        "rules": (*declared.network.rules, rule),
    })})
    if resolve_permission_sources(scope, candidate).network_decision(url) != "allow":
        raise ValueError("Network preauthorization is restricted by a permission ceiling")
    next_policy = (before or "") + "\n[[network.rules]]\ndomain=" + json.dumps(rule.domain) + '\ndecision="allow"\n'
    if permissions.suffix == ".toml":
        try:
            PermissionPolicy.model_validate(tomllib.loads(next_policy))
        except tomllib.TOMLDecodeError:
            # Inline TOML tables cannot be extended. Render the same validated
            # declaration as tables rather than writing an invalid configuration.
            next_policy = _policy_toml(candidate)
    if permissions.suffix == ".json":
        next_policy = json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    entry: dict[str, Any] = {"name": name, "transport": "http", "url": url, "enabled": True,
                             "read_only": read_only, "tools_allowlist": list(tools), "allow_all_tools": not tools}
    if name == "github" and read_only:
        entry["headers"] = {"X-MCP-Readonly": "true"}
    previous_credential: str | None = None
    use_gh = bearer_token_env_var is None and name == "github"
    if bearer_token_env_var:
        entry["bearer_token_env_var"] = bearer_token_env_var
    elif use_gh:
        authorize_github_cli_endpoint(url)
        try:
            previous_credential = _read_keychain(name)
        except ValueError:
            pass
        entry["keychain"] = True
    next_config = yaml.safe_dump({"servers": [*entries, entry]}, sort_keys=False, allow_unicode=True)
    credential_attempted = False
    try:
        if use_gh:
            token = github_cli_token()
            credential_attempted = True
            save_mcp_credential(name, url, token)
        _atomic_write(permissions, next_policy)
        _atomic_write(config, next_config)
    except BaseException:
        if before is None:
            permissions.unlink(missing_ok=True)
        else:
            _atomic_write(permissions, before)
        if credential_attempted:
            if previous_credential is None:
                _delete_keychain(name)
            else:
                _write_keychain(name, previous_credential)
        raise
    return config


def register_remote_mcp(name: str, *, url: str | None = None, bearer_token_env_var: str | None = None,
                        tools: tuple[str, ...] = (), read_only: bool = True,
                        workspace: Path | None = None) -> Path:
    with configuration_lock():
        return _register_remote_mcp(name, url=url, bearer_token_env_var=bearer_token_env_var,
                                    tools=tools, read_only=read_only, workspace=workspace)
