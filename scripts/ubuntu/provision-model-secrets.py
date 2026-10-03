"""One-time hidden key entry and explicit proxy rotation. Run as root on Ubuntu."""

import argparse
import getpass
import os
import pwd
import secrets
import tempfile
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path


def _secret(value: str) -> bool:
    return 16 <= len(value) <= 4096 and value.isascii() and value.isprintable() and not any(c.isspace() for c in value)


def _write(path: Path, value: str, uid: int, gid: int) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".credential-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(value + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o400)
            os.fchown(handle.fileno(), uid, gid)
        os.replace(temporary, path)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def hidden_prompt(label: str) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        return getpass.getpass(label)


def provision(
    directory: Path,
    token_path: Path,
    agent_uid: int,
    agent_gid: int,
    providers: Sequence[str],
    *,
    prompt: Callable[[str], str] = hidden_prompt,
) -> None:
    if os.path.lexists(directory) or os.path.lexists(token_path):
        raise FileExistsError("existing credential destination; initial setup refuses overwrite")
    if not providers or set(providers) - {"deepseek", "groq"}:
        raise ValueError("unsupported provider")
    # Validate every key before creating any destination. Never echo a value.
    keys = {provider: prompt(f"{provider} API key (hidden): ").strip() for provider in providers}
    if not all(_secret(value) for value in keys.values()):
        raise ValueError("invalid provider credential")
    directory.mkdir(mode=0o700)
    for provider, value in keys.items():
        _write(directory / f"{provider}-key", value, os.getuid(), os.getgid())
    token = secrets.token_urlsafe(32)
    _write(directory / "agent-token", token, os.getuid(), os.getgid())
    _write(token_path, token, agent_uid, agent_gid)


def rotate_token(directory: Path, token_path: Path, agent_uid: int, agent_gid: int) -> None:
    if directory.is_symlink() or not directory.is_dir() or not (directory / "agent-token").is_file():
        raise ValueError("credential directory not initialized")
    token = secrets.token_urlsafe(32)
    _write(directory / "agent-token", token, os.getuid(), os.getgid())
    _write(token_path, token, agent_uid, agent_gid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotate-token", action="store_true")
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("root is required to provision private service credentials")
    account = pwd.getpwnam("praxis-agent")
    directory = Path("/etc/praxis-model")
    token_path = Path("/var/lib/praxis-agent/proxy.token")
    try:
        if args.rotate_token:
            rotate_token(directory, token_path, account.pw_uid, account.pw_gid)
        else:
            provision(directory, token_path, account.pw_uid, account.pw_gid, ["deepseek", "groq"])
    except (OSError, ValueError, getpass.GetPassWarning):
        parser.exit(
            1, "Credential setup failed; check private destinations and enter valid keys. Values were not logged.\n"
        )
    print("Credentials installed. Restart praxis-model to load them; the daily budget is preserved.")


if __name__ == "__main__":
    main()
