#!/usr/bin/env python3
"""Direct CLI startup with one-time, private DeepSeek credential setup."""
import argparse
import getpass
import os
import stat
import sys
import tempfile
import warnings
from pathlib import Path


def hidden_prompt(label):
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        return getpass.getpass(label)


def validate_key(value):
    if (not 16 <= len(value) <= 4096 or not value.isascii() or not value.isprintable()
            or any(c.isspace() for c in value)):
        raise ValueError("Invalid or empty DeepSeek API Key.")
    return value


def read_key(path, *, validate=True):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        os.close(fd)
        raise ValueError("Credential must be a private regular file owned by this user.")
    with os.fdopen(fd, "rb") as handle:
        if not validate:
            return ""
        try:
            value = handle.read(4097).decode("ascii").strip()
        except UnicodeError:
            raise ValueError("Invalid credential encoding.") from None
    return validate_key(value)


def load_key(path, *, configure=False, prompt=None):
    path = Path(path).absolute()
    directory = path.parent
    if directory.is_symlink() or path.is_symlink():
        raise ValueError("Credential paths must not be symlinks.")
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = directory.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Credential directory must be owned by this user with mode 0700.")
    if path.exists():
        existing = read_key(path, validate=not configure)
        if not configure:
            return existing
    value = validate_key((prompt or hidden_prompt)("DeepSeek API Key (hidden; saved for later starts): "))
    fd, temporary = tempfile.mkstemp(prefix=".deepseek-", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(value + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if configure:
            os.replace(temporary, path)
        else:
            # First-time publication must not overwrite another simultaneous setup.
            os.link(temporary, path)
        print(f"Credential saved privately: {path}. Later starts will reuse it.", flush=True)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return value


def start(repo, key_path, *, configure=False):
    repo = Path(repo).resolve(strict=True)
    agent = repo / ".venv/bin/agent"
    if not agent.is_file() or not os.access(agent, os.X_OK):
        raise ValueError("Agent environment is missing; install the core dependencies first.")
    key = load_key(key_path, configure=configure)
    env = dict(os.environ)
    env["DEEPSEEK_API_KEY"] = key
    env["PRAXIS_DISABLE_DOTENV"] = "1"
    os.chdir(repo)
    os.execve(str(agent), [str(agent), "chat", "--model", "deepseek-flash"], env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--configure", action="store_true", help="Explicitly replace the saved Key via hidden input.")
    args = parser.parse_args()
    try:
        start(args.repo, Path.home() / ".config/praxis/credentials/deepseek.key", configure=args.configure)
    except (ValueError, OSError, EOFError, getpass.GetPassWarning) as error:
        print(f"Startup failed: {error if isinstance(error, ValueError) else type(error).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
