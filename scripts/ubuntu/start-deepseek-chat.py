#!/usr/bin/env python3
"""Direct CLI startup with one-time, private DeepSeek credential setup."""
import argparse
import getpass
import json
import os
import stat
import subprocess
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


def validate_workspace_readiness(repo, python):
    """Run the installed shell containment check before reading any credential."""
    code = (
        "import json,sys; from pathlib import Path; "
        "from agent_runtime.tools.builtins.shell import _workspace_sandbox_preflight_issue; "
        "print(json.dumps(_workspace_sandbox_preflight_issue(Path(sys.argv[1]), allow_workspace_write=False)))"
    )
    result = subprocess.run(
        [str(python), "-I", "-B", "-c", code, str(repo)], cwd=repo,
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise ValueError("Agent environment readiness check failed; reinstall the core dependencies.")
    try:
        issue = json.loads(result.stdout)
    except (TypeError, ValueError):
        raise ValueError("Agent environment readiness check returned invalid output.") from None
    if issue is not None:
        if not isinstance(issue, list) or len(issue) != 2 or not all(isinstance(value, str) for value in issue):
            raise ValueError("Agent environment readiness check returned invalid output.")
        raise ValueError(f"Agent workspace is not ready: {issue[0]}: {issue[1]}")


def start(repo, key_path, *, configure=False, venv=None, resume=None, continue_chat=False, recover_args=None):
    repo = Path(repo).resolve(strict=True)
    environment = Path(venv).resolve(strict=True) if venv is not None else repo / ".venv"
    agent = environment / "bin/agent"
    if not agent.is_file() or not os.access(agent, os.X_OK):
        raise ValueError("Agent environment is missing; install the core dependencies first.")
    validate_workspace_readiness(repo, environment / "bin/python")
    key = load_key(key_path, configure=configure)
    env = dict(os.environ)
    env["DEEPSEEK_API_KEY"] = key
    env["PRAXIS_DISABLE_DOTENV"] = "1"
    os.chdir(repo)
    if resume is not None:
        arguments = ["resume", *([resume] if resume else [])]
    elif continue_chat:
        arguments = ["chat", "--continue"]
    elif recover_args is not None:
        arguments = ["recover", *recover_args]
    else:
        arguments = ["chat", "--model", "deepseek-flash"]
    os.execve(str(agent), [str(agent), *arguments], env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--venv", type=Path, help="Runtime virtualenv; defaults to the repository's .venv.")
    parser.add_argument("--configure", action="store_true", help="Explicitly replace the saved Key via hidden input.")
    reopening = parser.add_mutually_exclusive_group()
    reopening.add_argument("--resume", nargs="?", const="", metavar="THREAD_ID",
                           help="Reopen a saved chat; omit its ID to choose a workspace conversation.")
    reopening.add_argument("--continue", dest="continue_chat", action="store_true",
                           help="Reopen the most recent chat in this launcher's workspace.")
    parser.add_argument("command", nargs="?", choices=["recover"], help="Explicitly recover an unfinished task.")
    parser.add_argument("recovery_arguments", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command == "recover" and (args.resume is not None or args.continue_chat):
        parser.error("recover cannot be combined with --resume or --continue")
    try:
        start(args.repo, Path.home() / ".config/praxis/credentials/deepseek.key",
              configure=args.configure, venv=args.venv, resume=args.resume, continue_chat=args.continue_chat,
              recover_args=args.recovery_arguments if args.command == "recover" else None)
    except (ValueError, OSError, EOFError, getpass.GetPassWarning, subprocess.SubprocessError) as error:
        print(f"Startup failed: {error if isinstance(error, ValueError) else type(error).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
