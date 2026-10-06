"""Launch with its proxy credential, optional search key path and fixed catalog."""

import os
import pwd
import stat
import sys
from pathlib import Path


def agent_environment(token_path: Path, home: Path, term: str) -> dict[str, str]:
    fd = os.open(token_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
            raise ValueError("insecure Agent proxy credential")
        token = handle.read(4098).decode("ascii").strip()
    if not 16 <= len(token) <= 4096 or not token.isprintable() or any(c.isspace() for c in token):
        raise ValueError("invalid Agent proxy credential")
    env = {
        "HOME": str(home),
        "USER": "praxis-agent",
        "LOGNAME": "praxis-agent",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "TERM": term,
        "PRAXIS_GATEWAY_TOKEN": token,
        "PRAXIS_DISABLE_DOTENV": "1",
        "RAG_AGENT_MODELS_PATH": "/opt/praxis/server-models.yaml",
        "PRAXIS_MODEL_REGISTRY_PATH": str(home / ".config/praxis/models.yaml"),
    }
    search_path = home / "search.key"
    if search_path.exists() or search_path.is_symlink():
        search_fd = os.open(search_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(search_fd, "rb") as handle:
            search_info = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(search_info.st_mode) or search_info.st_mode & 0o077
                or search_info.st_uid != os.getuid()
            ):
                raise ValueError("insecure search credential")
            search_key = handle.read(4097).decode("ascii").strip()
        if not 16 <= len(search_key) <= 4096 or not search_key.isprintable() or any(c.isspace() for c in search_key):
            raise ValueError("invalid search credential")
        env["PRAXIS_WEB_SEARCH_KEY_FILE"] = str(search_path)
    return env


def main() -> None:
    account = pwd.getpwnam("praxis-agent")
    if os.geteuid() != account.pw_uid or os.getuid() != account.pw_uid:
        raise SystemExit("Run as praxis-agent, which has no sudo permissions.")
    home = Path(account.pw_dir)
    try:
        env = agent_environment(home / "proxy.token", home, os.environ.get("TERM", "dumb"))
    except (OSError, UnicodeError, ValueError):
        raise SystemExit(
            "Missing or insecure Agent proxy/search credential; contact the server administrator."
        ) from None
    os.chdir(home / "workspace")
    argv = sys.argv[1:] or ["chat"]
    # -I excludes cwd, PYTHONPATH and user site; app is root-owned.
    code = "import sys; sys.path.insert(0, '/opt/praxis/app'); from agent_runtime.cli import agent_app; agent_app()"
    python = "/opt/praxis/agent-venv/bin/python"
    os.execve(python, [python, "-I", "-B", "-c", code, *argv], env)


if __name__ == "__main__":
    main()
