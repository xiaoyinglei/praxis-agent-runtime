"""Launch the Agent with only its proxy credential and a fixed service catalog."""

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
    return {
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


def main() -> None:
    account = pwd.getpwnam("praxis-agent")
    if os.geteuid() != account.pw_uid or os.getuid() != account.pw_uid:
        raise SystemExit("Run as praxis-agent, which has no sudo permissions.")
    home = Path(account.pw_dir)
    try:
        env = agent_environment(home / "proxy.token", home, os.environ.get("TERM", "dumb"))
    except (OSError, UnicodeError, ValueError):
        raise SystemExit("Missing or insecure Agent proxy credential; contact the server administrator.") from None
    os.chdir(home / "workspace")
    argv = sys.argv[1:] or ["chat"]
    # -I excludes cwd, PYTHONPATH and user site; app is root-owned.
    code = "import sys; sys.path.insert(0, '/opt/praxis/app'); from agent_runtime.cli import agent_app; agent_app()"
    python = "/opt/praxis/agent-venv/bin/python"
    os.execve(python, [python, "-I", "-B", "-c", code, *argv], env)


if __name__ == "__main__":
    main()
