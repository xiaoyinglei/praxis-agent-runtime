"""Location of host-owned personal settings, independent of the project cwd."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def user_config_directory() -> Path:
    configured = os.environ.get("PRAXIS_CONFIG_DIR")
    return (Path(configured).expanduser() if configured else Path.home() / ".config/praxis").resolve()


def command_state_directory() -> Path:
    """Host state, outside project grants; persists interpreter crashes, not reboot."""
    return Path(tempfile.gettempdir()).resolve() / f"praxis-command-jobs-{os.getuid()}"
