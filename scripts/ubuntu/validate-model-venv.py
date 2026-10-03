"""Reject import hooks or symlinks in the model service's fresh site-packages."""

import argparse
import os
import site
import sys
from pathlib import Path


def validate_site_packages(directory: Path) -> None:
    for current, directories, files in os.walk(directory, followlinks=False):
        for name in [*directories, *files]:
            path = Path(current) / name
            if path.is_symlink() or path.suffix in {".pth", ".egg-link"}:
                raise ValueError("model service venv contains an import hook or symlink")
            if os.geteuid() == 0 and (path.stat().st_uid != 0 or path.stat().st_mode & 0o022):
                raise ValueError("model service imports are not exclusively administrator-owned")


def validate_agent_config(config_path: Path) -> None:
    fields = dict(line.split("=", 1) for line in config_path.read_text().splitlines() if "=" in line)
    values = {key.strip(): value.strip() for key, value in fields.items()}
    version = values.get("version_info", values.get("version", ""))
    if version.split(".")[:2] != ["3", "12"]:
        raise ValueError("Agent environment requires Linux Python 3.12")


def relocate_agent_config(config_path: Path) -> None:
    version = ".".join(str(part) for part in sys.version_info[:3])
    config_path.write_text(
        f"home = /usr/bin\ninclude-system-site-packages = false\nversion = {version}\nexecutable = /usr/bin/python3\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-config", type=Path)
    parser.add_argument("--relocate-agent", type=Path)
    args = parser.parse_args()
    if args.agent_config:
        validate_agent_config(args.agent_config)
    elif args.relocate_agent:
        relocate_agent_config(args.relocate_agent)
    else:
        for directory in site.getsitepackages():
            validate_site_packages(Path(directory))
