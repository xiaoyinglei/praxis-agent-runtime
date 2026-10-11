#!/usr/bin/env python3
"""Install the pinned offline patch executor; standard library only."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

TAG = "rust-v0.162.1"
ARCHIVES = {
    "x86_64": ("x86_64-unknown-linux-musl", "86f268d81b898f3e144c5ecff2ad2fda2c5802e042d6195b72538fe0fa125057"),
    "aarch64": ("aarch64-unknown-linux-musl", "bf3ca2719f1143b1440fbd85c149379e696c782c50ad7991f8bfcec60b2600c1"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path, help="executable path outside the Agent workspace")
    parser.add_argument(
        "--archive", type=Path, help="optional already downloaded release archive (digest still checked)"
    )
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in ARCHIVES:
        parser.error("this installer supports Linux x86_64 and aarch64")
    target, digest = ARCHIVES[platform.machine()]
    destination = args.destination.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="codex-install-") as temporary:
        archive = args.archive
        if archive is None:
            archive = Path(temporary) / "codex.tar.gz"
            url = f"https://github.com/openai/codex/releases/download/{TAG}/codex-{target}.tar.gz"
            with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        with archive.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"release archive digest mismatch: {actual}")
        with tarfile.open(archive, "r:gz") as bundle:
            member = bundle.getmember(f"codex-{target}")
            if not member.isfile():
                raise ValueError("release executable is not a regular file")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError("release executable is missing")
            name = None
            try:
                with source, tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                    name = output.name
                    shutil.copyfileobj(source, output)
                    output.flush()
                    os.fsync(output.fileno())
                os.chmod(name, 0o755)
                os.replace(name, destination)
                name = None
            finally:
                if name is not None:
                    Path(name).unlink(missing_ok=True)
    print(f"Installed {TAG} at {destination}")
    print(f"Set PRAXIS_CODEX_APPLY_PATCH={destination}")


if __name__ == "__main__":
    main()
