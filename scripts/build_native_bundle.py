#!/usr/bin/env python3
"""Build a relocatable Agent distribution; uv is needed only on the builder."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path


def run(*args: str, cwd: Path, env: dict[str, str]) -> str:
    result = subprocess.run(args, cwd=cwd, env=env, check=True, stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-platform", help="Fail if the runner differs from the requested release target")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    system = {"Darwin": "darwin", "Linux": "linux"}.get(platform.system())
    arch = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x86_64"}.get(platform.machine())
    if system is None or arch is None or system == "linux" and platform.libc_ver()[0] != "glibc":
        parser.error("Native bundles support macOS and glibc Linux on arm64/x86_64")
    if args.expected_platform and args.expected_platform != f"{system}-{arch}":
        parser.error(f"Expected {args.expected_platform}, got {system}-{arch}")
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="praxis-native-build-") as directory:
        work = Path(directory).resolve()
        env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(work / "interpreters"), UV_PYTHON_INSTALL_BIN="0")
        print("Preparing standalone Python 3.12", flush=True)
        run("uv", "python", "install", "--no-bin", "3.12", cwd=root, env=env)
        python = run("uv", "python", "find", "--no-project", "--system", "--managed-python", "3.12",
                     cwd=work, env=env)
        prefix = Path(run(python, "-I", "-c", "import sys; print(sys.base_prefix)", cwd=work, env=env)).resolve()
        if not prefix.is_relative_to((work / "interpreters").resolve()):
            raise ValueError("Build interpreter must belong to this temporary build")
        bundle = work / "praxis"
        runtime = bundle / "python"
        shutil.copytree(prefix, runtime, symlinks=True)
        # Relocate every interpreter link and reject host-only references.
        for source in prefix.rglob("*"):
            if not source.is_symlink():
                continue
            target = source.resolve(strict=True)
            if not target.is_relative_to(prefix):
                raise ValueError(f"Standalone Python contains an external link: {source.relative_to(prefix)}")
            copied = runtime / source.relative_to(prefix)
            copied.unlink()
            copied.symlink_to(os.path.relpath(runtime / target.relative_to(prefix), copied.parent))
        bundled_python = str(runtime / "bin" / "python3")
        # The copied runtime belongs to this bundle, not to the host manager.
        (runtime / "lib/python3.12/EXTERNALLY-MANAGED").unlink(missing_ok=True)
        requirements = work / "requirements.txt"
        run("uv", "export", "--frozen", "--no-dev", "--no-default-groups", "--no-emit-project",
            "--format", "requirements-txt", "--output-file", str(requirements), cwd=root, env=env)
        run("uv", "build", "--wheel", "--no-sources", "--out-dir", str(work / "wheels"), cwd=root, env=env)
        wheel, = (work / "wheels").glob("*.whl")
        wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
        with requirements.open("a") as handle:
            handle.write(f"\npraxis-agent @ {wheel.as_uri()} --hash=sha256:{wheel_hash}\n")
        print("Installing locked dependencies into the bundled interpreter", flush=True)
        run("uv", "pip", "install", "--quiet", "--python", bundled_python, "--system", "--link-mode", "copy",
            "--only-binary", ":all:", "--require-hashes", "--requirements", str(requirements), cwd=root, env=env)
        # Generated Python console scripts must survive moving the whole bundle.
        for entry in (runtime / "bin").iterdir():
            if entry.is_symlink() or not entry.is_file():
                continue
            content = entry.read_bytes()
            if not content.startswith(b"#!"):
                continue
            lines = content.decode().splitlines(keepends=True)
            if "python" in lines[0]:
                body = "".join(lines[1:])
            elif len(lines) >= 3 and lines[1].startswith("'''exec'") and lines[2].strip() == "' '''":
                body = "".join(lines[3:])
            else:
                if str(runtime) in "".join(lines[:5]):
                    raise ValueError(f"Unrecognized non-relocatable entrypoint: {entry.name}")
                continue
            entry.write_text('#!/bin/sh\n\'\'\'exec\' "$(dirname -- "$0")/python3" "$0" "$@"\n\' \'\'\'\n' + body)
        bin_dir = bundle / "bin"
        bin_dir.mkdir()
        launcher = bin_dir / "agent"
        launcher.write_text(
            '#!/bin/sh\nset -eu\n'
            'root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)\n'
            'exec "$root/python/bin/python3" -I -c '
            "'from agent_runtime.cli import agent_app; agent_app(prog_name=\"agent\")' \"$@\"\n"
        )
        launcher.chmod(0o755)
        (bundle / "VERSION").write_text(version + "\n")
        shutil.copyfile(root / "LICENSE", bundle / "LICENSE")
        commit = run("git", "rev-parse", "HEAD", cwd=root, env=env)
        python_version = run(bundled_python, "-I", "-c", "import platform; print(platform.python_version())",
                             cwd=root, env=env)
        (bundle / "BUILD.json").write_text(json.dumps(
            {"version": version, "commit": commit, "python": python_version, "platform": f"{system}-{arch}"},
            indent=2,
        ) + "\n")
        for path in bundle.rglob("*"):
            if path.is_symlink() and not path.resolve(strict=True).is_relative_to(bundle):
                raise ValueError(f"Bundle contains an external link: {path.relative_to(bundle)}")
        name = f"praxis-{system}-{arch}.tar.gz"
        archive = output / name
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(bundle, arcname="praxis")
        checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
        (output / (name + ".sha256")).write_text(f"{checksum}  {name}\n")
        print(f"Built {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
