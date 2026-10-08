#!/usr/bin/env python3
"""Install the reduced Agent profile using locked hashes and a persistent uv wheel cache."""
import argparse
import fcntl
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


def prepare_cache(path):
    if path.is_symlink():
        raise ValueError("cache must not be a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    status = path.stat()
    if status.st_uid != os.getuid() or status.st_mode & 0o022:
        raise ValueError("cache must be owned by this user and not writable by others")


def locked_manifest(text, lock):
    """Use the resolver's dependency selection but the lock's immutable artifact hashes."""
    packages = {(p["name"], p["version"]): p for p in lock["package"] if "version" in p}
    logical = text.replace("\\\n", " ").splitlines()
    result = []
    for line in logical:
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        requirement = line.split("--hash=", 1)[0].strip()
        match = re.fullmatch(r"([\w.-]+)(\[[\w,.-]+\])?==([^\s;]+)(\s*;.*)?", requirement)
        if match is None:
            raise ValueError("compiled profile must contain only pinned registry requirements")
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        package = packages.get((name, match[3]))
        if package is None or package.get("source", {}).get("registry") != "https://pypi.org/simple":
            raise ValueError("compiled requirement is not pinned in uv.lock on PyPI")
        hashes = sorted({a["hash"] for a in package.get("wheels", [])})
        if not hashes or any(not re.fullmatch(r"sha256:[0-9a-f]{64}", h) for h in hashes):
            raise ValueError("locked package has no binary wheel hashes")
        result.append(requirement + " \\\n" + " \\\n".join("    --hash=" + h for h in hashes))
    if not result:
        raise ValueError("empty compiled profile")
    return "\n".join(result) + "\n"


def install(repo, python, cache, offline=False, wheel_dir=None):
    repo = Path(repo).resolve(strict=True)
    cache = Path(cache).absolute()
    prepare_cache(cache)
    env = dict(os.environ)
    for key in ("UV_NO_CACHE", "UV_NO_VERIFY_HASHES", "UV_EXTRA_INDEX_URL", "UV_INDEX_URL", "UV_INDEX"):
        env.pop(key, None)
    env["UV_HTTP_TIMEOUT"] = "20"
    env["UV_HTTP_RETRIES"] = "2"
    base = ["uv", "--no-config", "--cache-dir", str(cache / "uv")]
    if offline:
        base.append("--offline")
    source = (["--no-index", "--find-links", str(Path(wheel_dir).resolve(strict=True))]
              if wheel_dir is not None else ["--default-index", "https://pypi.org/simple"])

    def run(args):
        subprocess.run([*base, *args], cwd=repo, env=env, check=True, timeout=300)

    identity = subprocess.check_output([str(python), "-c",
        "import json,sys,sysconfig; print(json.dumps([sys.version_info[:3],sys.implementation.cache_tag,"
        "sysconfig.get_platform(),sysconfig.get_config_var('SOABI')]))"], text=True, timeout=10)
    lock_bytes = (repo / "uv.lock").read_bytes()
    profile_bytes = (repo / "scripts/ubuntu/requirements-core.in").read_bytes()
    key = hashlib.sha256(b"core-cache-v1\0" + lock_bytes + profile_bytes + identity.encode()).hexdigest()
    manifest = cache / (key + ".txt")
    lock = tomllib.loads(lock_bytes.decode())
    lock_file = cache / "install.lock"
    if lock_file.is_symlink() or manifest.is_symlink():
        raise ValueError("cache files must not be symlinks")
    with lock_file.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if not manifest.exists():
            print("Preparing locked core manifest (cached by lock, profile and Python platform).", flush=True)
            with tempfile.TemporaryDirectory(dir=cache) as staging:
                constraints = Path(staging) / "constraints.txt"
                compiled = Path(staging) / "requirements.txt"
                run(["export", "--quiet", "--locked", "--no-dev", "--no-hashes", "--no-emit-project",
                     "--format", "requirements-txt", "--output-file", str(constraints)])
                run(["pip", "compile", "--quiet", "--python", str(python), *source, "-c", str(constraints),
                     "scripts/ubuntu/requirements-core.in", "--output-file", str(compiled)])
                compiled.write_text(locked_manifest(compiled.read_text(), lock))
                os.replace(compiled, manifest)
        # Re-validate pins and immutable hashes on every use, including offline runs.
        if locked_manifest(manifest.read_text(), lock) != manifest.read_text():
            raise ValueError("cached manifest differs from uv.lock; remove the manifest and retry")
        print("Installing binary core dependencies from persistent wheel cache.", flush=True)
        run(["pip", "install", "--python", str(python), "--require-hashes", "--only-binary", ":all:",
             *source, "-r", str(manifest)])
        # uv also caches the editable build backend. --offline never downloads build tools.
        run(["pip", "install", "--python", str(python), "--no-deps", "-e", "."])
    print(f"Core dependencies ready. Cache: {cache}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/praxis-deps")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--wheel-dir", type=Path, help="Use only an explicitly supplied wheel directory (no index).")
    args = parser.parse_args()
    try:
        install(args.repo, args.python.absolute(), args.cache_dir, args.offline, args.wheel_dir)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Dependency installation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
