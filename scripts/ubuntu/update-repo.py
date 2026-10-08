#!/usr/bin/env python3
"""Bounded HTTPS fetch and fast-forward update; no reset, stash, or global config."""
import argparse
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit


class GitUpdateError(RuntimeError):
    pass


def git_environment():
    env = dict(os.environ)
    # -C does not override GIT_DIR/WORK_TREE/INDEX_FILE. Keep only explicit CA trust inputs.
    for key in list(env):
        if key.startswith("GIT_") and key not in {"GIT_SSL_CAINFO", "GIT_SSL_CAPATH"}:
            env.pop(key)
    env["GIT_TERMINAL_PROMPT"] = "0"
    # Environment takes precedence over even more-specific http.<url>.sslVerify config.
    env["GIT_SSL_NO_VERIFY"] = "false"
    env["GCM_INTERACTIVE"] = "never"
    return env


def classify(stderr):
    message = stderr.lower()
    for category, markers in (
        ("certificate", ("certificate verification", "certificate problem", "server certificate", "ssl certificate")),
        ("authentication", ("authentication failed", "could not read username", "403", "401")),
        ("repository", ("repository not found", "does not appear to be a git repository", "404")),
        ("ref", ("couldn't find remote ref", "invalid refspec")),
        ("tls", ("gnutls", "ssl_connect", "tls connection", "connection reset", "recv failure")),
        ("timeout", ("timed out", "operation too slow", "timeout")),
        ("network", ("could not resolve", "failed to connect", "unable to connect", "early eof",
                     "unexpected disconnect", "empty reply", "rpc failed", "http/2")),
    ):
        if any(marker in message for marker in markers):
            return category
    return "local_git_error"


def run_git(repo, args, deadline, *, attempt_timeout=None, allow_failure=False):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise GitUpdateError("timeout: total Git update deadline exhausted")
    process = subprocess.Popen(
        ["git", "-C", str(repo), *args], env=git_environment(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=min(remaining, attempt_timeout or remaining))
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise GitUpdateError("timeout: Git process group stopped") from None
    if process.returncode and not allow_failure:
        # Do not print raw Git stderr: it can contain credentials or signed URLs.
        raise GitUpdateError(classify(stderr))
    return process.returncode, stdout.strip()


def update(repo, branch="main", timeout=60, *, fetch_only=False, bundle=None, expected_commit=None):
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    deadline = time.monotonic() + timeout
    repo = Path(repo).resolve(strict=True)

    def git(*args, **kwargs):
        return run_git(repo, list(args), deadline, **kwargs)[1]

    git("check-ref-format", "--branch", branch)
    if git("symbolic-ref", "--short", "HEAD") != branch:
        raise GitUpdateError("checkout must already be on the requested branch")
    if git("status", "--porcelain"):
        raise GitUpdateError("dirty checkout: local changes preserved; update refused")
    origin = git("remote", "get-url", "origin")
    url = urlsplit(origin)
    if (url.scheme != "https" or url.hostname != "github.com" or url.port not in (None, 443)
            or url.username or url.password or url.query or url.fragment
            or not re.fullmatch(r"/[\w.-]+/[\w.-]+(?:\.git)?", url.path)):
        raise GitUpdateError("origin must resolve to a credential-free HTTPS GitHub repository")
    # Refuse matching rewrites even if the first rewrite still points at GitHub.
    _, rewrites = run_git(repo, ["config", "--get-regexp", r"^url\..*\.insteadof$"], deadline, allow_failure=True)
    if any(origin.startswith(line.split(" ", 1)[1]) for line in rewrites.splitlines() if " " in line):
        raise GitUpdateError("origin URL rewrite must be reviewed before update")
    if bundle is not None and not re.fullmatch(r"[0-9a-f]{40}", expected_commit or ""):
        raise GitUpdateError("bundle requires an explicitly approved full expected commit")

    temporary = "refs/praxis-update/" + uuid.uuid4().hex
    refspec = f"refs/heads/{branch}:{temporary}"
    configuration = [
        "-c", "http.version=HTTP/1.1", "-c", "http.sslVerify=true",
        "-c", f"http.{origin}.sslVerify=true", "-c", "http.lowSpeedLimit=1024",
        "-c", "http.lowSpeedTime=5", "-c", "http.followRedirects=false",
    ]
    network_deadline = deadline - min(10, timeout / 3) if bundle is not None else deadline
    try:
        for attempt in range(3):
            try:
                run_git(repo, configuration + ["fetch", "--no-tags", "--no-write-fetch-head", origin, refspec],
                        network_deadline,
                        attempt_timeout=min(15, max(0.01, (network_deadline - time.monotonic()) / (3 - attempt))))
                break
            except GitUpdateError as error:
                if str(error).split(":", 1)[0] not in {"network", "tls", "timeout"}:
                    raise
                print(f"Git attempt {attempt + 1}/3 failed: {str(error).split(':', 1)[0]}", file=sys.stderr, flush=True)
                if attempt == 2:
                    if bundle is None:
                        raise GitUpdateError(
                            "network unavailable after 3 attempts; use a trusted proxy or an approved bundle"
                        ) from None
                    bundle = str(Path(bundle).resolve(strict=True))
                    git("bundle", "verify", bundle)
                    heads = git("bundle", "list-heads", bundle, f"refs/heads/{branch}")
                    if heads != f"{expected_commit} refs/heads/{branch}":
                        raise GitUpdateError("bundle branch does not match expected commit") from None
                    git("fetch", "--no-tags", "--no-write-fetch-head", bundle, refspec)
        commit = git("rev-parse", temporary)
        if expected_commit is not None and commit != expected_commit:
            raise GitUpdateError("fetched branch does not match expected commit")
        status, _ = run_git(repo, ["merge-base", "--is-ancestor", "HEAD", temporary], deadline, allow_failure=True)
        if status:
            raise GitUpdateError("fast-forward refused: checkout is ahead or diverged; commits preserved")
        if not fetch_only:
            if git("status", "--porcelain") or git("symbolic-ref", "--short", "HEAD") != branch:
                raise GitUpdateError("dirty or changed checkout: update refused")
            git("-c", "core.hooksPath=/dev/null", "merge", "--ff-only", "--no-edit", temporary)
        git("update-ref", f"refs/remotes/origin/{branch}", commit)
        print(f"{'Fetched' if fetch_only else 'Updated'} {branch}: {commit}")
        return commit
    finally:
        # Bounded local cleanup even when the operation consumed its network budget.
        run_git(repo, ["update-ref", "-d", temporary], time.monotonic() + 2, allow_failure=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    try:
        update(args.repo, args.branch, args.timeout, fetch_only=args.fetch_only,
               bundle=args.bundle, expected_commit=args.expected_commit)
    except (GitUpdateError, ValueError, OSError) as error:
        print(f"Update failed: {error if isinstance(error, GitUpdateError) else type(error).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
