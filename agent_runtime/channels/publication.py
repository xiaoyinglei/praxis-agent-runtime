"""Explicit GitDelivery previews and independently journaled remote effects."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any
from urllib.parse import urlsplit

from agent_runtime.channels.providers import _secret
from agent_runtime.git_delivery import DeliveryInterruptedError, GitDelivery, _settle_delivery
from agent_runtime.harness import RuntimeStore
from agent_runtime.harness import channel_ops as ch
from agent_runtime.harness.operation_values import _json_object


def repository(remote: str) -> str:
    match = re.fullmatch(r"git@github\.com:([A-Za-z0-9_-]+/[A-Za-z0-9_.-]+?)(?:\.git)?", remote)
    if match:
        return match[1]
    url = urlsplit(remote)
    if (
        url.scheme != "https"
        or url.hostname != "github.com"
        or url.username
        or url.password
        or url.port not in {None, 443}
        or url.query
        or url.fragment
    ):
        raise PermissionError("publication remote must be the explicitly bound GitHub repository")
    name = url.path.strip("/").removesuffix(".git")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", name) or ".." in name:
        raise ValueError("invalid publication repository")
    return name


def authority(store: RuntimeStore, intent: dict[str, Any]) -> None:
    ref = intent["authority"]
    grant = ch.authorize(
        store,
        ref["source_id"],
        ref["principal"],
        ref["conversation_id"],
        intent["kind"],
        expected_revision=ref["revision"],
    )
    binding = ch.get(store, "binding", ch.identity(ref["source_id"], ref["conversation_id"]))
    source = ch.get(store, "source", ref["source_id"])
    if (
        binding is None
        or binding["revision"] != ref["binding_revision"]
        or grant["thread_id"] != intent["thread_id"]
        or source is None
        or source["revision"] != ref["source_revision"]
    ):
        raise PermissionError("publication authority changed")


async def prepare(
    channels: Any,
    *,
    source_id: str,
    principal: str,
    conversation_id: str,
    operation_id: str,
    kind: str,
    scope: dict[str, Any],
    expected_preview_id: str,
    git: GitDelivery | None = None,
) -> dict[str, Any]:
    if git is not None and git._workspace != channels.workspace:
        raise PermissionError("GitDelivery belongs to another workspace")
    if kind not in {"push", "open_pr", "review_reply"} or not operation_id or len(operation_id) > 256:
        raise ValueError("invalid publication intent")
    scope = _json_object(scope, field="publication scope")
    if kind == "push" and scope:
        raise ValueError("push scope is exactly the displayed branch and commit")
    if kind == "open_pr" and (
        set(scope) != {"title", "body"} or any(not isinstance(scope[k], str) or not scope[k].strip() for k in scope)
    ):
        raise ValueError("open_pr scope requires exact title and body")
    if kind == "review_reply" and (
        set(scope) != {"pull_request", "comment_id", "body"}
        or type(scope["pull_request"]) is not int
        or scope["pull_request"] < 1
        or type(scope["comment_id"]) is not int
        or scope["comment_id"] < 1
        or not isinstance(scope["body"], str)
        or not scope["body"].strip()
    ):
        raise ValueError("review_reply scope requires exact PR, comment and body")
    async with asyncio.timeout(60):
        preview = await (git or GitDelivery(channels.workspace)).pull_request_status()
    if preview.preview_id != expected_preview_id:
        raise ValueError("publication preview changed")
    repo = repository(preview.remote_url)
    async with await RuntimeStore.open(channels.database) as store:

        def commit() -> dict[str, Any]:
            with store.tx.transaction():
                grant = ch.authorize(store, source_id, principal, conversation_id, kind)
                channels._thread(store, grant["thread_id"])
                source = channels._source(store, source_id)
                if not source.outbound_token_env or not source.outbound_principal:
                    raise ValueError("publication requires explicit outbound token and identity references")
                if source.kind != "github" or source.repository != repo:
                    raise PermissionError("publication repository differs from explicit source")
                binding = ch.get(store, "binding", ch.identity(source_id, conversation_id))
                assert binding is not None
                if kind == "review_reply" and binding["destination"] != f"github:{repo}:{scope['pull_request']}":
                    raise PermissionError("review reply requires an explicit grant for its exact PR audience")
                ref = {k: grant[k] for k in ("source_id", "principal", "conversation_id", "revision")}
                ref.update(binding_revision=binding["revision"], source_revision=ch._source_revision(store, source_id))
                data = _json_object(
                    dict(
                        operation_id=operation_id,
                        kind=kind,
                        scope=scope,
                        authority=ref,
                        thread_id=grant["thread_id"],
                        repository=repo,
                        preview=asdict(preview),
                        correlation=ch.identity(operation_id, repo, preview.head_commit),
                    ),
                    field="publication",
                )
                prior = ch.get(store, "publication", operation_id)
                if prior is not None:
                    if prior["intent"] != data:
                        raise ValueError("publication identity conflict")
                    return prior
                row = {**data, "intent": dict(data), "status": "prepared", "result": {}}
                ch.put(store, grant["thread_id"], "publication", operation_id, row)
                return row

        return await store.run_async(commit)


def remote_effect(
    git: GitDelivery, row: dict[str, Any], *, reconcile: bool = False, deadline: float, authorize: Callable[[], Any]
) -> dict[str, Any] | None:
    """Blocking Git/gh execution is bounded and held through cancellation settlement."""

    def remaining() -> float:
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("publication deadline exceeded")
        return seconds

    environment = {
        "GH_HOST": "github.com",
        "GH_TOKEN": _secret(row.get("outbound_token_env")),
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "credential.helper",
        "GIT_CONFIG_VALUE_0": "",
        "GIT_CONFIG_KEY_1": "credential.https://github.com.helper",
        "GIT_CONFIG_VALUE_1": "!gh auth git-credential",
    }
    preview, scope, repo = row["preview"], row["scope"], row["repository"]
    marker = "<!-- praxis-publication:" + row["correlation"] + " -->"
    if row["kind"] == "push":
        if not reconcile:
            authorize()
            result = git._run_git(
                "push",
                preview["remote_url"],
                f"{preview['head_commit']}:refs/heads/{preview['branch']}",
                check=False,
                timeout_seconds=remaining(),
                environment=environment,
            )
            if result.returncode == 0:
                return {"head_sha": preview["head_commit"]}
        result = git._run_git(
            "ls-remote",
            preview["remote_url"],
            f"refs/heads/{preview['branch']}",
            check=False,
            timeout_seconds=remaining(),
            environment=environment,
        )
        if result.returncode == 0 and any(
            line.split() == [preview["head_commit"], "refs/heads/" + preview["branch"]]
            for line in result.stdout.splitlines()
        ):
            return {"head_sha": preview["head_commit"], "reconciled": True}
        return None

    def api(path: str, *args: str) -> Any:
        authorize()
        result = git._gh_output(
            "api", "--hostname", "github.com", path, *args, timeout_seconds=remaining(), environment=environment
        )
        if result.returncode != 0:
            raise RuntimeError("publication remote response unavailable")
        if len(result.stdout) > 1024 * 1024:
            raise ValueError("publication remote response too large")
        return json.loads(result.stdout)

    if row["kind"] == "open_pr":
        if not reconcile:
            remote = git._run_git(
                "ls-remote",
                preview["remote_url"],
                f"refs/heads/{preview['branch']}",
                check=False,
                timeout_seconds=remaining(),
                environment=environment,
            )
            if remote.returncode != 0 or not any(
                line.split() == [preview["head_commit"], "refs/heads/" + preview["branch"]]
                for line in remote.stdout.splitlines()
            ):
                raise ValueError("push the previewed commit before creating its PR")
            result = api(
                f"repos/{repo}/pulls",
                "--method",
                "POST",
                "-f",
                "title=" + scope["title"],
                "-f",
                "body=" + scope["body"] + "\n\n" + marker,
                "-f",
                "head=" + preview["branch"],
                "-f",
                "base=" + preview["base_branch"],
            )
            return {"remote_id": result["id"], "url": result["html_url"]}
        from urllib.parse import urlencode

        query = urlencode(
            {
                "state": "all",
                "head": repo.split("/")[0] + ":" + preview["branch"],
                "base": preview["base_branch"],
                "per_page": 100,
            }
        )
        results = api(f"repos/{repo}/pulls?{query}")
        for result in results:
            if (
                marker in (result.get("body") or "")
                and result.get("head", {}).get("sha") == preview["head_commit"]
                and str(result.get("user", {}).get("id")) == row.get("outbound_principal")
                and result.get("head", {}).get("repo", {}).get("full_name") == repo
            ):
                return {"remote_id": result["id"], "url": result["html_url"], "reconciled": True}
    else:
        if not reconcile:
            comment = api(f"repos/{repo}/pulls/comments/{scope['comment_id']}")
            if comment.get("commit_id") != preview["head_commit"]:
                raise ValueError("review position is stale for the previewed head")
            result = api(
                f"repos/{repo}/pulls/{scope['pull_request']}/comments/{scope['comment_id']}/replies",
                "--method",
                "POST",
                "-f",
                "body=" + scope["body"] + "\n\n" + marker,
            )
            return {"remote_id": result["id"], "url": result["html_url"]}
        for page in range(1, 11):
            results = api(f"repos/{repo}/pulls/{scope['pull_request']}/comments?per_page=100&page={page}")
            for result in results:
                if (
                    marker in (result.get("body") or "")
                    and result.get("in_reply_to_id") == scope["comment_id"]
                    and str(result.get("user", {}).get("id")) == row.get("outbound_principal")
                ):
                    return {"remote_id": result["id"], "url": result["html_url"], "reconciled": True}
            if len(results) < 100:
                break
    return None  # absence never proves safe to repeat a write


async def execute(
    channels: Any, operation_id: str, *, git: GitDelivery | None = None, reconcile: bool = False
) -> dict[str, Any]:
    git = git or GitDelivery(channels.workspace)
    deadline = time.monotonic() + 60
    async with await RuntimeStore.open(channels.database) as store:

        def read() -> dict[str, Any]:
            row = ch.get(store, "publication", operation_id)
            if row is None:
                raise KeyError("unknown publication intent")
            channels._thread(store, row["thread_id"])
            authority(store, row)
            source = channels._source(store, row["authority"]["source_id"])
            return {
                **row,
                "outbound_principal": source.outbound_principal,
                "outbound_token_env": source.outbound_token_env,
            }

        row = await store.run_async(read)
        if row["status"] == "succeeded":
            return row
        if reconcile and row["status"] not in {"sending", "unknown"}:
            raise ValueError("publication has no dispatched effect to reconcile")
        if row["status"] != "prepared" and not reconcile:
            raise ValueError("publication outcome unknown; reconcile before any new side effect")
        if not reconcile:
            async with asyncio.timeout(60):
                preview = await git.pull_request_status()
            if preview.preview_id != row["preview"]["preview_id"]:
                raise ValueError("publication preview changed")

        def claim() -> None:
            with store.tx.transaction():
                current = read()
                if current["status"] != row["status"]:
                    raise ValueError("publication already claimed")
                if not reconcile:
                    current["status"] = "sending"
                    ch.put(store, row["thread_id"], "publication", operation_id, current)

        await store.run_async(claim)
        result = None
        try:
            async with asyncio.timeout(60):
                result = await _settle_delivery(
                    lambda: remote_effect(
                        git, row, reconcile=reconcile, deadline=deadline, authorize=lambda: store.run_sync(read)
                    )
                )
        except DeliveryInterruptedError as error:
            result = error.result if isinstance(error.result, dict) else None
        except Exception:
            # A failed process/ACK is conservative unknown. One bounded positive
            # reconciliation may settle it; absence can never authorize a write.
            if not reconcile:
                try:
                    async with asyncio.timeout(60):
                        result = await _settle_delivery(
                            lambda: remote_effect(
                                git, row, reconcile=True, deadline=deadline, authorize=lambda: store.run_sync(read)
                            )
                        )
                except DeliveryInterruptedError as error:
                    result = error.result if isinstance(error.result, dict) else None
                except Exception:
                    pass
        finally:

            def finish() -> dict[str, Any]:
                with store.tx.transaction():
                    current = ch.get(store, "publication", operation_id)
                    assert current is not None
                    if current["status"] == "succeeded":
                        return current  # late absence/timeout cannot erase positive evidence
                    current.update(status="succeeded" if result is not None else "unknown", result=result or {})
                    ch.put(store, row["thread_id"], "publication", operation_id, current)
                    return current

            final = await asyncio.shield(store.run_async(finish))
        return final
