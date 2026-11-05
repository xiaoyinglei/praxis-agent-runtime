import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime.channels import Channels
from agent_runtime.channels.providers import Source
from agent_runtime.git_delivery import DeliveryError, GitDelivery, PullRequestStatus
from agent_runtime.harness import RuntimeStore, host_ops


@pytest.fixture(autouse=True)
def publication_credential(monkeypatch):
    monkeypatch.setenv("PUB_TOKEN", "token")
    monkeypatch.setenv("GH_HOST", "untrusted.invalid")


class GitRemote(GitDelivery):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.preview = PullRequestStatus(
            "praxis/fix",
            "https://github.com/o/r.git",
            "main",
            ("fix",),
            ("a.py",),
            0,
            None,
            head_commit="sha1",
            preview_id="preview",
        )
        self.effects = 0
        self.remote_sha = ""
        self.lose_ack = False

    async def pull_request_status(self):
        return self.preview

    def _run_git(self, *args, **kwargs):
        if args[0] == "push":
            self.effects += 1
            self.remote_sha = "sha1"
            if self.lose_ack:
                raise DeliveryError("timeout")
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[0] == "ls-remote":
            return subprocess.CompletedProcess(args, 0, self.remote_sha + "\trefs/heads/praxis/fix\n", "")
        raise AssertionError(args)


@pytest.mark.anyio
async def test_publication_preview_separate_intents_and_lost_push_ack(tmp_path: Path):
    channels = Channels(database=tmp_path / "r.db", workspace=tmp_path)
    async with await RuntimeStore.open(channels.database) as store:
        start = await store.run_async(
            host_ops.receive_command, store, workspace=tmp_path, kind="start", payload={"text": "work"}
        )
    await channels.configure(
        start.thread_id,
        Source(
            source_id="gh",
            kind="github",
            repository="o/r",
            webhook_secret_env="SECRET",
            outbound_token_env="PUB_TOKEN",
            outbound_principal="99",
        ),
    )
    await channels.bind(
        source_id="gh", conversation_id="github:o/r:1", destination="github:o/r:1", thread_id=start.thread_id
    )
    await channels.grant(
        source_id="gh",
        principal="7",
        conversation_id="github:o/r:1",
        thread_id=start.thread_id,
        role="owner",
        operations=["status", "push", "open_pr"],
    )
    git = GitRemote(tmp_path)
    intent = await channels.prepare_publication(
        source_id="gh",
        principal="7",
        conversation_id="github:o/r:1",
        operation_id="push1",
        kind="push",
        scope={},
        expected_preview_id="preview",
        git=git,
    )
    assert intent["status"] == "prepared"
    git.lose_ack = True
    result = await channels.execute_publication("push1", git=git)
    assert result["status"] == "succeeded"
    assert result["result"]["reconciled"] is True
    assert git.effects == 1
    assert (await channels.execute_publication("push1", git=git))["status"] == "succeeded"
    assert git.effects == 1
    git.preview = replace(git.preview, preview_id="changed")
    with pytest.raises(ValueError, match="preview"):
        await channels.prepare_publication(
            source_id="gh",
            principal="7",
            conversation_id="github:o/r:1",
            operation_id="other",
            kind="push",
            scope={},
            expected_preview_id="preview",
            git=git,
        )


def test_publication_rejects_lookalike_hosts():
    from agent_runtime.channels.publication import repository

    assert repository("git@github.com:o/r.git") == "o/r"
    assert repository("https://github.com/o/r.git") == "o/r"
    for remote in (
        "https://github.com.evil/o/r",
        "https://evil/github.com/o/r",
        "https://user:token@github.com/o/r",
        "https://github.com/o/r?host=evil",
    ):
        with pytest.raises((ValueError, PermissionError)):
            repository(remote)


@pytest.mark.anyio
async def test_unknown_pr_result_never_blind_repeats_and_revoke_blocks_cached_result(tmp_path):
    channels = Channels(database=tmp_path / "r.db", workspace=tmp_path)
    async with await RuntimeStore.open(channels.database) as store:
        start = await store.run_async(
            host_ops.receive_command, store, workspace=tmp_path, kind="start", payload={"text": "work"}
        )
    await channels.configure(
        start.thread_id,
        Source(
            source_id="gh",
            kind="github",
            repository="o/r",
            webhook_secret_env="SECRET",
            outbound_token_env="PUB_TOKEN",
            outbound_principal="99",
        ),
    )
    await channels.bind(
        source_id="gh", conversation_id="github:o/r:1", destination="github:o/r:1", thread_id=start.thread_id
    )
    await channels.grant(
        source_id="gh",
        principal="7",
        conversation_id="github:o/r:1",
        thread_id=start.thread_id,
        role="owner",
        operations=["open_pr"],
    )

    class PRRemote(GitRemote):
        def _gh_output(self, *args, **kwargs):
            if "--method" in args:
                self.effects += 1
                raise DeliveryError("ACK lost")
            return subprocess.CompletedProcess(args, 0, "[]", "")

    git = PRRemote(tmp_path)
    git.remote_sha = "sha1"
    await channels.prepare_publication(
        source_id="gh",
        principal="7",
        conversation_id="github:o/r:1",
        operation_id="pr1",
        kind="open_pr",
        scope={"title": "Fix", "body": "Details"},
        expected_preview_id="preview",
        git=git,
    )
    assert (await channels.execute_publication("pr1", git=git))["status"] == "unknown"
    with pytest.raises(ValueError, match="reconcile"):
        await channels.execute_publication("pr1", git=git)
    assert (await channels.execute_publication("pr1", git=git, reconcile=True))["status"] == "unknown"
    assert git.effects == 1
    await channels.grant(
        source_id="gh",
        principal="7",
        conversation_id="github:o/r:1",
        thread_id=start.thread_id,
        role="owner",
        operations=["open_pr"],
        enabled=False,
    )
    with pytest.raises(PermissionError):
        await channels.execute_publication("pr1", git=git, reconcile=True)
    assert git.effects == 1


@pytest.mark.anyio
async def test_late_negative_reconcile_cannot_erase_success(tmp_path, monkeypatch):
    from agent_runtime.channels import publication
    from agent_runtime.harness import channel_ops as ch

    channels = Channels(database=tmp_path / "r.db", workspace=tmp_path)
    async with await RuntimeStore.open(channels.database) as store:
        start = await store.run_async(
            host_ops.receive_command, store, workspace=tmp_path, kind="start", payload={"text": "work"}
        )
    await channels.configure(
        start.thread_id,
        Source(
            source_id="gh",
            kind="github",
            repository="o/r",
            webhook_secret_env="SECRET",
            outbound_token_env="PUB_TOKEN",
            outbound_principal="99",
        ),
    )
    await channels.bind(
        source_id="gh", conversation_id="github:o/r:1", destination="github:o/r:1", thread_id=start.thread_id
    )
    await channels.grant(
        source_id="gh",
        principal="7",
        conversation_id="github:o/r:1",
        thread_id=start.thread_id,
        role="owner",
        operations=["push"],
    )
    git = GitRemote(tmp_path)
    await channels.prepare_publication(
        source_id="gh",
        principal="7",
        conversation_id="github:o/r:1",
        operation_id="push",
        kind="push",
        scope={},
        expected_preview_id="preview",
        git=git,
    )
    with pytest.raises(ValueError, match="no dispatched"):
        await channels.execute_publication("push", git=git, reconcile=True)
    async with await RuntimeStore.open(channels.database) as store:

        def mark_unknown():
            row = ch.get(store, "publication", "push")
            row["status"] = "unknown"
            ch.put(store, start.thread_id, "publication", "push", row)

        await store.run_async(mark_unknown)

    def concurrent_success(*args, **kwargs):
        with RuntimeStore.open_sync(channels.database) as other:
            row = ch.get(other, "publication", "push")
            row.update(status="succeeded", result={"head_sha": "sha1", "reconciled": True})
            ch.put(other, start.thread_id, "publication", "push", row)
        return None

    monkeypatch.setattr(publication, "remote_effect", concurrent_success)
    final = await channels.execute_publication("push", git=git, reconcile=True)
    assert final["status"] == "succeeded"
    assert final["result"]["head_sha"] == "sha1"


@pytest.mark.anyio
async def test_interrupted_publication_keeps_known_settled_result(tmp_path, monkeypatch):
    from agent_runtime.channels import publication
    from agent_runtime.git_delivery import DeliveryInterruptedError

    channels = Channels(database=tmp_path / "r.db", workspace=tmp_path)
    async with await RuntimeStore.open(channels.database) as store:
        start = await store.run_async(
            host_ops.receive_command, store, workspace=tmp_path, kind="start", payload={"text": "work"}
        )
    await channels.configure(
        start.thread_id,
        Source(
            source_id="gh",
            kind="github",
            repository="o/r",
            webhook_secret_env="SECRET",
            outbound_token_env="PUB_TOKEN",
            outbound_principal="99",
        ),
    )
    await channels.bind(
        source_id="gh", conversation_id="github:o/r:1", destination="github:o/r:1", thread_id=start.thread_id
    )
    await channels.grant(
        source_id="gh",
        principal="7",
        conversation_id="github:o/r:1",
        thread_id=start.thread_id,
        role="owner",
        operations=["push"],
    )
    git = GitRemote(tmp_path)
    await channels.prepare_publication(
        source_id="gh",
        principal="7",
        conversation_id="github:o/r:1",
        operation_id="push",
        kind="push",
        scope={},
        expected_preview_id="preview",
        git=git,
    )
    calls = []

    async def settled_after_cancellation(operation):
        calls.append(operation)
        raise DeliveryInterruptedError({"head_sha": "sha1"})

    monkeypatch.setattr(publication, "_settle_delivery", settled_after_cancellation)
    result = await channels.execute_publication("push", git=git)
    assert result["status"] == "succeeded"
    assert result["result"]["head_sha"] == "sha1"
    assert len(calls) == 1


@pytest.mark.anyio
async def test_interrupted_fallback_keeps_positive_reconciliation(tmp_path, monkeypatch):
    from agent_runtime.channels import publication
    from agent_runtime.git_delivery import DeliveryInterruptedError

    channels = Channels(database=tmp_path / "r.db", workspace=tmp_path)
    async with await RuntimeStore.open(channels.database) as store:
        start = await store.run_async(
            host_ops.receive_command, store, workspace=tmp_path, kind="start", payload={"text": "work"}
        )
    await channels.configure(
        start.thread_id,
        Source(
            source_id="gh",
            kind="github",
            repository="o/r",
            webhook_secret_env="SECRET",
            outbound_token_env="PUB_TOKEN",
            outbound_principal="99",
        ),
    )
    await channels.bind(
        source_id="gh", conversation_id="github:o/r:1", destination="github:o/r:1", thread_id=start.thread_id
    )
    await channels.grant(
        source_id="gh",
        principal="7",
        conversation_id="github:o/r:1",
        thread_id=start.thread_id,
        role="owner",
        operations=["push"],
    )
    git = GitRemote(tmp_path)
    await channels.prepare_publication(
        source_id="gh",
        principal="7",
        conversation_id="github:o/r:1",
        operation_id="push",
        kind="push",
        scope={},
        expected_preview_id="preview",
        git=git,
    )
    calls = []

    async def settlement(operation):
        calls.append(operation)
        if len(calls) == 1:
            raise RuntimeError("ACK unknown")
        raise DeliveryInterruptedError({"head_sha": "sha1", "reconciled": True})

    monkeypatch.setattr(publication, "_settle_delivery", settlement)
    result = await channels.execute_publication("push", git=git)
    assert result["status"] == "succeeded"
    assert result["result"]["reconciled"] is True
    assert len(calls) == 2
