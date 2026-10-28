"""agent watch: CI observation that hands failures to the workspace's agent."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from agent_runtime.cli import _watch_command
from agent_runtime.git_delivery import CheckRun, GitDelivery, PullRequestChecks
from agent_runtime.harness import RuntimeState
from agent_runtime.result import AgentResult, AgentUsage

pytestmark = pytest.mark.anyio


def _checks_report(overall: str, checks: tuple[CheckRun, ...],
                   logs: tuple[tuple[str, str], ...] = ()) -> PullRequestChecks:
    return PullRequestChecks(
        branch="praxis/x",
        pull_request_url="https://github.com/o/r/pull/5",
        checks=checks,
        failure_logs=logs,
    )


class _FakeWatchSession:
    def __init__(self, submitted: list[tuple[str | None, str]], answer: AgentResult) -> None:
        self._submitted = submitted
        self._answer = answer
        self.thread_id: str | None = None

    async def __aenter__(self) -> _FakeWatchSession:
        return self

    async def __aexit__(self, *exception: object) -> None:
        return None

    async def submit(self, message: str, **kwargs: object) -> AgentResult:
        self._submitted.append((self.thread_id, message))
        return self._answer


class _FakeWatchFacade:
    def __init__(self, submitted: list[tuple[str | None, str]], answer: AgentResult) -> None:
        self._submitted = submitted
        self._answer = answer
        self.session_options: dict[str, object] = {}

    def session(self, **kwargs: object) -> _FakeWatchSession:
        self.session_options = dict(kwargs)
        inner = _FakeWatchSession(self._submitted, self._answer)
        inner.thread_id = kwargs.get("thread_id")
        return inner


def _result() -> AgentResult:
    return AgentResult(
        answer="已修复超时处理。", status="done", files=(), tool_calls=(), evidence=(),
        citations=(), usage=AgentUsage(), diagnostics=(), turn_id="turn_ci",
        stop_reason=None, pause=None, workspace_path=None, groundedness=False,
        insufficient_evidence=False, plan=None, plan_events=(),
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("base\n")
    return root


def _seed_thread(workspace: Path, database: Path) -> str:
    with RuntimeState.open_sync(database) as store:
        thread = store.create_thread(workspace=workspace)
        turn_id = store.start_turn(
            thread_id=thread.thread_id,
            user_message="修复超时",
            binding_manifest={"model_id": "test-model"},
        ).turn_id
        store.complete_turn(turn_id=turn_id, answer="done")
        return thread.thread_id


def _patch_delivery(
    monkeypatch: MonkeyPatch,
    *,
    current: str,
    latest: str | None,
    report: PullRequestChecks,
) -> None:
    monkeypatch.setattr(GitDelivery, "current_branch", lambda self: current)
    monkeypatch.setattr(GitDelivery, "latest_delivery_branch", lambda self: latest)

    async def pull_request_checks(self, branch=None, **kwargs):
        return report

    monkeypatch.setattr(GitDelivery, "pull_request_checks", pull_request_checks)


async def test_watch_pass_reports_success_without_agent(
    workspace: Path, tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    _patch_delivery(
        monkeypatch, current="praxis/x", latest=None,
        report=_checks_report("pass", (CheckRun("build", "pass", "CI", "u"),)),
    )
    submitted: list[tuple[str | None, str]] = []
    monkeypatch.setattr(
        "agent_runtime.cli._create_agent_facade",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("pass must not open a model")),
    )
    code = await _watch_command(
        workspace=workspace, checkpoint_db=tmp_path / "rollout.sqlite3", permissions=None,
        approval_mode="auto", branch=None, thread_id=None,
        timeout_seconds=60, poll_interval_seconds=5, fix=True,
    )
    assert code == 0
    assert submitted == []


async def test_watch_failure_hands_report_to_latest_thread(
    workspace: Path, tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    _patch_delivery(
        monkeypatch, current="main", latest="praxis/x",
        report=_checks_report(
            "fail",
            (CheckRun("build", "fail", "CI", "u"),),
            logs=(("CI", "AssertionError: boom"),),
        ),
    )
    database = tmp_path / "rollout.sqlite3"
    thread_id = (await asyncio.to_thread(_seed_thread, workspace, database))
    submitted: list[tuple[str | None, str]] = []
    monkeypatch.setattr(
        "agent_runtime.cli._create_agent_facade",
        lambda **kwargs: _FakeWatchFacade(submitted, _result()),
    )
    code = await _watch_command(
        workspace=workspace, checkpoint_db=database, permissions=None,
        approval_mode="auto", branch=None, thread_id=None,
        timeout_seconds=60, poll_interval_seconds=5, fix=True,
    )
    assert code == 1
    assert len(submitted) == 1
    resumed_thread, message = submitted[0]
    assert resumed_thread == thread_id
    assert message.startswith("CI 失败")
    assert "AssertionError: boom" in message
    assert "/commit" in message


async def test_watch_no_fix_only_reports(
    workspace: Path, tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    _patch_delivery(
        monkeypatch, current="praxis/x", latest=None,
        report=_checks_report("fail", (CheckRun("test", "fail", "CI", "u"),)),
    )
    submitted: list[tuple[str | None, str]] = []
    code = await _watch_command(
        workspace=workspace, checkpoint_db=tmp_path / "rollout.sqlite3", permissions=None,
        approval_mode="auto", branch=None, thread_id=None,
        timeout_seconds=60, poll_interval_seconds=5, fix=False,
    )
    assert code == 1
    assert submitted == []


async def test_watch_without_praxis_branch_is_refused(
    workspace: Path, tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    _patch_delivery(
        monkeypatch, current="main", latest=None,
        report=_checks_report("pass", ()),
    )
    code = await _watch_command(
        workspace=workspace, checkpoint_db=tmp_path / "rollout.sqlite3", permissions=None,
        approval_mode="auto", branch=None, thread_id=None,
        timeout_seconds=60, poll_interval_seconds=5, fix=True,
    )
    assert code == 2


async def test_watch_pending_timeout_keeps_waiting_state(
    workspace: Path, tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    _patch_delivery(
        monkeypatch, current="praxis/x", latest=None,
        report=_checks_report("pending", (CheckRun("build", "pending", "CI", "u"),)),
    )
    monkeypatch.setattr(
        "agent_runtime.cli._create_agent_facade",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("pending must not open a model")),
    )
    code = await _watch_command(
        workspace=workspace, checkpoint_db=tmp_path / "rollout.sqlite3", permissions=None,
        approval_mode="auto", branch=None, thread_id=None,
        timeout_seconds=60, poll_interval_seconds=5, fix=True,
    )
    assert code == 2


async def test_watch_failure_without_thread_reports_guidance(
    workspace: Path, tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    _patch_delivery(
        monkeypatch, current="praxis/x", latest=None,
        report=_checks_report("fail", (CheckRun("build", "fail", "CI", "u"),)),
    )
    submitted: list[tuple[str | None, str]] = []
    code = await _watch_command(
        workspace=workspace, checkpoint_db=tmp_path / "missing.sqlite3", permissions=None,
        approval_mode="auto", branch=None, thread_id=None,
        timeout_seconds=60, poll_interval_seconds=5, fix=True,
    )
    assert code == 2
    assert submitted == []
