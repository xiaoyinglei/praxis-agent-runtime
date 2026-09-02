from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_runtime.agent import Agent
from agent_runtime.harness import RolloutStore
from agent_runtime.models import ModelSpec
from agent_runtime.result import AgentResult


def _local_spec(model_id: str) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        provider="local_mlx_chat_8080",
        provider_model=model_id,
        context_window=32_768,
        supports_tools=True,
        supports_structured_output=True,
        location="local",
    )


@pytest.mark.anyio
async def test_run_opens_runtime_without_frozen_turn_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A new Turn must bootstrap the current selected model.

    run() must not treat a predecessor/frozen Turn as the model
    authority for the new Turn.
    """
    agent = Agent(
        checkpoint_db=tmp_path / "rollout.sqlite3",
        workspace_path=tmp_path,
        model_session_path=None,
    )

    opened_with: list[dict[str, object]] = []

    class FakeThreadManager:
        async def run(
            self,
            **_kwargs: object,
        ) -> object:
            return object()

    fake_runtime = SimpleNamespace(
        thread_manager=FakeThreadManager(),
        store=object(),
    )

    @asynccontextmanager
    async def fake_open_runtime(
        **kwargs: object,
    ):
        opened_with.append(dict(kwargs))
        yield fake_runtime

    monkeypatch.setattr(
        agent,
        "_open_harness_runtime",
        fake_open_runtime,
    )

    sentinel = object()

    monkeypatch.setattr(
        AgentResult,
        "_from_harness",
        staticmethod(
            lambda *_args, **_kwargs: sentinel
        ),
    )

    result = await agent.run(
        "Inspect the repository.",
        require_workspace_change=False,
    )

    assert result is sentinel
    assert len(opened_with) == 1

    # Critical lifecycle contract:
    # run() creates a new Turn, so it must not request
    # frozen-Turn provider bootstrap.
    assert opened_with[0].get(
        "frozen_turn_id"
    ) is None


@pytest.mark.anyio
async def test_model_bootstrap_uses_current_selected_model_for_new_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_spec = _local_spec(
        "current-model"
    )

    class FakeControlPlane:
        def current_model(self) -> ModelSpec:
            return current_spec

    agent = Agent(
        checkpoint_db=tmp_path / "rollout.sqlite3",
        workspace_path=tmp_path,
        model_session_path=None,
    )

    monkeypatch.setattr(
        agent,
        "_get_model_control_plane",
        lambda: FakeControlPlane(),
    )

    ready: list[ModelSpec] = []

    async def fake_ensure_ready(
        spec: ModelSpec,
    ) -> None:
        ready.append(spec)

    monkeypatch.setattr(
        "agent_runtime.local_runtime."
        "ensure_local_provider_ready",
        fake_ensure_ready,
    )

    await agent._bootstrap_model_provider()

    assert ready == [current_spec]


@pytest.mark.anyio
async def test_model_bootstrap_uses_frozen_turn_model_for_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    database = tmp_path / "rollout.sqlite3"

    frozen_binding = {
    "authentication_schema_version": 1,
    "test_marker": "frozen-binding",}

    with RolloutStore(database) as store:
        thread = store.create_thread(
            workspace=workspace,
        )

        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="Original task",
            binding_manifest=frozen_binding,
        )

    frozen_spec = _local_spec(
        "frozen-model"
    )

    reviewed: list[
        tuple[dict[str, object], str, str]
    ] = []

    class FakeControlPlane:
        def current_model(self) -> ModelSpec:
            raise AssertionError(
                "resume bootstrap must not use "
                "the current selected model"
            )

        def model_spec_for_frozen_binding(
            self,
            binding: object,
            *,
            thread_id: str,
            turn_id: str,
        ) -> ModelSpec:
            assert isinstance(binding, Mapping)

            reviewed.append(
                (
                    dict(binding),
                    thread_id,
                    turn_id,
                )
            )

            return frozen_spec

    agent = Agent(
        checkpoint_db=database,
        workspace_path=workspace,
        model_session_path=None,
    )

    monkeypatch.setattr(
        agent,
        "_get_model_control_plane",
        lambda: FakeControlPlane(),
    )

    ready: list[ModelSpec] = []

    async def fake_ensure_ready(
        spec: ModelSpec,
    ) -> None:
        ready.append(spec)

    monkeypatch.setattr(
        "agent_runtime.local_runtime."
        "ensure_local_provider_ready",
        fake_ensure_ready,
    )

    await agent._bootstrap_model_provider(
        frozen_turn_id=turn.turn_id,
    )

    assert ready == [frozen_spec]

    assert reviewed == [
        (
            frozen_binding,
            thread.thread_id,
            turn.turn_id,
        )
    ]


@pytest.mark.anyio
async def test_provider_bootstrap_completes_before_runtime_composition_opens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Provider readiness belongs to async bootstrap.

    RuntimeComposition / Session construction must happen only
    after that bootstrap has completed.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    agent = Agent(
        checkpoint_db=tmp_path / "rollout.sqlite3",
        workspace_path=workspace,
        model_session_path=None,
        enable_workspace_mcp=False,
    )

    lifecycle: list[str] = []

    async def fake_bootstrap(
        *,
        frozen_turn_id: str | None = None,
    ) -> None:
        assert frozen_turn_id is None
        lifecycle.append(
            "provider_ready"
        )

    monkeypatch.setattr(
        agent,
        "_bootstrap_model_provider",
        fake_bootstrap,
    )

    # Avoid constructing the real model adapter.
    monkeypatch.setattr(
        agent,
        "_harness_model",
        lambda: object(),
    )

    class FakeRuntime:
        def close(self) -> None:
            lifecycle.append(
                "runtime_closed"
            )

    def fake_runtime_open(
        **_kwargs: object,
    ) -> FakeRuntime:
        # This is the important assertion:
        # RuntimeComposition cannot exist yet
        # if provider bootstrap has not completed.
        assert lifecycle == [
            "provider_ready"
        ]

        lifecycle.append(
            "runtime_opened"
        )

        return FakeRuntime()

    from agent_runtime.harness import (
        RuntimeComposition,
    )

    monkeypatch.setattr(
        RuntimeComposition,
        "open",
        fake_runtime_open,
    )

    async with agent._open_harness_runtime(
        require_workspace_change=False,
        allow_write_tools=False,
        allow_execute_tools=False,
        max_steps=1,
        max_tokens_total=None,
    ):
        lifecycle.append(
            "inside_runtime"
        )

    assert lifecycle == [
        "provider_ready",
        "runtime_opened",
        "inside_runtime",
        "runtime_closed",
    ]