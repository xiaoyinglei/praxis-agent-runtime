from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from agent_runtime.agent import Agent
from agent_runtime.harness import RolloutStore, Session
from agent_runtime.models import ModelSessionState, ModelSpec
from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel


def _local_spec(model_id: str) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        provider="local_mlx_chat_8080",
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

    sentinel = object()

    class FakeSession:
        async def submit(self, *args, **kwargs):
            return sentinel

    @asynccontextmanager
    async def fake_session(**kwargs):
        opened_with.append(dict(kwargs))
        yield FakeSession()

    monkeypatch.setattr(agent, "session", fake_session)

    result = await agent.run(
        "Inspect the repository.",
        require_workspace_change=False,
    )

    assert result is sentinel
    assert len(opened_with) == 1

    # Critical lifecycle contract:
    # run() creates a new Turn, so it must not request
    # frozen-Turn provider bootstrap.
    assert opened_with[0].get("frozen_turn_id") is None


@pytest.mark.anyio
async def test_model_bootstrap_uses_current_selected_model_for_new_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_spec = _local_spec("current-model")

    class FakeControlPlane:
        state = ModelSessionState(current_model_id="current-model")
        def ensure_model_binding_trust(self, *, has_existing_bindings: bool) -> None:
            assert isinstance(has_existing_bindings, bool)

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
        "agent_runtime.local_runtime.ensure_local_provider_ready",
        fake_ensure_ready,
    )

    monkeypatch.setattr(agent, "_harness_model", PublicHarnessModel)
    async with agent.session(require_workspace_change=False):
        pass

    assert ready == [current_spec]


@pytest.mark.anyio
async def test_resume_bootstraps_only_the_operation_binding_when_prepared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    database = tmp_path / "rollout.sqlite3"

    frozen_binding = {
        "authentication_schema_version": 2,
        "model_id": "frozen-model",
        "test_marker": "frozen-binding",
    }

    with RolloutStore(database) as store:
        thread = store.create_thread(
            workspace=workspace,
        )

        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="Original task",
            binding_manifest=frozen_binding,
        )

    frozen_spec = _local_spec("frozen-model")

    reviewed: list[tuple[dict[str, object], str, str]] = []

    class FakeControlPlane:
        state = ModelSessionState(current_model_id="current-model")
        def ensure_model_binding_trust(self, *, has_existing_bindings: bool) -> None:
            assert isinstance(has_existing_bindings, bool)

        def current_model(self) -> ModelSpec:
            raise AssertionError("resume bootstrap must not use the current selected model")

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
        "agent_runtime.local_runtime.ensure_local_provider_ready",
        fake_ensure_ready,
    )

    monkeypatch.setattr(agent, "_harness_model", PublicHarnessModel)
    operation_binding = {**frozen_binding, "thread_id": thread.thread_id, "turn_id": turn.turn_id}
    async with await Session.open(agent=agent, frozen_turn_id=turn.turn_id) as session:
        assert ready == []
        assert reviewed == []
        await session._prepare_step_binding(operation_binding)

    assert ready == [frozen_spec]

    assert reviewed == [
        (
            operation_binding,
            thread.thread_id,
            turn.turn_id,
        )
    ]


@pytest.mark.anyio
async def test_provider_bootstrap_uses_session_settings_and_is_reused_across_submissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_runtime.harness import session as session_module

    agent = Agent(
        checkpoint_db=tmp_path / "rollout.sqlite3",
        workspace_path=tmp_path,
        model_session_path=None,
        enable_workspace_mcp=False,
    )
    lifecycle = []

    async def bootstrap(spec):
        assert spec.id
        lifecycle.append("provider_ready")

    configure = session_module.configure_services

    def configure_after_bootstrap(session, **kwargs):
        assert lifecycle == []
        configure(session, **kwargs)
        lifecycle.append("services_created")

    monkeypatch.setattr("agent_runtime.local_runtime.ensure_local_provider_ready", bootstrap)
    monkeypatch.setattr(agent, "_harness_model", PublicHarnessModel)
    monkeypatch.setattr(session_module, "configure_services", configure_after_bootstrap)
    async with agent.session(require_workspace_change=False) as session:
        assert lifecycle == ["services_created", "provider_ready"]
        await session.submit("first")
        await session.submit("second")
        assert lifecycle == ["services_created", "provider_ready"]
    assert session._closed
