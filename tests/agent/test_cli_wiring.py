from __future__ import annotations

import json
from pathlib import Path

import click
import pytest
from pydantic import ValidationError
from typer.main import get_command
from typer.testing import CliRunner

from agent_runtime import RAGKnowledgeConfig
from agent_runtime import cli as cli_module
from agent_runtime.agent import Agent
from agent_runtime.cli import (
    _CLIToolEventDisplay,
    _display_agent_result,
    _load_knowledge_config,
    agent_app,
)
from agent_runtime.harness import RolloutStore
from agent_runtime.planning import AgentPlan, PlanEvent, PlanStep
from agent_runtime.result import AgentPause, AgentResult, AgentToolCall, AgentUsage
from agent_runtime.streaming.events import (
    EventType,
    StreamEvent,
    recovery_event,
    text_delta,
    tool_use_error,
    tool_use_progress,
    tool_use_result,
    tool_use_start,
)


def test_cli_defaults_use_praxis_runtime_paths() -> None:
    assert cli_module.DEFAULT_MODEL_SESSION_PATH == Path(".praxis/model_session.json")
    assert cli_module.DEFAULT_CHECKPOINT_PATH == Path(".praxis/checkpoints.sqlite")


def test_cli_defaults_leave_legacy_rag_agent_state_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    legacy_root = tmp_path / ".rag"
    legacy_checkpoint = legacy_root / "agent_checkpoints.sqlite"
    legacy_session = legacy_root / "agent_model_session.json"
    legacy_session_text = '{"current_model_id":"legacy-sentinel"}'
    legacy_root.mkdir()
    legacy_checkpoint.write_bytes(b"legacy checkpoint sentinel")
    legacy_session.write_text(legacy_session_text, encoding="utf-8")

    agent = Agent()
    assert agent.checkpoint_db == Path(".praxis/checkpoints.sqlite")
    assert agent.model_session_path == Path(".praxis/model_session.json")
    try:
        current = agent.current_model()
        switched = agent.switch_model(current.id)

        assert switched.id == current.id
        assert not cli_module.DEFAULT_CHECKPOINT_PATH.exists()
        assert json.loads(cli_module.DEFAULT_MODEL_SESSION_PATH.read_text(encoding="utf-8")) == {
            "current_model_id": current.id
        }
        assert legacy_checkpoint.read_bytes() == b"legacy checkpoint sentinel"
        assert legacy_session.read_text(encoding="utf-8") == legacy_session_text
    finally:
        if agent._model_control_plane is not None:
            agent._model_control_plane.close()


@pytest.mark.parametrize(
    ("command", "present", "removed"),
    [
        pytest.param(
            "run",
            (
                "--previous-turn-id",
                "--last",
                "--knowledge-config",
                "--file",
                "--max-tokens-total",
                "--allow-write-tools",
                "--allow-execute-tools",
                "--require-workspace-change",
                "--no-require-workspace-change",
                "--model-session-path",
            ),
            (
                "--agent",
                "--turn-id",
                "--run-id",
                "--knowledge",
                "--input-file",
                "--tool",
                "--disable-tool",
                "--allow-discovery-tools",
                "--budget",
                "--vector-dsn",
            ),
            id="run",
        ),
        pytest.param(
            "chat",
            (
                "--previous-turn-id",
                "--last",
                "--knowledge-config",
                "--max-tokens-total",
            ),
            (
                "--agent",
                "--budget",
                "--vector-dsn",
                "--storage-root",
                "--embedding-model",
                "--reranker-model",
            ),
            id="chat",
        ),
        pytest.param(
            "resume",
            ("--last", "--all", "--action", "--input"),
            ("--decision", "--vector-dsn"),
            id="resume",
        ),
    ],
)
def test_agent_command_options_match_the_clean_public_contract(
    command: str,
    present: tuple[str, ...],
    removed: tuple[str, ...],
) -> None:
    root_command = get_command(agent_app)
    command_info = root_command.get_command(click.Context(root_command), command)

    assert command_info is not None
    option_names = {
        option
        for parameter in command_info.params
        for option in (*parameter.opts, *parameter.secondary_opts)
    }
    for option in present:
        assert option in option_names
    for option in removed:
        assert option not in option_names


def test_workspace_change_help_discloses_post_change_verification_gate() -> None:
    result = CliRunner().invoke(
        agent_app,
        ["run", "--help"],
        env={"COLUMNS": "240"},
    )

    assert result.exit_code == 0
    assert "最后一次真实变更后" in result.output
    assert "测试、lint、类型检查或构建" in result.output


def test_agent_run_rejects_missing_input_without_internal_traceback(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.txt"

    result = CliRunner().invoke(
        agent_app,
        [
            "run",
            "Read the file.",
            "--file",
            str(missing),
            "--non-interactive",
        ],
        env={"COLUMNS": "240"},
    )

    assert result.exit_code == 2
    assert "输入文件不存在" in result.output
    assert "Traceback" not in result.output


def test_agent_run_can_store_model_session_outside_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facade_options: list[dict[str, object]] = []

    class _Facade:
        async def arun(self, _task: str, **_kwargs: object) -> AgentResult:
            return _result()

    def create_facade(**kwargs: object) -> _Facade:
        facade_options.append(kwargs)
        return _Facade()

    monkeypatch.setattr(cli_module, "_create_agent_facade", create_facade)
    model_session_path = tmp_path / "artifacts" / "model-session.json"

    cli_module.agent_run(
        task="Inspect the repository.",
        checkpoint_db=tmp_path / "artifacts" / "checkpoints.sqlite",
        model_session_path=model_session_path,
        non_interactive=True,
    )

    assert facade_options[0]["model_session_path"] == model_session_path


def test_agent_run_can_disable_workspace_mcp_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facade_options: list[dict[str, object]] = []

    class _Facade:
        async def arun(self, _task: str, **_kwargs: object) -> AgentResult:
            return _result()

    def create_facade(**kwargs: object) -> _Facade:
        facade_options.append(kwargs)
        return _Facade()

    monkeypatch.setattr(cli_module, "_create_agent_facade", create_facade)

    cli_module.agent_run(
        task="Evaluate built-in tools only.",
        checkpoint_db=tmp_path / "checkpoints.sqlite",
        model_session_path=tmp_path / "model-session.json",
        disable_workspace_mcp=True,
        non_interactive=True,
    )

    assert facade_options[0]["enable_workspace_mcp"] is False


def test_agent_chat_restores_the_previous_turn_runtime_before_model_switching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_workspace = tmp_path / "caller"
    turn_workspace = tmp_path / "turn-workspace"
    caller_workspace.mkdir()
    turn_workspace.mkdir()
    database = tmp_path / "agent.sqlite"
    knowledge = RAGKnowledgeConfig(
        storage_root=tmp_path / "knowledge",
        vector_backend="sqlite",
    )
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=turn_workspace)
        previous = store.start_turn(
            thread_id=thread.thread_id,
            user_message="remember cobalt",
            binding_manifest={
                "model_alias": "qwen3_5_9b_mlx_4bit",
                "knowledge_config": knowledge.model_dump(mode="json"),
            },
        )
        previous = store.complete_turn(
            turn_id=previous.turn_id,
            answer="remembered cobalt",
        )
    facade_options: list[dict[str, object]] = []
    loop_options: list[dict[str, object]] = []

    def create_facade(**kwargs: object) -> object:
        facade_options.append(kwargs)
        return object()

    async def chat_loop(_facade: object, **kwargs: object) -> None:
        loop_options.append(kwargs)

    monkeypatch.chdir(caller_workspace)
    monkeypatch.setattr(cli_module, "_create_agent_facade", create_facade)
    monkeypatch.setattr(cli_module, "_chat_facade_loop", chat_loop)

    cli_module.agent_chat(
        previous_turn_id=previous.turn_id,
        checkpoint_db=database,
    )

    assert facade_options == [
        {
            "model": "qwen3_5_9b_mlx_4bit",
            "checkpoint_db": database,
            "workspace_path": str(turn_workspace.resolve()),
            "model_session_path": cli_module.DEFAULT_MODEL_SESSION_PATH,
            "knowledge": knowledge,
        }
    ]
    assert loop_options[0]["previous_turn_id"] == previous.turn_id


def test_agent_run_forwards_workspace_change_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_options: list[dict[str, object]] = []

    class _Facade:
        async def arun(self, _task: str, **kwargs: object) -> AgentResult:
            run_options.append(kwargs)
            return _result()

    monkeypatch.setattr(
        cli_module,
        "_create_agent_facade",
        lambda **_kwargs: _Facade(),
    )

    cli_module.agent_run(
        task="Fix the implementation.",
        checkpoint_db=tmp_path / "checkpoints.sqlite",
        model_session_path=tmp_path / "model-session.json",
        non_interactive=True,
    )

    assert run_options[0]["require_workspace_change"] is True

    cli_module.agent_run(
        task="Explain the implementation.",
        checkpoint_db=tmp_path / "checkpoints.sqlite",
        model_session_path=tmp_path / "model-session.json",
        require_workspace_change=False,
        non_interactive=True,
    )

    assert run_options[1]["require_workspace_change"] is False


def _result(
    *,
    status: str = "done",
    answer: str | None = None,
    tool_calls: tuple[AgentToolCall, ...] = (),
    plan: AgentPlan | None = None,
    plan_events: tuple[PlanEvent, ...] = (),
    pause: AgentPause | None = None,
    needs_user_input: str | None = None,
) -> AgentResult:
    return AgentResult(
        answer=answer,
        status=status,  # type: ignore[arg-type]
        files=(),
        tool_calls=tool_calls,
        evidence=(),
        citations=(),
        usage=AgentUsage(),
        diagnostics=(),
        turn_id="turn-test",
        stop_reason=None,
        pause=pause,
        workspace_path=None,
        groundedness=False,
        insufficient_evidence=False,
        plan=plan,
        plan_events=plan_events,
        needs_user_input=needs_user_input,
    )


def test_cli_shows_called_tool_names_without_verbose(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _display_agent_result(
        _result(
            tool_calls=(
                AgentToolCall(
                    tool_call_id="call_search",
                    tool_name="search_text",
                ),
            ),
        ),
        verbose=False,
    )

    assert "✓ search_text" in capsys.readouterr().out


def test_cli_does_not_repeat_an_answer_that_was_already_streamed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _display_agent_result(
        _result(answer="already visible"),
        verbose=False,
        answer_streamed=True,
    )

    assert "already visible" not in capsys.readouterr().out


def test_cli_shows_untyped_pause_reason_exactly_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    reason = "Choose a target branch before continuing."

    _display_agent_result(
        _result(
            status="paused",
            needs_user_input=reason,
        ),
        verbose=False,
    )

    output = capsys.readouterr().out
    assert f"暂停原因: {reason}" in output
    assert output.count(reason) == 1


def test_cli_leaves_typed_pause_question_to_pause_handler(
    capsys: pytest.CaptureFixture[str],
) -> None:
    question = "Allow apply_patch?"

    _display_agent_result(
        _result(
            status="paused",
            pause=AgentPause(
                request_id="request-approval",
                kind="tool_approval",
                question=question,
            ),
            needs_user_input=question,
        ),
        verbose=False,
    )

    assert question not in capsys.readouterr().out


def test_cli_shows_the_persisted_update_plan_without_verbose(
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = AgentPlan(
        objective="Ship durable plans.",
        revision=3,
        active_step_id="step_verify",
        steps=[
            PlanStep(
                step_id="step_store",
                title="Store the plan",
                status="completed",
            ),
            PlanStep(
                step_id="step_verify",
                title="Verify CLI exposure",
                status="in_progress",
            ),
        ],
    )
    event = PlanEvent(
        event_id="plan_event_cli",
        event_type="llm_update",
        plan_revision=plan.revision,
        message="Applied update_plan tool update.",
    )

    _display_agent_result(
        _result(
            status="paused",
            plan=plan,
            plan_events=(event,),
        ),
        verbose=False,
    )

    output = capsys.readouterr().out
    assert "计划 (revision 3)" in output
    assert "✓ Store the plan" in output
    assert "→ Verify CLI exposure" in output


@pytest.mark.anyio
async def test_cli_displays_canonical_tool_start_with_bounded_preview(
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _CLIToolEventDisplay().emit(
        tool_use_start(
            "read_file",
            "call_read",
            input_preview="path='src/service.py'",
        )
    )

    assert "→ read_file: path='src/service.py'" in capsys.readouterr().out


@pytest.mark.anyio
async def test_cli_displays_one_start_for_resumed_tool_call(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = _CLIToolEventDisplay()
    event = tool_use_start(
        "apply_patch",
        "call_patch",
        input_preview="file_path='notes.txt'",
    )

    await display.emit(event)
    await display.emit(event)

    assert capsys.readouterr().out.count("→ apply_patch") == 1


@pytest.mark.anyio
async def test_cli_streams_text_deltas_without_inserting_newlines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = _CLIToolEventDisplay()

    await display.emit(text_delta("hello"))
    await display.emit(text_delta(" world"))

    assert capsys.readouterr().out == "hello world"
    assert display.answer_streamed is True

    display.begin_turn()

    assert display.answer_streamed is False


@pytest.mark.anyio
async def test_cli_displays_correlated_tool_lifecycle_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = _CLIToolEventDisplay()
    start = tool_use_start(
        "read_file",
        "call_read",
        input_preview="path='src/service.py'",
    )
    result = tool_use_result(
        "read_file",
        "call_read",
        {"path": "src/service.py", "size_bytes": 420},
    )

    await display.emit(start)
    await display.emit(tool_use_progress("call_read", "reading", percent=50))
    await display.emit(result)
    await display.emit(result)

    output = capsys.readouterr().out
    assert "→ read_file: path='src/service.py'" in output
    assert "… read_file: reading (50%)" in output
    assert "✓ read_file:" in output
    assert "size_bytes" in output
    assert output.count("✓ read_file:") == 1


@pytest.mark.anyio
async def test_cli_displays_correlated_tool_error_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = _CLIToolEventDisplay()
    error = tool_use_error("call_read", "file not found")

    await display.emit(tool_use_start("read_file", "call_read"))
    await display.emit(error)
    await display.emit(error)

    output = capsys.readouterr().out
    assert "✗ read_file: file not found" in output
    assert output.count("✗ read_file:") == 1


@pytest.mark.anyio
async def test_cli_displays_patch_diff_from_existing_result_event(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = _CLIToolEventDisplay()
    event = StreamEvent(
        type=EventType.TOOL_USE_RESULT,
        data={
            "tool_name": "apply_patch",
            "tool_id": "call_patch",
            "result": {"replaced": True},
            "details": {
                "file_path": "src/example.py",
                "diff": ("--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n-old\n+new"),
                "diff_truncated": False,
            },
        },
    )

    await display.emit(event)

    output = capsys.readouterr().out
    assert "✓ apply_patch:" in output
    assert "--- a/src/example.py" in output
    assert "+++ b/src/example.py" in output
    assert "-old" in output
    assert "+new" in output


@pytest.mark.anyio
async def test_cli_displays_plan_and_recovery_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = _CLIToolEventDisplay()
    plan_event = StreamEvent(
        type=EventType.PLAN_UPDATED,
        data={
            "plan": {
                "revision": 2,
                "steps": [
                    {"title": "Inspect source", "status": "completed"},
                    {"title": "Wire CLI", "status": "in_progress"},
                ],
            }
        },
    )

    await display.emit(plan_event)
    await display.emit(plan_event)
    await display.emit(recovery_event("model_retry", "attempt 2 of 3"))

    output = capsys.readouterr().out
    assert output.count("计划 (revision 2)") == 1
    assert "✓ Inspect source" in output
    assert "→ Wire CLI" in output
    assert "↻ 恢复: model_retry — attempt 2 of 3" in output


def test_knowledge_config_is_serializable_and_forbids_unknown_fields(
    tmp_path: Path,
) -> None:
    config = RAGKnowledgeConfig(
        storage_root=tmp_path / "knowledge",
        embedding_model="embed-v1",
        vector_backend="sqlite",
        vector_namespace="docs",
    )

    restored = RAGKnowledgeConfig.model_validate_json(config.model_dump_json())

    assert restored == config
    with pytest.raises(ValidationError):
        RAGKnowledgeConfig.model_validate({"storage_root": ".rag", "source_name": "legacy"})


def test_cli_loads_one_explicit_yaml_knowledge_config(tmp_path: Path) -> None:
    path = tmp_path / "knowledge.yaml"
    path.write_text(
        "storage_root: /tmp/index\nvector_backend: sqlite\n",
        encoding="utf-8",
    )

    config = _load_knowledge_config(path)

    assert config == RAGKnowledgeConfig(
        storage_root=Path("/tmp/index"),
        vector_backend="sqlite",
    )
