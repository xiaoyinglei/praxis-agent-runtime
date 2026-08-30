import json
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch
from typer.testing import CliRunner

import agent_runtime.cli as agent_cli
import rag.cli as cli
from agent_runtime.cli import agent_app
from rag import StorageConfig
from rag.cli import app
from rag.retrieval.models import BuiltContext, PublicQueryResult
from rag.schema.query import GroundedAnswer
from rag.schema.runtime import RetrievalDiagnostics
from tests.support import make_runtime

runner = CliRunner()
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _plain_help(output: str) -> str:
    return _ANSI_RE.sub("", output)


def _use_isolated_cli_runtime(monkeypatch: MonkeyPatch) -> None:
    def _runtime(
        storage_root: Path,
        *,
        require_chat: bool = False,
        require_rerank: bool = False,
        model: str | None = None,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
        vector_backend: str = "milvus",
        vector_dsn: str | None = None,
        vector_namespace: str | None = None,
        vector_collection_prefix: str | None = None,
    ):
        del (
            model,
            embedding_model,
            reranker_model,
            require_rerank,
            vector_backend,
            vector_dsn,
            vector_namespace,
            vector_collection_prefix,
        )
        return make_runtime(storage=StorageConfig(root=storage_root), require_chat=require_chat)

    monkeypatch.setattr(cli, "_runtime", _runtime)


def test_cli_ingest_query_round_trip(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _use_isolated_cli_runtime(monkeypatch)
    storage_root = tmp_path / ".rag"

    ingest = runner.invoke(
        app,
        [
            "ingest",
            "--storage-root",
            str(storage_root),
            "--source-type",
            "plain_text",
            "--location",
            "memory://note-1",
            "--content",
            "Alpha Engine handles ingestion. Beta Service depends on Alpha Engine.",
        ],
    )

    query = runner.invoke(
        app,
        [
            "query",
            "--storage-root",
            str(storage_root),
            "--query",
            "What does Alpha Engine handle?",
            "--json",
        ],
    )
    payload = json.loads(query.stdout)

    assert ingest.exit_code == 0
    assert query.exit_code == 0
    assert payload["answer"]["answer_text"]
    assert payload["context"]["evidence"]


def test_cli_delete_and_rebuild_use_new_runtime_contract(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _use_isolated_cli_runtime(monkeypatch)
    storage_root = tmp_path / ".rag"

    ingest = runner.invoke(
        app,
        [
            "ingest",
            "--storage-root",
            str(storage_root),
            "--source-type",
            "plain_text",
            "--location",
            "memory://note-1",
            "--content",
            "Alpha Engine handles ingestion.",
        ],
    )
    delete = runner.invoke(
        app,
        [
            "delete",
            "--storage-root",
            str(storage_root),
            "--location",
            "memory://note-1",
        ],
    )
    rebuild = runner.invoke(
        app,
        [
            "rebuild",
            "--storage-root",
            str(storage_root),
            "--location",
            "memory://note-1",
        ],
    )

    assert ingest.exit_code == 0
    assert delete.exit_code == 0
    delete_payload = json.loads(delete.stdout)
    assert delete_payload["deleted_doc_ids"]
    assert delete_payload["deleted_source_ids"]
    assert delete_payload["deleted_vector_count"] >= 2
    assert rebuild.exit_code == 0
    rebuild_payload = json.loads(rebuild.stdout)
    assert rebuild_payload["rebuilt_doc_ids"] == delete_payload["deleted_doc_ids"]
    assert rebuild_payload["results"][0]["indexed_object_count"] >= 2


def test_cli_rejects_missing_source_payload_for_ingest(tmp_path: Path) -> None:
    storage_root = tmp_path / ".rag"

    result = runner.invoke(
        app,
        [
            "ingest",
            "--storage-root",
            str(storage_root),
            "--source-type",
            "plain_text",
            "--location",
            "memory://note-1",
        ],
    )

    assert result.exit_code != 0
    assert "content" in result.stdout.lower() or "content" in result.stderr.lower()


def test_cli_main_delegates_to_typer_app(monkeypatch: MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeApp:
        def __call__(self) -> None:
            calls.append("called")

    monkeypatch.setattr(cli, "app", FakeApp())

    cli.main()

    assert calls == ["called"]


def test_project_metadata_exposes_agent_as_primary_console_script() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["name"] == "praxis-agent-runtime"
    assert pyproject["project"]["scripts"]["agent"] == "agent_runtime.cli:agent_app"
    assert pyproject["project"]["scripts"]["rag"] == "rag.cli:app"


def test_assembly_profile_cli_surface_is_removed() -> None:
    help_env = {"COLUMNS": "240"}
    root_help = runner.invoke(app, ["--help"], env=help_env)
    query_help = runner.invoke(app, ["query", "--help"], env=help_env)
    agent_run_help = runner.invoke(agent_app, ["run", "--help"], env=help_env)
    agent_chat_help = runner.invoke(agent_app, ["chat", "--help"], env=help_env)
    agent_resume_help = runner.invoke(agent_app, ["resume", "--help"], env=help_env)

    assert root_help.exit_code == 0
    assert query_help.exit_code == 0
    assert agent_run_help.exit_code == 0
    assert agent_chat_help.exit_code == 0
    assert agent_resume_help.exit_code == 0

    root_output = _plain_help(root_help.output)
    query_output = _plain_help(query_help.output)
    agent_run_output = _plain_help(agent_run_help.output)
    agent_chat_output = _plain_help(agent_chat_help.output)
    agent_resume_output = _plain_help(agent_resume_help.output)

    assert "profiles" not in root_output
    assert "--profile" not in query_output
    assert "--profile" not in agent_run_output
    assert "--profile" not in agent_chat_output
    assert "--profile" not in agent_resume_output

    for output in (agent_run_output, agent_chat_output):
        assert "--model" in output
    assert "--model" not in agent_resume_output
    for output in (agent_run_output, agent_chat_output, agent_resume_output):
        assert "--storage-root" not in output
        assert "--embedding-model" not in output
        assert "--reranker-model" not in output
        assert "--vector-backend" not in output
    for output in (agent_run_output, agent_chat_output, agent_resume_output):
        assert "--budget" not in output
    assert "--max-tokens-total" in agent_run_output
    assert "--max-tokens-total" in agent_chat_output
    assert "--max-tokens-total" not in agent_resume_output


def test_agent_cli_is_the_top_level_agent_entrypoint() -> None:
    help_env = {"COLUMNS": "240"}
    root_help = runner.invoke(agent_app, ["--help"], env=help_env)
    run_help = runner.invoke(agent_app, ["run", "--help"], env=help_env)
    chat_help = runner.invoke(agent_app, ["chat", "--help"], env=help_env)
    resume_help = runner.invoke(agent_app, ["resume", "--help"], env=help_env)

    assert root_help.exit_code == 0
    assert run_help.exit_code == 0
    assert chat_help.exit_code == 0
    assert resume_help.exit_code == 0

    root_output = _plain_help(root_help.output)
    assert "run" in root_output
    assert "chat" in root_output
    assert "resume" in root_output
    assert "model" in root_output
    assert "--agent" not in _plain_help(run_help.output)


def test_chat_model_switch_is_not_blocked_by_turn_lineage(capsys) -> None:
    switched: list[str] = []

    class Agent:
        def switch_model(self, model_id: str):
            switched.append(model_id)
            return SimpleNamespace(id=model_id)

    agent_cli._handle_model_slash_command(
        "/model switch qwen3_5_9b_mlx_4bit",
        agent=Agent(),  # type: ignore[arg-type]
    )

    assert switched == ["qwen3_5_9b_mlx_4bit"]
    assert "已切换模型: qwen3_5_9b_mlx_4bit" in capsys.readouterr().out


def test_rag_cli_no_longer_exposes_agent_or_analyze_task() -> None:
    root_help = runner.invoke(app, ["--help"], env={"COLUMNS": "240"})
    agent = runner.invoke(app, ["agent", "--help"])
    analyze = runner.invoke(app, ["analyze-task", "--help"])

    assert root_help.exit_code == 0
    root_output = _plain_help(root_help.output)
    assert "agent" not in root_output
    assert "analyze-task" not in root_output

    assert agent.exit_code != 0
    assert analyze.exit_code != 0


def test_cli_query_uses_public_query_contract(monkeypatch: MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    class _FakeRuntime:
        def __enter__(self) -> "_FakeRuntime":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def query(self, *_args, **_kwargs):
            raise AssertionError("cli query should not use runtime.query")

        def query_public(self, query_text: str, *, options=None) -> PublicQueryResult:
            calls.append((query_text, getattr(options, "retrieval_profile", None)))
            return PublicQueryResult(
                query=query_text,
                retrieval_profile="auto",
                answer=GroundedAnswer(
                    answer_text="Alpha answer",
                    groundedness_flag=True,
                    insufficient_evidence_flag=False,
                ),
                context=BuiltContext(
                    evidence=[],
                    token_budget=1200,
                    token_count=12,
                    grounded_candidate="alpha",
                    prompt="prompt",
                ),
                routing_decision={},
                retrieval_diagnostics=RetrievalDiagnostics(),
            )

    monkeypatch.setattr(cli, "_runtime", lambda *args, **kwargs: _FakeRuntime())

    result = runner.invoke(
        app,
        [
            "query",
            "--storage-root",
            ".rag",
            "--query",
            "What does Alpha Engine do?",
            "--retrieval-profile",
            "auto",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["answer"]["answer_text"] == "Alpha answer"
    assert "retrieval" not in payload
    assert calls == [("What does Alpha Engine do?", "auto")]


def test_cli_query_help_uses_new_retrieval_profile_option() -> None:
    result = runner.invoke(app, ["query", "--help"], env={"COLUMNS": "240"})

    assert result.exit_code == 0
    output = _plain_help(result.output)
    assert "--retrieval-profile" in output
    assert "--model" in output
    assert "--vector-collection-prefix" in output


def test_agent_run_help_exposes_clean_one_shot_options() -> None:
    result = runner.invoke(agent_app, ["run", "--help"], env={"COLUMNS": "240"})

    assert result.exit_code == 0
    output = _plain_help(result.output)
    assert "--agent" not in output
    assert "--input-file" not in output
    assert "--file" in output
    assert "--knowledge-config" in output
    assert "--max-tokens-total" in output
    assert "--vector-collection-prefix" not in output
    assert "--budget" not in output


def test_agent_resume_help_uses_persisted_runtime_metadata() -> None:
    result = runner.invoke(agent_app, ["resume", "--help"], env={"COLUMNS": "240"})

    assert result.exit_code == 0
    output = _plain_help(result.output)
    assert "--agent" not in output
    assert "--model" not in output
    assert "--checkpoint-db" in output
    assert "Turn ID" in output


def test_cli_benchmark_help_defaults_to_new_milvus_backend() -> None:
    help_env = {"COLUMNS": "240"}
    ingest_help = runner.invoke(app, ["benchmark-ingest", "--help"], env=help_env)
    evaluate_help = runner.invoke(app, ["benchmark-evaluate", "--help"], env=help_env)

    assert ingest_help.exit_code == 0
    assert evaluate_help.exit_code == 0
    ingest_output = _plain_help(ingest_help.output)
    evaluate_output = _plain_help(evaluate_help.output)
    assert "--retrieval-profile" in evaluate_output
    assert "--mode" not in evaluate_output
    assert "[default: milvus]" in ingest_output
    assert "[default: milvus]" in evaluate_output
