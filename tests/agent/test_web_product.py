from __future__ import annotations

import base64
import json

import httpx
import pytest

from agent_runtime import Agent
from agent_runtime.harness import HarnessModelResponse, HarnessToolCall, RolloutStore
from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel


class FetchThenAnswer(PatchThenAnswerModel):
    def __init__(self, args, tool_name="web_fetch"):
        self.args = args
        self.tool_name = tool_name
        self.requests = []

    def prepare(self, request):
        self.requests.append(request)
        return super().prepare(request)

    async def dispatch(self, prepared):
        if prepared.request_ref["request_id"].endswith(":step:1"):
            return HarnessModelResponse(
                text="", provider_response_id="fetch-call", usage={},
                tool_calls=(HarnessToolCall(id="fetch-1", name=self.tool_name, arguments=self.args),),
            )
        return HarnessModelResponse(text="Research complete", provider_response_id="answer", usage={})


@pytest.mark.anyio
async def test_product_fetch_persists_source_and_reopens_without_network(tmp_path, monkeypatch):
    import agent_runtime.tools.web_http as http_module
    from agent_runtime.tools.web_http import PublicWebClient

    requests = []
    real_client = PublicWebClient

    def handler(request):
        requests.append(request)
        return httpx.Response(200, text="line one\nline two", headers={"content-type": "text/plain"})

    monkeypatch.setattr(http_module, "PublicWebClient", lambda **options: real_client(
        **options, transport=httpx.MockTransport(handler)))
    workspace = tmp_path / "workspace"
    database = tmp_path / "checkpoints.sqlite"
    model = FetchThenAnswer({"url": "https://example.com/doc", "max_lines": 1})
    agent = Agent(workspace_path=workspace, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_harness_model", lambda: model)
    result = await agent.run("Read external docs", allow_web_tools=True, require_workspace_change=False)
    assert result.status == "done"
    assert {"web_search", "web_fetch"}.issubset({t.definition.name for t in model.requests[0].tools})
    assert model.requests[0].binding_manifest["tool_execution_policy"]["allow_web_tools"] is True
    with RolloutStore(database) as store:
        artifacts = store.list_artifacts(result.turn_id)
        assert len(artifacts) == 2
        tool_output = next(i.payload['structured_content'] for i in store.list_items(result.turn_id)
                           if i.kind == 'tool_result')
        source_id = tool_output['source_id']
        source = json.loads(store.read_artifact(source_id))
        original = next(json.loads(store.read_artifact(a.artifact_id)) for a in artifacts
                        if a.artifact_id != source_id)
        assert original['extraction_method'] == 'not_attempted'
        assert base64.b64decode(original['raw_body_base64']) == b'line one\nline two'
        assert source["text"] == "line one\nline two"
        assert store.verify().valid
    model2 = FetchThenAnswer({"source_id": source_id, "start_line": 2})
    agent2 = Agent(workspace_path=workspace, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(agent2, "_harness_model", lambda: model2)
    restored = await agent2.run("Continue saved source", require_workspace_change=False)
    assert restored.status == "done"
    assert len(requests) == 1
    with RolloutStore(database) as store:
        outputs = [item.payload for item in store.list_items(restored.turn_id) if item.kind == "tool_result"]
        assert len(outputs) == 1
        assert outputs[0]["is_error"] is False
        assert "line two" in outputs[0]["structured_content"]["content"]
    other = Agent(workspace_path=tmp_path / "other-workspace", checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(other, "_harness_model", lambda: FetchThenAnswer({"source_id": source_id}))
    denied = await other.run("Read another workspace snapshot", require_workspace_change=False)
    with RolloutStore(database) as store:
        outputs = [item.payload for item in store.list_items(denied.turn_id) if item.kind == "tool_result"]
        assert outputs[0]["error_code"] == "web_source_unavailable"
    assert len(requests) == 1


@pytest.mark.anyio
async def test_unapproved_product_fetch_pauses_then_resume_uses_original_permission(tmp_path, monkeypatch):
    import agent_runtime.tools.web_http as http_module
    from agent_runtime.tools.web_http import PublicWebClient

    requests = []
    real_client = PublicWebClient
    monkeypatch.setattr(http_module, "PublicWebClient", lambda **options: real_client(
        **options, transport=httpx.MockTransport(
        lambda req: requests.append(req) or httpx.Response(200, text="hello", headers={"content-type": "text/plain"})
    )))
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "state.sqlite", enable_workspace_mcp=False)
    model = FetchThenAnswer({"url": "https://example.com"})
    monkeypatch.setattr(agent, "_harness_model", lambda: model)
    paused = await agent.run("Read a page", require_workspace_change=False)
    assert paused.status == "paused"
    assert not requests
    resumed = await agent.resume(paused.turn_id, "allow_once")
    assert resumed.status == "done"
    assert len(requests) == 1


@pytest.mark.anyio
async def test_search_resume_in_new_agent_uses_explicit_external_credential(tmp_path, monkeypatch):
    import agent_runtime.tools.web_http as http_module
    from agent_runtime.tools.web_http import PublicWebClient

    requests = []
    real_client = PublicWebClient
    monkeypatch.setattr(http_module, "PublicWebClient", lambda **options: real_client(
        **options, transport=httpx.MockTransport(
        lambda req: requests.append(req) or httpx.Response(200, json={"web": {"results": []}})
    )))
    key = tmp_path / "search.key"
    key.write_text("fake-search-secret-credential")
    key.chmod(0o400)
    options = dict(workspace_path=tmp_path / "workspace", checkpoint_db=tmp_path / "state.sqlite",
                   enable_workspace_mcp=False, web_search_key_file=key)
    first = Agent(**options)
    monkeypatch.setattr(first, "_harness_model", lambda: FetchThenAnswer({"query": "public docs"}, "web_search"))
    paused = await first.run("Search public docs", require_workspace_change=False)
    assert paused.status == "paused"
    assert requests == []
    second = Agent(**options)
    monkeypatch.setattr(second, "_harness_model", lambda: FetchThenAnswer({"query": "public docs"}, "web_search"))
    resumed = await second.resume(paused.turn_id, "allow_once")
    assert resumed.status == "done"
    assert len(requests) == 1
    with RolloutStore(options["checkpoint_db"]) as store:
        outputs = [item.payload for item in store.list_items(resumed.turn_id) if item.kind == "tool_result"]
        assert outputs[-1]["is_error"] is False
        assert "fake-search-secret-credential" not in repr(store.list_items(resumed.turn_id))


def test_cli_resume_forwards_key_file_from_environment(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from agent_runtime import cli
    from tests.agent.test_agent_cli_resume import _persist_cli_turn, _result

    options = []

    class Facade:
        async def resume(self, turn_id, action, **kwargs):
            return _result(turn_id=turn_id, answer="resumed")

    def create_facade(**kwargs):
        options.append(kwargs)
        return Facade()

    monkeypatch.setattr(cli, "_create_agent_facade", create_facade)
    database = tmp_path / "state.sqlite"
    turn_id = _persist_cli_turn(database, tmp_path / "workspace")
    path = tmp_path / "search.key"
    result = CliRunner().invoke(cli.agent_app, ["resume", turn_id, "--checkpoint-db", str(database),
                                               "--action", "allow_once"],
                                env={"PRAXIS_WEB_SEARCH_KEY_FILE": str(path)})
    assert result.exit_code == 0, result.output
    assert options[0]["web_search_key_file"] == path
    assert "allow_web_tools" not in options[0]


def test_legacy_step_never_inherits_live_web_authorization(monkeypatch):
    from agent_runtime.harness.tool_orchestrator import ToolOrchestrator
    from agent_runtime.tools.permissions import ToolExecutionContext
    from tests.agent.test_web_tools import call

    orchestrator = object.__new__(ToolOrchestrator)
    orchestrator._execution_context = ToolExecutionContext(allow_web_tools=True)
    monkeypatch.setattr(orchestrator, "_step_snapshot", lambda *_: {
        "tools": [], "binding_manifest": {"tool_execution_policy": {"allow_execute_tools": False}},
    })
    monkeypatch.setattr(orchestrator, "_context_for_turn", lambda _, base=None: base)
    context = orchestrator._context_for_call("turn", call("web_fetch", {"url": "https://example.com"}))
    assert context.allow_web_tools is False


def test_search_credential_is_explicit_protected_and_outside_workspace(tmp_path):
    from agent_runtime.tools.builtins.web import load_search_key

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    credential = tmp_path / "search.key"
    credential.write_text("fake-search-secret-credential")
    credential.chmod(0o400)
    assert load_search_key(credential, workspace=workspace) == "fake-search-secret-credential"
    assert load_search_key(None, workspace=workspace) is None
    credential.chmod(0o644)
    with pytest.raises(ValueError, match="protected"):
        load_search_key(credential, workspace=workspace)
    credential.chmod(0o400)
    alias = tmp_path / "alias.key"
    alias.symlink_to(credential)
    with pytest.raises(OSError):
        load_search_key(alias, workspace=workspace)
    inside = workspace / "search.key"
    inside.write_text("fake-search-secret-credential")
    inside.chmod(0o400)
    with pytest.raises(ValueError, match="outside"):
        load_search_key(inside, workspace=workspace)


def test_cli_exposes_web_authorization_and_search_credential_path():
    from typer.main import get_command

    from agent_runtime.cli import agent_app

    for command in ("run", "chat"):
        params = {p.name for p in get_command(agent_app).commands[command].params}
        assert {"allow_web_tools", "web_search_key_file"}.issubset(params)
        web_option = next(p for p in get_command(agent_app).commands[command].params if p.name == "allow_web_tools")
        assert web_option.default is True
        assert "--no-web-tools" in web_option.secondary_opts
    assert "web_search_key_file" in {p.name for p in get_command(agent_app).commands["resume"].params}


def test_server_launcher_forwards_only_optional_protected_search_key_path(tmp_path):
    from tests.agent.test_server_model_deployment import load_script

    launcher = load_script("launch-server-agent.py")
    proxy = tmp_path / "proxy.token"
    proxy.write_text("fake-agent-proxy-token")
    proxy.chmod(0o400)
    search_key = tmp_path / "search.key"
    search_key.write_text("fake-brave-search-key")
    search_key.chmod(0o400)
    env = launcher.agent_environment(proxy, tmp_path, "dumb")
    assert env["PRAXIS_WEB_SEARCH_KEY_FILE"] == str(search_key)
    assert "fake-brave-search-key" not in repr(env)
    search_key.chmod(0o644)
    with pytest.raises(ValueError):
        launcher.agent_environment(proxy, tmp_path, "dumb")


@pytest.mark.anyio
async def test_product_search_budget_is_persisted_and_credential_is_not(tmp_path, monkeypatch):
    import agent_runtime.tools.web_http as http_module
    from agent_runtime.tools.web_http import PublicWebClient

    hits = []
    real_client = PublicWebClient
    monkeypatch.setattr(http_module, "PublicWebClient", lambda **options: real_client(
        **options, transport=httpx.MockTransport(
        lambda req: hits.append(req) or httpx.Response(200, json={"web": {"results": []}})
    )))
    key = tmp_path / "search.key"
    key.write_text("fake-search-secret-credential")
    key.chmod(0o400)

    class RepeatedSearch(PatchThenAnswerModel):
        async def dispatch(self, prepared):
            step = int(prepared.request_ref["request_id"].rsplit(":", 1)[-1])
            if step <= 33:
                return HarnessModelResponse(text="", provider_response_id=f"search-{step}", usage={},
                    tool_calls=(HarnessToolCall(
                        id=f"search-{step}", name="web_search", arguments={"query": f"docs {step}"}),))
            return HarnessModelResponse(text="Complete", provider_response_id="answer", usage={})

    database = tmp_path / "state.sqlite"
    agent = Agent(workspace_path=tmp_path / "workspace", checkpoint_db=database,
                  enable_workspace_mcp=False, web_search_key_file=key)
    monkeypatch.setattr(agent, "_harness_model", lambda: RepeatedSearch())
    result = await agent.run("Look up docs", allow_web_tools=True, require_workspace_change=False, max_turns=40)
    assert result.status == "done"
    assert len(hits) == 32
    with RolloutStore(database) as store:
        outputs = [item.payload for item in store.list_items(result.turn_id) if item.kind == "tool_result"]
        assert outputs[-1]["error_code"] == "web_request_budget_exceeded"
        assert len(store.list_tool_operations(result.turn_id)) == 33
        assert "fake-search-secret-credential" not in repr(store.list_items(result.turn_id))
