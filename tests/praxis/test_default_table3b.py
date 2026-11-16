"""Task 20:默认工具表 15 builtin + web 装配 + 渲染探针 + 宿主隔离。

阶段4 T6 口径同步:默认表 15+0(enable_subagents 缺省关,旧 parity 零扰动);
``enable_subagents=True`` 后 15+5=20(agent 五件套追加在表尾)。

钉住四件事:
1. 默认表恰好 15 个 builtin(集合),MCP 动态工具不进入(只经 extra_tools);
2. scripted turn 的 wire tools 里同时有 web_fetch/web_search(不触网),
   证明默认装配把 web 工具真的交给了模型;
3. renderer 探针是 monkeypatch 接缝:False ⇒ 纯文本面(无 EXECUTE_PROCESS),
   True ⇒ render-capable(T14 形状);
4. ``tools=`` 整体替换时不构造 web client(capability coupling,廉价路径),
   以及宿主 ``PRAXIS_WEB_SEARCH_PROVIDER`` 不会漏进默认配置读取。
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path

import pytest

import praxis.composition as composition
from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.adapters.web.client import PublicHttpClient, PublicWebError
from praxis.interfaces.sdk import (
    TurnStatus,
    load_web_search_config,
    open_session,
)
from praxis.protocol.models import ModelResponse, Usage
from praxis.protocol.permission import PermissionUpdate
from praxis.protocol.policy import NetworkRule
from praxis.protocol.tools import ToolEffect
from praxis.tools.builtins.filesystem import create_read_tool

_EXPECTED_BUILTINS = {
    "read",
    "write",
    "bash",
    "manage_command",
    "list_files",
    "search_text",
    "inspect_data_file",
    "apply_patch",
    "update_plan",
    "run_python",
    "read_artifact",
    "web_fetch",
    "web_search",
    "invoke_skill",
    "materialize_skill_asset",
}


def _resp(content: str) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=(),
        usage=Usage(input_tokens=1, output_tokens=1),
        finish_reason="stop",
    )


def test_default_table_has_exactly_fifteen_builtins(workspace: Path) -> None:
    session = open_session(workspace, ScriptedAdapter([]))
    try:
        names = set(session._tools)  # noqa: SLF001 — 装配契约:集合可断言
        assert names == _EXPECTED_BUILTINS
        assert len(names) == 15
        assert not any(name.startswith("mcp__") for name in names)
    finally:
        session.close()


_AGENT_FIVE = {
    "spawn_subagent",
    "wait_subagent",
    "send_to_subagent",
    "cancel_subagent",
    "integrate_child_workspace",
}


def test_enable_subagents_adds_exactly_five_to_default_table(workspace: Path) -> None:
    """T6 口径:默认 15+0;开关后 15+5=20,且五件真的交给模型(wire tools)。"""
    adapter = ScriptedAdapter([_resp("done")])
    session = open_session(workspace, adapter, enable_subagents=True)
    try:
        names = set(session._tools)  # noqa: SLF001 — 装配契约:集合可断言
        assert names == _EXPECTED_BUILTINS | _AGENT_FIVE
        assert len(names) == 20
        result = asyncio.run(session.submit("hello"))
        assert result.status is TurnStatus.COMPLETED
        assert adapter.last_request is not None
        wire_names = {tool.name for tool in adapter.last_request.tools}
        assert _AGENT_FIVE <= wire_names
    finally:
        session.close()


def test_scripted_turn_offers_web_tools_without_network(workspace: Path) -> None:
    """MCP 缺席 + web 工具在场:wire tools 含两件 web 工具,剧本零网络。"""
    adapter = ScriptedAdapter([_resp("done")])
    session = open_session(workspace, adapter)
    try:
        result = asyncio.run(session.submit("hello"))
        assert result.status is TurnStatus.COMPLETED
        assert adapter.last_request is not None
        wire_names = {tool.name for tool in adapter.last_request.tools}
        assert {"web_fetch", "web_search"} <= wire_names
        assert {"invoke_skill", "materialize_skill_asset"} <= wire_names
        assert not any(name.startswith("mcp__") for name in wire_names)
    finally:
        session.close()


def test_renderer_probe_false_keeps_text_only_surface(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(composition, "_playwright_importable", lambda: False)
    session = open_session(workspace, ScriptedAdapter([]))
    try:
        tool = session._tools["web_fetch"]  # noqa: SLF001 — 装配契约
        assert ToolEffect.EXECUTE_PROCESS not in tool.static_effects
    finally:
        session.close()


def test_renderer_probe_true_wires_render_capable_tool(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(composition, "_playwright_importable", lambda: True)
    session = open_session(workspace, ScriptedAdapter([]))
    try:
        tool = session._tools["web_fetch"]  # noqa: SLF001 — 装配契约
        assert ToolEffect.EXECUTE_PROCESS in tool.static_effects
    finally:
        session.close()


def test_full_replacement_skips_web_client_construction(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``tools=`` 整体替换 ⇒ 默认装配路径不执行(不读 web limits/web.json)。"""

    def explode() -> None:
        raise AssertionError("web limits must not load when tools are replaced")

    monkeypatch.setattr(composition, "load_web_limits", explode)
    session = open_session(
        workspace, ScriptedAdapter([]), tools=(create_read_tool(workspace),)
    )
    try:
        assert set(session._tools) == {"read"}  # noqa: SLF001 — 装配契约
    finally:
        session.close()


def test_host_search_provider_env_does_not_leak_into_default_config(
    workspace: Path,
) -> None:
    """隔离 fixture 剥掉宿主 env;默认配置读取只看到 per-test 空配置目录。"""
    assert os.environ.get("PRAXIS_WEB_SEARCH_PROVIDER") is None
    assert load_web_search_config() is None


def test_wired_default_authorize_reads_live_session_policy(workspace: Path) -> None:
    """默认表**实际装配**的 authorize 必须是 live 视图,不是打开时快照(S8)。

    Reach path(整分支审查可复核):``session._integrations`` 里的
    ``PublicHttpClient``(T20 把 web client 作为 integration 注入)→
    ``client._authorize``(即 ``make_web_authorizer`` 产出的同一闭包)。
    默认策略 ask ⇒ 先 denied;``update_permission`` 授域后**同一实例**立即
    allow。若未来在 ``_default_tools`` 把 provider 改成策略快照,本测试失败。
    """
    session = open_session(workspace, ScriptedAdapter([]))
    try:
        client = next(
            item
            for item in session._integrations  # noqa: SLF001 — 装配契约:client 收尾面
            if isinstance(item, PublicHttpClient)
        )
        authorize = client._authorize  # noqa: SLF001 — 唯一授权闭包的接线点
        url = "https://live.example/path"
        with pytest.raises(PublicWebError) as denied:
            authorize(url)
        assert denied.value.code == "denied"
        assert denied.value.verdict == "ask"  # 默认 ask,而非 fail-closed deny

        session.update_permission(
            PermissionUpdate(
                rule=NetworkRule(domain="live.example", decision="allow"),
                destination="session",
            )
        )
        assert authorize(url) is None
    finally:
        session.close()


@pytest.mark.parametrize("error_type", (ModuleNotFoundError, ValueError))
def test_playwright_probe_is_total_when_dotted_lookup_raises(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    """回归:base 安装(playwright 缺失)时 dotted find_spec 会抛异常。

    不 monkeypatch ``_playwright_importable`` 本身;而是替换底层
    ``importlib.util.find_spec``,让**任何 dotted 名**(含
    ``playwright.async_api``)抛出父包缺失/异常 spec 的真实错误类型。探针
    必须 total ⇒ False,默认 build_session 必须成功,web_fetch 保持纯文本面
    (无 EXECUTE_PROCESS)。pyproject `browser` extra 可选;
    scripts/agent_install_smoke.sh 的 base-install disjoint 断言即此回归面。
    """
    real_find_spec = importlib.util.find_spec

    def raising_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if "." in name:
            raise error_type(f"stubbed dotted lookup failure: {name}")
        return real_find_spec(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", raising_find_spec)
    assert composition._playwright_importable() is False
    session = open_session(workspace, ScriptedAdapter([]))
    try:
        assert len(session._tools) == 15  # noqa: SLF001 — base 安装仍可开会话
        tool = session._tools["web_fetch"]  # noqa: SLF001 — 装配契约
        assert ToolEffect.EXECUTE_PROCESS not in tool.static_effects
    finally:
        session.close()
