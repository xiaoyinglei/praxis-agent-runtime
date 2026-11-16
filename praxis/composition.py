"""Praxis 唯一组合根(设计 §4.1;CI 规则 7:Session 构造只允许出现在本文件)。

build_session 把分层组件装配成一个可用会话,顺序即依赖序:
建 ``.praxis/`` 目录 → Store(单写者线程) → EventBus → **sweep**(打开即收编
崩溃遗留的 running turn → interrupted,§5.8;活执行者持锁,Sweep 跳过,绝不误伤)
→ **CommandSessions + 孤儿命令进程组清扫**(Task 15:同一实例注入 bash 与
manage_command;marker 清扫结果只记日志) → 默认内置工具(read/write/bash/
manage_command/list_files/search_text/inspect_data_file/apply_patch/
update_plan/run_python/read_artifact/web_fetch/web_search/invoke_skill/
materialize_skill_asset = 15 builtin;顺序稳定,可由 tools 参数整体替换)
→ PermissionContext(``probe_backend()`` 真实沙箱探测)→ 线程 ensure/create →
Session(注入 ``compile_sandbox``/``finalize_sandbox``、后台会话收尾面与唯一
重试机制 ``retry_stream``——core 经结构签名消费,不 import adapters)。

thread_id 是外部输入且直接参与 lockfile 命名:在这里(边界)先验拒绝空串、路径
分隔符、NUL 与 ``..`` 成分——locks.py 会再拦一次,但边界拒绝让错误在碰文件系统
与建库之前发生(fail early at the boundary)。

任何装配失败都要关掉已建的 Store,绝不留写者线程。
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from praxis.adapters.config.credentials import (
    resolve_model_credential,
    resolve_web_search_credential,
)
from praxis.adapters.config.store import ModelConfigStore
from praxis.adapters.hooks.config import load_host_hooks
from praxis.adapters.hooks.runner import spawn_hook
from praxis.adapters.mcp.manager import McpBundle
from praxis.adapters.mcp.manager import connect_mcp_servers as _connect_mcp_servers
from praxis.adapters.models.ollama import OllamaAdapter
from praxis.adapters.models.openai_compat import OpenAICompatAdapter
from praxis.adapters.models.retry import make_retry_stream
from praxis.adapters.models.tokenization import measure_request
from praxis.adapters.sandbox.base import probe_backend
from praxis.adapters.sandbox.compile import compile_sandbox, finalize_sandbox
from praxis.adapters.web.client import PublicHttpClient, PublicWebError
from praxis.adapters.web.limits import WebLimits, load_web_limits
from praxis.adapters.web.search import (
    SEARCH_ENDPOINTS,
    SearchProviderConfig,
    create_web_search_tools,
    load_web_search_config,
)
from praxis.adapters.web.tools import create_web_fetch_tools
from praxis.core.compaction import (
    SUMMARY_MAX_OUTPUT_TOKENS,
    SUMMARY_PROMPT_MAX_CHARS,
    default_compact_limit,
    default_summary_max_output_tokens,
    default_summary_prompt_max_chars,
)
from praxis.core.events import EventBus
from praxis.core.permission import network_verdict
from praxis.core.recovery import sweep, sweep_orphan_command_jobs
from praxis.core.session import AsyncCloser, Session
from praxis.protocol.budget import UNLIMITED, Limits
from praxis.protocol.hooks import HookOutcome, HookSetup, HookSpec
from praxis.protocol.json import JsonValue
from praxis.protocol.models import (
    CostMicros,
    ModelAdapter,
    ModelMessage,
    ModelSettings,
    ModelSpec,
    WireRequest,
)
from praxis.protocol.permission import PermissionContext, PermissionMode
from praxis.protocol.policy import PermissionPolicy, PolicyVerdict, ToolRule, url_host
from praxis.protocol.retry import RetryPolicy
from praxis.protocol.skills import SkillCatalog
from praxis.protocol.tools import Tool, ToolEffect, ToolSource
from praxis.protocol.workspaces import WorkspaceMode
from praxis.store.artifacts import ArtifactStore
from praxis.store.log import Store
from praxis.tools.builtins.artifacts_tool import create_read_artifact_tool
from praxis.tools.builtins.child_env import child_env
from praxis.tools.builtins.command import create_bash_tool
from praxis.tools.builtins.command_sessions import (
    CommandSessions,
    create_manage_command_tool,
)
from praxis.tools.builtins.data import create_inspect_data_file_tool
from praxis.tools.builtins.filesystem import (
    create_list_files_tool,
    create_read_tool,
    create_write_tool,
)
from praxis.tools.builtins.patch import create_apply_patch_tool
from praxis.tools.builtins.planning import create_update_plan_tool
from praxis.tools.builtins.process import signal_group
from praxis.tools.builtins.python_tool import create_run_python_tool
from praxis.tools.builtins.search import create_search_text_tool
from praxis.tools.builtins.skills import (
    create_invoke_skill_tool,
    create_materialize_skill_asset_tool,
    load_skill_catalog,
)

_VALID_MODES: tuple[PermissionMode, ...] = ("on-request", "never", "bypass")

logger = logging.getLogger(__name__)


def missing_credential_message(spec: ModelSpec) -> str:
    """缺凭据的唯一文案产地(D6/R-T6-6):模型/provider、环境变量名、下一步。

    只拼配置事实(spec 里的变量名,绝不是值),因此 ``models add`` 的干跑、
    ``build_session`` 与 ``models check`` 的报错天然同源。``api_key_env`` 未配置时
    如实说明并指向 ``models add --api-key-env``;否则给出 export 与 keychain 两条
    出路。凭据值永远不进入消息。
    """
    head = f"no credential for model {spec.model_id!r} (provider {spec.provider})"
    if spec.api_key_env is None:
        return (
            f"{head}: no api_key_env configured; "
            f"register one with: praxis models add {spec.model_id} --api-key-env <NAME>"
        )
    env = spec.api_key_env
    return (
        f"{head}: set environment variable {env} (export {env}=...), "
        f"or run: praxis models login {spec.model_id}"
    )


def _validate_thread_id(thread_id: str) -> None:
    if not thread_id:
        raise ValueError("thread_id must be non-empty")
    if "/" in thread_id or "\\" in thread_id:
        raise ValueError(f"thread_id must not contain path separators: {thread_id!r}")
    if "\x00" in thread_id:
        raise ValueError("thread_id must not contain NUL bytes")
    if ".." in Path(thread_id).parts:
        raise ValueError(f"thread_id must not contain '..' components: {thread_id!r}")


def _adapter_for(
    spec: ModelSpec,
    api_key: str | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> ModelAdapter:
    """按 provider 的唯一分派点:openai-compatible 与 ollama 在此装配。

    ``scripted`` 永远由 CLI/测试显式注入(它没有远端配置可解析),这里明确拒绝;
    openai-compatible 缺 key 属于边界错误——凭据只经参数进入,适配器不回读环境。
    ``base_url`` 必须在构造期绝对化:注册表里的模型不允许悄悄回落到 SDK 默认端点。
    """
    if spec.provider == "ollama":
        return OllamaAdapter(spec, http_client=http_client)
    if spec.provider == "openai-compatible":
        if not spec.base_url:
            raise ValueError(
                f"provider openai-compatible requires base_url for model {spec.model_id!r}"
            )
        parsed = urlsplit(spec.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(
                "provider openai-compatible requires an absolute http(s) base_url, "
                f"got {spec.base_url!r}"
            )
        if api_key is None:
            raise ValueError(missing_credential_message(spec))
        return OpenAICompatAdapter(spec, api_key=api_key, http_client=http_client)
    raise ValueError(
        f"provider {spec.provider!r} must be injected explicitly (scripted), "
        "not constructed via composition"
    )


def model_adapter_for(
    spec: ModelSpec,
    api_key: str | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> ModelAdapter:
    """公开的 provider 分派入口(R-T6-2):cli/sdk 只经这里构造模型适配器。

    唯一实现在 ``_adapter_for``;本函数只是可被 sdk re-export 的稳定名字,
    不引入第二条分派路径。
    """
    return _adapter_for(spec, api_key, http_client=http_client)


def validate_model_spec(spec: ModelSpec, api_key: str | None = None) -> None:
    """``praxis models add`` 的写前干跑(R-T6-1):只 build_request,不发网络。

    走与运行期完全相同的 adapter 构造 + ``build_request``,因此越界
    ``provider_options``、坏 ``base_url``、缺凭据等问题在写库之前暴露;
    ``ollama`` 无凭据是正常形态(provider 本身没有凭据概念)。
    """
    adapter = _adapter_for(spec, api_key)
    adapter.build_request(
        [ModelMessage(role="user", content="praxis model registration dry run")],
        (),
        ModelSettings(),
    )


def _temporary_spec_from_env(base_url: str | None) -> ModelSpec | None:
    """未注册临时模型逃生门(仅构造一个 ModelSpec,不是第二装配入口)。

    ``PRAXIS_MODEL_ID`` 是开关,``PRAXIS_BASE_URL`` 提供端点;凭据值由
    ``PRAXIS_API_KEY`` 在调用方直接交给 ``_adapter_for``。只要注册库里有模型,
    就不会走到这里——逃生门只服务"零注册"的临时运行。
    """
    model_id = os.environ.get("PRAXIS_MODEL_ID")
    if not model_id:
        return None
    return ModelSpec(
        model_id=model_id,
        provider="openai-compatible",
        base_url=base_url or os.environ.get("PRAXIS_BASE_URL") or None,
    )


def _adapter_and_spec_from_registry(
    model: str | None,
    config_store: ModelConfigStore | None,
) -> tuple[ModelAdapter, ModelSpec]:
    """从全局配置库解析 ModelSpec 并交给唯一分派点(R-T6-2)。

    自建的 store 在解析完即关闭(CLI 单次调用);调用方注入的 store 由其自己
    管理生命周期。返回 spec 是为了让组合根计算自动压缩阈值(D15)——
    ModelSpec 只在这里可解析,core 不接触配置库。
    """
    owned = config_store is None
    store = config_store if config_store is not None else ModelConfigStore()
    try:
        if model is None:
            try:
                spec = store.resolve_spec(None)
            except KeyError:
                # 只有"库里一个模型都没有"才允许逃生门;有模型但缺默认模型时
                # 继续报错(set-default),绝不静默改用环境变量。
                if store.list_models():
                    raise
                temporary = _temporary_spec_from_env(None)
                if temporary is None:
                    raise
                api_key = os.environ.get("PRAXIS_API_KEY") or None
                return _adapter_for(temporary, api_key), temporary
        else:
            spec = store.resolve_spec(model)
    finally:
        if owned:
            store.close()
    api_key = (
        resolve_model_credential(spec.api_key_env, spec.base_url or "")
        if spec.api_key_env is not None
        else None
    )
    return _adapter_for(spec, api_key), spec


def _measure_request_tokens(req: WireRequest) -> int:
    """D2 唯一测量入口的注入形式(core 不得 import adapters)。"""
    return measure_request(req).count


def make_web_authorizer(
    permissions_provider: Callable[[], PermissionContext],
) -> Callable[[str], None]:
    """T10 公共 web 客户端的 authorize 回调生产点(唯一组合根;T12 注入)。

    ``permissions_provider`` 每次调用都重新取 live 权限上下文——绝不缓存策略
    快照:session 中途授权后,下一次/下一跳立刻按新策略裁决。裁决与 decide
    第 1/2/6 步共用 ``core.permission.network_verdict``(同一 fold,无第二套代数)。

    契约(T10 authorize:任意异常原样传播、先于 validate/connect):
    - allow ⇒ 返回 None;
    - ask ⇒ raise ``PublicWebError("denied", "approval required for <host>", verdict="ask")``
      ——逐跳/子资源是**执行期检查点**:ask 不再被无视放行,而是中止本次请求
      并指向 NetworkRule 授权路径;初始 URL 的 ask 由 decide 正常暂停审批;
    - deny ⇒ raise ``PublicWebError("denied", verdict="deny")``。
    ``verdict`` 机器字段(T14)供浏览器子资源闸分别计数 blocked_ask/blocked_deny。
    URL 原样交给 ``network_verdict``(归一化完全由 protocol.policy 承担);
    闭包内不再改写/重解析待裁决的 URL 串。
    """

    def authorize(url: str) -> None:
        verdict = network_verdict(permissions_provider(), url)
        if verdict == "allow":
            return
        if verdict == "ask":
            raise PublicWebError("denied", f"approval required for {_web_url_host(url)}", verdict="ask")
        raise PublicWebError("denied", verdict="deny")

    return authorize


def _web_url_host(url: str) -> str:
    """仅用于拒绝对话文案的主机名(R-T11-4 唯一归一化;不参与规则匹配)。"""
    return url_host(url) or url


def _playwright_importable() -> bool:
    """T20 renderer 探针:查 ``playwright.async_api`` 子模块是否存在。

    **必须是 total**:dotted ``find_spec`` 会导入父包,playwright 缺失时抛
    ``ModuleNotFoundError``(父包 spec 异常时还可能 ``ValueError``)——base
    (text-only)安装每次默认 ``build_session`` 都会调本探针,任何异常都会
    让会话打不开。异常 ⇒ False(纯文本面,render 如实 render_unavailable)。
    ``find_spec`` 不执行 async_api 本身;composition 模块级无 playwright
    import 链。探针是 monkeypatch 接缝(测试钉两种装配结果)。
    """
    try:
        return importlib.util.find_spec("playwright.async_api") is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _browser_renderer_factory(
    client: PublicHttpClient,
    authorize: Callable[[str], None],
    limits: WebLimits,
) -> Callable[[], Any] | None:
    """按探针结果绑定 T14 渲染器工厂(同一 client/authorize/limits 实例)。

    browser 模块的 import 推迟到工厂构造时(规则①禁止函数内 import 语句,
    故用 ``importlib.import_module``);playwright 缺失 ⇒ ``None``,web_fetch
    保持纯文本面、render 请求如实 ``render_unavailable``。chromium 二进制
    缺失但模块在的情况由 T14 的渲染 start 相位映射 ``render_unavailable``。
    """
    if not _playwright_importable():
        return None
    browser: Any = importlib.import_module("praxis.adapters.web.browser")
    return lambda: browser.PageRenderer(
        client, authorize=authorize, limits=limits, signal_process_group=signal_group
    )


def _web_search_credential(cfg: SearchProviderConfig) -> str | None:
    """R-T13-1:provider 与 endpoint 取自**同一个 live cfg**,原子绑定。"""
    return resolve_web_search_credential(cfg.api_key_env, SEARCH_ENDPOINTS[cfg.provider])


def _default_tools(
    root: Path,
    sessions: CommandSessions,
    artifact_store: ArtifactStore,
    skills_catalog: SkillCatalog,
    session_holder: list[Session],
) -> tuple[tuple[Tool, ...], PublicHttpClient]:
    """T20 默认工具表(15 builtin)与配套 web client。

    顺序是稳定契约(测试钉死)。web client 与 web_fetch/web_search 共用**同一**
    ``WebLimits`` 与**同一** authorize 闭包;authorize 经 ``session.permissions``
    读取 live 权限(与 T11/T12 同一单源,不缓存策略快照)。``session_holder``
    是延迟绑定:Session 在工具表之后构造,填充后权限更新立即可见(web 工具只
    在 submit 内被调用)。

    web_search 的 ``config_provider`` 每次调用现读宿主 web.json(live),
    凭据面按 R-T13-1 由 cfg 原子解析。``tools=`` 整体替换时本函数不执行
    ——不构造 web client、不装渲染器,保持替换路径廉价(capability coupling
    先例:没有 web 工具就不需要 web 能力)。
    """
    limits = load_web_limits()
    authorize = make_web_authorizer(lambda: session_holder[0].permissions)
    client = PublicHttpClient(authorize=authorize, limits=limits)
    tools: tuple[Tool, ...] = (
        create_read_tool(root),
        create_write_tool(root),
        create_bash_tool(root, sessions=sessions),
        create_manage_command_tool(sessions),
        create_list_files_tool(root),
        create_search_text_tool(root),
        create_inspect_data_file_tool(root),
        create_apply_patch_tool(root),
        create_update_plan_tool(),
        create_run_python_tool(root),
        create_read_artifact_tool(artifact_store),
        *create_web_fetch_tools(
            client,
            limits=limits,
            renderer_factory=_browser_renderer_factory(client, authorize, limits),
        ),
        *create_web_search_tools(
            client,
            config_provider=load_web_search_config,
            credential_provider=_web_search_credential,
        ),
        create_invoke_skill_tool(skills_catalog),
        create_materialize_skill_asset_tool(root),
    )
    return tools, client


async def _spawn_host_hook(
    spec: HookSpec, request: Mapping[str, JsonValue]
) -> HookOutcome:
    """T4 runner 的生产绑定:整组 SIGKILL 语义由 injected signal_group 提供。"""
    return await spawn_hook(spec, request, kill_group=signal_group)


def _host_hook_setup(root: Path) -> HookSetup:
    """装配 Host command hook(T5):workspace 一律以已 resolve 的绝对路径进入。

    T3 留档的相对路径角落在此收口——``load_host_hooks`` 的快照与真实
    工作区必须同源。``spawn`` 只做机制;阀门/结算在 core 管道。
    """
    specs = load_host_hooks(root.expanduser().resolve(), None)
    return HookSetup(specs=specs, spawn=_spawn_host_hook)


def resolve_declared_policy(root: Path, explicit: PermissionPolicy | None) -> PermissionPolicy:
    """policy 来源序(Task 8):显式参数 > ``.praxis/permissions.json`` > 默认。

    文件是策略**声明**(always 授权由 Session 原子写合并于此);坏文件/坏 JSON
    一律 ValueError 边界拒绝,绝不静默回退默认(fail early at the boundary;
    pydantic strict 的 ValidationError 就是 ValueError 的子类)。

    T19:名字转正——MCP connect 的声明快照与 ``build_session`` 必须来自**同一
    加载实现**;宿主(CLI)先取快照,再同时交给 connect 与 open_session。
    """
    if explicit is not None:
        return explicit
    path = root / ".praxis" / "permissions.json"
    if not path.exists():
        return PermissionPolicy()
    try:
        return PermissionPolicy.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load policy file {path}: {exc}") from exc


async def connect_mcp_servers(
    workspace: Path,
    *,
    policy: PermissionPolicy | None = None,
    network_verdict_at_connect: Callable[[str], PolicyVerdict] | None = None,
) -> McpBundle:
    """MCP eager connect 的公开装配入口(T19;sdk/CLI 只经这里)。

    本函数是 ``adapters.mcp.manager.connect_mcp_servers`` 的**唯一绑定点**:
    ``sandbox_compiler`` 与 ``env_builder``(``child_env`` 的唯一实现)在这里
    注入,adapters 层因此不 import tools(R-T15-7)。``policy`` 是
    ``resolve_declared_policy`` 的声明快照;``None`` 表示调用方未声明。
    """
    return await _connect_mcp_servers(
        workspace,
        policy=policy,
        sandbox_compiler=compile_sandbox,
        env_builder=child_env,
        network_verdict_at_connect=network_verdict_at_connect,
    )


def build_session(
    workspace: Path,
    adapter: ModelAdapter | None = None,
    *,
    model: str | None = None,
    config_store: ModelConfigStore | None = None,
    policy: PermissionPolicy | None = None,
    mode: PermissionMode | None = None,
    thread_id: str | None = None,
    tools: Sequence[Tool] | None = None,
    max_steps: int = 50,
    budget: Limits = UNLIMITED,
    hooks: HookSetup | None = None,
    skills_catalog: SkillCatalog | None = None,
    extra_tools: Sequence[Tool] = (),
    integrations: Sequence[AsyncCloser] = (),
    db_path: Path | None = None,
    unit_costs: CostMicros | None = None,
    ceilings: Sequence[PermissionPolicy] = (),
    run_sweep: bool = True,
) -> Session:
    """装配一个 Session:唯一组合根。CLI/TUI/host 一律经 sdk.open_session 走到这里。

    ``adapter is None`` 时从全局配置库解析:``ModelConfigStore.resolve_spec`` →
    ``resolve_model_credential`` → ``_adapter_for``(设计 §5.15;R-T6-2)。显式
    ``adapter``(测试/``--scripted``)完全绕过注册库。未注册临时模型只能经
    ``PRAXIS_MODEL_ID/PRAXIS_API_KEY/PRAXIS_BASE_URL`` 构造一个 ModelSpec,
    适配器仍然只由 ``_adapter_for`` 装配——不是第二装配入口。

    Task 8 入口:policy 见 resolve_declared_policy 来源序;mode 来源序 = 显式参数 >
    policy.approval,正交于策略(C5:bypass 只活在运行上下文,永不进 policy、
    永不落盘;切换 mode 不触碰任何策略规则),非法值边界拒绝。

    artifact 外部化与回读能力耦合:``tools`` 整体替换后的工具表里没有
    ``read_artifact`` 时,**不**注入 ArtifactStore,管道自动退回旧字节截断
    语义——不会给模型一个无法回读的指针。

    skill 目录(§5.14):``skills_catalog=None``(缺省)⇒ 本组合根从 workspace
    双根调用 ``load_skill_catalog``(PROJECT 上溯 + ``$SKILL_PATH``);显式参数
    (测试)完全绕过文件系统扫描。空目录是常态,注入步骤在 core 里自然恒等——
    不设 ``SKILLS_ENABLED`` 开关。composition 可 import tools,core 不可。

    T19:``extra_tools``(MCP 投影)并入默认/显式工具表,任何**名字冲突**都在
    这里边界拒绝(跨 server 同名也在 merge 处暴露,T18 交接);``integrations``
    (MCP bundle)注入 Session,``close()/aclose()`` 在 command_sessions 之后、
    store 之前收尾。

    T20:默认表 15 builtin(web_fetch/web_search/invoke_skill/
    materialize_skill_asset 在此装配)。web 面与 Session 同源:client 的
    authorize 闭包读 live ``session.permissions``;``tools`` 整体替换时**不**
    构造 web client、不装渲染器(capability coupling:没有 web 工具就没有 web
    能力,替换路径保持廉价)。web client 自身作为 integration 收尾。

    阶段4 T3(R-T3-4):``budget`` 是 root limits(缺省 UNLIMITED = 不 enforce,
    历史路径零扰动),经 Session 写进 turn binding;``unit_costs`` 从注册库
    解析出的 ModelSpec.cost **取一次**注入——显式 adapter(测试/--scripted)
    没有可解析的 spec → None,cost 限额生效时由 admit_dispatch fail-closed
    (pricing_unavailable)兜底。core 不 import adapters/config,与
    measure_tokens 注入同纪律。

    阶段4 T5(子代理装配,全部尾部默认零扰动):
    - ``db_path``:Store 库文件位置覆盖(缺省 workspace/.praxis/praxis.sqlite)。
      子 Session 与父**同库**(§5.13)——write 子工作区是独立目录时仍指向父库;
    - ``unit_costs``:显式单价覆盖(子继承父单价;缺省从 ModelSpec.cost 解析);
    - ``ceilings``:权限上限策略(只否决永不授权;read_only 子的 WRITE/
      DESTRUCTIVE 拒绝经此注入,与父 live policy 取交集);
    - 恒注入 ``child_session_factory`` 闭包(经 session_holder 延迟绑定,
      与 web authorizer 同先例)——SubagentManager 经它构造子 Session,
      内部仍调 build_child_session → build_session(Session 构造唯一点不破);
    - ``run_sweep=False``:跳过打开即清扫(子 Session 构造用——同库已由父
      打开时 sweep 覆盖;见 build_child_session 的锁文件族说明)。
    """
    if thread_id is not None:
        _validate_thread_id(thread_id)
    if mode is not None and mode not in _VALID_MODES:
        raise ValueError(f"unknown permission mode: {mode!r} (expected on-request/never/bypass)")
    if adapter is None:
        adapter, resolved_spec = _adapter_and_spec_from_registry(model, config_store)
    else:
        # 显式注入(测试/--scripted)没有可解析的 ModelSpec → 关闭自动压缩,
        # 既有测试的 item 序列保持完全不变。
        resolved_spec = None
    compact_limit_tokens = (
        default_compact_limit(resolved_spec) if resolved_spec is not None else None
    )
    # R-T3-4:单价只在这里取一次(spec.cost 全零 = 未配置计价,admit 判据
    # 对 cost 限额 fail-closed);显式 adapter 无 spec → None。
    # T5:显式 unit_costs 参数优先(子 Session 继承父单价,不重解析注册库)。
    spec_unit_costs = resolved_spec.cost if resolved_spec is not None else None
    effective_unit_costs = unit_costs if unit_costs is not None else spec_unit_costs
    summary_max_output_tokens = (
        default_summary_max_output_tokens(resolved_spec)
        if resolved_spec is not None
        else SUMMARY_MAX_OUTPUT_TOKENS
    )
    # K-4:摘要请求正文上限也按模型上下文窗口推导——小窗口本地模型上 16k
    # 字符的常量上限会让摘要请求被 provider 拒收(静默不压缩)。
    summary_prompt_max_chars = (
        default_summary_prompt_max_chars(resolved_spec)
        if resolved_spec is not None
        else SUMMARY_PROMPT_MAX_CHARS
    )
    root = Path(workspace)
    # T5:默认从 Host 配置装配 command hooks(显式 hooks 参数=测试路径,完全
    # 绕过加载)。workspace 传已 resolve 的绝对路径(T3 相对路径角落收口)。
    hook_setup = hooks if hooks is not None else _host_hook_setup(root)
    # §5.14:skill 目录缺省从 workspace 双根装载;显式参数(测试)绕过 fs 扫描。
    # 默认开启、空目录自然恒等——不需要 SKILLS_ENABLED 总开关。
    resolved_skills = (
        skills_catalog if skills_catalog is not None else load_skill_catalog(root)
    )
    praxis_dir = root / ".praxis"
    praxis_dir.mkdir(parents=True, exist_ok=True)
    # T5 同库纪律:db_path 缺省 = 本工作区会话库;子 Session 经父 Store.db_path
    # 注入同一库文件(独立目录 ≠ 独立库)。
    store = Store(db_path if db_path is not None else praxis_dir / "praxis.sqlite")
    try:
        bus = EventBus()
        if run_sweep:
            sweep(store, bus, root)
        # Task 15:后台命令会话注册表(一次实例化,注入 bash 与 manage_command)。
        sessions = CommandSessions(root)
        # 崩溃遗留的命令进程组清扫:结果只记日志,不影响会话可用性(函数自身不抛,
        # 这里再加一层边界围栏,保证任何清扫意外都不会让会话打不开)。
        try:
            orphan_summaries = sweep_orphan_command_jobs(kill_group=signal_group)
        except Exception:  # noqa: BLE001 — 清扫不是会话可用性的前置条件
            logger.warning("orphan command job sweep failed", exc_info=True)
            orphan_summaries = ()
        if orphan_summaries:
            logger.info(
                "swept %d orphan command job marker(s): %s",
                len(orphan_summaries),
                ", ".join(f"{s.session_id}={s.state}" for s in orphan_summaries),
            )
        # 内容寻址 artifact 目录(D10):.praxis 是地板,模型只能经 read_artifact 回读。
        artifact_store = ArtifactStore(praxis_dir / "artifacts")
        # T20:web 工具需要 web client;authorize 经 session_holder 延迟绑定到
        # live ``session.permissions``。tools 整体替换 ⇒ 两个都不构造。
        session_holder: list[Session] = []
        web_client: PublicHttpClient | None = None
        if tools is not None:
            tool_seq = tools
        else:
            tool_seq, web_client = _default_tools(
                root, sessions, artifact_store, resolved_skills, session_holder
            )
        tool_map: dict[str, Tool] = {tool.definition.name: tool for tool in tool_seq}
        # T19:MCP 工具在仓库边界并入;任何重名都是装配错误(extra↔builtin 或
        # 跨 server canonical 名冲突)。先查后写,错误点名字与双方来源,绝不
        # 静默覆盖(覆盖会让一个 server 的工具永远不可达)。
        for extra in extra_tools:
            name = extra.definition.name
            existing = tool_map.get(name)
            if existing is not None:
                origin = (
                    "a built-in/configured tool of the same name"
                    if existing.source is ToolSource.BUILTIN
                    else "another extra tool (two MCP servers expose the same name)"
                )
                raise ValueError(
                    f"tool name {name!r} from extra tools collides with {origin}"
                )
            tool_map[name] = extra
        # 外部化与回读能力耦合:工具表被整体替换且没有 read_artifact 时,
        # 不注入 artifacts——否则模型只会拿到打不开的指针,比旧字节截断更差。
        effective_artifacts = artifact_store if "read_artifact" in tool_map else None
        effective_policy = resolve_declared_policy(root, policy)
        effective_mode: PermissionMode = mode if mode is not None else effective_policy.approval
        permissions = PermissionContext(
            workspace=str(root),
            policy=effective_policy,
            mode=effective_mode,
            # 真实探测:darwin seatbelt;其他平台不可用 → decide 第 7 步对
            # EXECUTE_PROCESS fail-closed deny(bwrap 延后,D2/D7)。
            sandbox=probe_backend(),
            # T5:ceiling 只否决永不授权(read_only 子的收窄经此与父策略取交集)。
            ceilings=tuple(ceilings),
        )
        resolved_thread = thread_id if thread_id is not None else store.create_thread()
        store.ensure_thread(resolved_thread)
        # T20:web client 是 Session 的 integration(aclose 在 command_sessions
        # 之后、store 之前;与 MCP bundle 同一条收尾链)。
        effective_integrations: tuple[AsyncCloser, ...] = tuple(integrations) + (
            (web_client,) if web_client is not None else ()
        )

        def _child_session_factory(
            *,
            workspace: Path,
            thread_id: str,
            limits: Limits,
            tool_subset: Sequence[Tool],
            workspace_mode: WorkspaceMode,
        ) -> Session:
            # T5:parent 经 session_holder 延迟绑定(与 web authorizer 同先例);
            # 内部仍调 build_child_session → build_session,构造唯一点不破。
            return build_child_session(
                parent=session_holder[0],
                workspace=workspace,
                thread_id=thread_id,
                limits=limits,
                tool_subset=tool_subset,
                workspace_mode=workspace_mode,
            )

        session = Session(
            store,
            bus,
            root,
            adapter,
            tool_map,
            permissions,
            resolved_thread,
            max_steps,
            compile_sandbox=compile_sandbox,
            finalize_sandbox=finalize_sandbox,
            # 唯一重试机制在此绑定:真实时钟 + asyncio.sleep;core 只见结构签名。
            retry_stream=make_retry_stream(
                policy=RetryPolicy(), clock=time.time, sleep=asyncio.sleep
            ),
            artifacts=effective_artifacts,
            compact_limit_tokens=compact_limit_tokens,
            measure_tokens=_measure_request_tokens,
            summary_max_output_tokens=summary_max_output_tokens,
            summary_prompt_max_chars=summary_prompt_max_chars,
            command_sessions=sessions,
            hooks=hook_setup,
            skills_catalog=resolved_skills,
            integrations=effective_integrations,
            budget=budget,
            unit_costs=effective_unit_costs,
            child_session_factory=_child_session_factory,
        )
        # 延迟绑定完成:此后 web authorize 闭包读到的就是 live 权限上下文
        # (Session.update_permission 替换 _permissions,property 每次现取)。
        session_holder.append(session)
        return session
    except BaseException:
        store.close()
        raise


# read_only 子拒绝的副作用维(brief §5.13:intersection + 拒 WRITE/DESTRUCTIVE)。
_READ_ONLY_DENIED_EFFECTS: frozenset[ToolEffect] = frozenset(
    {ToolEffect.WRITE_WORKSPACE, ToolEffect.DESTRUCTIVE}
)


def build_child_session(
    *,
    parent: Session,
    workspace: Path,
    thread_id: str,
    limits: Limits,
    tool_subset: Sequence[Tool],
    workspace_mode: WorkspaceMode = "read_only",
) -> Session:
    """子 Session 工厂(阶段4 T5;内部仍调 build_session——构造唯一点不破)。

    装配纪律:
    - **同库**(§5.13):Store 经 ``parent.store.db_path`` 指向父库文件——
      write 子的独立工作区目录不等于独立库;子 items 写自己的 thread;
    - 适配器/单价继承父(生产适配器无状态可共享;子的 cost 限额与父同单价);
    - 权限 = 父 live policy ∩ read_only 收窄:``workspace_mode="read_only"``
      时追加一个 ``profile="read-only"`` ceiling(恒追加,与工具表组成无关),
      承载两层否决:
      (a) **沙箱写路径否决(承重层,R-T5-6)**:compile_sandbox 的 allow_write
      逐 ceiling AND ``path_access(write=True)``;read-only profile 在规则
      匹配之前即拒写 → bash(EXECUTE_PROCESS,无 path target,工具规则/地板
      全不触)的沙箱工作区写路径关闭。注意不可用 ``PathRule(root,"read")``
      表达——resolve 的 profile 基线同路径生成 write 规则,_ACCESS_ORDER
      同特异性取 write,read 规则被吸收;profile 形态才是有效否决;
      (b) 工具规则 deny(纵深防御):对 tool_subset 中声明 WRITE_WORKSPACE/
      DESTRUCTIVE 的工具逐名 ToolRule deny(dispatch 前拒,先于 bypass)。
      读不受影响:read-only profile 基线授权工作区读。父 ceilings 一并继承。
      write 子不加收窄(其隔离由 T7 worktree 供给);
    - ``workspace_mode`` 是 brief 签名之外的尾部默认参数(read_only 缺省):
      收窄判据需要它,既有签名调用零破坏。

    ``run_sweep=False``:子构造不是"打开库"事件——同库已由父打开时 sweep
    覆盖,且子根 turn 此刻尚未写入。更关键:write 子的锁文件族在子工作区,
    若以子工作区对全库 running turn 试锁,父/兄弟的锁(在父工作区)会被
    误判空闲而错标 interrupted。独立目录子的全库 sweep 工作区解析(按
    allocation 的 workspace 试锁)属 T7/T9 接线,本任务不开第二套 sweep。

    已知边界(如实,交 T6):read_only 子与父共享 permissions.json——子的
    always 授权落盘会写同一文件(子自身被 ceiling 否决不变,但父下次打开会
    读到子的 always 规则)。收口属后续任务,不在本任务重开已拍板语义。
    """
    ceilings = list(parent.permissions.ceilings)
    if workspace_mode == "read_only":
        denied = tuple(
            ToolRule(pattern=tool.definition.name, decision="deny")
            for tool in tool_subset
            if tool.static_effects & _READ_ONLY_DENIED_EFFECTS
        )
        # R-T5-6:read-only profile 是沙箱写否决的承重层(见 docstring (a));
        # 恒追加——即使工具表没有静态 WRITE 工具,bash 的写路径也必须关闭。
        ceilings.append(PermissionPolicy(profile="read-only", tools=denied))
    return build_session(
        workspace,
        parent.adapter,
        thread_id=thread_id,
        tools=tuple(tool_subset),
        policy=parent.permissions.policy,
        mode=parent.permissions.mode,
        budget=limits,
        unit_costs=parent.unit_costs,
        ceilings=tuple(ceilings),
        db_path=parent.store.db_path,
        run_sweep=False,
    )
