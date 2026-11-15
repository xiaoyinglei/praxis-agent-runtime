"""Praxis 唯一组合根(设计 §4.1;CI 规则 7:Session 构造只允许出现在本文件)。

build_session 把分层组件装配成一个可用会话,顺序即依赖序:
建 ``.praxis/`` 目录 → Store(单写者线程) → EventBus → **sweep**(打开即收编
崩溃遗留的 running turn → interrupted,§5.8;活执行者持锁,Sweep 跳过,绝不误伤)
→ **CommandSessions + 孤儿命令进程组清扫**(Task 15:同一实例注入 bash 与
manage_command;marker 清扫结果只记日志) → 默认内置工具(read/write/bash/
manage_command/list_files/search_text/inspect_data_file/apply_patch/
update_plan/run_python/read_artifact;顺序稳定,可由 tools 参数整体替换)
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
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from praxis.adapters.config.credentials import resolve_model_credential
from praxis.adapters.config.store import ModelConfigStore
from praxis.adapters.hooks.config import load_host_hooks
from praxis.adapters.hooks.runner import spawn_hook
from praxis.adapters.models.ollama import OllamaAdapter
from praxis.adapters.models.openai_compat import OpenAICompatAdapter
from praxis.adapters.models.retry import make_retry_stream
from praxis.adapters.models.tokenization import measure_request
from praxis.adapters.sandbox.base import probe_backend
from praxis.adapters.sandbox.compile import compile_sandbox, finalize_sandbox
from praxis.adapters.web.client import PublicWebError
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
from praxis.core.session import Session
from praxis.protocol.hooks import HookOutcome, HookSetup, HookSpec
from praxis.protocol.json import JsonValue
from praxis.protocol.models import (
    ModelAdapter,
    ModelMessage,
    ModelSettings,
    ModelSpec,
    WireRequest,
)
from praxis.protocol.permission import PermissionContext, PermissionMode
from praxis.protocol.policy import PermissionPolicy
from praxis.protocol.retry import RetryPolicy
from praxis.protocol.skills import SkillCatalog
from praxis.protocol.tools import Tool
from praxis.store.artifacts import ArtifactStore
from praxis.store.log import Store
from praxis.tools.builtins.artifacts_tool import create_read_artifact_tool
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
from praxis.tools.builtins.skills import load_skill_catalog

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
    - ask ⇒ raise ``PublicWebError("denied", "approval required for <host>")``
      ——逐跳/子资源是**执行期检查点**:ask 不再被无视放行,而是中止本次请求
      并指向 NetworkRule 授权路径;初始 URL 的 ask 由 decide 正常暂停审批;
    - deny ⇒ raise ``PublicWebError("denied")``。
    URL 原样交给 ``network_verdict``(归一化完全由 protocol.policy 承担);
    闭包内不再改写/重解析待裁决的 URL 串。
    """

    def authorize(url: str) -> None:
        verdict = network_verdict(permissions_provider(), url)
        if verdict == "allow":
            return
        if verdict == "ask":
            raise PublicWebError("denied", f"approval required for {_web_url_host(url)}")
        raise PublicWebError("denied")

    return authorize


def _web_url_host(url: str) -> str:
    """仅用于拒绝对话文案的主机名(不参与规则匹配——那是 network_decision 的事)。"""
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".") or url
    except ValueError:
        return url


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


def _resolve_policy(root: Path, explicit: PermissionPolicy | None) -> PermissionPolicy:
    """policy 来源序(Task 8):显式参数 > ``.praxis/permissions.json`` > 默认。

    文件是策略**声明**(always 授权由 Session 原子写合并于此);坏文件/坏 JSON
    一律 ValueError 边界拒绝,绝不静默回退默认(fail early at the boundary;
    pydantic strict 的 ValidationError 就是 ValueError 的子类)。
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
    hooks: HookSetup | None = None,
    skills_catalog: SkillCatalog | None = None,
) -> Session:
    """装配一个 Session:唯一组合根。CLI/TUI/host 一律经 sdk.open_session 走到这里。

    ``adapter is None`` 时从全局配置库解析:``ModelConfigStore.resolve_spec`` →
    ``resolve_model_credential`` → ``_adapter_for``(设计 §5.15;R-T6-2)。显式
    ``adapter``(测试/``--scripted``)完全绕过注册库。未注册临时模型只能经
    ``PRAXIS_MODEL_ID/PRAXIS_API_KEY/PRAXIS_BASE_URL`` 构造一个 ModelSpec,
    适配器仍然只由 ``_adapter_for`` 装配——不是第二装配入口。

    Task 8 入口:policy 见 _resolve_policy 来源序;mode 来源序 = 显式参数 >
    policy.approval,正交于策略(C5:bypass 只活在运行上下文,永不进 policy、
    永不落盘;切换 mode 不触碰任何策略规则),非法值边界拒绝。

    artifact 外部化与回读能力耦合:``tools`` 整体替换后的工具表里没有
    ``read_artifact`` 时,**不**注入 ArtifactStore,管道自动退回旧字节截断
    语义——不会给模型一个无法回读的指针。

    skill 目录(§5.14):``skills_catalog=None``(缺省)⇒ 本组合根从 workspace
    双根调用 ``load_skill_catalog``(PROJECT 上溯 + ``$SKILL_PATH``);显式参数
    (测试)完全绕过文件系统扫描。空目录是常态,注入步骤在 core 里自然恒等——
    不设 ``SKILLS_ENABLED`` 开关。composition 可 import tools,core 不可。
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
    store = Store(praxis_dir / "praxis.sqlite")
    try:
        bus = EventBus()
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
        tool_seq = (
            tools
            if tools is not None
            else (
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
            )
        )
        tool_map: dict[str, Tool] = {tool.definition.name: tool for tool in tool_seq}
        # 外部化与回读能力耦合:工具表被整体替换且没有 read_artifact 时,
        # 不注入 artifacts——否则模型只会拿到打不开的指针,比旧字节截断更差。
        effective_artifacts = artifact_store if "read_artifact" in tool_map else None
        effective_policy = _resolve_policy(root, policy)
        effective_mode: PermissionMode = mode if mode is not None else effective_policy.approval
        permissions = PermissionContext(
            workspace=str(root),
            policy=effective_policy,
            mode=effective_mode,
            # 真实探测:darwin seatbelt;其他平台不可用 → decide 第 7 步对
            # EXECUTE_PROCESS fail-closed deny(bwrap 延后,D2/D7)。
            sandbox=probe_backend(),
        )
        resolved_thread = thread_id if thread_id is not None else store.create_thread()
        store.ensure_thread(resolved_thread)
        return Session(
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
        )
    except BaseException:
        store.close()
        raise
