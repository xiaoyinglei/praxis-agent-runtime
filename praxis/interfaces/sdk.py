"""Praxis SDK 公开面(设计 §5.12;findings R2:CLI/TUI/host 只经此模块触达内核)。

本模块是 praxis 对外唯一 import 面:会话与结果类型(Session/TurnResult/
TurnStatus/ApprovalSummary)、事件订阅(Subscription)、并发与守卫异常
(ThreadBusy/ConversationBusy/ResumeNotConfirmed)、装配入口
(``open_session = build_session``),以及 CLI 构造模型所需的适配器与协议类型
(OpenAICompatAdapter/ScriptedAdapter/ModelSpec/ModelResponse/ModelToolCall/
Usage/PermissionPolicy)——providers 细节对 CLI 只经 sdk 暴露(R2)。
R14 扩展:只读面 ``Session.items/turns/verify`` 配套 re-export
Item/ItemKind/Turn/VerifyReport——CLI 的 log/status/verify 命令仍不直依 core/store。
Task 8(R6-7)扩展:权限更新面 re-export PermissionUpdate/PathRule/ToolRule/
NetworkRule/PermissionMode/apply_update——宿主/UI 构造与合并更新只经此出口。
Task 11 扩展:权限更新披露 PermissionNotice(update_permission 的 D9 提示/
TurnResult.notices)同经 sdk 暴露。
D9 提醒:session/always 的 NetworkRule 授权放行 URL 级工具,但 bash
(EXECUTE_PROCESS)的命令网络因规则在场一律 deny(阶段 3 web 前无命令网络授权)。
Task 6 扩展:全局模型注册表与凭据面 re-export——ModelConfigStore/ModelEntry/
ConnectionEntry/CostMicros/ConfigVersionConflict、resolve/save_model_credential、
model_adapter_for/validate_model_spec/probe_model/ProbeOutcome。CLI 的
``praxis models`` 因此不直依 adapters(provider 分派仍只有 composition 一处)。
Task 17 扩展:artifact/计划/流式事件面 re-export——ArtifactStore/PlanSnapshot/
latest_plan、ModelEvent 与三个 delta 型、assemble_model_events;宿主/UI 读取
这些结构仍只经 sdk,不直依 core/store/protocol 内部模块。
Task 19 扩展:MCP 接线面 re-export——connect_mcp_servers/McpBundle/
McpServerStatus、resolve_declared_policy(声明快照唯一加载实现),以及
``praxis mcp`` 只读管理所需的配置面(load_mcp_configs/resolve_mcp_config_path/
decide_mcp_config_trust/mcp_config_write_path/normalize_mcp_name)与
save_mcp_bearer。宿主构造 typed ``extra_tools``/hooks/skills 面所需的
``ToolSource``/``SkillCatalog``/``load_host_hooks`` 一并出口(修复轮 #1;
无豁免)。CLI 因此不直依 adapters(契约 praxis_cli_via_sdk)。
Task 20 扩展:web 宿主配置面 ``WebLimits``/``load_web_limits`` 与 provider
覆盖 env 名 ``WEB_SEARCH_PROVIDER_ENV`` 经 sdk 出口(宿主配置限度、CLI
归因展示都不直依 adapters)。

``ConversationBusy``/``ResumeNotConfirmed`` 定义在 core.session 并由这里
re-export:core 不得 import interfaces(import-linter praxis_core_purity),
异常类必须定义在使用它们的最低层;sdk 仍是用户可见的唯一出口。

H-5:``OpenAICompatAdapter`` 保留导出(CLI/host 显式装配需要它),但它绕过
注册库装配路径(``composition._adapter_for``)。嵌入方直接构造时**须自行
负责** ``base_url`` 与凭据校验——``model_adapter_for``/``validate_model_spec``
才是带这些边界检查的唯一装配入口。
"""

from __future__ import annotations

from praxis.adapters.config.credentials import (
    credential_source,
    delete_web_search_credential,
    resolve_model_credential,
    resolve_web_search_credential,
    save_mcp_bearer,
    save_model_credential,
    save_web_search_credential,
    web_search_credential_source,
)
from praxis.adapters.config.store import (
    ConfigVersionConflict,
    ConnectionEntry,
    ModelConfigStore,
    ModelEntry,
)
from praxis.adapters.hooks.config import load_host_hooks
from praxis.adapters.mcp.config import (
    HttpServerConfig,
    McpServerConfig,
    StdioServerConfig,
    decide_mcp_config_trust,
    load_mcp_configs,
    mcp_config_write_path,
    resolve_mcp_config_path,
)
from praxis.adapters.mcp.manager import McpBundle, McpServerStatus
from praxis.adapters.mcp.names import normalize_mcp_name
from praxis.adapters.models.openai_compat import OpenAICompatAdapter
from praxis.adapters.models.probe import ProbeOutcome, probe_model
from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.adapters.web.limits import WebLimits, load_web_limits, web_configuration_path
from praxis.adapters.web.search import (
    API_KEY_ENV_NAME,
    SEARCH_ENDPOINTS,
    SEARCH_PROVIDERS,
    WEB_SEARCH_PROVIDER_ENV,
    SearchProvider,
    SearchProviderConfig,
    load_web_search_config,
)
from praxis.composition import (
    build_session,
    connect_mcp_servers,
    model_adapter_for,
    resolve_declared_policy,
    validate_model_spec,
)
from praxis.core.events import Subscription
from praxis.core.locks import ThreadBusy
from praxis.core.permission import apply_update
from praxis.core.planning import latest_plan
from praxis.core.session import ConversationBusy, ResumeNotConfirmed, Session
from praxis.core.statemachine import TurnStatus
from praxis.core.turn import ApprovalSummary, TurnResult
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.models import (
    CostMicros,
    ModelEvent,
    ModelReasoningDelta,
    ModelResponse,
    ModelSpec,
    ModelTextDelta,
    ModelToolCall,
    ModelToolCallDelta,
    Usage,
    assemble_model_events,
)
from praxis.protocol.permission import PermissionMode, PermissionNotice, PermissionUpdate
from praxis.protocol.plan import PlanSnapshot
from praxis.protocol.policy import NetworkRule, PathRule, PermissionPolicy, ToolRule
from praxis.protocol.skills import SkillCatalog, SkillManifest
from praxis.protocol.tools import ToolSource
from praxis.store.artifacts import ArtifactStore
from praxis.store.turns import Turn
from praxis.store.verify import VerifyReport
from praxis.tools.builtins.skills import load_skill_catalog

open_session = build_session

__all__ = [
    "API_KEY_ENV_NAME",
    "ApprovalSummary",
    "ArtifactStore",
    "ConfigVersionConflict",
    "ConnectionEntry",
    "ConversationBusy",
    "CostMicros",
    "HttpServerConfig",
    "Item",
    "ItemKind",
    "McpBundle",
    "McpServerConfig",
    "McpServerStatus",
    "ModelConfigStore",
    "ModelEntry",
    "ModelEvent",
    "ModelReasoningDelta",
    "ModelResponse",
    "ModelSpec",
    "ModelTextDelta",
    "ModelToolCall",
    "ModelToolCallDelta",
    "NetworkRule",
    "OpenAICompatAdapter",
    "PathRule",
    "PermissionMode",
    "PermissionNotice",
    "PermissionPolicy",
    "PermissionUpdate",
    "PlanSnapshot",
    "ProbeOutcome",
    "ResumeNotConfirmed",
    "SEARCH_ENDPOINTS",
    "SEARCH_PROVIDERS",
    "ScriptedAdapter",
    "SearchProvider",
    "SearchProviderConfig",
    "Session",
    "SkillCatalog",
    "SkillManifest",
    "StdioServerConfig",
    "Subscription",
    "ThreadBusy",
    "ToolRule",
    "ToolSource",
    "Turn",
    "TurnResult",
    "TurnStatus",
    "Usage",
    "VerifyReport",
    "WEB_SEARCH_PROVIDER_ENV",
    "WebLimits",
    "apply_update",
    "assemble_model_events",
    "connect_mcp_servers",
    "credential_source",
    "decide_mcp_config_trust",
    "delete_web_search_credential",
    "latest_plan",
    "load_host_hooks",
    "load_mcp_configs",
    "load_skill_catalog",
    "load_web_limits",
    "load_web_search_config",
    "mcp_config_write_path",
    "model_adapter_for",
    "normalize_mcp_name",
    "open_session",
    "probe_model",
    "resolve_declared_policy",
    "resolve_mcp_config_path",
    "resolve_model_credential",
    "resolve_web_search_credential",
    "save_mcp_bearer",
    "save_model_credential",
    "save_web_search_credential",
    "validate_model_spec",
    "web_configuration_path",
    "web_search_credential_source",
]
