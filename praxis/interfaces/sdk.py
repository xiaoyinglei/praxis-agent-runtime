"""Praxis SDK 公开面(设计 §5.12;findings R2:CLI/TUI/host 只经此模块触达内核)。

本模块是 praxis 对外唯一 import 面:会话与结果类型(Session/TurnResult/
TurnStatus/ApprovalSummary)、事件订阅(Subscription)、并发与守卫异常
(ThreadBusy/ConversationBusy/ResumeNotConfirmed)、装配入口
(``open_session = build_session``),以及 CLI 构造模型所需的适配器与协议类型
(OpenAICompatAdapter/ScriptedAdapter/ModelSpec/ModelResponse/ModelToolCall/
Usage/PermissionPolicy)——providers 细节对 CLI 只经 sdk 暴露(R2)。
R14 扩展:只读面 ``Session.items/turns/verify`` 配套 re-export
Item/ItemKind/Turn/VerifyReport——CLI 的 log/status/verify 命令仍不直依 core/store。

``ConversationBusy``/``ResumeNotConfirmed`` 定义在 core.session 并由这里
re-export:core 不得 import interfaces(import-linter praxis_core_purity),
异常类必须定义在使用它们的最低层;sdk 仍是用户可见的唯一出口。
"""

from __future__ import annotations

from praxis.adapters.models.openai_compat import OpenAICompatAdapter
from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.composition import build_session
from praxis.core.events import Subscription
from praxis.core.locks import ThreadBusy
from praxis.core.session import ConversationBusy, ResumeNotConfirmed, Session
from praxis.core.statemachine import TurnStatus
from praxis.core.turn import ApprovalSummary, TurnResult
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.models import ModelResponse, ModelSpec, ModelToolCall, Usage
from praxis.protocol.policy import PermissionPolicy
from praxis.store.turns import Turn
from praxis.store.verify import VerifyReport

open_session = build_session

__all__ = [
    "ApprovalSummary",
    "ConversationBusy",
    "Item",
    "ItemKind",
    "ModelResponse",
    "ModelSpec",
    "ModelToolCall",
    "OpenAICompatAdapter",
    "PermissionPolicy",
    "ResumeNotConfirmed",
    "ScriptedAdapter",
    "Session",
    "Subscription",
    "ThreadBusy",
    "Turn",
    "TurnResult",
    "TurnStatus",
    "Usage",
    "VerifyReport",
    "open_session",
]
