"""Praxis 测试公共 fixture(Task 4 起):workspace 目录与独立 db 的 Store。

Task 7: make_turn reaches requested status via transition() (the sole state-write
entry point), eliminating duplicate write_status/event-emit boilerplate and the
loop-body import that was hiding inside the for-loop.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

import pytest

from praxis.store.log import Store
from praxis.store.turns import Turn, create_turn, get_turn

# Sentinel — default "no status given" becomes RUNNING inside the function.
_UNSET: object = object()

# N-6 可断言证据:本进程最近一次隔离决策(供 test_host_isolation 读取)。
_ISOLATION_DECISIONS: dict[str, Any] = {}

# TMPDIR 会被本 fixture 重定向,而 tiktoken 的默认 blob 缓存是
# ``tempfile.gettempdir()/data-gym-cache``。M5 又剥掉了操作员代理:若每次
# 测试都从冷缓存重新下载,离线/代理依赖环境下 token 测量会永久退回 heuristic
# (test_compaction 的 BPE 口径断言会因此误红)。这里在 fixture 改写前捕获
# 宿主临时缓存目录,per-test 钉到 ``TIKTOKEN_CACHE_DIR``(已有 blob 直接复用;
# 无 blob 时 tiktoken 仍按自身语义下载到该目录)。
_HOST_TIKTOKEN_CACHE = Path(tempfile.gettempdir()) / "data-gym-cache"


@pytest.fixture(autouse=True)
def _isolate_host_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """N-6:每条测试都不能读写真实宿主目录(运维显式环境除外)。

    隔离两处全局状态:
    - ``PRAXIS_CONFIG_DIR``:``ModelConfigStore()``/凭据库的默认位置
      (``~/.config/praxis``);eager mkdir+建表会污染真实注册库。
    - ``TMPDIR``/``TEMP``/``TMP`` + ``tempfile.tempdir``:同时重定向
      ``command_jobs_dir()``(崩溃 marker 目录)、``compile_sandbox`` 的
      ``praxis-scratch-*`` 与一切 ``mkdtemp``(子进程经环境继承)。

    **运维显式设置 ``PRAXIS_CONFIG_DIR``(例如授权 live 冒烟指向自己的临时
    注册库)时一律尊重原环境,不注入任何重定向**——否则模型注册表会被换到
    空目录,live 冒烟永远找不到模型。这就是 ``host_isolation_updates`` 的
    纯函数契约(可单测,不依赖 fixture 时序)。
    """
    updates = host_isolation_updates(tmp_path, os.environ)
    if updates:
        Path(updates["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        Path(updates["PRAXIS_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    _ISOLATION_DECISIONS.clear()
    _ISOLATION_DECISIONS.update(
        reused_operator_env=not updates,
        operator_config_dir=os.environ.get("PRAXIS_CONFIG_DIR"),
        updates=updates,
    )
    for key, value in updates.items():
        monkeypatch.setenv(key, value)
    if "TMPDIR" in updates:
        # tempfile.gettempdir() 缓存模块级 tempdir;仅改 env 不够。
        monkeypatch.setattr(tempfile, "tempdir", updates["TMPDIR"], raising=False)
    # T20 累积转记:宿主配置/技能/搜索凭据的显式环境一律从测试进程剥掉,
    # 否则真实宿主文件/密钥会经默认加载路径漏进单测。测试需要时自行 setenv
    # (fixture 先执行,per-test monkeypatch 仍可覆盖)。
    for name in _HOST_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(_HOST_TIKTOKEN_CACHE))
    # 不在此删除 per-test host 目录:``child_env.scratch_dir_for`` 会把无沙箱
    # 单测路径的 fallback scratch 缓存在进程级;删掉旧 test 的目录会让后续
    # 测试拿到断链路径。目录都在 pytest basetemp 内,由 pytest 回收旧会话。
    yield


# T20 隔离名单:技能/宿主配置文件选择、web 限制与搜索 provider 覆盖,
# 以及测试套件使用的 *SEARCH_KEY 凭据哨兵(env→Keychain 解析会读它们)。
# M5:默认 open 现在每次都会构造 web client,``web_proxy_configuration`` 会读
# 代理 env;操作员代理设置不得渗进单测。运维授权 live 运行若需要代理/密钥,
# 在具体测试内自行 setenv(fixture 先执行,per-test monkeypatch 可覆盖)。
_HOST_ENV_NAMES: tuple[str, ...] = (
    "SKILL_PATH",
    "PRAXIS_MCP_CONFIG",
    "PRAXIS_HOOKS_CONFIG",
    "PRAXIS_WEB_LIMITS",
    "PRAXIS_WEB_SEARCH_PROVIDER",
    "DEEPSEEK_SEARCH_KEY",
    "BRAVE_SEARCH_KEY",
    "TAVILY_SEARCH_KEY",
    "PRAXIS_WEB_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "http_proxy",
    "HTTP_PROXY",
    "no_proxy",
    "NO_PROXY",
)


def host_isolation_updates(base: Path, environ: Mapping[str, str]) -> dict[str, str]:
    """计算 per-test 隔离要注入的环境。

    外部已给 ``PRAXIS_CONFIG_DIR`` 时返回空 dict(运维授权运行:live 冒烟使用
    自己的注册库;TMPDIR 也一并尊重)。否则注入 per-test 的 config 与 tmp,
    覆盖 ``PRAXIS_CONFIG_DIR`` / ``TMPDIR`` / ``TEMP`` / ``TMP``。
    """
    if environ.get("PRAXIS_CONFIG_DIR"):
        return {}
    host_root = base.with_name(f"{base.name}-host-state")
    return {
        "PRAXIS_CONFIG_DIR": str(host_root / "config"),
        "TMPDIR": str(host_root / "tmp"),
        "TEMP": str(host_root / "tmp"),
        "TMP": str(host_root / "tmp"),
    }

if TYPE_CHECKING:
    # Stubs for type-hints only; resolved at runtime inside functions.
    class _EventBus(Protocol):
        def emit(self, event: object) -> None: ...

    class _TransitionFn(Protocol):
        def __call__(
            self,
            store: Store,
            bus: _EventBus,
            turn_id: str,
            to: str,  # noqa: PYI041 — str annotation avoids forward-reference pain
            *,
            reason: str,
        ) -> Turn: ...


def make_turn(
    store: Store,
    status: str | None = None,  # type: ignore[assignment]
    thread_id: str = "t",
    _bus: object | None = None,
    *,
    _transition: object | None = None,
) -> Turn:
    """Create a new turn and (if needed) reach the requested status via transition().

    By default (_transition=None) delegates to
    ``praxis.core.statemachine.transition``.
    Callers may inject ``_transition`` or ``_bus`` for testing without duplicating logic.
    """
    tid = f"turn-{uuid4().hex}"
    # Lazy import — avoids any circular import dance at module load time.
    from praxis.core.events import EventBus  # noqa: PLC0415
    from praxis.core.statemachine import TurnStatus as TS  # noqa: PLC0415,N817
    from praxis.core.statemachine import transition as _default_transition  # noqa: PLC0415

    target: TS
    if status is None:
        target = TS.RUNNING
    else:
        target = TS(status)

    transition_fn: _TransitionFn = _transition or _default_transition

    turn = create_turn(store, turn_id=tid, thread_id=thread_id, binding={})
    assert turn.status == "running"
    if target != TS.RUNNING:
        bus = _bus or EventBus()
        transition_fn(store, bus, tid, target, reason=f"reach_{target.value}")
    result = get_turn(store, tid)
    assert result is not None and result.status == target.value, (
        f"expected {target.value}, got {result.status}"
    )
    return result


def make_minimal_tool(**overrides: object) -> Any:
    """构造一个最小可用的合法 Tool(阶段 2 协议层测试助手,沿用 make_turn 的惰性导入风格)。

    默认:只读玩具,run 为双参(签名 ToolRunner)返回 OK 结果,
    check_permission 不设(None)。任意 Tool 字段可用关键字 overrides 替换。
    """
    from praxis.protocol.tools import (  # noqa: PLC0415
        CancellationMode,
        InterruptBehavior,
        ResolvedUse,
        Tool,
        ToolDefinition,
        ToolEffect,
        ToolResult,
    )

    kwargs: dict[str, object] = dict(
        definition=ToolDefinition(
            name="minimal", description="minimal test tool", input_schema={"type": "object"}
        ),
        validate_input=lambda arguments: arguments,
        run=lambda arguments, exec_ctx: ToolResult(tool_call_id="pending", tool_name="minimal"),
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda arguments: ResolvedUse(
            effects=frozenset({ToolEffect.READ_WORKSPACE}), targets=()
        ),
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30.0,
        max_model_output_bytes=4096,
    )
    kwargs.update(overrides)
    return Tool(**kwargs)


# --- Helpers exposed for tests that need extra stores -----------------------


def stub_store() -> Store:
    """Create a temporary standalone store."""
    s = Store(tempfile.mktemp(suffix=".db"))
    return s


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def store(workspace: Path) -> Iterator[Store]:
    s = Store(workspace / "store.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def bus() -> Iterator[_EventBus]:  # noqa: F821
    from praxis.core.events import EventBus  # noqa: PLC0415

    eb = EventBus()
    yield eb
