#!/usr/bin/env python3
"""Scan rag/ for hardcoded model names, providers, and base URLs in business code.

Scans all .py files under rag/ except:
  - rag/models/        (model catalog — the config system itself)
  - rag/assembly/      (provider factory — legitimately references model names)

Also scans praxis/ for architecture hard-rule violations (阶段1 CI 硬规则):
  1. 函数内 import(0 容忍)
  2. 跨模块私有访问(_private import / attribute)
  3. UPDATE turns SQL 只允许写入 store/turns.py
  4. write_status() 调用只允许 core/statemachine.py
  5. transition(..., TurnStatus.INTERRUPTED) 只允许 core/recovery.py(设计 §5.2)
  6. 禁止 import agent_runtime(旧 runtime)
  7. Session(...) 构造只允许 composition.py(设计 §4.1)

阶段2 追加(单一来源机械强制,设计 §8 / findings S1/S2):
  8. 沙箱 profile 构造字面量("(deny default)"/"(version 1)"/"sandbox-exec"/
     "--unshare-all"/"bwrap")只允许出现在 adapters/sandbox/(S1 唯一编译器)。
     启发式:只看 AST 代码字符串常量——注释根本不进 AST,docstring 显式排除,
     因此协议层 docstring"解释"后端名词不误报(与验收 test_s1 同一口径)。
  9. def path_access/resolve_policy/tool_decision/network_decision/
     command_network_decision 只允许 protocol/policy.py(唯一合并器)。
 10. def decide 只允许 core/permission.py(唯一权限决策入口)。
 11. def floor_reason 与 FLOOR_ENTRIES 赋值只允许 protocol/floor.py(唯一地板)。

阶段3A 追加(Task 17:单一来源继续收口):
 12. provider SDK 面(chat.completions.create/AsyncOpenAI/text/event-stream)
     只允许 adapters/models/;
 13. def measure_request/count_text_tokens 只允许 adapters/models/tokenization.py;
 14. 重试机制(classify_failure/stream_with_retry/make_retry_stream)只允许
     adapters/models/retry.py;重试策略类型(RetryPolicy/ModelFailure/RetryPlan/
     NoRetryPlan/AttemptRecord/clock_floor/ModelUnavailableError)只允许
     protocol/retry.py(两侧各唯一);
 15. CREATE TABLE/sqlite3.connect 只允许 store/log.py 与 adapters/config/store.py;
 16. 凭据存储(find-generic-password//usr/bin/security/0o600 凭据读写)只允许
     adapters/config/credentials.py;
 17. def put( artifact 落盘)只允许 store/artifacts.py;
 18. MAX_PLAN_STEPS 定义只允许 protocol/plan.py。

阶段3B 追加(Task 20:CI 单一来源硬规则与工具表收口):
 19. ``mcp__`` 字符串字面量只允许 adapters/mcp/names.py(旧 6 处拼装漂移的
     结构预防;代码字符串,docstring 排除);
 20. S7 argv 密钥重置形态全仓禁止:代码字符串含 ``/usr/bin/env``,或同一
     序列字面量里 ``-i`` 与 env 可执行名共现(``/usr/bin/env -i`` 形态);
 21. B33 宿主 env 直接进子进程:``env={**os.environ}``、``os.environ.copy()``、
     ``dict(os.environ)`` 与 ``env=os.environ`` 全仓禁止(常量白名单之外无路);
 22. 唯一定义点:child_env/signal_group/await_main_exit/drain_stream/
     skill_fingerprint/atomic_write_bytes/atomic_write_text/load_host_hooks/
     spawn_hook/connect_mcp_servers/web_configuration_path 各自唯一
     (retry 机制三项沿用规则 14,不重复登记);
 23. 宿主配置文件读取面(host_config_dir() 调用与
     PRAXIS_{MCP_CONFIG,HOOKS_CONFIG,WEB_LIMITS,WEB_SEARCH_PROVIDER}
     字面量)只允许适配器配置层(adapters/{hooks,config}/、web limits/search、
     mcp config)与 protocol/floor.py(路径定义点)。

``--praxis-only``:只扫 praxis 硬规则、跳过 rag 存量扫描(CI 门 R5:阶段1 只强制 praxis 规则)。

Exit code 0: no anti-patterns found.
Exit code 1: anti-patterns detected.
"""

from __future__ import annotations

import argparse
import ast
import re
from collections.abc import Sequence
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RAG_DIR = _PROJECT_ROOT / "rag"
_PRAXIS_DIR = _PROJECT_ROOT / "praxis"

_EXCLUDED_DIRS = {
    "models",
    "__pycache__",
    ".mypy_cache",
}


def _is_infrastructure_file(filepath: str) -> bool:
    """Provider implementations and assembly factory legitimately reference model names."""
    infra_prefixes = ("rag/assembly/", "rag/providers/")
    return any(filepath.startswith(p) for p in infra_prefixes)


def _scan_file(path: Path) -> list[str]:
    violations: list[str] = []
    content = path.read_text(encoding="utf-8")
    rel = str(path.relative_to(_PROJECT_ROOT))

    # 1. Hardcoded model name defaults
    if m := re.search(r'DEFAULT_SUMMARY_MODEL\s*=\s*"([^"]+)"', content):
        violations.append(f"{rel}: DEFAULT_SUMMARY_MODEL = \"{m.group(1)}\" — use ModelRuntimeConfig instead")

    if m := re.search(r'DEFAULT_SUMMARY_PROVIDER_KIND\s*=\s*"([^"]+)"', content):
        violations.append(f"{rel}: DEFAULT_SUMMARY_PROVIDER_KIND = \"{m.group(1)}\" — use ModelRuntimeConfig instead")

    if m := re.search(r'DEFAULT_SUMMARY_BACKEND\s*=\s*"([^"]+)"', content):
        violations.append(f"{rel}: DEFAULT_SUMMARY_BACKEND = \"{m.group(1)}\" — use ModelRuntimeConfig instead")

    # 2. Hardcoded model name strings (skip infrastructure files)
    if not _is_infrastructure_file(rel):
        model_patterns = [
            (r'"deepseek-chat"', "hardcoded model name 'deepseek-chat'"),
            (r'"deepseek-reasoner"', "hardcoded model name 'deepseek-reasoner'"),
            (r'"qwen3-embedding', "hardcoded model name containing 'qwen3-embedding'"),
            (r'"Qwen/', "hardcoded model name containing 'Qwen/'"),
            (r'"BAAI/', "hardcoded model name containing 'BAAI/'"),
            (r'"mlx-community/', "hardcoded model name containing 'mlx-community/'"),
        ]
        # 3. DEFAULT_TOKENIZER_FALLBACK_MODEL is infrastructure, not business code
        content_without_fallback = content.replace(
            'DEFAULT_TOKENIZER_FALLBACK_MODEL = "BAAI/bge-m3"', ""
        )

        for pattern, message in model_patterns:
            if re.search(pattern, content_without_fallback):
                violations.append(f"{rel}: {message} — use configs/models.yaml")

        # 3. Hardcoded base URLs
        url_patterns = [
            (r'"https?://api\.deepseek\.com', "hardcoded DeepSeek base URL"),
            (r'"https?://api\.openai\.com', "hardcoded OpenAI base URL"),
        ]
        for pattern, message in url_patterns:
            if re.search(pattern, content):
                violations.append(f"{rel}: {message} — use configs/models.yaml")

        # 4. Hardcoded api_key_env in business code
        if m := re.search(r'api_key_env\s*=\s*"([^"]+)"', content):
            violations.append(
                f"{rel}: hardcoded api_key_env=\"{m.group(1)}\" — model keys belong in configs/models.yaml"
            )

    return violations


# ---------------------------------------------------------------------------
# Praxis 硬规则扫描(阶段1)
#
# 唯一写入点白名单严格等于"相对 praxis 根的路径":praxis/composition.py 豁免,
# praxis/flow/composition.py 不豁免。scan_praxis_file 仍为纯函数、可接受任意路径:
# 优先用显式 praxis_root,否则在路径中定位名为 praxis 的目录段,再否则视为"父目录即根"
# (仅 basename 参与比较,便于 tmp fixture)。
# ---------------------------------------------------------------------------

_PRAXIS_TURN_SQL_FILE = ("store", "turns.py")
_PRAXIS_WRITE_STATUS_FILE = ("core", "statemachine.py")
_PRAXIS_INTERRUPTED_FILE = ("core", "recovery.py")
_PRAXIS_SESSION_FILE = ("composition.py",)

# 阶段2 单一来源豁免路径(同样严格等于"相对 praxis 根的路径";沙箱是"整个包"豁免)
_PRAXIS_SANDBOX_PKG = ("adapters", "sandbox")
_PRAXIS_POLICY_FILE = ("protocol", "policy.py")
_PRAXIS_DECIDE_FILE = ("core", "permission.py")
_PRAXIS_FLOOR_FILE = ("protocol", "floor.py")

# profile 构造字面量(S1):seatbelt SBPL / bwrap argv 的构造痕迹,唯一编译器持有
_PROFILE_CONSTRUCTION_LITERALS = ("(deny default)", "(version 1)", "sandbox-exec", "--unshare-all", "bwrap")
# 规则代数函数(Task 2 定义,Task 4/5/6 消费):第二合并器 = 审查直接拒绝(§5.7)
_RULE_ALGEBRA_FUNCS = frozenset(
    {"path_access", "resolve_policy", "tool_decision", "network_decision", "command_network_decision"}
)

# 阶段3A(Task 17)单一来源白名单:同样严格等于"相对 praxis 根的路径"。
_PRAXIS_MODELS_PKG = ("adapters", "models")
_PRAXIS_TOKENIZATION_FILE = ("adapters", "models", "tokenization.py")
_PRAXIS_ADAPTERS_RETRY_FILE = ("adapters", "models", "retry.py")
_PRAXIS_PROTOCOL_RETRY_FILE = ("protocol", "retry.py")
_PRAXIS_GLOBAL_SQL_FILES = frozenset(
    {("store", "log.py"), ("adapters", "config", "store.py")}
)
_PRAXIS_CREDENTIALS_FILE = ("adapters", "config", "credentials.py")
_PRAXIS_ARTIFACTS_FILE = ("store", "artifacts.py")
_PRAXIS_PLAN_FILE = ("protocol", "plan.py")

_RETRY_MECHANISM_FUNCS = frozenset({"classify_failure", "stream_with_retry", "make_retry_stream"})
_RETRY_POLICY_CLASSES = frozenset(
    {"RetryPolicy", "ModelFailure", "RetryPlan", "NoRetryPlan", "AttemptRecord", "ModelUnavailableError"}
)
# 规则 16 的凭据信号:命令/Keychain 字面量与"凭据路径 + 0600"组合。
_CREDENTIAL_COMMAND_MARKERS = ("find-generic-password",)
_SECURITY_BASENAMES = frozenset({"security", "/usr/bin/security"})

# 阶段3B(Task 20)单一来源白名单:同样严格等于"相对 praxis 根的路径"。
_PRAXIS_MCP_NAMES_FILE = ("adapters", "mcp", "names.py")
_PRAXIS_CHILD_ENV_FILE = ("tools", "builtins", "child_env.py")
_PRAXIS_PROCESS_FILE = ("tools", "builtins", "process.py")
_PRAXIS_SKILLS_FILE = ("tools", "builtins", "skills.py")
_PRAXIS_PATHS_FILE = ("tools", "builtins", "paths.py")
_PRAXIS_WEB_LIMITS_FILE = ("adapters", "web", "limits.py")
_PRAXIS_HOOKS_CONFIG_FILE = ("adapters", "hooks", "config.py")
_PRAXIS_HOOKS_RUNNER_FILE = ("adapters", "hooks", "runner.py")
_PRAXIS_MCP_MANAGER_FILE = ("adapters", "mcp", "manager.py")

# 规则 20(S7):argv 密钥重置形态的两个字面量。``-i`` 单独出现(如
# ``security -i`` 的 stdin 模式)不是重置形态——只有与 env 可执行名共处同一
# 序列字面量时才拒绝;``/usr/bin/env`` 本身在任何代码字符串里都拒绝。
# 盲区(文档化,非静默):拆分为多个字面量再拼("exe" + "nv")、或把
# ``"/usr/bin/env"`` 与 ``"-i"`` 分两次 append 进同一列表,静态字面量扫描
# 无法跨表达式/跨语句关联——与规则 8 的代码字符串口径同类的固有限制。
_S7_ENV_RESET_BINARY = "/usr/bin/env"
_S7_ENV_NAME = "env"
_S7_RESET_FLAG = "-i"

# 规则 22(阶段3B):单一定义点。retry 机制三项已由 _RETRY_MECHANISM_FUNCS
# (规则 14)钉住,这里不重复登记。connect_mcp_servers 的**唯一实现**在
# adapters/mcp/manager.py;composition.py 是注入绑定 facade(R-T15-7,env_builder/
# sandbox_compiler 的唯一绑定点),同属允许面。
# 盲区(文档化,非静默):``f = lambda ...:`` 形式的同名重绑定不是 def 节点,
# 静态 def 扫描看不到——与旧规则 4/5/7 同一机制口径,交给整分支审查台账。
_SINGLE_DEFINITION_SITES: dict[str, frozenset[tuple[str, ...]]] = {
    "child_env": frozenset({_PRAXIS_CHILD_ENV_FILE}),
    "signal_group": frozenset({_PRAXIS_PROCESS_FILE}),
    "await_main_exit": frozenset({_PRAXIS_PROCESS_FILE}),
    "drain_stream": frozenset({_PRAXIS_PROCESS_FILE}),
    "skill_fingerprint": frozenset({_PRAXIS_SKILLS_FILE}),
    "atomic_write_bytes": frozenset({_PRAXIS_PATHS_FILE}),
    "atomic_write_text": frozenset({_PRAXIS_PATHS_FILE}),
    "load_host_hooks": frozenset({_PRAXIS_HOOKS_CONFIG_FILE}),
    "spawn_hook": frozenset({_PRAXIS_HOOKS_RUNNER_FILE}),
    "connect_mcp_servers": frozenset({_PRAXIS_MCP_MANAGER_FILE, _PRAXIS_SESSION_FILE}),
    "web_configuration_path": frozenset({_PRAXIS_WEB_LIMITS_FILE}),
}

# 规则 23(阶段3B):宿主配置文件读取面。adapters/{hooks,config}/ 是整包允许
# (hooks.json 与 config.sqlite/credentials 同属宿主配置层);web 的两个加载器
# 与 mcp 配置各自唯一;protocol/floor.py 是 host_config_dir 的定义点。
_HOST_CONFIG_DIR_PACKAGE_PREFIXES = frozenset(
    {("adapters", "hooks"), ("adapters", "config")}
)
_HOST_CONFIG_DIR_FILES = frozenset(
    {
        _PRAXIS_WEB_LIMITS_FILE,
        ("adapters", "web", "search.py"),
        ("adapters", "mcp", "config.py"),
        _PRAXIS_FLOOR_FILE,
    }
)
_HOST_CONFIG_ENV_LITERALS = (
    "PRAXIS_MCP_CONFIG",
    "PRAXIS_HOOKS_CONFIG",
    "PRAXIS_WEB_LIMITS",
    "PRAXIS_WEB_SEARCH_PROVIDER",
)


def _rel_to_praxis_parts(path: Path, praxis_root: Path | None) -> tuple[str, ...]:
    if praxis_root is not None:
        try:
            return path.relative_to(praxis_root).parts
        except ValueError:
            pass
    parts = path.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "praxis":
            return parts[i + 1 :]
    return parts[-1:]


def _has_update_turns_literal(tree: ast.Module) -> bool:
    # 只看字符串字面量(AST Constant str),注释/文档外的文本不触发
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if re.search(r"UPDATE\s+turns", node.value):
                return True
    return False


def _docstring_constant_ids(tree: ast.Module) -> set[int]:
    """docstring 字符串常量的 id() 集合(规则 8 排除用)。

    启发式与验收 test_s1_no_second_profile_builder 同一口径:规则 8 只钉"第二构造器"
    ——注释根本不进 AST;模块/类/函数体首表达式的 str 常量是 docstring,排除之,
    其余 str 常量(f-string 的组成部分同样是 Constant)视为代码字符串。
    """
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _is_private(name: str) -> bool:
    # dunder(__init__/__version__ 等)不算跨模块私有名
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _dotted_chain(node: ast.AST) -> list[str] | None:
    """把 a.b.c 形式的 Attribute/Name 链解析为 ['a','b','c'];无法解析则返回 None。"""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return parts
    return None


def _call_is(node: ast.Call, name: str) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == name


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
        cur = parents.get(cur)
    return False


def _host_config_reader_allowed(rel_parts: tuple[str, ...]) -> bool:
    """规则 23:宿主配置文件读取面白名单(包前缀或精确文件)。"""
    if rel_parts[:2] in _HOST_CONFIG_DIR_PACKAGE_PREFIXES:
        return True
    return rel_parts in _HOST_CONFIG_DIR_FILES


def _is_environment_items(node: ast.AST, aliases: set[str]) -> bool:
    """``os.environ.items()`` / ``<alias>.items()``(dict-comp 与 dict(...) 的源)。"""
    if not isinstance(node, ast.Call):
        return False
    chain = _dotted_chain(node.func)
    if chain == ["os", "environ", "items"]:
        return True
    return chain is not None and len(chain) == 2 and chain[0] in aliases and chain[1] == "items"


def _is_environment_source(node: ast.AST, aliases: set[str]) -> bool:
    """规则 21(B33):宿主 env 源判定(含一层别名,无数据流:保守失败方向)。

    覆盖 ``os.environ`` 本身及其别名、``.copy()``、``dict(...)``、
    ``{**...}``、``|`` 合并(BinOp BitOr)与 ``dict(os.environ.items())``。
    别名检测是单遍、无作用域/重赋值追踪的保守近似:宁可拒绝一次合法复用,
    也不放过宿主 env 进子进程;别名后又被常量覆盖的误报面已由测试记录。
    """
    chain = _dotted_chain(node)
    if chain == ["os", "environ"]:
        return True
    if chain is not None and len(chain) == 1 and chain[0] in aliases:
        return True
    if isinstance(node, ast.Call):
        call_chain = _dotted_chain(node.func)
        if call_chain == ["os", "environ", "copy"]:
            return True
        if (
            call_chain is not None
            and len(call_chain) == 2
            and call_chain[0] in aliases
            and call_chain[1] == "copy"
        ):
            return True
        if call_chain == ["dict"] and node.args:
            first = node.args[0]
            if _is_environment_source(first, aliases) or _is_environment_items(first, aliases):
                return True
    if isinstance(node, ast.Dict):
        return any(
            key is None and _is_environment_source(value, aliases)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_environment_source(node.left, aliases) or _is_environment_source(
            node.right, aliases
        )
    if isinstance(node, ast.DictComp):
        return any(_is_environment_items(generator.iter, aliases) for generator in node.generators)
    return False


def _environment_aliases(tree: ast.Module) -> set[str]:
    """收集 ``x = <宿主 env 源>`` 的别名(规则 21 的一层别名面)。"""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        if not _is_environment_source(node.value, aliases):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                aliases.add(target.id)
    return aliases


def scan_praxis_file(path: Path, praxis_root: Path | None = None) -> list[str]:
    """扫描单个 praxis 文件,返回违规描述列表(纯函数,可接受任意路径)。"""
    violations: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"line {exc.lineno or 0}: syntax error: {exc.msg}"]

    rel_parts = _rel_to_praxis_parts(path, praxis_root)

    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    # 顶层 import 绑定的名字,用于判定"跨模块私有访问"的属性链根
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", None)
            names = [alias.name for alias in node.names]
            # 规则 6:praxis 任何文件不得 import agent_runtime
            roots: list[str] = []
            if isinstance(node, ast.Import):
                roots = names
            elif module and node.level == 0:
                roots = [module]
            for root in roots:
                if root == "agent_runtime" or root.startswith("agent_runtime."):
                    violations.append(
                        f"line {node.lineno}: praxis must not import agent_runtime "
                        f"(found import of {root}) — 设计红线:praxis 不得依赖旧 runtime"
                    )
            # 规则 1:0 处函数内 import
            if _enclosing_function(node, parents):
                what = module if isinstance(node, ast.ImportFrom) else ", ".join(names)
                violations.append(
                    f"line {node.lineno}: function-level import ({what}) — imports 必须放模块顶层"
                )
            # 规则 2a:跨模块私有 import(dunder 除外)
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if _is_private(alias.name):
                        violations.append(
                            f"line {node.lineno}: cross-module private import "
                            f"({module} import {alias.name}) — 只走公开接口"
                        )

    # 规则 2b:对已 import 模块的私有属性访问
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = _dotted_chain(node)
            if chain and len(chain) >= 2 and chain[0] in imported_names and _is_private(chain[-1]):
                violations.append(
                    f"line {node.lineno}: cross-module private access ({'.'.join(chain)}) — 只走公开接口"
                )

    # 规则 3:UPDATE turns 只允许 store/turns.py(仅字符串字面量)
    if rel_parts != _PRAXIS_TURN_SQL_FILE and _has_update_turns_literal(tree):
        violations.append("UPDATE turns SQL outside store/turns.py — turn 状态写入只允许 store 层唯一入口")

    # 规则 8(阶段2/S1):沙箱 profile 构造字面量只允许 adapters/sandbox/(代码字符串,排除 docstring)
    if rel_parts[:2] != _PRAXIS_SANDBOX_PKG:
        docstrings = _docstring_constant_ids(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)) or id(node) in docstrings:
                continue
            for marker in _PROFILE_CONSTRUCTION_LITERALS:
                if marker in node.value:
                    violations.append(
                        f"line {node.lineno}: sandbox profile construction outside adapters/sandbox "
                        f"(S1 single source) — 代码字符串含 profile/后端字面量 {marker!r},"
                        "沙箱 profile 只允许 adapters/sandbox/ 唯一编译器构造"
                    )

    # 规则 9-11(阶段2):规则代数/决策/地板的唯一定义点
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _RULE_ALGEBRA_FUNCS and rel_parts != _PRAXIS_POLICY_FILE:
                violations.append(
                    f"line {node.lineno}: rule algebra outside protocol/policy.py (single merger) — "
                    f"def {node.name}() 权限规则代数只允许在 policy.py"
                )
            if node.name == "decide" and rel_parts != _PRAXIS_DECIDE_FILE:
                violations.append(
                    f"line {node.lineno}: permission decision outside core/permission.py (single entry) — "
                    "def decide()(含同名方法)权限决策唯一入口"
                )
            if node.name == "floor_reason" and rel_parts != _PRAXIS_FLOOR_FILE:
                violations.append(
                    f"line {node.lineno}: floor definition outside protocol/floor.py (single floor) — "
                    "def floor_reason() 地板判定只允许在 floor.py"
                )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is None:  # 裸注解无定义,不算第二地板清单
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assigns_floor = any(isinstance(t, ast.Name) and t.id == "FLOOR_ENTRIES" for t in targets)
            if assigns_floor and rel_parts != _PRAXIS_FLOOR_FILE:
                violations.append(
                    f"line {node.lineno}: floor definition outside protocol/floor.py (single floor) — "
                    "FLOOR_ENTRIES 地板清单单一数据源(FloorSpec),后端只能 import 消费"
                )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # 规则 4:write_status() 只允许 core/statemachine.py
        if _call_is(node, "write_status") and rel_parts != _PRAXIS_WRITE_STATUS_FILE:
            violations.append(
                f"line {node.lineno}: write_status() called outside core/statemachine.py — 权限决策/状态转换唯一入口"
            )
        # 规则 7:Session() 构造只允许 praxis/composition.py
        if _call_is(node, "Session") and rel_parts != _PRAXIS_SESSION_FILE:
            violations.append(
                f"line {node.lineno}: Session(...) constructed outside composition.py — 组合根唯一(设计 §4.1)"
            )
        # 规则 5:transition(..., TurnStatus.INTERRUPTED) 只允许 core/recovery.py
        if _call_is(node, "transition") and rel_parts != _PRAXIS_INTERRUPTED_FILE:
            args = list(node.args) + [kw.value for kw in node.keywords]
            for arg in args:
                if _dotted_chain(arg) == ["TurnStatus", "INTERRUPTED"]:
                    violations.append(
                        f"line {node.lineno}: transition(..., TurnStatus.INTERRUPTED) outside core/recovery.py — "
                        "interrupted 只能由 recovery 写入(设计 §5.2)"
                    )
                    break

    # 规则 12-18(阶段3A/Task 17):provider SDK、token 测量、重试分层、
    # 全局 SQL、凭据存储、artifact put 与计划常量各自唯一。纯增:不改上方旧规则。
    docstrings = _docstring_constant_ids(tree)
    in_models_pkg = rel_parts[:2] == _PRAXIS_MODELS_PKG
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = _dotted_chain(node)
            if (
                chain is not None
                and chain[-3:] == ["chat", "completions", "create"]
                and not in_models_pkg
            ):
                violations.append(
                    f"line {node.lineno}: provider SDK surface outside adapters/models (B6/B25) — "
                    "chat.completions.create 只允许在 adapters/models/ 内"
                )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            if "text/event-stream" in node.value and not in_models_pkg:
                violations.append(
                    f"line {node.lineno}: provider SDK surface outside adapters/models (B6/B25) — "
                    "text/event-stream 字面量只允许在 adapters/models/ 内"
                )
            if "CREATE TABLE" in node.value and rel_parts not in _PRAXIS_GLOBAL_SQL_FILES:
                violations.append(
                    f"line {node.lineno}: global SQL outside store/log.py and adapters/config/store.py — "
                    "CREATE TABLE 只允许两个全局库单一来源"
                )
            if rel_parts != _PRAXIS_CREDENTIALS_FILE:
                if any(marker in node.value for marker in _CREDENTIAL_COMMAND_MARKERS):
                    violations.append(
                        f"line {node.lineno}: credential storage outside adapters/config/credentials.py — "
                        "find-generic-password 只允许凭据层"
                    )
                if node.value in _SECURITY_BASENAMES:
                    violations.append(
                        f"line {node.lineno}: credential storage outside adapters/config/credentials.py — "
                        "security 子进程只允许凭据层"
                    )
        elif isinstance(node, ast.Call):
            chain = _dotted_chain(node.func)
            if chain is not None and chain[-1] == "AsyncOpenAI" and not in_models_pkg:
                violations.append(
                    f"line {node.lineno}: provider SDK surface outside adapters/models (B6/B25) — "
                    "AsyncOpenAI 只在 adapters/models/ 构造"
                )
            if chain == ["sqlite3", "connect"] and rel_parts not in _PRAXIS_GLOBAL_SQL_FILES:
                violations.append(
                    f"line {node.lineno}: global SQL outside store/log.py and adapters/config/store.py — "
                    "sqlite3.connect 只允许两个全局库单一来源"
                )
            if rel_parts != _PRAXIS_CREDENTIALS_FILE and chain in (["os", "open"], ["os", "chmod"]):
                args = list(node.args) + [kw.value for kw in node.keywords]
                has_private_mode = any(
                    isinstance(arg, ast.Constant) and type(arg.value) is int and arg.value == 0o600
                    for arg in args
                )
                has_credential_path = any(
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and "credential" in arg.value.casefold()
                    for arg in args
                )
                if has_private_mode and has_credential_path:
                    violations.append(
                        f"line {node.lineno}: credential storage outside adapters/config/credentials.py — "
                        "0o600 凭据读写只允许凭据层"
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in ("measure_request", "count_text_tokens") and rel_parts != _PRAXIS_TOKENIZATION_FILE:
                violations.append(
                    f"line {node.lineno}: token measurement outside adapters/models/tokenization.py — "
                    f"def {node.name}() 只允许 tokenization.py"
                )
            if node.name in _RETRY_MECHANISM_FUNCS and rel_parts != _PRAXIS_ADAPTERS_RETRY_FILE:
                violations.append(
                    f"line {node.lineno}: retry mechanism outside adapters/models/retry.py — "
                    f"def {node.name}() 只允许 adapters/models/retry.py"
                )
            if node.name == "clock_floor" and rel_parts != _PRAXIS_PROTOCOL_RETRY_FILE:
                violations.append(
                    f"line {node.lineno}: retry policy definition outside protocol/retry.py — "
                    "def clock_floor() 只允许 protocol/retry.py"
                )
            if node.name == "put" and rel_parts != _PRAXIS_ARTIFACTS_FILE:
                violations.append(
                    f"line {node.lineno}: artifact put() outside store/artifacts.py — "
                    "artifact 落盘 put() 只允许 store/artifacts.py"
                )
        elif isinstance(node, ast.ClassDef):
            if node.name in _RETRY_POLICY_CLASSES and rel_parts != _PRAXIS_PROTOCOL_RETRY_FILE:
                violations.append(
                    f"line {node.lineno}: retry policy definition outside protocol/retry.py — "
                    f"class {node.name} 只允许 protocol/retry.py"
                )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is None:  # 裸注解无定义,不算第二来源
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if (
                any(isinstance(t, ast.Name) and t.id == "MAX_PLAN_STEPS" for t in targets)
                and rel_parts != _PRAXIS_PLAN_FILE
            ):
                violations.append(
                    f"line {node.lineno}: MAX_PLAN_STEPS definition outside protocol/plan.py — "
                    "计划上限只允许 protocol/plan.py 单一来源"
                )

    # 规则 19-23(阶段3B/Task 20):MCP 名字、S7 argv 重置、B33 宿主 env、
    # 单一定义点与宿主配置文件读取面。纯增:不改上方旧规则。
    environment_aliases = _environment_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            # 规则 19:MCP 工具名前缀只允许 names.py 唯一拼装
            # 盲区(文档化,非静默):拼接/格式化("mcp_" + "_x")可绕开字面量
            # 扫描——与规则 8 的代码字符串口径同类的固有限制,不是遗漏。
            if "mcp__" in node.value and rel_parts != _PRAXIS_MCP_NAMES_FILE:
                violations.append(
                    f"line {node.lineno}: MCP tool-name literal outside adapters/mcp/names.py — "
                    "mcp__ 前缀只允许 names.py(canonical_tool_name/server_wildcard/global_wildcard)"
                )
            # 规则 20(S7):env 重置可执行名本身全仓禁止
            if _S7_ENV_RESET_BINARY in node.value:
                violations.append(
                    f"line {node.lineno}: S7 argv env reset outside the envp-only posture — "
                    f"代码字符串含 {_S7_ENV_RESET_BINARY!r};配置 env 只允许经 StdioServerParameters(env=...)"
                )
            # 规则 23:宿主配置 env 名只允许配置层读取
            if not _host_config_reader_allowed(rel_parts):
                for literal in _HOST_CONFIG_ENV_LITERALS:
                    if literal in node.value:
                        violations.append(
                            f"line {node.lineno}: host configuration env read outside the "
                            f"config adapters — {literal} 只允许宿主配置加载层"
                        )
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            # 规则 20(S7):``/usr/bin/env -i`` / ``env -i`` 的 argv 重置形态
            values = [
                element.value
                for element in node.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
            if _S7_RESET_FLAG in values and (
                any(_S7_ENV_RESET_BINARY in value for value in values)
                or _S7_ENV_NAME in values
            ):
                violations.append(
                    f"line {node.lineno}: S7 argv env reset outside the envp-only posture — "
                    "argv 序列里的 env -i 形态把配置密钥暴露给 ps;envp 之外无路"
                )
        elif isinstance(node, ast.Dict):
            # 规则 21(B33):``{**os.environ}`` / ``{**alias}`` 直接展开进子进程 env
            for key, value in zip(node.keys, node.values, strict=True):
                if key is None and _is_environment_source(value, environment_aliases):
                    violations.append(
                        f"line {node.lineno}: host environment inherited into a subprocess (B33) — "
                        "宿主 env 展开/copy/dict/union/别名全仓禁止;子进程 env 只允许常量白名单/child_env"
                    )
        elif isinstance(node, ast.BinOp):
            # 规则 21(B33):``os.environ | {...}`` / ``{...} | os.environ``(BitOr 合并)
            if isinstance(node.op, ast.BitOr) and (
                _is_environment_source(node.left, environment_aliases)
                or _is_environment_source(node.right, environment_aliases)
            ):
                violations.append(
                    f"line {node.lineno}: host environment inherited into a subprocess (B33) — "
                    "宿主 env 展开/copy/dict/union/别名全仓禁止;子进程 env 只允许常量白名单/child_env"
                )
        elif isinstance(node, ast.DictComp):
            # 规则 21(B33):``{k: v for k, v in os.environ.items()}``
            if any(
                _is_environment_items(generator.iter, environment_aliases)
                for generator in node.generators
            ):
                violations.append(
                    f"line {node.lineno}: host environment inherited into a subprocess (B33) — "
                    "宿主 env 展开/copy/dict/union/别名全仓禁止;子进程 env 只允许常量白名单/child_env"
                )
        elif isinstance(node, ast.Call):
            # 规则 21(B33):os.environ.copy() / dict(os.environ[.items()]) /
            # 别名的同类形态;env= 关键字再覆盖 Attribute/Name/BinOp/Dict 值。
            if _is_environment_source(node, environment_aliases):
                violations.append(
                    f"line {node.lineno}: host environment inherited into a subprocess (B33) — "
                    "宿主 env 展开/copy/dict/union/别名全仓禁止;子进程 env 只允许常量白名单/child_env"
                )
            for keyword in node.keywords:
                if keyword.arg == "env" and _is_environment_source(
                    keyword.value, environment_aliases
                ):
                    violations.append(
                        f"line {node.lineno}: host environment inherited into a subprocess (B33) — "
                        "宿主 env 展开/copy/dict/union/别名全仓禁止;子进程 env 只允许常量白名单/child_env"
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # 规则 22:单一定义点
            allowed_sites = _SINGLE_DEFINITION_SITES.get(node.name)
            if allowed_sites is not None and rel_parts not in allowed_sites:
                sites = ", ".join("/".join(site) for site in sorted(allowed_sites))
                violations.append(
                    f"line {node.lineno}: def {node.name}() outside its single definition site — "
                    f"只允许 {sites}"
                )

    # 规则 23:host_config_dir() 调用只允许配置加载层与 floor.py(定义点)。
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and _call_is(node, "host_config_dir")
            and not _host_config_reader_allowed(rel_parts)
        ):
            violations.append(
                f"line {node.lineno}: host configuration path constructed outside the config "
                "adapters — host_config_dir() 只允许 adapters/{hooks,config}/、web、mcp config "
                "与 protocol/floor.py"
            )

    return list(dict.fromkeys(violations))


def scan_praxis_tree(root: Path) -> list[str]:
    """扫描整棵 praxis 树,返回带相对路径前缀的违规列表。"""
    violations: list[str] = []
    if not root.is_dir():
        return violations
    for py_file in sorted(root.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        try:
            rel = str(py_file.relative_to(_PROJECT_ROOT))
        except ValueError:
            rel = str(py_file)
        violations.extend(f"{rel}: {v}" for v in scan_praxis_file(py_file, praxis_root=root))
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--praxis-only",
        action="store_true",
        help="只扫描 praxis 硬规则,跳过 rag 存量扫描(CI 门只强制 praxis 规则)",
    )
    args = parser.parse_args(argv)

    violations: list[str] = []

    if not args.praxis_only:
        for py_file in sorted(_RAG_DIR.rglob("*.py")):
            rel = str(py_file.relative_to(_RAG_DIR))
            parts = Path(rel).parts
            if any(part in _EXCLUDED_DIRS for part in parts):
                continue

            violations.extend(_scan_file(py_file))

    violations.extend(scan_praxis_tree(_PRAXIS_DIR))

    if violations:
        print(f"{len(violations)} anti-pattern(s) found:\n")
        for v in violations:
            print(f"  ✗ {v}")
        print(
            "\nModel names, providers, and base URLs must live in configs/models.yaml.\n"
            "Business code should consume CapabilityBinding, not reference specific models.\n"
            "Praxis violations are architecture hard rules (see this file's docstring)."
        )
        return 1

    if not args.praxis_only:
        print("✓ no hardcoded model names found in business code")
    print("✓ praxis hard-rule scan clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
