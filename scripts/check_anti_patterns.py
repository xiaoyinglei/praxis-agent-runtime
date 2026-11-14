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
