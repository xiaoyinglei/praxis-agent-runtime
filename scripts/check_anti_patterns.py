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

Exit code 0: no anti-patterns found.
Exit code 1: anti-patterns detected.
"""

from __future__ import annotations

import ast
import re
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
# 唯一写入点白名单按"相对 praxis 根的后缀路径"匹配,这样 scan_praxis_file
# 是纯函数、可接受任意路径(tmp fixture 也能模拟 store/turns.py 等允许位置)。
# ---------------------------------------------------------------------------

_PRAXIS_TURN_SQL_FILE = ("store", "turns.py")
_PRAXIS_WRITE_STATUS_FILE = ("core", "statemachine.py")
_PRAXIS_INTERRUPTED_FILE = ("core", "recovery.py")
_PRAXIS_SESSION_FILE = ("composition.py",)


def _has_suffix(path: Path, suffix: tuple[str, ...]) -> bool:
    parts = path.parts
    return len(parts) >= len(suffix) and parts[-len(suffix) :] == suffix


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


def scan_praxis_file(path: Path) -> list[str]:
    """扫描单个 praxis 文件,返回违规描述列表(纯函数,可接受任意路径)。"""
    violations: list[str] = []
    content = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(content, filename=str(path))
    except SyntaxError as exc:
        return [f"line {exc.lineno or 0}: syntax error: {exc.msg}"]

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

    # 规则 3:UPDATE turns 只允许 store/turns.py
    if re.search(r"UPDATE\s+turns", content) and not _has_suffix(path, _PRAXIS_TURN_SQL_FILE):
        violations.append("UPDATE turns SQL outside store/turns.py — turn 状态写入只允许 store 层唯一入口")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # 规则 4:write_status() 只允许 core/statemachine.py
        if _call_is(node, "write_status") and not _has_suffix(path, _PRAXIS_WRITE_STATUS_FILE):
            violations.append(
                f"line {node.lineno}: write_status() called outside core/statemachine.py — 权限决策/状态转换唯一入口"
            )
        # 规则 7:Session() 构造只允许 composition.py
        if _call_is(node, "Session") and not _has_suffix(path, _PRAXIS_SESSION_FILE):
            violations.append(
                f"line {node.lineno}: Session(...) constructed outside composition.py — 组合根唯一(设计 §4.1)"
            )
        # 规则 5:transition(..., TurnStatus.INTERRUPTED) 只允许 core/recovery.py
        if _call_is(node, "transition") and not _has_suffix(path, _PRAXIS_INTERRUPTED_FILE):
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
        violations.extend(f"{rel}: {v}" for v in scan_praxis_file(py_file))
    return violations


def main() -> int:
    violations: list[str] = []

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

    print("✓ no hardcoded model names found in business code")
    print("✓ praxis hard-rule scan clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
