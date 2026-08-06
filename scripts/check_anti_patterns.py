#!/usr/bin/env python3
"""Scan rag/ for hardcoded model names, providers, and base URLs in business code.

Scans all .py files under rag/ except:
  - rag/models/        (model catalog — the config system itself)
  - rag/assembly/      (provider factory — legitimately references model names)

Exit code 0: no anti-patterns found.
Exit code 1: anti-patterns detected.
"""

from __future__ import annotations

import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RAG_DIR = _PROJECT_ROOT / "rag"

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


def main() -> int:
    violations: list[str] = []

    for py_file in sorted(_RAG_DIR.rglob("*.py")):
        rel = str(py_file.relative_to(_RAG_DIR))
        parts = Path(rel).parts
        if any(part in _EXCLUDED_DIRS for part in parts):
            continue

        violations.extend(_scan_file(py_file))

    if violations:
        print(f"{len(violations)} anti-pattern(s) found:\n")
        for v in violations:
            print(f"  ✗ {v}")
        print(
            "\nModel names, providers, and base URLs must live in configs/models.yaml.\n"
            "Business code should consume CapabilityBinding, not reference specific models."
        )
        return 1

    print("✓ no hardcoded model names found in business code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
