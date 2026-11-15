"""skills loader:双根扫描 → frontmatter 解析 → 指纹/去重/排序(设计 §5;3B 笔记 §5)。

本模块只有 loader(T6);T8 的 ``invoke_skill`` 与 T9 的
``materialize_skill_asset`` 会追加在其后的独立小节,请保持这一分区。

扫描顺序(扫描序决定"同一 resolved SKILL.md 首次出现"的胜者):

1. ``$SKILL_PATH``(冒号分隔目录,已存在者)→ :attr:`SkillSource.EXTERNAL`;
2. 从**显式 workspace** 逐级上溯到最近的 ``.git``,每一级取
   ``.agents/skills/`` → :attr:`SkillSource.PROJECT`。

与旧 ``skills/loader.py`` 的有意差异:旧实现从 ``os.getcwd()`` 上溯,praxis
的 loader 必须 cwd 无关——workspace 由调用方传入(CLI resume/composition 不会
因为 cwd 不同得到不同目录)。文件布局与指纹/去重/排序语义逐字移植:

- ``<root>/<name>/SKILL.md`` 与一层命名空间 ``<root>/<ns>/<name>/SKILL.md``
  (后者 name 前缀 ``<ns>:``);
- frontmatter ``^---\\n…\\n---\\n`` + ``yaml.safe_load`` 必须是 mapping;
  必填 ``name``/``description`` 非空;
- ``unsupported_constraints``(非空 ``allowed_tools``/``paths``)在
  :func:`_build_manifest` **一处**裁决:skip + ``logger.warning``,不进目录;
- fingerprint = ``sha256(yaml.dump(sorted frontmatter) + "\\n" + body)``;
- 按 resolved SKILL.md 路径去重(不是 name);
- 按 ``(SOURCE_PRIORITY, name)`` 稳定排序;
- 单个坏 skill ``log+skip``,绝不炸整个 loader。
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from praxis.protocol.skills import (
    SOURCE_PRIORITY,
    SkillCatalog,
    SkillManifest,
    SkillSource,
)

logger = logging.getLogger(__name__)

# ── frontmatter 解析 ─────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


class _SkillLoadError(ValueError):
    """单个 SKILL.md 的结构性错误;loader 捕获后 log+skip。"""


def _parse_frontmatter(content: str, skill_file: Path) -> tuple[dict[str, Any], str]:
    """返回 ``(frontmatter, body)``;结构性错误抛 :class:`_SkillLoadError`。"""
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        raise _SkillLoadError(f"{skill_file}: missing YAML frontmatter (--- ... ---)")
    raw = match.group(1)
    body = content[match.end():]
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise _SkillLoadError(f"{skill_file}: invalid YAML frontmatter: {error}") from error
    if not isinstance(loaded, dict):
        raise _SkillLoadError(
            f"{skill_file}: frontmatter must be a YAML mapping, got {type(loaded).__name__}"
        )
    if any(not isinstance(key, str) for key in loaded):
        raise _SkillLoadError(f"{skill_file}: frontmatter keys must be strings")
    return loaded, body


def _required_text(frontmatter: dict[str, Any], field_name: str, skill_file: Path) -> str:
    if field_name not in frontmatter:
        raise _SkillLoadError(f"{skill_file}: missing required field '{field_name}'")
    value = frontmatter[field_name]
    if not isinstance(value, str) or not value.strip():
        raise _SkillLoadError(f"{skill_file}: '{field_name}' must be a non-empty string")
    return value.strip()


def _optional_text(frontmatter: dict[str, Any], field_name: str) -> str | None:
    value = frontmatter.get(field_name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(frontmatter: dict[str, Any], field_name: str, skill_file: Path) -> tuple[str, ...]:
    value = frontmatter.get(field_name)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _SkillLoadError(f"{skill_file}: '{field_name}' must be a list of strings")
    return tuple(str(item).strip() for item in value)


def _boolean_field(frontmatter: dict[str, Any], field_name: str, skill_file: Path) -> bool:
    value = frontmatter.get(field_name)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise _SkillLoadError(f"{skill_file}: '{field_name}' must be a boolean")
    return value


def _compute_fingerprint(frontmatter: dict[str, Any], body: str) -> str:
    """稳定指纹:frontmatter 键序规范化后与正文拼接再 sha256。"""
    canonical = yaml.dump(
        {key: frontmatter[key] for key in sorted(frontmatter)},
        sort_keys=True,
        default_flow_style=False,
    )
    return hashlib.sha256(f"{canonical}\n{body}".encode()).hexdigest()


def _build_manifest(
    frontmatter: dict[str, Any],
    body: str,
    skill_file: Path,
    source: SkillSource,
) -> SkillManifest | None:
    """构造清单;非空约束在这一处 skip+warning 并返回 ``None``。"""
    name = _required_text(frontmatter, "name", skill_file)
    description = _required_text(frontmatter, "description", skill_file)
    allowed_tools = _string_list(frontmatter, "allowed_tools", skill_file)
    path_patterns = _string_list(frontmatter, "paths", skill_file)

    unsupported = tuple(
        field_name
        for field_name, values in (("allowed_tools", allowed_tools), ("paths", path_patterns))
        if values
    )
    if unsupported:
        logger.warning(
            "Skipping skill with unsupported constraints: %s (%s)",
            skill_file,
            ", ".join(unsupported),
        )
        return None

    return SkillManifest(
        name=name,
        description=description,
        when_to_use=_optional_text(frontmatter, "when_to_use"),
        version=_optional_text(frontmatter, "version"),
        skill_file=str(skill_file.resolve()),
        root_dir=str(skill_file.parent.resolve()),
        source=source,
        fingerprint=_compute_fingerprint(frontmatter, body),
        disable_model_invocation=_boolean_field(
            frontmatter, "disable_model_invocation", skill_file
        ),
    )


def _load_skill_file(
    skill_file: Path,
    source: SkillSource,
    skill_root: Path,
) -> SkillManifest | None:
    """读并解析单个文件;命名空间前缀在此按相对路径施加。"""
    content = skill_file.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(content, skill_file)

    try:
        relative = skill_file.parent.relative_to(skill_root)
        parts = relative.parts
    except ValueError:
        parts = ()
    # parts == () 或 (name,) ⇒ 直接 skill;("ns", "name") ⇒ 一层命名空间。
    if len(parts) >= 2:
        raw_name = frontmatter.get("name")
        if isinstance(raw_name, str):
            frontmatter = dict(frontmatter)
            frontmatter["name"] = f"{parts[0]}:{raw_name}"

    return _build_manifest(frontmatter, body, skill_file, source)


# ── 路径扫描 ─────────────────────────────────────────────────────────


def _scan_skill_dir(root: Path) -> list[Path]:
    """收集 ``<root>/<name>/SKILL.md`` 与一层 ``<root>/<ns>/<name>/SKILL.md``。"""
    if not root.is_dir():
        return []
    try:
        entries = sorted(root.iterdir())
    except OSError:
        logger.warning("Cannot read skills directory: %s", root)
        return []

    skill_files: list[Path] = []
    for entry in entries:
        if not entry.is_dir():
            continue
        direct = entry / "SKILL.md"
        if direct.is_file():
            skill_files.append(direct)
        try:
            subentries = sorted(entry.iterdir())
        except OSError:
            continue
        for subentry in subentries:
            namespaced = subentry / "SKILL.md"
            if subentry.is_dir() and namespaced.is_file():
                skill_files.append(namespaced)
    return skill_files


def _external_roots() -> list[Path]:
    """``$SKILL_PATH`` 冒号分隔目录(expanduser + resolve;不存在则忽略)。"""
    roots: list[Path] = []
    for part in os.environ.get("SKILL_PATH", "").split(":"):
        part = part.strip()
        if not part:
            continue
        path = Path(part).expanduser().resolve()
        if path.is_dir():
            roots.append(path)
    return roots


def _find_repo_root(workspace: Path) -> Path | None:
    """从 workspace 上溯找最近的 ``.git``(目录或 worktree 的 gitfile)。"""
    current = workspace
    while True:
        if (current / ".git").exists():
            return current
        if current == current.parent:
            return None
        current = current.parent


def _project_roots(workspace: Path) -> list[Path]:
    """workspace → repo root(含端点)每级的 ``.agents/skills/``;无 repo 时仅 workspace。"""
    resolved = workspace.resolve()
    repo_root = _find_repo_root(resolved)
    if repo_root is None:
        return [resolved / ".agents" / "skills"]

    roots: list[Path] = []
    current = resolved
    while True:
        roots.append(current / ".agents" / "skills")
        if current == repo_root:
            return roots
        current = current.parent


# ── 公共 API ─────────────────────────────────────────────────────────


def load_skill_catalog(workspace: Path) -> SkillCatalog:
    """扫描双根并返回按 ``(SOURCE_PRIORITY, name)`` 排序的目录。

    ``workspace`` 是显式传入的装载基点(绝对化后向上找 repo root),**不是
    ``os.getcwd()``**——同一 workspace 在任何 cwd 下得到同一目录。单个坏
    skill 只 ``log+skip``;重复 resolved 路径只保留扫描序首个(``$SKILL_PATH``
    先于 project walk ⇒ 同一物理文件双见时记 EXTERNAL)。
    """
    roots: list[tuple[Path, SkillSource]] = [
        (path, SkillSource.EXTERNAL) for path in _external_roots()
    ]
    roots.extend((path, SkillSource.PROJECT) for path in _project_roots(workspace))

    seen: set[Path] = set()
    manifests: list[SkillManifest] = []
    for root, source in roots:
        for skill_file in _scan_skill_dir(root):
            try:
                resolved = skill_file.resolve()
            except OSError:
                logger.warning("Cannot resolve skill path: %s", skill_file)
                continue
            if resolved in seen:
                logger.debug("Skipping duplicate skill: %s", resolved)
                continue
            seen.add(resolved)

            try:
                manifest = _load_skill_file(skill_file, source, root)
            except _SkillLoadError as error:
                logger.warning("Skipping invalid skill: %s", error)
                continue
            except Exception:
                logger.exception("Unexpected error loading skill: %s", skill_file)
                continue
            if manifest is not None:
                manifests.append(manifest)

    manifests.sort(key=lambda manifest: (SOURCE_PRIORITY[manifest.source], manifest.name))
    return SkillCatalog(tuple(manifests))