"""Skill loader — scan roots, parse SKILL.md, dedupe, fingerprint.

Design principle: **accept unknown, degrade gracefully**.  Unknown frontmatter
fields are stored in ``extra`` and never cause rejection.

Skill roots (in order):
  1. $SKILL_PATH env var (colon-separated dirs)
  2. .agents/skills/ walking from CWD up to repo root
  3. (future) ~/.agents/skills/
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from agent_runtime.skills.models import (
    SOURCE_PRIORITY,
    SkillManifest,
    SkillSource,
    is_understood_field,
)

logger = logging.getLogger(__name__)

# ── Frontmatter parsing ──────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Required in every SKILL.md regardless of source
_REQUIRED_FIELDS = frozenset({"name", "description"})


class SkillLoadError(ValueError):
    """Raised when a SKILL.md has a structural problem (not unknown fields)."""


# ── Frontmatter helpers ───────────────────────────────────────────────


def _parse_frontmatter(content: str, file_path: Path) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter and body from a SKILL.md file.

    Returns (frontmatter_dict, body_str).  Raises SkillLoadError on
    structural problems only.
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        raise SkillLoadError(
            f"{file_path}: missing YAML frontmatter (--- ... ---)"
        )

    raw = match.group(1)
    body = content[match.end():]

    try:
        frontmatter: dict[str, Any] = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise SkillLoadError(f"{file_path}: invalid YAML frontmatter: {e}") from e

    if not isinstance(frontmatter, dict):
        raise SkillLoadError(
            f"{file_path}: frontmatter must be a YAML mapping, "
            f"got {type(frontmatter).__name__}"
        )

    return frontmatter, body


def _build_manifest(
    frontmatter: dict[str, Any],
    body: str,
    skill_file: Path,
    source: SkillSource,
) -> SkillManifest:
    """Build a SkillManifest from parsed frontmatter.

    Only raises SkillLoadError for structural problems (missing required
    fields, wrong types for understood fields).  Unknown fields are
    silently stored in ``extra``.
    """
    # Required fields
    for field in _REQUIRED_FIELDS:
        if field not in frontmatter:
            raise SkillLoadError(
                f"{skill_file}: missing required field '{field}'"
            )

    raw_name = frontmatter["name"]
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise SkillLoadError(f"{skill_file}: 'name' must be a non-empty string")
    name = raw_name.strip()

    raw_desc = frontmatter["description"]
    if not isinstance(raw_desc, str) or not raw_desc.strip():
        raise SkillLoadError(f"{skill_file}: 'description' must be a non-empty string")
    description = raw_desc.strip()

    # ── Understood optional fields (type-checked) ──────────────────

    when_to_use: str | None = None
    if "when_to_use" in frontmatter:
        raw = frontmatter["when_to_use"]
        if raw is not None and str(raw).strip():
            when_to_use = str(raw).strip()

    version: str | None = None
    if "version" in frontmatter:
        raw = frontmatter["version"]
        if raw is not None and str(raw).strip():
            version = str(raw).strip()

    allowed_tools: tuple[str, ...] = ()
    if "allowed_tools" in frontmatter:
        raw = frontmatter["allowed_tools"]
        if raw is not None:
            if not isinstance(raw, list):
                raise SkillLoadError(
                    f"{skill_file}: 'allowed_tools' must be a list of strings"
                )
            allowed_tools = tuple(str(t).strip() for t in raw)

    path_patterns: tuple[str, ...] = ()
    if "paths" in frontmatter:
        raw = frontmatter["paths"]
        if raw is not None:
            if not isinstance(raw, list):
                raise SkillLoadError(
                    f"{skill_file}: 'paths' must be a list of glob patterns"
                )
            path_patterns = tuple(str(p).strip() for p in raw)

    disable_model_invocation: bool = False
    if "disable_model_invocation" in frontmatter:
        raw = frontmatter["disable_model_invocation"]
        if raw is not None and not isinstance(raw, bool):
            raise SkillLoadError(
                f"{skill_file}: 'disable_model_invocation' must be a boolean"
            )
        if raw is not None:
            disable_model_invocation = bool(raw)

    # ── Unknown fields → extra ─────────────────────────────────────

    extra: dict[str, Any] = {}
    for key, value in frontmatter.items():
        if not is_understood_field(key):
            extra[key] = value

    if extra:
        logger.debug(
            "%s: %d unrecognized field(s) stored in extra: %s",
            skill_file, len(extra), ", ".join(sorted(extra)),
        )

    # ── Fingerprint ────────────────────────────────────────────────

    fingerprint = _compute_fingerprint(frontmatter, body)

    skill_id = f"{source.value}:{name}"
    return SkillManifest(
        skill_id=skill_id,
        name=name,
        description=description,
        source=source,
        skill_file=skill_file.resolve(),
        root_dir=skill_file.parent.resolve(),
        when_to_use=when_to_use,
        version=version,
        allowed_tools=allowed_tools,
        path_patterns=path_patterns,
        disable_model_invocation=disable_model_invocation,
        content_fingerprint=fingerprint,
        extra=extra,
    )


def _compute_fingerprint(frontmatter: dict[str, Any], body: str) -> str:
    """Stable hash of the entire SKILL.md content including frontmatter."""
    canonical = yaml.dump(
        {k: frontmatter[k] for k in sorted(frontmatter)},
        sort_keys=True,
        default_flow_style=False,
    )
    combined = f"{canonical}\n{body}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# ── Path scanning ────────────────────────────────────────────────────


def _scan_skill_dir(root: Path, source: SkillSource) -> list[Path]:
    """Find all SKILL.md files under a skills root directory.

    Only supports directory format: <root>/<skill-name>/SKILL.md
    Also supports namespace format: <root>/<namespace>/<skill-name>/SKILL.md
    (one level of nesting).
    """
    if not root.is_dir():
        return []

    skill_files: list[Path] = []
    try:
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            # Direct skill: <root>/<name>/SKILL.md
            direct = entry / "SKILL.md"
            if direct.is_file():
                skill_files.append(direct)
            # Namespaced skill: <root>/<namespace>/<name>/SKILL.md
            try:
                for sub in sorted(entry.iterdir()):
                    if sub.is_dir():
                        ns_skill = sub / "SKILL.md"
                        if ns_skill.is_file():
                            # Use namespace:name format
                            skill_files.append(ns_skill)
            except PermissionError:
                continue
    except PermissionError:
        logger.warning("Cannot read skills directory: %s", root)

    return skill_files


def _resolve_skill_roots(
    cwd: Path,
    repo_root: Path | None = None,
) -> list[tuple[Path, SkillSource]]:
    """Compute the ordered list of (path, source) tuples to scan for skills.

    Order:
      1. SKILL_PATH env var dirs (EXTERNAL source)
      2. .agents/skills/ walking from CWD up to repo root (PROJECT source)
    """
    roots: list[tuple[Path, SkillSource]] = []

    # ── SKILL_PATH env var ─────────────────────────────────────────
    skill_path = os.environ.get("SKILL_PATH", "")
    if skill_path:
        for part in skill_path.split(":"):
            part = part.strip()
            if part:
                p = Path(part).expanduser().resolve()
                if p.is_dir():
                    roots.append((p, SkillSource.EXTERNAL))

    # ── Repo-scoped .agents/skills/ ────────────────────────────────

    if repo_root is None:
        current = cwd.resolve()
        while current != current.parent:
            if (current / ".git").is_dir():
                repo_root = current
                break
            current = current.parent

    if repo_root is not None:
        repo_root = repo_root.resolve()
        try:
            cwd.resolve().relative_to(repo_root)
        except ValueError:
            repo_root = None  # cwd not under repo_root

    if repo_root is not None:
        cwd_resolved = cwd.resolve()
        current = cwd_resolved
        while True:
            skills_dir = current / ".agents" / "skills"
            roots.append((skills_dir, SkillSource.PROJECT))
            if current == repo_root:
                break
            current = current.parent
    else:
        # No repo — still scan CWD
        cwd_skills = cwd.resolve() / ".agents" / "skills"
        roots.append((cwd_skills, SkillSource.PROJECT))

    return roots


# ── Public API ────────────────────────────────────────────────────────


def load_skill_from_file(
    skill_file: Path,
    source: SkillSource,
    skill_root: Path | None = None,
) -> SkillManifest:
    """Parse a single SKILL.md file and return its SkillManifest.

    Only raises SkillLoadError for structural problems (missing required
    fields, wrong types on understood fields).  Unknown fields are stored
    in ``extra``.

    If ``skill_root`` is provided, the relative path from root to the
    skill directory is used to derive the namespace.  For example::

        root = /path/to/skills
        file = /path/to/skills/acme/pdf-tool/SKILL.md
        → namespace "acme", name "acme:pdf-tool"

    A skill directly under root has no namespace.
    """
    if not skill_file.is_file():
        raise SkillLoadError(f"{skill_file}: file not found")

    content = skill_file.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(content, skill_file)

    # Derive namespace from relative path within skill root
    if skill_root is not None:
        try:
            rel = skill_file.parent.relative_to(skill_root)
        except ValueError:
            rel = skill_file.parent
        parts = rel.parts
        # parts = () → direct in root
        # parts = ("acme", "pdf-tool") → namespace "acme"
        if len(parts) >= 2:
            ns = parts[0]  # first segment is the namespace
            raw = frontmatter.get("name", skill_file.parent.name)
            frontmatter["name"] = f"{ns}:{raw}"
        # len(parts) == 1 → direct skill, no namespace

    return _build_manifest(frontmatter, body, skill_file, source)


def load_skill_body(manifest: SkillManifest) -> str:
    """Read the body (after frontmatter) of a skill from disk."""
    content = manifest.skill_file.read_text(encoding="utf-8")
    _, body = _parse_frontmatter(content, manifest.skill_file)
    return body


def load_skill_body_from_file(skill_file: Path) -> str:
    """Read the body (after frontmatter) directly from a SKILL.md path."""
    content = skill_file.read_text(encoding="utf-8")
    _, body = _parse_frontmatter(content, skill_file)
    return body


def scan_and_load_skills(
    cwd: Path,
    repo_root: Path | None = None,
    extra_dirs: list[Path] | None = None,
) -> list[SkillManifest]:
    """Scan all skill roots and return loaded manifests.

    Dedupes by resolved SKILL.md path.  Non-fatal load errors are
    logged and skipped — one bad skill never breaks the whole catalog.

    ``extra_dirs`` is for programmatic injection (e.g. CLI --skill-dir).
    """
    roots = _resolve_skill_roots(cwd, repo_root)

    # Inject extra dirs as EXTERNAL source
    if extra_dirs:
        for d in extra_dirs:
            p = Path(d).expanduser().resolve()
            if p.is_dir():
                roots.append((p, SkillSource.EXTERNAL))

    seen: set[Path] = set()
    manifests: list[SkillManifest] = []

    for root_path, source in roots:
        skill_files = _scan_skill_dir(root_path, source)
        for skill_file in skill_files:
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
                manifest = load_skill_from_file(
                    skill_file, source, skill_root=root_path,
                )
                manifests.append(manifest)
            except SkillLoadError as e:
                logger.warning("Skipping invalid skill: %s", e)
            except Exception:
                logger.exception("Unexpected error loading skill: %s", skill_file)

    manifests.sort(key=lambda m: (
        SOURCE_PRIORITY.get(m.source, 99),
        m.name,
    ))

    return manifests
