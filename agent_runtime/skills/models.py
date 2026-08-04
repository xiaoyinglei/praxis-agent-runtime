"""Data contracts for the skill layer.

Design principle: **accept unknown, degrade gracefully**.  The frontmatter
schema is open — fields we don't understand are stored in ``extra`` so
they survive round-trips and can be used by future code without re-parsing.
We never reject a skill because of an unknown field.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ── Skill source ────────────────────────────────────────────────────


class SkillSource(StrEnum):
    """Where a skill was loaded from.  Ordered by trust (lowest first)."""

    BUNDLED = "bundled"    # shipped with the agent runtime
    PROJECT = "project"    # .agents/skills/ under repo root
    USER = "user"          # ~/.agents/skills/
    EXTERNAL = "external"  # SKILL_PATH or --skill-dir override
    MANAGED = "managed"    # org/admin policy skills
    MCP = "mcp"            # provided by an MCP server
    PLUGIN = "plugin"      # packaged in a plugin


SOURCE_PRIORITY: dict[SkillSource, int] = {
    SkillSource.BUNDLED: 0,
    SkillSource.PROJECT: 1,
    SkillSource.USER: 2,
    SkillSource.EXTERNAL: 2,
    SkillSource.MANAGED: 2,
    SkillSource.MCP: 3,
    SkillSource.PLUGIN: 3,
}


# ── Fields we actively understand ────────────────────────────────────

# These fields have defined behaviour in the current implementation.
# Everything else goes into ``extra``.
_UNDERSTOOD_FIELDS = frozenset({
    "name",
    "description",
    "when_to_use",
    "version",
    "allowed_tools",
    "paths",
    "disable_model_invocation",
})


# ── Skill manifest ──────────────────────────────────────────────────


@dataclass(frozen=True)
class SkillManifest:
    """Metadata parsed from a SKILL.md file.  Immutable after loading.

    The ``extra`` dict carries all frontmatter fields we don't actively
    understand.  This is the forward-compatibility mechanism: a future
    implementation can read ``extra`` without re-parsing the file.
    """

    skill_id: str
    name: str
    description: str
    source: SkillSource
    skill_file: Path              # absolute path to SKILL.md
    root_dir: Path                # absolute path to skill directory
    when_to_use: str | None = None
    version: str | None = None
    allowed_tools: tuple[str, ...] = ()
    path_patterns: tuple[str, ...] = ()
    disable_model_invocation: bool = False
    content_fingerprint: str = ""  # stable hash of SKILL.md + frontmatter
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_path_filter(self) -> bool:
        return len(self.path_patterns) > 0

    @property
    def namespace(self) -> str | None:
        """Return namespace if the name contains ':' (e.g. 'acme:pdf')."""
        if ":" in self.name:
            return self.name.split(":", 1)[0]
        return None

    @property
    def basename(self) -> str:
        """Return the name without namespace prefix."""
        return self.name.split(":", 1)[-1]


# ── Skill summary ────────────────────────────────────────────────────


@dataclass(frozen=True)
class SkillSummary:
    """Compact entry for the skill listing injected into the model prompt."""

    name: str
    description: str
    skill_id: str = ""
    when_to_use: str | None = None
    source: SkillSource = SkillSource.PROJECT

    def render(self) -> str:
        """Render a single-line listing entry."""
        text = self.description
        if self.when_to_use:
            text = f"{text} — {self.when_to_use}"
        display_name = self.skill_id or self.name
        return f"- {display_name}: {text}"


# ── Loaded skill ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class LoadedSkill:
    """A skill whose body has been read from disk and is ready for injection."""

    manifest: SkillManifest
    content: str                   # full SKILL.md body (after frontmatter)
    referenced_base_dir: Path
    loaded_at_iteration: int

    def to_ref(self, *, args: str | None = None) -> LoadedSkillRef:
        """Return a checkpoint-friendly reference to this loaded skill."""
        return LoadedSkillRef(
            skill_id=self.manifest.skill_id,
            name=self.manifest.name,
            source=self.manifest.source.value,
            skill_file=str(self.manifest.skill_file),
            root_dir=str(self.manifest.root_dir),
            fingerprint=self.manifest.content_fingerprint,
            body_fingerprint=hashlib.sha256(self.content.encode("utf-8")).hexdigest(),
            loaded_at_iteration=self.loaded_at_iteration,
            args=args,
        )


class LoadedSkillRef(BaseModel):
    """Checkpoint-friendly active skill reference.

    The full body is intentionally reloaded from disk during prompt assembly.
    """

    skill_id: str
    name: str
    source: str
    skill_file: str
    root_dir: str
    fingerprint: str
    body_fingerprint: str = ""
    loaded_at_iteration: int = 0
    args: str | None = None


# ── Skill invocation record ──────────────────────────────────────────


@dataclass(frozen=True)
class SkillInvocation:
    """Audit record for a single skill invocation."""

    name: str
    source: str                    # SkillSource.value
    skill_file: str                # absolute path string
    fingerprint: str
    invoked_at_iteration: int
    args: str | None = None
    skill_id: str = ""


# ── Skill state ──────────────────────────────────────────────────────


class SkillState(BaseModel):
    """Checkpointable skill sub-state stored in LoopState."""

    visible_skill_ids: tuple[str, ...] = ()
    invoked: tuple[SkillInvocation, ...] = ()
    active: dict[str, LoadedSkillRef] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "visible_skill_ids" not in data and "visible_skill_names" in data:
            data["visible_skill_ids"] = data.get("visible_skill_names") or ()
        if "active" not in data and "loaded_skills" in data:
            active: dict[str, LoadedSkillRef] = {}
            for _key, loaded in (data.get("loaded_skills") or {}).items():
                if isinstance(loaded, LoadedSkillRef):
                    active[loaded.skill_id] = loaded
                elif isinstance(loaded, LoadedSkill):
                    ref = loaded.to_ref()
                    active[ref.skill_id] = ref
                elif isinstance(loaded, dict):
                    try:
                        ref = LoadedSkillRef.model_validate(loaded)
                    except Exception:
                        continue
                    active[ref.skill_id] = ref
            data["active"] = active
        return data

    @property
    def visible_skill_names(self) -> tuple[str, ...]:
        """Backward-compatible alias for older tests/callers."""
        return self.visible_skill_ids

    @property
    def loaded_skills(self) -> dict[str, LoadedSkillRef]:
        """Backward-compatible alias for active loaded skills."""
        return self.active


# ── Error codes ──────────────────────────────────────────────────────

SkillErrorCode = Literal[
    "skill_not_found",
    "skill_disabled",
    "invalid_skill_manifest",
    "skill_source_untrusted",
    "skill_content_changed_on_resume",
]


# ── Helpers ──────────────────────────────────────────────────────────

def is_understood_field(name: str) -> bool:
    """Check whether a frontmatter field is actively understood."""
    return name in _UNDERSTOOD_FIELDS
