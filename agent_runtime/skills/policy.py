"""Skill policy — source allowlist and trust configuration.

Design principle: **accept unknown, degrade gracefully**.  Unknown
sources are treated as EXTERNAL and subject to the same trust rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_runtime.skills.models import SkillManifest, SkillSource


@dataclass(frozen=True)
class SkillPolicy:
    """Read-only policy for skill visibility and trust decisions.

    By default, PROJECT and EXTERNAL sources are enabled.  USER and
    MANAGED sources are disabled (phase 2).  External skills require
    explicit opt-in via ``trust_external_skills`` or per-skill allowlist.
    """

    # Which sources are enabled
    enabled_sources: frozenset[SkillSource] = field(default_factory=lambda: frozenset({
        SkillSource.PROJECT,
        SkillSource.EXTERNAL,
    }))

    # Explicitly disabled skill names (source-agnostic)
    disabled_skills: frozenset[str] = frozenset()

    # Whether to trust external skills (SKILL_PATH, --skill-dir)
    trust_external_skills: bool = True

    # Per-skill allowlist (if non-empty, only these skills are visible)
    allowed_skills: frozenset[str] = frozenset()

    def is_source_enabled(self, source: SkillSource) -> bool:
        return source in self.enabled_sources

    def is_skill_enabled(self, manifest: SkillManifest) -> bool:
        """Check whether a skill should be visible to the model."""
        # Per-skill allowlist takes precedence
        if self.allowed_skills:
            if manifest.name not in self.allowed_skills:
                return False

        if manifest.name in self.disabled_skills:
            return False
        if not self.is_source_enabled(manifest.source):
            return False
        return True

    def can_autoload(self, manifest: SkillManifest) -> bool:
        """Whether a skill can be auto-loaded without user approval.

        PROJECT: always auto-load.
        EXTERNAL: auto-load if trust_external_skills is True.
        USER/MANAGED/MCP: require explicit approval (phase 2).
        """
        if manifest.source == SkillSource.PROJECT:
            return True
        if manifest.source == SkillSource.EXTERNAL:
            return self.trust_external_skills
        return False
