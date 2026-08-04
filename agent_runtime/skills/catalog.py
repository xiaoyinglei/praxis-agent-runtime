"""Skill catalog — searchable/budgeted skill index.

The catalog is the runtime authority for which skills are visible and
how they are listed in the model prompt.  It wraps a flat list of
SkillManifest objects and provides:

  - list_visible() — all skills, minus those filtered by state
  - find(name)    — exact lookup
  - listing_for_prompt(max_chars) — budget-aware single-line listing
  - load(name)    — read full body from disk
"""

from __future__ import annotations

import logging
from collections.abc import Collection

from agent_runtime.skills.loader import load_skill_body
from agent_runtime.skills.models import (
    SOURCE_PRIORITY,
    LoadedSkill,
    SkillManifest,
    SkillSource,
    SkillSummary,
)

logger = logging.getLogger(__name__)

# ── Listing budget ───────────────────────────────────────────────────

DEFAULT_MAX_LISTING_CHARS = 2000
MIN_DESC_LENGTH = 20  # below this, fall back to name-only


class SkillCatalog:
    """Searchable index of loaded skills for one run.

    The catalog owns the parsed manifest list; the runtime reads from it
    each turn to produce the skill listing and to resolve invoke_skill calls.
    """

    def __init__(self, manifests: list[SkillManifest] | None = None) -> None:
        self._manifests: list[SkillManifest] = []
        self._by_id: dict[str, SkillManifest] = {}
        self._by_name: dict[str, list[SkillManifest]] = {}
        self._by_basename: dict[str, list[SkillManifest]] = {}
        self._seen_paths: set[str] = set()
        if manifests:
            for m in manifests:
                self.add(m)

    # ── Mutation (startup only) ───────────────────────────────────

    def add(self, manifest: SkillManifest) -> None:
        """Register a manifest without overwriting same-name skills."""
        path_key = str(manifest.skill_file.resolve())
        if path_key in self._seen_paths:
            return
        self._seen_paths.add(path_key)
        if manifest.skill_id in self._by_id:
            logger.warning(
                "Skipping duplicate skill id %s at %s",
                manifest.skill_id, manifest.skill_file,
            )
            return
        self._manifests.append(manifest)
        self._by_id[manifest.skill_id] = manifest
        self._by_name.setdefault(manifest.name, []).append(manifest)
        self._by_basename.setdefault(manifest.basename, []).append(manifest)

    def clear(self) -> None:
        self._manifests.clear()
        self._by_id.clear()
        self._by_name.clear()
        self._by_basename.clear()
        self._seen_paths.clear()

    # ── Queries ───────────────────────────────────────────────────

    def list_all(self) -> list[SkillManifest]:
        """All manifests, sorted by source priority then name."""
        return sorted(
            self._manifests,
            key=lambda m: (SOURCE_PRIORITY.get(m.source, 99), m.name),
        )

    def find(self, name: str) -> SkillManifest | None:
        """Resolve a skill id or unambiguous name to a manifest.

        Ambiguous bare names return None.  Use ``candidates_for`` to explain
        ambiguity to the model.
        """
        query = name.strip()
        if query in self._by_id:
            return self._by_id[query]

        candidates = self.candidates_for(query)
        if len(candidates) == 1:
            return candidates[0]
        return None

    def candidates_for(self, name: str) -> list[SkillManifest]:
        """Return all manifests matching a skill id, name, or basename."""
        query = name.strip()
        if query in self._by_id:
            return [self._by_id[query]]
        candidates = [*self._by_name.get(query, [])]
        for manifest in self._by_basename.get(query, []):
            if manifest not in candidates:
                candidates.append(manifest)
        return sorted(
            candidates,
            key=lambda m: (SOURCE_PRIORITY.get(m.source, 99), m.skill_id),
        )

    def load(self, name: str, iteration: int = 0) -> LoadedSkill | None:
        """Load a skill body from disk.

        Returns None if the skill is not in the catalog.  The caller should
        verify the skill exists before calling this.
        """
        manifest = self.find(name)
        if manifest is None:
            return None
        content = load_skill_body(manifest)
        return LoadedSkill(
            manifest=manifest,
            content=content,
            referenced_base_dir=manifest.root_dir,
            loaded_at_iteration=iteration,
        )

    def search(
        self,
        query: str,
        limit: int = 8,
    ) -> list[SkillSummary]:
        """Simple substring search over name + description + when_to_use.

        Phase 1 uses naive matching.  A BM25 index can be added in phase 2
        when the skill count grows beyond ~50.
        """
        query_lower = query.lower()
        scored: list[tuple[int, SkillSummary]] = []

        for m in self.list_all():
            if m.disable_model_invocation:
                continue
            score = 0
            text = f"{m.name} {m.description} {m.when_to_use or ''}".lower()
            if query_lower in m.name.lower():
                score += 100
            for word in query_lower.split():
                if word in text:
                    score += 1
            if score > 0:
                scored.append((
                    score,
                    SkillSummary(
                        name=m.name,
                        skill_id=m.skill_id,
                        description=m.description,
                        when_to_use=m.when_to_use,
                        source=m.source,
                    ),
                ))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:limit]]

    # ── Prompt listing ────────────────────────────────────────────

    def listing_for_prompt(
        self,
        max_chars: int = DEFAULT_MAX_LISTING_CHARS,
        *,
        exclude_skill_ids: Collection[str] = (),
    ) -> str:
        """Render a budget-aware skill listing for the model prompt.

        Rules (matches Claude Code pattern):
        1. Sort by source priority: bundled > project > user
        2. Bundled skills always get full descriptions
        3. Non-bundled descriptions are proportionally truncated when the
           full listing exceeds the budget
        4. If the listing still does not fit, fall back to name-only for
           non-bundled skills
        5. If still over budget, drop lowest-priority entries and append
           an omitted-count line
        """
        manifests = self.list_all()
        excluded = set(exclude_skill_ids)

        # Filter out disabled and path-conditional (unmatched) skills
        active: list[SkillManifest] = []
        for m in manifests:
            if m.disable_model_invocation:
                continue
            if m.skill_id in excluded:
                continue
            # Phase 1: path-conditional skills not yet implemented,
            # but when they are, they'll be filtered here via state
            active.append(m)

        if not active:
            return ""

        summaries = [
            SkillSummary(
                name=m.name,
                skill_id=m.skill_id,
                description=m.description,
                when_to_use=m.when_to_use,
                source=m.source,
            )
            for m in active
        ]

        # Step 1: compute full listing size
        full_lines = [s.render() for s in summaries]
        full_size = sum(len(line) + 1 for line in full_lines)  # +1 for \n

        if full_size <= max_chars:
            return "\n".join(full_lines)

        # Step 2: bundled keep full descriptions; rest are truncated
        bundled: list[tuple[int, str]] = []
        non_bundled: list[tuple[int, str]] = []
        for i, s in enumerate(summaries):
            if s.source == SkillSource.BUNDLED:
                bundled.append((i, s.render()))
            else:
                non_bundled.append((i, s.render()))

        # Compute space consumed by bundled skills
        bundled_chars = sum(len(line) + 1 for _, line in bundled)
        remaining_budget = max_chars - bundled_chars

        if remaining_budget <= 0:
            # Extreme: bundled skills alone exceed budget.  Keep bundled
            # with full descriptions and drop everything else.
            return "\n".join(line for _, line in bundled)

        # Compute max description length for non-bundled
        name_overhead_per_skill = sum(
            len(f"- {_summary_display_name(summaries[i])}: ") + 1  # +1 for \n
            for i, _ in non_bundled
        )
        available_for_descs = remaining_budget - name_overhead_per_skill
        max_desc_len = (
            available_for_descs // len(non_bundled)
            if non_bundled else 0
        )

        if max_desc_len < MIN_DESC_LENGTH:
            # Fall back to name-only for non-bundled
            result_lines: list[str] = [line for _, line in bundled]
            for i, _ in non_bundled:
                result_lines.append(f"- {_summary_display_name(summaries[i])}")
            joined = "\n".join(result_lines)
            if len(joined) <= max_chars:
                return joined
            # Still too big — drop lowest-priority non-bundled entries
            while non_bundled and len(joined) > max_chars:
                non_bundled.pop()
                result_lines = [line for _, line in bundled]
                for i, _ in non_bundled:
                    result_lines.append(f"- {_summary_display_name(summaries[i])}")
                if non_bundled:
                    omitted = len(summaries) - len(bundled) - len(non_bundled)
                    result_lines.append(f"... and {omitted} more skills")
                joined = "\n".join(result_lines)
            return joined

        # Proportional truncation
        result_lines = [line for _, line in bundled]
        for i, _ in non_bundled:
            s = summaries[i]
            desc = s.description
            if s.when_to_use:
                desc = f"{desc} — {s.when_to_use}"
            if len(desc) > max_desc_len:
                desc = desc[:max_desc_len - 1] + "…"
            result_lines.append(f"- {_summary_display_name(s)}: {desc}")

        return "\n".join(result_lines)

    def __len__(self) -> int:
        return len(self._manifests)


def _summary_display_name(summary: SkillSummary) -> str:
    return summary.skill_id or summary.name
