"""Runtime bridge for skills and model prompt assembly."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

from agent_runtime.skills.catalog import SkillCatalog
from agent_runtime.skills.models import SkillManifest
from agent_runtime.skills.policy import SkillPolicy

_SKILL_DIR_RE = re.compile(r"\$SKILL_DIR\b")
_ARGUMENTS_RE = re.compile(r"\$ARGUMENTS\b")


class SkillRuntime:
    """Service-scoped runtime facade for skill prompt context."""

    def __init__(
        self,
        catalog: SkillCatalog,
        *,
        policy: SkillPolicy | None = None,
        max_listing_chars: int = 2000,
    ) -> None:
        self.policy = policy or SkillPolicy()
        self.catalog = SkillCatalog(
            [
                manifest
                for manifest in catalog.list_all()
                if self.policy.is_skill_enabled(manifest)
            ]
        )
        self.max_listing_chars = max_listing_chars

    @property
    def has_model_invocable_skills(self) -> bool:
        return bool(self.model_invocable_skill_ids)

    @property
    def catalog_revision(self) -> str:
        payload = "\n".join(
            f"{manifest.skill_id}:{manifest.content_fingerprint}"
            for manifest in self.catalog.list_all()
            if self._invocable(manifest)
        )
        return "skills_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def invoke_skill(self, arguments: Mapping[str, object]) -> dict[str, object]:
        """Resolve one model-visible skill into a canonical activation event."""

        name = str(arguments.get("name", "")).strip()
        args_value = arguments.get("args")
        args = None if args_value is None else str(args_value)
        manifest = self.catalog.find(name)
        if manifest is None:
            return _activation_error(
                name=name,
                code="skill_not_found",
                message="skill is not available in the active catalog",
            )
        if manifest.disable_model_invocation:
            return _activation_error(
                name=name,
                code="skill_disabled",
                message="skill is not available for model invocation",
            )
        if not self.policy.can_autoload(manifest):
            return _activation_error(
                name=name, code="skill_source_untrusted",
                message="skill source is not trusted for model invocation",
            )
        if manifest.unsupported_constraints:
            return _activation_error(
                name=name, code="unsupported_skill_constraints",
                message="unsupported skill constraints: " + ", ".join(manifest.unsupported_constraints),
            )
        loaded = self.catalog.load(manifest.skill_id)
        if loaded is None:
            return _activation_error(
                name=name,
                code="skill_not_found",
                message="skill could not be loaded from the active catalog",
            )
        # Replacement text is literal data (paths, user arguments): it must
        # never be interpreted as a regex replacement template.
        base_dir = str(loaded.referenced_base_dir)
        instructions = loaded.content.replace("${SKILL_DIR}", base_dir)
        instructions = _SKILL_DIR_RE.sub(lambda _match: base_dir, instructions)
        if args is not None:
            instructions = _ARGUMENTS_RE.sub(lambda _match: args, instructions)
        return {
            "success": True,
            "name": manifest.name,
            "skill_id": manifest.skill_id,
            "source": manifest.source.value,
            "fingerprint": manifest.content_fingerprint,
            "instructions": instructions,
            "args": args,
        }

    @property
    def model_invocable_skill_ids(self) -> tuple[str, ...]:
        return tuple(
            manifest.skill_id
            for manifest in self.catalog.list_all()
            if self._invocable(manifest)
        )

    def skill_root(self, skill_id: str) -> Path | None:
        manifest = self.catalog.find(skill_id)
        if manifest is None or not self._invocable(manifest):
            return None
        return manifest.root_dir

    def _invocable(self, manifest: SkillManifest) -> bool:
        return (not manifest.disable_model_invocation and self.policy.can_autoload(manifest)
                and not manifest.unsupported_constraints)

def _activation_error(*, name: str, code: str, message: str) -> dict[str, object]:
    return {
        "success": False,
        "name": name,
        "skill_id": "",
        "source": "",
        "fingerprint": "",
        "instructions": "",
        "error_code": code,
        "error_message": message,
    }


__all__ = ["SkillRuntime"]
