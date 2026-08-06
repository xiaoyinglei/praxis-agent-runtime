"""Runtime bridge for skills and model prompt assembly."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from agent_runtime.skills.catalog import SkillCatalog
from agent_runtime.skills.context import build_skills_prompt_section
from agent_runtime.skills.models import SkillInvocation, SkillState
from agent_runtime.skills.policy import SkillPolicy

if TYPE_CHECKING:
    from agent_runtime.loop.state import LoopState


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
        return any(
            not manifest.disable_model_invocation
            for manifest in self.catalog.list_all()
        )

    @property
    def catalog_revision(self) -> str:
        payload = "\n".join(
            f"{manifest.skill_id}:{manifest.content_fingerprint}"
            for manifest in self.catalog.list_all()
            if not manifest.disable_model_invocation
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
        loaded = self.catalog.load(manifest.skill_id)
        if loaded is None:
            return _activation_error(
                name=name,
                code="skill_not_found",
                message="skill could not be loaded from the active catalog",
            )
        return {
            "success": True,
            "name": manifest.name,
            "skill_id": manifest.skill_id,
            "source": manifest.source.value,
            "fingerprint": manifest.content_fingerprint,
            "instructions": loaded.content,
            "args": args,
        }

    def apply_activation_event(
        self,
        state: LoopState,
        event: Mapping[str, object],
        *,
        iteration: int,
    ) -> bool:
        """Persist one successful activation in checkpointable loop state."""

        if not bool(event.get("success")):
            return False
        skill_id = str(event.get("skill_id", ""))
        manifest = self.catalog.find(skill_id)
        if manifest is None or manifest.disable_model_invocation:
            return False
        if (
            str(event.get("source", "")) != manifest.source.value
            or str(event.get("fingerprint", ""))
            != manifest.content_fingerprint
        ):
            return False
        loaded = self.catalog.load(skill_id, iteration=iteration)
        if loaded is None:
            return False
        args_value = event.get("args")
        args = None if args_value is None else str(args_value)
        skill_state = _skill_state(state)
        ref = loaded.to_ref(args=args)
        skill_state.active[skill_id] = ref
        skill_state.invoked = (
            *skill_state.invoked,
            SkillInvocation(
                name=manifest.name,
                skill_id=skill_id,
                source=manifest.source.value,
                skill_file=str(manifest.skill_file),
                fingerprint=manifest.content_fingerprint,
                invoked_at_iteration=iteration,
                args=args,
            ),
        )
        skill_state.visible_skill_ids = self.model_invocable_skill_ids
        return True

    @property
    def model_invocable_skill_ids(self) -> tuple[str, ...]:
        return tuple(
            manifest.skill_id
            for manifest in self.catalog.list_all()
            if not manifest.disable_model_invocation
        )

    def skill_root(self, skill_id: str) -> Path | None:
        manifest = self.catalog.find(skill_id)
        if manifest is None or manifest.disable_model_invocation:
            return None
        return manifest.root_dir

    def validated_active_skill_ids(self, state: LoopState) -> frozenset[str]:
        """Return active ids whose checkpoint identity still matches the catalog."""

        active: set[str] = set()
        for skill_id, ref in _skill_state(state).active.items():
            manifest = self.catalog.find(skill_id)
            if manifest is None or manifest.disable_model_invocation:
                continue
            if ref.skill_id != skill_id:
                continue
            if ref.name != manifest.name or ref.source != manifest.source.value:
                continue
            if _resolved(ref.skill_file) != manifest.skill_file.resolve():
                continue
            if _resolved(ref.root_dir) != manifest.root_dir.resolve():
                continue
            active.add(skill_id)
        return frozenset(active)

    def render_prompt_context(self, state: LoopState) -> str:
        skill_state = _skill_state(state)
        skill_state.visible_skill_ids = self.model_invocable_skill_ids
        validated_ids = self.validated_active_skill_ids(state)
        prompt_state = skill_state.model_copy(deep=True)
        prompt_state.active = {
            skill_id: ref
            for skill_id, ref in prompt_state.active.items()
            if skill_id in validated_ids
        }
        return build_skills_prompt_section(
            self.catalog,
            max_listing_chars=self.max_listing_chars,
            skill_state=prompt_state,
        )


def _skill_state(state: LoopState) -> SkillState:
    skill_state = state.get("skill_state")
    if not isinstance(skill_state, SkillState):
        skill_state = SkillState.model_validate(skill_state or {})
        state["skill_state"] = skill_state
    return skill_state


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


def _resolved(value: str) -> Path:
    return Path(value).expanduser().resolve()


__all__ = ["SkillRuntime"]
