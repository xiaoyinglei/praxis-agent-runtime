"""Skill context rendering — prompt listing and loaded-skill injection.

This module is the bridge between the catalog and the model prompt.
It produces:
  - The skill listing block (injected into system prompt each turn)
  - The loaded-skill XML block (injected after invoke_skill is called)
  - The prompt guidance text that teaches the model how to use skills
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection
from pathlib import Path

from agent_runtime.skills.catalog import SkillCatalog
from agent_runtime.skills.loader import load_skill_body_from_file
from agent_runtime.skills.models import LoadedSkill, LoadedSkillRef, SkillState

# ── Prompt guidance text ─────────────────────────────────────────────

SKILL_PROMPT_GUIDANCE = """\
Available skills are reusable workflows. When a skill matches the user's
request, this is a BLOCKING REQUIREMENT: invoke the relevant skill BEFORE
generating any other response about the task. The full skill instructions
are loaded only after invocation.

How to invoke:
- Use the invoke_skill tool with the listed skill id and optional arguments
- Example: invoke_skill(name="project:code-review", args="check the last commit")

Important:
- Available skills are listed in <available_skills> blocks in the conversation
- NEVER mention a skill without actually calling invoke_skill
- Do not invoke a skill that is already loaded in the current conversation
- Do not invent unavailable skills — only use skill ids from the listing
- Loaded skill content supersedes only generic workflow guidance, not user
  instructions or safety policy"""


# ── Skill listing block ──────────────────────────────────────────────

_SKILL_DIR_RE = re.compile(r"\$SKILL_DIR\b")
_ARGUMENTS_RE = re.compile(r"\$ARGUMENTS\b")


def render_skill_listing(
    catalog: SkillCatalog,
    max_chars: int = 2000,
    *,
    exclude_skill_ids: Collection[str] = (),
) -> str:
    """Render the <available_skills> block for the model prompt.

    Returns an empty string when there are no skills to list.
    """
    listing = catalog.listing_for_prompt(
        max_chars=max_chars,
        exclude_skill_ids=exclude_skill_ids,
    )
    if not listing:
        return ""

    return f"""<available_skills>
{listing}
</available_skills>"""


# ── Loaded skill block ───────────────────────────────────────────────


def _expand_skill_body(body: str, args: str | None, base_dir: Path) -> str:
    """Apply textual substitution to a skill body before injection.

    This is plain textual substitution only.  It does NOT imply shell
    interpolation or script execution.  If a skill wants to run a script,
    the expanded instructions must still call an ordinary approved tool
    such as run_command.
    """
    body = body.replace("${SKILL_DIR}", str(base_dir))
    body = _SKILL_DIR_RE.sub(str(base_dir), body)
    if args is not None:
        body = _ARGUMENTS_RE.sub(args, body)
    return body


def render_loaded_skill(
    loaded: LoadedSkill,
    args: str | None = None,
) -> str:
    """Render a loaded skill as a model-visible context injection.

    The output is wrapped in <loaded_skill> tags with metadata attributes
    so the model and transcript tools can identify it.
    """
    expanded = _expand_skill_body(loaded.content, args, loaded.referenced_base_dir)
    fp = loaded.manifest.content_fingerprint[:16]
    args_block = _render_skill_arguments(args)
    attrs = (
        f'id="{loaded.manifest.skill_id}" '
        f'name="{loaded.manifest.name}" '
        f'source="{loaded.manifest.source.value}" '
        f'fingerprint="{fp}"'
    )

    return f"""<loaded_skill {attrs}>
Base directory for this skill: {loaded.referenced_base_dir}
{args_block}

{expanded}
</loaded_skill>"""


def render_loaded_skill_ref(ref: LoadedSkillRef) -> str:
    """Render a checkpoint-friendly loaded skill reference for prompt injection."""
    body = load_skill_body_from_file(Path(ref.skill_file))
    expanded = _expand_skill_body(body, ref.args, Path(ref.root_dir))
    fp = ref.fingerprint[:16]
    args_block = _render_skill_arguments(ref.args)
    warning_block = _render_content_changed_warning(ref, body)
    attrs = f'id="{ref.skill_id}" name="{ref.name}" source="{ref.source}" fingerprint="{fp}"'
    return f"""<loaded_skill {attrs}>
Base directory for this skill: {ref.root_dir}
{args_block}
{warning_block}

{expanded}
</loaded_skill>"""


def _render_skill_arguments(args: str | None) -> str:
    if args is None:
        return ""
    return f"""
Invocation arguments:
<skill_arguments>
{args}
</skill_arguments>"""


def _render_content_changed_warning(ref: LoadedSkillRef, body: str) -> str:
    if not ref.body_fingerprint:
        return ""
    current = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if current == ref.body_fingerprint:
        return ""
    return """
<skill_warning code="skill_content_changed_on_resume">
The skill file changed after it was invoked. Re-read these active skill
instructions before continuing.
</skill_warning>"""


def render_active_loaded_skills(state: SkillState) -> str:
    """Render all active loaded skills from SkillState."""
    if not state.active:
        return ""
    blocks = [render_loaded_skill_ref(ref) for ref in state.active.values()]
    return "<loaded_skills>\n" + "\n\n".join(blocks) + "\n</loaded_skills>"


# ── Full prompt section ──────────────────────────────────────────────


def build_skills_prompt_section(
    catalog: SkillCatalog,
    max_listing_chars: int = 2000,
    *,
    skill_state: SkillState | None = None,
) -> str:
    """Build the complete skills section for the system prompt.

    Includes skill invocation guidance only when at least one visible skill is
    available. Loaded skills are injected independently so they survive resume
    and compaction even when no new skill remains invokable.
    """
    active_skill_ids = frozenset(skill_state.active) if skill_state is not None else frozenset()
    listing = render_skill_listing(
        catalog,
        max_chars=max_listing_chars,
        exclude_skill_ids=active_skill_ids,
    )
    loaded = render_active_loaded_skills(skill_state) if skill_state is not None else ""

    if listing:
        parts = [SKILL_PROMPT_GUIDANCE, listing]
    else:
        parts = []
    if loaded:
        parts.append(loaded)

    return "\n\n".join(parts)
