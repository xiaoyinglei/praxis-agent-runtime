"""Skill layer — reusable workflow knowledge for the agent runtime.

Skills are discoverable and invocable. They live beside
tools, not inside ToolRegistry: a skill says *how* to work; a tool performs
a typed action.

Architecture (phase 1 — inline local skills):
  models.py    — immutable SkillManifest, SkillSource, SkillSummary, LoadedSkill
  loader.py    — scan skill roots, parse SKILL.md frontmatter
  catalog.py   — searchable/budgeted skill index
  tools/integrations/skills.py — canonical skill tools
  policy.py    — source allowlist, path trust, enabled/disabled config
"""
