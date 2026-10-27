from dataclasses import replace

import pytest

from agent_runtime.skills.catalog import SkillCatalog
from agent_runtime.skills.loader import SkillLoadError, load_skill_from_file, scan_and_load_skills
from agent_runtime.skills.models import SkillSource
from agent_runtime.skills.policy import SkillPolicy
from agent_runtime.skills.runtime import SkillRuntime
from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.integrations.skills import create_invoke_skill_tool
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import ToolCall, ToolCallOrigin


def write_skill(tmp_path, *, body="original", metadata=""):
    path = tmp_path / "demo" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: demo\ndescription: demo skill\n{metadata}---\n{body}")
    return path


def test_loaded_body_and_identity_remain_one_frozen_version(tmp_path):
    path = write_skill(tmp_path)
    manifest = load_skill_from_file(path, SkillSource.PROJECT)
    # The file can change between discovery and construction of the runtime.
    path.write_text("---\nname: demo\ndescription: changed\ndisable_model_invocation: true\n---\nreplacement")
    runtime = SkillRuntime(SkillCatalog([manifest]))
    event = runtime.invoke_skill({"name": manifest.skill_id})
    assert event["success"] is True
    assert event["instructions"] == "original"
    assert event["fingerprint"] == manifest.content_fingerprint
    fresh = SkillRuntime(SkillCatalog([load_skill_from_file(path, SkillSource.PROJECT)]))
    assert fresh.catalog_revision != runtime.catalog_revision
    assert fresh.invoke_skill({"name": manifest.skill_id})["error_code"] == "skill_disabled"
    path.unlink()
    assert runtime.invoke_skill({"name": manifest.skill_id})["instructions"] == "original"


@pytest.mark.anyio
async def test_strict_external_policy_is_enforced_by_the_public_tool(tmp_path):
    path = write_skill(tmp_path)
    manifest = load_skill_from_file(path, SkillSource.EXTERNAL)
    runtime = SkillRuntime(SkillCatalog([manifest]), policy=SkillPolicy(trust_external_skills=False))
    tool = create_invoke_skill_tool(runtime.invoke_skill, available_skill_ids=runtime.model_invocable_skill_ids)
    call = ToolCall(
        tool_call_id="invoke",
        tool_name="invoke_skill",
        arguments={"name": manifest.skill_id},
        origin=ToolCallOrigin(request_id="request", toolset_revision="skills", exposed_tool_names=("invoke_skill",)),
    )
    execution = await ToolExecutor({"invoke_skill": tool}).execute(call, context=ToolExecutionContext())
    assert execution.result.is_error is True
    assert execution.result.structured_content["error_code"] == "skill_source_untrusted"
    assert runtime.skill_root(manifest.skill_id) is None
    trusted = SkillRuntime(SkillCatalog([manifest]))
    assert trusted.invoke_skill({"name": manifest.skill_id})["success"] is True
    project = replace(manifest, skill_id="project:demo", source=SkillSource.PROJECT)
    assert (
        SkillRuntime(SkillCatalog([project]), policy=SkillPolicy(trust_external_skills=False)).invoke_skill(
            {"name": project.skill_id}
        )["success"]
        is True
    )


@pytest.mark.parametrize("metadata", ["allowed_tools: [read_file]\n", 'paths: ["*.xlsx"]\n'])
def test_unimplemented_constraints_are_rejected_instead_of_silently_accepted(tmp_path, metadata, caplog):
    path = write_skill(tmp_path, metadata=metadata)
    with pytest.raises(SkillLoadError, match="unsupported skill constraints"):
        load_skill_from_file(path, SkillSource.PROJECT)
    assert scan_and_load_skills(tmp_path, repo_root=tmp_path, extra_dirs=[tmp_path]) == []
    assert "unsupported skill constraints" in caplog.text


@pytest.mark.parametrize("fields", [{"allowed_tools": ("read_file",)}, {"path_patterns": ("*.xlsx",)}])
def test_programmatic_manifest_cannot_bypass_unsupported_constraints(tmp_path, fields):
    manifest = replace(load_skill_from_file(write_skill(tmp_path), SkillSource.PROJECT), **fields)
    runtime = SkillRuntime(SkillCatalog([manifest]))
    assert runtime.invoke_skill({"name": manifest.skill_id})["error_code"] == "unsupported_skill_constraints"
    assert runtime.skill_root(manifest.skill_id) is None


def test_explicit_empty_constraints_keep_plain_skill_usable(tmp_path):
    path = write_skill(tmp_path, metadata="allowed_tools: []\npaths: []\n")
    manifest = load_skill_from_file(path, SkillSource.PROJECT)
    assert SkillRuntime(SkillCatalog([manifest])).invoke_skill({"name": manifest.skill_id})["success"] is True


@pytest.mark.parametrize("source", [SkillSource.PROJECT, SkillSource.BUNDLED])
def test_enabled_trusted_sources_do_not_depend_on_external_trust(tmp_path, source):
    manifest = load_skill_from_file(write_skill(tmp_path), source)
    runtime = SkillRuntime(
        SkillCatalog([manifest]), policy=SkillPolicy(enabled_sources=frozenset({source}), trust_external_skills=False)
    )
    assert runtime.invoke_skill({"name": manifest.skill_id})["success"] is True
    assert runtime.model_invocable_skill_ids == (manifest.skill_id,)


def test_external_allowlist_is_an_explicit_per_skill_trust_grant(tmp_path):
    manifest = load_skill_from_file(write_skill(tmp_path), SkillSource.EXTERNAL)
    runtime = SkillRuntime(
        SkillCatalog([manifest]),
        policy=SkillPolicy(trust_external_skills=False, allowed_skills=frozenset({manifest.name})),
    )
    assert runtime.invoke_skill({"name": manifest.skill_id})["success"] is True
