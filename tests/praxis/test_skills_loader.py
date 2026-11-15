"""T6:skills loader 与目录(设计 §5;3B 笔记 §5;剪枝移植旧 skills/loader.py)。

钉住六件事:
1. 根顺序:``$SKILL_PATH``(冒号分隔 → EXTERNAL)先扫,再从 **workspace** 上溯到
   最近的 ``.git`` 收集 ``.agents/skills/``(→ PROJECT;cwd 无关,旧实现从
   ``os.getcwd()`` 走——此处是有意改进);支持 ``<root>/<name>/SKILL.md`` 与
   一层命名空间 ``<root>/<ns>/<name>/SKILL.md``(name 前缀 ``<ns>:``);
2. 去重按 **resolved SKILL.md 路径**(不是 name):同一物理文件经 ``$SKILL_PATH``
   与 project walk 双见 ⇒ 只留先扫到的 EXTERNAL;不同路径同名 ⇒ 都保留,
   ``SkillCatalog.get`` 按排序后首个命中;
3. 确定性排序 ``(SOURCE_PRIORITY{PROJECT:0,EXTERNAL:1}, name)``;
4. 单个坏 skill(frontmatter 非 mapping / 缺 name/description)⇒ log+skip;
5. ``unsupported_constraints``(``allowed_tools``/``paths`` 非空)⇒ 只在 loader
   一处 skip + ``logger.warning``,绝不进目录(空列表不拒);
6. 指纹 ``sha256(yaml.dump(sorted frontmatter) + "\n" + body)``:正文敏感、
   frontmatter 键序不敏感、64 位小写 hex。

真实语料形状:``xlsx``/``docx`` 的 frontmatter 逐字复制进本文件(不在测试期读
仓库真语料),证明长双引号 description 等真实世界 YAML 可解析。
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import pytest
import yaml

from praxis.protocol.skills import (
    MAX_INVOKE_ARGS_CHARS,
    MAX_SKILL_DESCRIPTION_CHARS,
    MAX_SKILL_INSTRUCTIONS_CHARS,
    MAX_SKILL_NAME_CHARS,
    SOURCE_PRIORITY,
    SkillCatalog,
    SkillManifest,
    SkillSource,
)
from praxis.tools.builtins.skills import load_skill_catalog

_LOGGER = "praxis.tools.builtins.skills"

# ---------------------------------------------------------------------------
# 真实仓库语料形状(frontmatter 逐字;正文取代表性片段)
# ---------------------------------------------------------------------------

_XLSX_DESCRIPTION = (
    r"Use this skill any time a spreadsheet file is the primary input or output. "
    r"This means any task where the user wants to: open, read, edit, or fix an existing .xlsx, .xlsm, .csv, or .tsv "
    r"file (e.g., adding columns, computing formulas, formatting, charting, cleaning messy data); create a new "
    r"spreadsheet from scratch or from other data sources; or convert between tabular file formats. Trigger "
    r"especially when the user references a spreadsheet file by name or path — even casually (like \"the xlsx in my "
    r"downloads\") — and wants something done to it or produced from it. Also trigger for cleaning or restructuring "
    r"messy tabular data files (malformed rows, misplaced headers, junk data) into proper spreadsheets. The "
    r"deliverable must be a spreadsheet file. Do NOT trigger when the primary deliverable is a Word document, HTML "
    r"report, standalone Python script, database pipeline, or Google Sheets API integration, even if tabular data is "
    r"involved."
)

_DOCX_DESCRIPTION = (
    r"Use this skill whenever the user wants to create, read, edit, or manipulate Word documents (.docx files) or "
    r"Word templates (.dotx files). Triggers include: any mention of 'Word doc', 'word document', '.docx', "
    r"'.dotx', or requests to produce professional documents with formatting like tables of contents, headings, "
    r"page numbers, or letterheads. Also use when extracting or reorganizing content from .docx or .dotx files, "
    r"inserting or replacing images in documents, performing find-and-replace in Word files, working with tracked "
    r"changes or comments, or converting content into a polished Word document. If the user asks for a 'report', "
    r"'memo', 'letter', 'template', or similar deliverable as a Word or .docx file, use this skill. Do NOT use for "
    r"PDFs, spreadsheets, Google Docs, or general coding tasks unrelated to document generation."
)

_XLSX_SKILL_MD = (
    "---\n"
    "name: xlsx\n"
    'description: "' + _XLSX_DESCRIPTION + '"\n'
    "license: Proprietary. LICENSE.txt has complete terms\n"
    "---\n"
    "\n"
    "# Requirements for Outputs\n"
    "\n"
    "## All Excel files\n"
    "\n"
    "### Professional Font\n"
    "- Use a consistent, professional font (e.g., Arial) for all deliverables\n"
    "\n"
    "### Zero Formula Errors\n"
    "- Every Excel model MUST be delivered with ZERO formula errors (#REF!, #DIV/0!)\n"
)

_DOCX_SKILL_MD = (
    "---\n"
    "name: docx\n"
    'description: "' + _DOCX_DESCRIPTION + '"\n'
    "license: Proprietary. LICENSE.txt has complete terms\n"
    "---\n"
    "\n"
    "# DOCX creation, editing, and analysis\n"
    "\n"
    "A `.docx` is a ZIP archive of XML files. Choose your approach by task:\n"
    "\n"
    "| Task | Approach |\n"
    "|---|---|\n"
    "| **Create** a new document | Write a `docx` (npm) script |\n"
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_external_skill_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认无外部根;需要 ``$SKILL_PATH`` 的测试自行 setenv。"""
    monkeypatch.delenv("SKILL_PATH", raising=False)


def _repo_workspace(tmp_path: Path) -> tuple[Path, Path]:
    """repo/.git + repo/packages/app 三层 workspace(向上 walk 有两个候选根)。"""
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    workspace = repo / "packages" / "app"
    workspace.mkdir(parents=True)
    return repo, workspace


def _standalone_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "standalone"
    workspace.mkdir()
    return workspace


def _write_raw(directory: Path, content: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(content, encoding="utf-8")
    return path


def _write_skill(
    directory: Path,
    *,
    frontmatter: str,
    body: str = "Body.\n",
) -> Path:
    return _write_raw(
        directory,
        f"---\n{frontmatter.strip()}\n---\n{body}",
    )


def _write_named_skill(
    root: Path,
    directory: str,
    name: str,
    *,
    description: str = "test skill",
    extra_frontmatter: str = "",
    body: str = "Body.\n",
) -> Path:
    frontmatter = f"name: {name}\ndescription: {description}"
    if extra_frontmatter.strip():
        frontmatter += f"\n{extra_frontmatter.strip()}"
    return _write_skill(root / directory, frontmatter=frontmatter, body=body)


def _by_dir(catalog: SkillCatalog, directory: str) -> SkillManifest:
    matches = [m for m in catalog.skills if Path(m.skill_file).parent.name == directory]
    assert len(matches) == 1, f"expected one skill under {directory}, got {matches}"
    return matches[0]


# ---------------------------------------------------------------------------
# ① PROJECT/EXTERNAL 双根、命名空间一层、排序确定性、cwd 无关
# ---------------------------------------------------------------------------


def test_project_and_external_roots_namespace_and_sorting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, workspace = _repo_workspace(tmp_path)
    repo_skills = repo / ".agents" / "skills"
    workspace_skills = workspace / ".agents" / "skills"

    _write_raw(repo_skills / "xlsx", _XLSX_SKILL_MD)
    _write_named_skill(workspace_skills, "zzz", "zeta")
    _write_named_skill(
        workspace_skills,
        "acme/pdf-tool",
        "pdf-tool",
        extra_frontmatter="disable_model_invocation: true",
    )

    external = tmp_path / "external"
    _write_named_skill(external, "alpha-skill", "alpha")
    monkeypatch.setenv("SKILL_PATH", str(external))

    # cwd 与 workspace 无关:loader 只从显式 workspace 上溯(旧实现从 cwd 走)。
    monkeypatch.chdir(tmp_path)

    catalog = load_skill_catalog(workspace)

    assert [m.name for m in catalog.skills] == ["acme:pdf-tool", "xlsx", "zeta", "alpha"]
    assert [m.source for m in catalog.skills] == [
        SkillSource.PROJECT,
        SkillSource.PROJECT,
        SkillSource.PROJECT,
        SkillSource.EXTERNAL,
    ]
    # 一层命名空间: <root>/<ns>/<name>/SKILL.md → "<ns>:<frontmatter name>"
    assert _by_dir(catalog, "pdf-tool").name == "acme:pdf-tool"
    assert _by_dir(catalog, "pdf-tool").disable_model_invocation is True
    assert _by_dir(catalog, "pdf-tool").root_dir == str((workspace_skills / "acme" / "pdf-tool").resolve())
    # repo 级与 workspace 级 project 根都被 walk 到
    assert _by_dir(catalog, "xlsx").root_dir == str((repo_skills / "xlsx").resolve())
    assert _by_dir(catalog, "zzz").root_dir == str((workspace_skills / "zzz").resolve())

    again = load_skill_catalog(workspace)
    assert [m.name for m in again.skills] == [m.name for m in catalog.skills]
    assert [m.fingerprint for m in again.skills] == [m.fingerprint for m in catalog.skills]


# ---------------------------------------------------------------------------
# ② $SKILL_PATH 优先级语义:去重按 resolved path,不是 name
# ---------------------------------------------------------------------------


def test_same_resolved_path_via_skill_path_and_project_keeps_external_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _standalone_workspace(tmp_path)
    project_skills = workspace / ".agents" / "skills"
    _write_named_skill(project_skills, "shared", "shared")
    # 同一物理目录同时经 $SKILL_PATH(EXTERNAL,先扫)与 project walk(PROJECT,后扫)可见。
    monkeypatch.setenv("SKILL_PATH", str(project_skills))

    catalog = load_skill_catalog(workspace)

    assert len(catalog.skills) == 1
    assert catalog.skills[0].source is SkillSource.EXTERNAL


def test_same_name_at_distinct_paths_are_both_kept_and_get_prefers_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo, workspace = _repo_workspace(tmp_path)
    _write_named_skill(workspace / ".agents" / "skills", "project-dup", "dup")
    ext1 = tmp_path / "ext1"
    ext2 = tmp_path / "ext2"
    _write_named_skill(ext1, "dup-a", "dup")
    _write_named_skill(ext2, "dup-b", "dup")
    monkeypatch.setenv("SKILL_PATH", f"{ext1}:{ext2}")

    catalog = load_skill_catalog(workspace)

    # 同名不被 name 去重;排序 (priority, name) 稳定保留扫描序(ext1 → ext2)。
    assert [m.name for m in catalog.skills] == ["dup", "dup", "dup"]
    assert [m.source for m in catalog.skills] == [
        SkillSource.PROJECT,
        SkillSource.EXTERNAL,
        SkillSource.EXTERNAL,
    ]
    assert Path(catalog.skills[1].skill_file).parent.parent == ext1.resolve()
    assert Path(catalog.skills[2].skill_file).parent.parent == ext2.resolve()
    assert catalog.get("dup") is catalog.skills[0]
    assert catalog.get("missing") is None


# ---------------------------------------------------------------------------
# ③ frontmatter 非 mapping / 缺必填 ⇒ log+skip,不炸整个 loader
# ---------------------------------------------------------------------------


def test_invalid_frontmatter_and_missing_required_fields_skip_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    workspace = _standalone_workspace(tmp_path)
    root = workspace / ".agents" / "skills"
    _write_skill(root / "non-mapping", frontmatter="- one\n- two")
    _write_skill(root / "missing-name", frontmatter="description: no name")
    _write_skill(root / "missing-description", frontmatter="name: only-name")
    _write_raw(root / "no-frontmatter", "# plain markdown, no YAML frontmatter\n")
    _write_named_skill(root, "good", "good")

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        catalog = load_skill_catalog(workspace)

    assert [m.name for m in catalog.skills] == ["good"]
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(messages) == 4
    assert all("Skipping invalid skill" in m for m in messages)
    for directory in ("non-mapping", "missing-name", "missing-description", "no-frontmatter"):
        assert any(directory in m for m in messages), f"warning should name {directory}: {messages}"


# ---------------------------------------------------------------------------
# ④ unsupported constraints:一处处理(skip + warning),空列表不拒
# ---------------------------------------------------------------------------


def test_unsupported_constraints_skip_once_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    workspace = _standalone_workspace(tmp_path)
    root = workspace / ".agents" / "skills"
    _write_named_skill(root, "allowed-tools", "tool-filtered", extra_frontmatter="allowed_tools:\n  - Read")
    _write_named_skill(root, "path-filtered", "path-filtered", extra_frontmatter="paths:\n  - src/**")
    _write_named_skill(root, "empty-allowed", "empty-ok", extra_frontmatter="allowed_tools: []")
    _write_named_skill(root, "empty-paths", "empty-paths-ok", extra_frontmatter="paths: []")

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        catalog = load_skill_catalog(workspace)

    assert [m.name for m in catalog.skills] == ["empty-ok", "empty-paths-ok"]
    # 每个被拒 skill 恰好一条 warning(唯一处理点,不重复告警)。
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert all("unsupported constraints" in message for message in warnings)
    assert any("allowed_tools" in message for message in warnings)
    assert any("paths" in message for message in warnings)


# ---------------------------------------------------------------------------
# ⑤ 指纹:正文敏感、frontmatter 键序不敏感、sha256 形状
# ---------------------------------------------------------------------------


def test_fingerprint_tracks_body_and_ignores_frontmatter_key_order(tmp_path: Path) -> None:
    workspace = _standalone_workspace(tmp_path)
    root = workspace / ".agents" / "skills"

    _write_skill(root / "a", frontmatter="name: dup\ndescription: d", body="body one\n")
    _write_skill(root / "b", frontmatter="description: d\nname: dup", body="body one\n")

    catalog = load_skill_catalog(workspace)
    first = _by_dir(catalog, "a")
    reordered = _by_dir(catalog, "b")

    assert first.fingerprint == reordered.fingerprint
    assert re.fullmatch(r"[0-9a-f]{64}", first.fingerprint)
    canonical = yaml.dump({"description": "d", "name": "dup"}, sort_keys=True, default_flow_style=False)
    expected = hashlib.sha256(f"{canonical}\nbody one\n".encode()).hexdigest()
    assert first.fingerprint == expected

    _write_skill(root / "a", frontmatter="name: dup\ndescription: d", body="body two\n")
    changed = _by_dir(load_skill_catalog(workspace), "a")
    assert changed.fingerprint != first.fingerprint


# ---------------------------------------------------------------------------
# 真实语料形状(frontmatter 逐字复制,不在测试期读 live repo)
# ---------------------------------------------------------------------------


def test_repo_corpus_shaped_skills_parse(tmp_path: Path) -> None:
    workspace = _standalone_workspace(tmp_path)
    root = workspace / ".agents" / "skills"
    _write_raw(root / "xlsx", _XLSX_SKILL_MD)
    _write_raw(root / "docx", _DOCX_SKILL_MD)

    catalog = load_skill_catalog(workspace)

    assert [m.name for m in catalog.skills] == ["docx", "xlsx"]
    xlsx = catalog.get("xlsx")
    docx = catalog.get("docx")
    assert xlsx is not None and docx is not None
    assert xlsx.description.startswith("Use this skill any time a spreadsheet")
    assert 'even casually (like "the xlsx in my downloads")' in xlsx.description
    assert docx.description.startswith("Use this skill whenever the user wants to create")
    assert xlsx.when_to_use is None and xlsx.version is None
    assert xlsx.disable_model_invocation is False
    assert re.fullmatch(r"[0-9a-f]{64}", xlsx.fingerprint)


# ---------------------------------------------------------------------------
# 协议面常量与校验(brief 数值逐字)
# ---------------------------------------------------------------------------


def test_protocol_constants_exact() -> None:
    assert MAX_SKILL_NAME_CHARS == 500
    assert MAX_SKILL_DESCRIPTION_CHARS == 200
    assert MAX_INVOKE_ARGS_CHARS == 4000
    assert MAX_SKILL_INSTRUCTIONS_CHARS == 40_000
    assert SkillSource.PROJECT.value == "project"
    assert SkillSource.EXTERNAL.value == "external"
    assert SOURCE_PRIORITY == {SkillSource.PROJECT: 0, SkillSource.EXTERNAL: 1}


def test_manifest_validation_and_catalog_lookup(tmp_path: Path) -> None:
    base: dict[str, object] = {
        "name": "n",
        "description": "d",
        "when_to_use": None,
        "version": None,
        "skill_file": str(tmp_path / "SKILL.md"),
        "root_dir": str(tmp_path),
        "source": SkillSource.PROJECT,
        "fingerprint": "a" * 64,
        "disable_model_invocation": False,
    }
    manifest = SkillManifest(**base)  # type: ignore[arg-type]
    assert manifest.name == "n"

    # name 1..500 边界
    assert SkillManifest(**{**base, "name": "x" * 500}).name == "x" * 500  # type: ignore[arg-type]

    invalid_overrides: list[dict[str, object]] = [
        {"name": ""},
        {"name": "x" * 501},
        {"description": ""},
        {"when_to_use": ""},
        {"version": ""},
        {"skill_file": "relative/SKILL.md"},
        {"root_dir": ""},
        {"source": "project"},
        {"fingerprint": "a" * 129},
        {"disable_model_invocation": 1},
    ]
    for overrides in invalid_overrides:
        with pytest.raises(ValueError):
            SkillManifest(**{**base, **overrides})  # type: ignore[arg-type]

    catalog = SkillCatalog((manifest,))
    assert catalog.get("n") is manifest
    assert catalog.get("missing") is None