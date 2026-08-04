# Praxis Hard Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Praxis the repository's single public identity, move the complete Agent runtime out of `rag.agent`, clear all repository Ruff debt, publish honest demo/benchmark evidence, merge the verified PR, and rename the GitHub repository.

**Architecture:** Preserve the existing `AgentService -> AgentLoop -> ToolExecutor` runtime while moving ownership to `agent_runtime`. `rag` remains an optional knowledge subsystem reached only through `agent_runtime.knowledge_providers.rag`; neutral model contracts move once into `agent_runtime.modeling` and may be consumed inward by RAG. Agent state moves to `.praxis`, while `.rag` remains untouched knowledge state.

**Tech Stack:** Python 3.12, uv, Hatchling, Typer, Pydantic, LangGraph checkpointing, Ruff, mypy, pytest, import-linter, Pillow, GitHub Actions, Groq `openai/gpt-oss-120b`.

---

## Execution rules

- Work only in `/Users/leixiaoying/LLM/RAG学习-worktrees/praxis-hard-cutover` on `codex/praxis-hard-cutover`.
- Never reset, clean, edit, or fast-forward `/Users/leixiaoying/LLM/RAG学习`; its dirty `configs/models.yaml` belongs to the user.
- Use red-green-refactor for behavior and boundary changes. Watch each focused test fail for the intended reason before production edits.
- Use `apply_patch` for authored file changes. `git mv`, Ruff's safe fixer, `uv lock`, generated GIF rendering, and other bounded mechanical transformations are allowed.
- Preserve runtime behavior; do not add a second loop, service, executor, registry, provider gateway, or checkpoint system.
- Keep `evals/code_agent/benchmark_v1.json`, `tests/agent/test_code_agent_benchmark.py` historical path assertions, and `docs/superpowers/**` out of blind `rag.agent` replacements.
- Use only Groq alias `groq_gpt_oss_120b` for cloud evidence. Never expose the API key in output or artifacts.

## Target file map

### Runtime ownership

- Move `rag/agent/builtin/` to `agent_runtime/builtin/`.
- Move `rag/agent/core/` to `agent_runtime/core/`.
- Move `rag/agent/loop/` to `agent_runtime/loop/`.
- Move `rag/agent/memory/` to `agent_runtime/memory/`.
- Move `rag/agent/skills/` to `agent_runtime/skills/`.
- Move `rag/agent/streaming/` to `agent_runtime/streaming/`.
- Move `rag/agent/tools/` to `agent_runtime/tools/`.
- Move `rag/agent/cli.py`, `file_manifest.py`, `primitive_ops.py`, `service.py`, `turns.py`, and `workspace.py` to the `agent_runtime/` root.
- Merge required exports deliberately into the existing `agent_runtime/__init__.py`; remove `rag/agent/__init__.py`.

### Neutral shared contracts

- Create `agent_runtime/modeling/__init__.py`.
- Move `rag/providers/llm_gateway.py` to `agent_runtime/modeling/gateway.py`.
- Move `rag/models/config.py` to `agent_runtime/modeling/config.py`.
- Move `rag/schema/llm.py` to `agent_runtime/modeling/contracts.py`.
- Move `rag/assembly/tokenizer.py` to `agent_runtime/modeling/tokenization.py`.
- Move `rag/utils/text.py` to `agent_runtime/text.py`.
- Recreate the old non-Agent RAG modules only as explicit RAG-facing re-exports when current RAG tests require them.
- Add Agent-owned evidence/citation DTOs to `agent_runtime/knowledge.py`; convert RAG DTOs only in `agent_runtime/knowledge_providers/rag.py`.

### Public artifacts

- Create `LICENSE`.
- Create `scripts/render_praxis_demo.py` and `docs/assets/praxis-demo.gif`.
- Create `scripts/render_model_quality_report.py`.
- Create `docs/benchmark.md` and `docs/runs/groq-gpt-oss-120b.md`.
- Create a redacted report in `evals/model_quality/runs/` after the live gate.
- Rewrite `README.md` around the approved recruiter/runtime path.

## Task 1: Establish the hard namespace and package cutover

**Files:**

- Create: `tests/agent/test_praxis_namespace_contract.py`
- Move: every tracked file below `rag/agent/` to its mapped `agent_runtime/` destination
- Modify: `agent_runtime/__init__.py`
- Modify: `rag/__init__.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: active imports under `agent_runtime/`, `rag/`, `tests/`, and `scripts/`
- Modify: active path references in `README.md`, `docs/RUNBOOK.md`, and `docs/design/*.md`

- [ ] **Step 1: Write failing namespace contract tests**

Add tests with these essential assertions:

```python
def test_distribution_and_console_entrypoint_use_praxis_namespace() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["name"] == "praxis-agent-runtime"
    assert config["project"]["scripts"]["agent"] == "agent_runtime.cli:agent_app"
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"] == {
        "configs/models.yaml": "agent_runtime/_data/models.yaml"
    }


def test_legacy_rag_agent_package_is_absent() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rag.agent")


def test_rag_root_no_longer_exports_agent_runtime_objects() -> None:
    assert not hasattr(rag, "AgentService")
    assert not hasattr(rag, "ToolRegistry")
```

Also add a source scan that fails on `rag.agent` in active code/tests/scripts,
with an explicit allowlist for `tests/agent/test_code_agent_benchmark.py` and
the frozen manifest.

- [ ] **Step 2: Run the contract test and confirm RED**

Run:

```bash
uv run pytest -q tests/agent/test_praxis_namespace_contract.py
```

Expected: failures show the old distribution name, old console target, existing
`rag.agent` package, and old root exports.

- [ ] **Step 3: Move the runtime tree with Git history preserved**

Run the exact directory moves:

```bash
git mv rag/agent/builtin agent_runtime/builtin
git mv rag/agent/core agent_runtime/core
git mv rag/agent/loop agent_runtime/loop
git mv rag/agent/memory agent_runtime/memory
git mv rag/agent/skills agent_runtime/skills
git mv rag/agent/streaming agent_runtime/streaming
git mv rag/agent/tools agent_runtime/tools
git mv rag/agent/cli.py agent_runtime/cli.py
git mv rag/agent/file_manifest.py agent_runtime/file_manifest.py
git mv rag/agent/primitive_ops.py agent_runtime/primitive_ops.py
git mv rag/agent/service.py agent_runtime/service.py
git mv rag/agent/turns.py agent_runtime/turns.py
git mv rag/agent/workspace.py agent_runtime/workspace.py
git rm rag/agent/__init__.py
```

Use a bounded mechanical rewrite from `rag.agent` to `agent_runtime` and from
`rag/agent` to `agent_runtime` only in active source, tests, scripts, README,
RUNBOOK, and current design docs. Exclude `docs/superpowers/**`, the frozen
manifest, and its historical-path validation test. Inspect the resulting diff
before continuing.

- [ ] **Step 4: Update public exports and packaging**

Apply these changes:

- keep `agent_runtime.Agent`, result DTOs, planning DTOs, and stream events as
  deliberate public exports;
- remove every Agent lazy export from `rag/__init__.py`;
- set project name to `praxis-agent-runtime` and approved description;
- point the `agent` script at `agent_runtime.cli:agent_app`;
- keep `rag = "rag.cli:app"`;
- force-include `configs/models.yaml` at `agent_runtime/_data/models.yaml`;
- change bundled-config package lookup from `rag.agent` to `agent_runtime`;
- run `uv lock` to update local package metadata without changing unrelated
  dependency versions.

- [ ] **Step 5: Run focused namespace and public API tests**

Run:

```bash
uv run pytest -q \
  tests/agent/test_praxis_namespace_contract.py \
  tests/agent/test_agent_runtime_imports.py \
  tests/agent/test_agent_runtime_facade.py \
  tests/agent/test_cli_wiring.py \
  tests/ui/test_cli.py
```

Expected: PASS; `rag.agent` cannot import, while `agent_runtime.Agent`, `agent
--help`, and `rag --help` remain valid.

- [ ] **Step 6: Run moved runtime tests and commit**

Run:

```bash
uv run pytest -q tests/agent tests/ui
uv run mypy agent_runtime
git diff --check
```

Expected: Agent/UI tests and mypy pass. Commit:

```bash
git add agent_runtime rag tests scripts pyproject.toml uv.lock README.md docs/RUNBOOK.md docs/design
git commit -m "refactor: move Agent runtime out of RAG namespace"
```

## Task 2: Make the Agent-to-RAG boundary real

**Files:**

- Create: `agent_runtime/modeling/__init__.py`
- Create: `agent_runtime/modeling/config.py`
- Create: `agent_runtime/modeling/contracts.py`
- Create: `agent_runtime/modeling/gateway.py`
- Create: `agent_runtime/modeling/tokenization.py`
- Create: `agent_runtime/text.py`
- Modify: `agent_runtime/knowledge.py`
- Modify: `agent_runtime/result.py`
- Modify: `agent_runtime/knowledge_providers/rag.py`
- Modify: `agent_runtime/service.py`
- Modify: `agent_runtime/core/observations.py`
- Modify: `agent_runtime/core/checkpointing.py`
- Modify: Agent modules that import `rag.models`, `rag.schema.llm`, `rag.providers.llm_gateway`, `rag.assembly.tokenizer`, or `rag.utils.text`
- Modify: `rag/providers/llm_gateway.py`, `rag/models/config.py`, `rag/schema/llm.py`, `rag/assembly/tokenizer.py`, `rag/utils/text.py`
- Modify: RAG consumers of the moved canonical modules
- Modify: `.importlinter`
- Create: `tests/agent/test_runtime_rag_boundary.py`

- [ ] **Step 1: Write failing dependency and identity tests**

Add tests that AST-scan every `agent_runtime/**/*.py` file except
`agent_runtime/knowledge_providers/rag.py` and fail on any import beginning
with `rag`. Add identity tests such as:

```python
def test_rag_llm_contract_reexports_runtime_identity() -> None:
    from agent_runtime.modeling.contracts import LLMUsage as RuntimeUsage
    from rag.schema.llm import LLMUsage as RagUsage

    assert RagUsage is RuntimeUsage
    assert RuntimeUsage.__module__ == "agent_runtime.modeling.contracts"
```

Add result tests proving internal evidence and citations use Agent-owned DTOs,
while `LazyRAGKnowledgeProvider` converts RAG query values at the adapter edge.

- [ ] **Step 2: Run the boundary tests and confirm RED**

Run:

```bash
uv run pytest -q tests/agent/test_runtime_rag_boundary.py
```

Expected: failures list current imports from `rag.schema`, `rag.models`,
`rag.providers`, `rag.assembly`, and `rag.utils`.

- [ ] **Step 3: Move each neutral implementation once**

Use `git mv` for the five canonical modules, then add explicit RAG-facing
re-export modules with their existing `__all__` contracts. Update Agent imports
to the canonical `agent_runtime` modules. Update every RAG production import to
the canonical modules so monkeypatching and type identity are not split across
wrappers. The old RAG paths exist only as external-facing re-exports and in
focused compatibility tests.

Do not copy the 1,134-line gateway. There must be one `LLMGateway`
implementation.

- [ ] **Step 4: Introduce Agent-owned knowledge DTOs**

In `agent_runtime/knowledge.py`, add frozen DTOs or Pydantic models for the
runtime's evidence, citation, and grounding target values. Preserve the fields
currently projected by `AgentResult`.

Change `agent_runtime/service.py` and `agent_runtime/core/observations.py` to
validate Agent-owned DTOs. Change `agent_runtime/result.py` to project those
DTOs without importing `rag.schema.query`. Keep all `rag.schema.query`
conversion inside `agent_runtime/knowledge_providers/rag.py`.

- [ ] **Step 5: Make checkpoint identities current-only**

Update the checkpoint allowlist to canonical `agent_runtime` module identities.
Remove old `rag.agent` aliases and do not add aliases for moved
`rag.schema.llm`/`rag.models.config` identities. Preserve data-shape migrations
that do not depend on removed module paths.

Add a serialization assertion:

```python
encoded = agent_checkpoint_serde().dumps_typed(LLMUsage(input_tokens=1))
assert b"rag.schema.llm" not in encoded[1]
assert agent_checkpoint_serde().loads_typed(encoded) == LLMUsage(input_tokens=1)
```

- [ ] **Step 6: Enforce the import contract**

Configure import-linter with both roots and a forbidden contract from
`agent_runtime` to `rag`, ignoring only direct imports made by
`agent_runtime.knowledge_providers.rag`. Keep the existing schema/tools/memory
contracts expressed against their new package paths.

- [ ] **Step 7: Verify focused boundaries and commit**

Run:

```bash
uv run pytest -q \
  tests/agent/test_runtime_rag_boundary.py \
  tests/agent/test_checkpointing.py \
  tests/agent/test_loop_state.py \
  tests/agent/test_rag_answer_tool.py \
  tests/agent/test_agent_service.py \
  tests/core/test_model_runtime.py \
  tests/core/test_query_pipeline.py
uv run lint-imports
uv run mypy agent_runtime rag
git diff --check
```

Expected: PASS and all import contracts kept. Commit:

```bash
git add agent_runtime rag tests .importlinter
git commit -m "refactor: isolate runtime from optional RAG subsystem"
```

## Task 3: Separate Agent state from RAG state

**Files:**

- Modify: `agent_runtime/cli.py`
- Modify: `agent_runtime/workspace.py`
- Modify: `agent_runtime/agent.py`
- Modify: `agent_runtime/knowledge.py`
- Modify: `.gitignore`
- Modify: `tests/agent/test_workspace.py`
- Modify: `tests/agent/test_cli_wiring.py`
- Modify: `tests/agent/test_agent_cli_resume.py`
- Modify: `tests/agent/test_checkpointing.py`
- Modify: `tests/agent/test_praxis_namespace_contract.py`
- Modify: `docs/RUNBOOK.md`

- [ ] **Step 1: Write failing `.praxis` state tests**

Assert these defaults:

```python
assert DEFAULT_CHECKPOINT_PATH == Path(".praxis/checkpoints.sqlite")
assert DEFAULT_MODEL_SESSION_PATH == Path(".praxis/model_session.json")
assert workspace.runtime_root == root / ".praxis" / "runtime"
assert RAGKnowledgeConfig().storage_root == Path(".rag")
```

Create old `.rag/agent_checkpoints.sqlite` and `.rag/agent_model_session.json`
sentinels in a temporary workspace and assert opening the new Agent neither
reads nor deletes them.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
uv run pytest -q \
  tests/agent/test_workspace.py \
  tests/agent/test_cli_wiring.py \
  tests/agent/test_agent_cli_resume.py \
  tests/agent/test_praxis_namespace_contract.py
```

Expected: old `.rag/agent_*` defaults fail.

- [ ] **Step 3: Implement the state-root cutover**

Set Agent defaults to `.praxis`, set workspace runtime root to
`.praxis/runtime`, retain `.rag` for `RAGKnowledgeConfig`, add `.praxis/` to
`.gitignore`, and update active docs/help text. Do not add auto-migration or
deletion code.

- [ ] **Step 4: Verify GREEN and commit**

Run the focused tests above plus:

```bash
uv run pytest -q tests/agent/test_checkpointing.py tests/agent/test_public_turn_api.py
git diff --check
```

Expected: PASS. Commit:

```bash
git add agent_runtime tests .gitignore docs/RUNBOOK.md
git commit -m "refactor: separate Praxis runtime state from RAG data"
```

## Task 4: Clear all 78 Ruff errors without weakening policy

**Files:**

- Modify: `rag/ingest/header_detector.py`
- Modify: `rag/ingest/parsers/dispatcher.py`
- Modify: `rag/ingest/parsers/excel_parser_repo.py`
- Modify: `rag/ingest/parsers/util.py`
- Modify: `rag/retrieval/graph.py`
- Modify: `rag/schema/graph.py`
- Modify: `rag/schema/model_protocols.py`
- Modify: `rag/schema/runtime.py`
- Modify: `rag/storage/repositories/postgres_metadata_repo.py`
- Modify: `rag/storage/storage_lifecycle_service.py`
- Modify: `rag/utils/guard.py`
- Modify: `scripts/check_anti_patterns.py`
- Modify: Ruff-reported tests under `tests/agent`, `tests/core`, `tests/provider`, and `tests/service`

- [ ] **Step 1: Capture the full RED baseline after namespace moves**

Run:

```bash
uv run ruff check . --output-format concise
```

Expected: non-zero with the current historical violations plus any move-induced
issues. Save the terminal output as evidence; do not add it to Git.

- [ ] **Step 2: Apply only Ruff's safe fixes**

Run:

```bash
uv run ruff check . --fix
git diff --check
```

Inspect every auto-fix. Do not use `--unsafe-fixes`, global ignores, per-file
blankets, or bulk `noqa` comments.

- [ ] **Step 3: Fix semantic findings manually**

- wrap E501 lines while preserving strings and SQL semantics;
- replace constant `getattr` with direct access for B009;
- replace B017 blind exceptions with the exact expected exception;
- repair the F821 `query_understanding` reference using the intended fixture or
  explicit construction;
- remove genuinely unused imports without altering assertions.

- [ ] **Step 4: Run affected tests before the full linter**

Run the test files touched by manual fixes, including:

```bash
uv run pytest -q \
  tests/core/test_cli_runtime_model_loading.py \
  tests/core/test_citation_formatter.py \
  tests/core/test_excel_parser_repo.py \
  tests/core/test_guard.py \
  tests/core/test_header_detector.py \
  tests/core/test_index_sync_service.py \
  tests/core/test_table_executor.py \
  tests/service/test_retrieval_adapter.py
```

Expected: PASS.

- [ ] **Step 5: Verify Ruff GREEN and commit**

Run:

```bash
uv run ruff check .
uv run pytest -q
git diff --check
```

Expected: Ruff reports zero errors and the full suite passes. Commit:

```bash
git add rag agent_runtime scripts tests
git commit -m "style: clear repository Ruff debt"
```

## Task 5: Upgrade CI and prove the built distribution

**Files:**

- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Create: `tests/repo/test_distribution_contract.py`
- Modify: `scripts/agent_cli_smoke.py`
- Modify: `tests/agent/test_cli_smoke_script.py`

- [ ] **Step 1: Write failing distribution tests**

Build the project in a temporary output directory and inspect the wheel:

```python
def test_wheel_contains_runtime_and_excludes_legacy_agent(tmp_path: Path) -> None:
    wheel = build_wheel(tmp_path)
    names = wheel_members(wheel)
    assert "agent_runtime/cli.py" in names
    assert not any(name.startswith("rag/agent/") for name in names)
    assert entry_point(wheel, "agent") == "agent_runtime.cli:agent_app"
```

Add a CLI smoke assertion for installed `agent --help` and `rag --help`.

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```bash
uv run pytest -q tests/repo/test_distribution_contract.py tests/agent/test_cli_smoke_script.py
```

Expected: failure until packaging and installed-smoke helpers use the Praxis
metadata and runtime path.

- [ ] **Step 3: Replace changed-file CI gates with full gates**

Make the workflow run, in order:

```bash
uv sync --locked
uv run ruff check .
uv run mypy
uv run pytest -q
uv run lint-imports
uv build
uv run python scripts/agent_cli_smoke.py
uv run python scripts/agent_delivery_smoke.py --fake-model --verbose
uv run python scripts/agent_tool_aci_eval.py --fake-model --json
uv run python scripts/agent_code_benchmark.py validate evals/code_agent/benchmark_v1.json --repository .
```

For the installed-wheel smoke, create a temporary environment under
`$RUNNER_TEMP`, install the built wheel with uv, and run `agent --help` and
`rag --help`. Do not create a repository-local environment that could be
committed.

- [ ] **Step 4: Verify the local CI equivalent and commit**

Run every deterministic command above locally. Expected: all exit zero. Commit:

```bash
git add .github/workflows/ci.yml pyproject.toml scripts tests
git commit -m "ci: enforce full repository quality gates"
```

## Task 6: Bind live model evidence to a clean source tree

**Files:**

- Modify: `scripts/agent_model_quality_gate.py`
- Modify: `tests/agent/test_model_quality_gate.py`
- Create: `scripts/render_model_quality_report.py`
- Create: `tests/agent/test_model_quality_report.py`

- [ ] **Step 1: Write failing repository fingerprint tests**

Create a temporary Git repository in tests and assert:

```python
fingerprint = repository_fingerprint(clean_repo)
assert fingerprint.dirty is False
assert fingerprint.source_commit == git(clean_repo, "rev-parse", "HEAD")
assert fingerprint.source_tree == git(clean_repo, "rev-parse", "HEAD^{tree}")

(clean_repo / "untracked.txt").write_text("dirty", encoding="utf-8")
with pytest.raises(DirtyRepositoryError):
    repository_fingerprint(clean_repo)
```

Add report-shape assertions for `source_commit`, `source_tree`, `dirty: false`,
suite revision, evaluator version, and redacted paths.

- [ ] **Step 2: Run the fingerprint/report tests and confirm RED**

Run:

```bash
uv run pytest -q tests/agent/test_model_quality_gate.py tests/agent/test_model_quality_report.py
```

Expected: missing fingerprint API and fields.

- [ ] **Step 3: Implement fail-closed preflight**

Add a small immutable fingerprint value and a function accepting an explicit
repository path. It must use Git commands without a shell, fail on any tracked
or untracked non-ignored change, and run before the first provider call.

Add `source_commit`, `source_tree`, `dirty: false`, fixture revision, and an
explicit evaluator version to gate reports. Keep infrastructure failures as
exit 2 / inconclusive.

- [ ] **Step 4: Implement the Markdown renderer**

`scripts/render_model_quality_report.py` reads one redacted gate report and
writes `docs/benchmark.md` plus the expanded approval-continuation run record.
It must not invent metrics, must surface failed/inconclusive cases, and must
reject reports whose `dirty` field is not false.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```bash
uv run pytest -q tests/agent/test_model_quality_gate.py tests/agent/test_model_quality_report.py
uv run ruff check scripts/agent_model_quality_gate.py scripts/render_model_quality_report.py tests/agent/test_model_quality_gate.py tests/agent/test_model_quality_report.py
uv run mypy scripts/agent_model_quality_gate.py scripts/render_model_quality_report.py
git diff --check
```

Expected: PASS. Commit:

```bash
git add scripts/agent_model_quality_gate.py scripts/render_model_quality_report.py tests/agent
git commit -m "feat: bind model evidence to committed source"
```

## Task 7: Build a reproducible, clearly labelled demo GIF

**Files:**

- Modify: `scripts/agent_delivery_smoke.py`
- Create: `scripts/render_praxis_demo.py`
- Modify: `tests/agent/test_delivery_smoke_script.py`
- Create: `tests/repo/test_praxis_demo.py`
- Create: `docs/assets/praxis-demo.gif`

- [ ] **Step 1: Write failing demo scenario and renderer tests**

Add a deterministic public-path case that emits a bounded inspect, patch,
verification, and completion trace. Assert the result passes, the workspace
diff is correct, and every rendered frame receives these banners:

```text
PRAXIS — DETERMINISTIC DEMO
FAKE MODEL — NOT MODEL QUALITY EVIDENCE
```

Open the generated GIF with Pillow and assert `n_frames >= 5`, non-zero
dimensions, and a bounded file size suitable for GitHub README rendering.

- [ ] **Step 2: Run demo tests and confirm RED**

Run:

```bash
uv run pytest -q tests/agent/test_delivery_smoke_script.py tests/repo/test_praxis_demo.py
```

Expected: missing demo case/renderer/GIF.

- [ ] **Step 3: Implement the deterministic renderer**

Use the public Agent facade and existing fake provider path; do not render a
hand-authored success unrelated to runtime execution. Capture sanitized event
lines, then use Pillow and a portable monospace font to render terminal-style
frames. Never include credentials or absolute local paths.

- [ ] **Step 4: Generate and inspect the GIF**

Run:

```bash
uv run python scripts/render_praxis_demo.py --output docs/assets/praxis-demo.gif
uv run pytest -q tests/repo/test_praxis_demo.py
```

Open the generated GIF locally for visual inspection. Expected: readable
labels, no clipping, and a deterministic completion trace.

- [ ] **Step 5: Verify and commit**

Run focused Ruff/mypy/tests and `git diff --check`. Commit:

```bash
git add scripts/agent_delivery_smoke.py scripts/render_praxis_demo.py tests docs/assets/praxis-demo.gif
git commit -m "docs: add deterministic Praxis demo"
```

## Task 8: Rewrite the public repository story and add MIT license

**Files:**

- Rewrite: `README.md`
- Create: `LICENSE`
- Create initially: `docs/benchmark.md`
- Create initially: `docs/runs/groq-gpt-oss-120b.md`
- Modify: `docs/RUNBOOK.md`
- Modify: current non-archived docs containing product identity or active paths
- Create: `tests/repo/test_readme_presentation.py`

- [ ] **Step 1: Write failing presentation tests**

Assert README contains the approved title/tagline, demo link, architecture,
evidence, Quickstart, optional RAG, safety/limitations, and MIT link. Assert it
does not contain `Private-RAG-Agent`, a `production-ready` claim, or active
`rag.agent` paths.

- [ ] **Step 2: Run the presentation tests and confirm RED**

Run:

```bash
uv run pytest -q tests/repo/test_readme_presentation.py
```

Expected: old title/clone path and missing assets fail.

- [ ] **Step 3: Add the license and concise README**

Create the standard MIT license with `Copyright (c) 2026 xiaoyinglei`.
Rewrite README in the approved recruiter/runtime order. Keep code examples on
the public `agent` and `agent_runtime.Agent` APIs. Link deep details instead of
retaining a near-thousand-line combined manual.

Create honest benchmark/run pages with methodology, exact evidence fields, and
a clearly marked pre-live state that cannot be mistaken for a PASS. These pages
are filled from the live renderer in Task 10 before merge.

- [ ] **Step 4: Verify active naming and commit**

Run:

```bash
uv run pytest -q tests/repo/test_readme_presentation.py
rg -n "Private-RAG-Agent|rag\.agent|rag/agent" README.md docs/RUNBOOK.md docs/design scripts agent_runtime rag tests \
  --glob '!docs/superpowers/**' \
  --glob '!tests/agent/test_code_agent_benchmark.py'
git diff --check
```

Expected: presentation tests pass; any remaining search hits are reviewed and
limited to the explicit frozen benchmark exception. Commit:

```bash
git add README.md LICENSE docs tests/repo/test_readme_presentation.py
git commit -m "docs: present Praxis as a workspace agent runtime"
```

## Task 9: Freeze and verify the live-evidence source commit

**Files:** No authored changes.

- [ ] **Step 1: Run the complete deterministic local gate**

Run fresh:

```bash
uv run ruff check .
uv run mypy
uv run pytest -q
uv run lint-imports
uv build
uv run python scripts/agent_cli_smoke.py
uv run python scripts/agent_delivery_smoke.py --fake-model --verbose
uv run python scripts/agent_tool_aci_eval.py --fake-model --json
uv run python scripts/agent_code_benchmark.py validate evals/code_agent/benchmark_v1.json --repository .
uv run pytest -q tests/agent/test_run_command_safety.py
git diff --check
git status --short
```

Expected: every command exits zero and Git status is clean.

- [ ] **Step 2: Record the candidate identities outside Git**

Run:

```bash
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
```

The live gate must record these exact values itself. Do not hand-copy them into
the report before execution.

## Task 10: Run Groq 5-case x 3-trial evidence and publish the report

**Files:**

- Create: `evals/model_quality/runs/2026-08-04-groq-gpt-oss-120b.json`
- Update: `docs/benchmark.md`
- Update: `docs/runs/groq-gpt-oss-120b.md`
- Update: `README.md` evidence table with generated metrics only

- [ ] **Step 1: Run the clean-tree live gate**

With the linked-worktree `.env` fallback intact, run:

```bash
uv run python scripts/agent_model_quality_gate.py gate \
  --env-file .env \
  --model groq_gpt_oss_120b \
  --report evals/model_quality/runs/2026-08-04-groq-gpt-oss-120b.json
```

Expected: exit 0 and 5 capabilities x 3 trials in the report. Exit 2 is
inconclusive and must be diagnosed/retried without changing provider. Exit 1
is a real quality failure and blocks completion until the generic runtime/model
issue is understood; do not tune prompts to fixture IDs.

- [ ] **Step 2: Validate and render committed evidence**

Run:

```bash
uv run python scripts/render_model_quality_report.py \
  evals/model_quality/runs/2026-08-04-groq-gpt-oss-120b.json \
  --benchmark docs/benchmark.md \
  --run-record docs/runs/groq-gpt-oss-120b.md
```

Inspect the JSON and Markdown for secrets, absolute temporary paths, provider
headers, and invented metrics. The approval-continuation record must include
the task, tool trace, approval/resume evidence, diff assertion, final answer,
and evaluator verdict.

- [ ] **Step 3: Run evidence tests and commit without production changes**

Run:

```bash
uv run pytest -q tests/agent/test_model_quality_gate.py tests/agent/test_model_quality_report.py tests/repo/test_readme_presentation.py
git diff --check
git diff --name-only
```

Expected: only report/README documentation artifacts changed; no production
code changed after the measured source commit. Commit:

```bash
git add README.md docs/benchmark.md docs/runs evals/model_quality/runs
git commit -m "docs: publish current Groq model evidence"
```

- [ ] **Step 4: Re-run all deterministic gates on evidence HEAD**

Repeat every command from Task 9. Expected: all exit zero and tree clean.

## Task 11: Review, PR, merge, and GitHub repository rename

**Files:** No new product changes unless review or CI finds a defect.

- [ ] **Step 1: Perform requirement and code review**

Use the requesting-code-review skill with the spec, plan, base commit, and
current HEAD. Independently inspect `git diff origin/main...HEAD`, commits,
untracked files, and every acceptance criterion. Fix real issues with focused
tests and rerun affected/full gates.

- [ ] **Step 2: Push and create a Draft PR**

Use `apply_patch` to write `/tmp/praxis-pr-body.md` with this reviewed content:

```markdown
## Summary

- hard-cut Agent ownership from `rag.agent` to `agent_runtime`
- rename the project to Praxis and separate `.praxis` runtime state from `.rag` knowledge data
- clear full-repository Ruff debt and enforce full deterministic CI gates
- add a labelled fake-model demo plus commit-bound Groq model-quality evidence

## Breaking changes

- `rag.agent.*` imports and old Agent checkpoints are intentionally unsupported
- the distribution metadata is `praxis-agent-runtime`; no PyPI publication is part of this PR

## Evidence

- full pytest, Ruff, mypy, import contracts, package/CLI smoke, deterministic delivery, benchmark manifest, and macOS Seatbelt gates
- Groq `groq_gpt_oss_120b`: 5 cases x 3 trials, with source commit/tree provenance in `docs/benchmark.md`

## Limits

- trusted-local macOS/Python runtime, not a production-ready general autonomous Agent
- the frozen 30-task Code Agent suite is documented but not claimed as a passing release gate here
```

Then run:

```bash
git push -u origin codex/praxis-hard-cutover
gh pr create --draft \
  --base main \
  --head codex/praxis-hard-cutover \
  --title "Praxis hard cutover and repository polish" \
  --body-file /tmp/praxis-pr-body.md
```

The PR body summarizes the breaking namespace/state boundary, deterministic
gates, live evidence provenance, limitations, and explicitly states no PyPI
publication.

- [ ] **Step 3: Wait for PR CI and resolve failures**

Use `gh pr checks --watch`. Do not merge on partial or pending checks. For each
failure, reproduce locally, write a regression test when behavior is involved,
fix the smallest cause, and rerun all relevant gates.

- [ ] **Step 4: Mark ready and merge**

After fresh local verification and green PR CI:

```bash
gh pr ready
gh pr merge --merge --delete-branch
```

Record the exact merge commit on `main`.

- [ ] **Step 5: Wait for the push workflow on merged `main`**

Query Actions for the exact merge commit and wait until it concludes success.
Do not rename the repository before this run is green.

- [ ] **Step 6: Rename and update repository metadata**

Run:

```bash
gh repo rename praxis-agent-runtime --repo xiaoyinglei/Private-RAG-Agent --yes
git remote set-url origin https://github.com/xiaoyinglei/praxis-agent-runtime.git
gh repo edit xiaoyinglei/praxis-agent-runtime \
  --description "Praxis — a trusted-local workspace agent runtime for files, code, data, and private knowledge." \
  --add-topic python \
  --add-topic ai-agent \
  --add-topic agent-runtime \
  --add-topic llm \
  --add-topic tool-use \
  --add-topic rag \
  --add-topic mcp
```

- [ ] **Step 7: Verify final external and local state**

Verify:

- `gh repo view xiaoyinglei/praxis-agent-runtime` reports public, `main`, the
  approved description, and topics;
- the old GitHub URL redirects to the new repository;
- `origin` fetch/push URLs use the new slug;
- `origin/main` contains the merge commit;
- no PR remains open;
- the implementation worktree is clean;
- the original dirty `main` still has its user-owned `configs/models.yaml` and
  was not synchronized or cleaned;
- no PyPI project or GitHub Release was created.

Only after these checks may the work be reported complete.
