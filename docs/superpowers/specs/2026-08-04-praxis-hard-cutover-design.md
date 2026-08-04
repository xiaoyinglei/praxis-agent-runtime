# Praxis Hard Cutover and Repository Polish Design

## Status

Approved interactively on 2026-08-04. The user selected:

- product name: **Praxis**;
- tagline: **a trusted-local workspace agent runtime**;
- repository slug: `praxis-agent-runtime`;
- Python import package: `agent_runtime`;
- hard removal of `rag.agent.*` without compatibility shims;
- `rag` retained as an optional knowledge/evidence subsystem;
- MIT license, `Copyright (c) 2026 xiaoyinglei`;
- a deterministic fake-model GIF plus separate real-model evidence;
- the real-model gate uses Groq alias `groq_gpt_oss_120b`;
- one PR with staged commits, followed by merge and GitHub repository rename.

This specification consolidates the five design sections the user reviewed and
approved. It does not reopen product scope.

## Context and baseline

The repository currently presents mixed identities:

- GitHub is `xiaoyinglei/Private-RAG-Agent`;
- the README leads with a generic Code Agent;
- the distribution is named `agent-runtime`;
- the public Python facade is `agent_runtime`;
- most Agent implementation still lives under `rag/agent`;
- `rag.__init__` still exports Agent objects;
- the `agent` console entry point still targets `rag.agent.cli`;
- Agent-owned local state is stored below `.rag`.

At the start of this work, the isolated worktree baseline is:

- source commit: `f955685aa4524f929e748f5c93655721e143f58e`;
- tests: 1478 passed, 4 skipped;
- full-repository Ruff: 78 errors;
- CI: full pytest and import contracts, but Ruff and mypy only on changed files;
- no repository LICENSE, demo GIF, or current one-page benchmark report.

The user's original `main` worktree at
`/Users/leixiaoying/LLM/RAG学习` is 20 commits behind `origin/main` and has an
uncommitted `configs/models.yaml`. This work must not modify, clean, reset, or
fast-forward that worktree.

## Goals

1. Give the repository one coherent public identity: Praxis.
2. Make `agent_runtime` the actual owner of Agent execution code, not only a
   facade over `rag.agent`.
3. Keep RAG available as an explicit, lazy knowledge provider and independent
   `rag` CLI, rather than the identity of the whole product.
4. Separate Agent-owned local state from RAG knowledge-store state.
5. Remove the existing full-repository Ruff debt and make it impossible for CI
   to silently reintroduce it.
6. Give a recruiter or interviewer a concise, evidence-backed README, stable
   demo, current real-model result, and honest limitations.
7. Deliver through one reviewable PR, then rename the public GitHub repository
   only after all gates pass and the PR is merged.

## Non-goals

- Do not publish a PyPI distribution.
- Do not create a GitHub Release or version tag.
- Do not add a second runtime, loop, executor, registry, event system, or
  checkpoint store.
- Do not expand the tool set or add product features unrelated to the rename
  and ownership cleanup.
- Do not promise production readiness or general autonomous-agent readiness.
- Do not run the 30-task Code Agent suite as a release claim in this change.
- Do not delete old local checkpoints, Agent state, RAG indexes, or user data.
- Do not preserve `rag.agent` imports, modules, or checkpoint type aliases.
- Do not use DeepSeek, Kimi, or another cloud model for the live evidence.

## Product identity

The public identity is:

```text
Praxis
A trusted-local workspace agent runtime
```

The complete naming map is:

| Surface | Target |
| --- | --- |
| GitHub repository | `xiaoyinglei/praxis-agent-runtime` |
| Distribution metadata | `praxis-agent-runtime` |
| Python import | `agent_runtime` |
| Primary CLI | `agent` |
| Optional knowledge CLI | `rag` |
| Agent local state | `.praxis/` |
| RAG knowledge state | `.rag/` |

The distribution name is metadata for local builds in this task. No upload to
PyPI is authorized.

## Target architecture and ownership

### Agent runtime

`agent_runtime` owns the product API and complete execution runtime:

```text
agent_runtime/
├── __init__.py
├── agent.py
├── cli.py
├── result.py
├── models.py
├── planning.py
├── turns.py
├── workspace.py
├── file_manifest.py
├── primitive_ops.py
├── service.py
├── builtin/
├── capabilities/
├── core/
├── loop/
├── memory/
├── skills/
├── streaming/
├── tools/
├── runtime/
└── knowledge_providers/
    └── rag.py
```

The existing `AgentService -> AgentLoop -> ToolExecutor` chain remains the
single canonical runtime. Moving modules must not introduce replacement
abstractions or change verified execution semantics.

The current `rag/agent/**` files move into the corresponding
`agent_runtime/**` locations. Where a destination file already exists, the
implementation plan must merge responsibilities deliberately instead of
overwriting either file. The public facade remains `agent_runtime.Agent`.

### RAG subsystem

`rag` owns only the optional knowledge/evidence engine:

```text
rag/
├── assembly/
├── ingest/
├── models/
├── providers/
├── retrieval/
├── schema/
├── storage/
├── utils/
└── cli.py
```

It continues to provide ingestion, retrieval, reranking, grounding, citations,
storage, and its own `rag` CLI. RAG is not auto-started by the presence of a
`.rag` directory.

### Dependency direction

The core contract is:

```text
agent_runtime core -> no rag imports
agent_runtime.knowledge_providers.rag -> rag
```

Only `agent_runtime.knowledge_providers.rag` may import the RAG subsystem. It
translates RAG citation/evidence values into Agent-owned public result types.

Agent execution currently consumes several neutral model, usage, token, and
query-result contracts through `rag.*` paths. These must stop leaking the RAG
identity into the runtime. The implementation plan must place Agent-owned
contracts under `agent_runtime` and keep RAG-specific values behind the
adapter. It may update RAG consumers to use runtime-owned neutral protocols,
but it must not create a new third top-level `core` package or duplicate a
second model gateway.

Import-linter must enforce the allowed direction. Active source, tests,
scripts, and README must contain no `rag.agent` import or path. Historical
design documents under `docs/superpowers` may retain old names when clearly
treated as archived implementation history.

## Namespace and package migration

The migration is atomic on `main`:

| Current | Target | Policy |
| --- | --- | --- |
| `rag/agent/**` | `agent_runtime/**` | Move implementation; remove old tree |
| `rag.agent.*` | `agent_runtime.*` | Rewrite source, tests, scripts, active docs |
| `agent-runtime` | `praxis-agent-runtime` | Rename project metadata only |
| `rag.agent.cli:agent_app` | `agent_runtime.cli:agent_app` | Keep `agent` command |
| `rag/agent/_data/models.yaml` | `agent_runtime/_data/models.yaml` | Update wheel include and loader |
| Agent exports from `rag.__init__` | none | Remove exports and lazy aliases |

No `rag.agent` forwarding package, deprecation shim, import alias, or lazy
compatibility export remains. Tests must explicitly prove that `rag.agent`
cannot be imported and that the built wheel exposes `agent_runtime`, `agent`,
and `rag` as intended.

## Local state and checkpoint boundary

Agent-owned state moves to:

```text
.praxis/
├── checkpoints.sqlite
├── model_session.json
└── runtime/
    ├── input_files/
    ├── scratch/
    ├── artifacts/
    ├── reports/
    └── logs/
```

RAG knowledge data remains under `.rag/`. Agent rename work must not move,
rewrite, or delete RAG indexes or metadata.

The old `.rag/agent_checkpoints.sqlite`, `.rag/agent_model_session.json`, and
`.rag/agent_runtime/` files remain untouched on disk. The new runtime does not
auto-read or auto-migrate them.

Checkpoint type identities change from `rag.agent.*` to `agent_runtime.*`.
This is an intentional pre-1.0 compatibility break. The serializer allowlist
must contain only current type identities, and the implementation must not add
dozens of old module aliases. Existing migrations for still-supported data
shape versions remain tested; only the removed namespace compatibility is
dropped.

`.gitignore` must ignore `.praxis/`. No runtime state, API key, live temporary
workspace, or unredacted provider artifact may enter Git.

## README and repository presentation

The README becomes a concise product entry point instead of a near-thousand-line
combined manual. Its order is:

1. Praxis title, tagline, badges, and one-sentence positioning.
2. Deterministic demo GIF with an always-visible `FAKE MODEL` label.
3. Why Praxis: knowledge to controlled action to verifiable result.
4. Runtime architecture: Turn, Loop, ACI, approval, checkpoint, verification.
5. Current evidence at a glance.
6. Quickstart for `agent` and `agent_runtime.Agent`.
7. Capability map: files/code, data/documents, private knowledge, extensions.
8. Optional RAG usage and the explicit provider boundary.
9. Safety model and limitations.
10. Development gates and links to deeper documents.

Long operational, architecture, and reference material moves into focused
documents under `docs/`; it is not silently deleted when still useful.

The README must not claim `production-ready`, must not treat test count as Agent
task success, and must not claim the 30-task benchmark passed. Numeric claims
must be generated from the final candidate and include their measurement date
or provenance.

The repository gains:

- `LICENSE`: MIT, `Copyright (c) 2026 xiaoyinglei`;
- `docs/assets/praxis-demo.gif`;
- `docs/benchmark.md`;
- `docs/runs/groq-gpt-oss-120b.md`;
- a redacted machine-readable live result under `evals/model_quality/runs/`.

The GitHub description becomes:

> Praxis — a trusted-local workspace agent runtime for files, code, data, and
> private knowledge.

Topics should reflect actual scope, such as `python`, `ai-agent`,
`agent-runtime`, `llm`, `tool-use`, `rag`, and `mcp`.

## Deterministic demo

The GIF exercises the public Agent path with a deterministic fake model. It
shows a bounded sequence such as plan, inspect, patch, verify, and finish. It
must be reproducible without credentials and must display `DETERMINISTIC DEMO`
and `FAKE MODEL` throughout.

The GIF proves presentation and runtime wiring, not model quality. Its source
scenario must be represented by a tested script or existing deterministic
delivery case so the checked-in media cannot drift silently from the public
API.

## Real-model evidence and benchmark page

The only authorized cloud model is:

```text
alias: groq_gpt_oss_120b
provider: groq
provider model: openai/gpt-oss-120b
```

The current model-quality fixture has five capabilities:

1. exact file read;
2. search then read;
3. missing-file recovery;
4. approval continuation and workspace mutation;
5. repeated-failure control.

The existing baseline uses three trials, so the current report runs five cases
times three trials. It records the tested source commit, UTC timestamp, model
identity, environment, task success, capability rates, tool/model call counts,
latency, token usage, failures, and infrastructure status.

`approval_continuation` is the expanded human-readable real run because it
demonstrates model tool choice, destructive approval, continuation, file
mutation, and result validation. The run record includes the task, redacted
tool trace, approval event, before/after diff, final answer, evaluator verdict,
and a link to the redacted raw JSON.

Infrastructure failure is `inconclusive`; it is not converted into PASS or a
model-quality score. Secrets, provider headers, absolute temporary paths, and
unnecessary model content must be redacted before committing evidence.

The 30-task Code Agent suite remains visible as a protocol. The page may report
manifest validation, but must state that this change did not run it as a
release gate and must not reuse a historical failed or incomplete run as a
current score.

## Ruff cleanup and CI

The existing 78 Ruff errors are in scope. Fixes must preserve behavior:

- use Ruff's safe fixes only where the diff is understood;
- wrap long code instead of increasing the 120-character limit;
- fix the existing undefined-name test defect with the intended fixture or
  parameter;
- replace blind `Exception` assertions with the specific expected exception;
- use direct attribute access where B009 requires it;
- do not add global ignores, broad per-file ignores, or mass `# noqa` comments;
- do not change tests merely to hide production regressions.

CI must run on every PR and push to `main`:

1. `uv run ruff check .`;
2. full strict mypy for both `agent_runtime` and `rag`;
3. full pytest;
4. import-linter contracts;
5. wheel/sdist build and installed-package/CLI smoke;
6. deterministic fake-model delivery checks;
7. Code Agent benchmark manifest validation.

The live Groq gate is not ordinary CI: it requires a secret, incurs external
cost, and can be affected by provider limits. It is a manual final-candidate
gate after deterministic checks and macOS safety evidence pass.

The local final gate also runs real macOS Seatbelt tests for workspace and
`.git` protection. Ubuntu CI skips platform-specific Seatbelt execution rather
than simulating it and claiming equivalent evidence.

## Test strategy

Namespace and behavior changes use red-green-refactor cycles. Before moving
production modules, add focused tests that establish:

- `agent_runtime` contains the runtime modules and CLI target;
- `rag.agent` is unavailable after the cutover;
- `rag.__init__` no longer exports Agent runtime objects;
- Agent defaults use `.praxis` while RAG defaults remain `.rag`;
- old Agent checkpoint paths are not auto-read or deleted;
- the checkpoint serializer writes current `agent_runtime` type identities;
- the import boundary permits only the RAG adapter to import `rag`;
- the built distribution and console scripts use the new metadata and paths.

Existing focused tests are then migrated with their production modules. Test
renames do not authorize weakening assertions. Runtime, approval, resume,
streaming, safety, provider-wire, RAG, and packaging behavior remain covered by
the full suite.

## Delivery sequence and provenance

Implementation occurs only in:

```text
worktree: /Users/leixiaoying/LLM/RAG学习-worktrees/praxis-hard-cutover
branch:   codex/praxis-hard-cutover
base:     origin/main
```

The intended commit sequence is:

1. boundary tests plus namespace, package, and local-state cutover;
2. full-repository Ruff cleanup and CI gate upgrade;
3. README, MIT license, deterministic demo, and static evidence pages;
4. live Groq evidence recorded against a stable source commit;
5. any evidence-only update followed by a fresh deterministic full gate.

The machine-readable live report stores the exact tested source commit. The
commit that adds the report necessarily has a later hash; it must not pretend
the report was measured on itself. No production code changes are allowed
between the measured source commit and the evidence commit. After adding
evidence, all deterministic gates run again on final HEAD.

After local verification:

1. push `codex/praxis-hard-cutover`;
2. open a Draft PR;
3. wait for GitHub CI;
4. fix failures only on the branch and rerun local gates;
5. mark ready and merge only when CI and required local evidence are green;
6. rename the GitHub repository to `praxis-agent-runtime`;
7. update the shared local `origin` URL;
8. update GitHub description and topics;
9. verify the new URL, old URL redirect, default branch, Actions, and clean
   remote state.

The user explicitly authorized merge and repository rename after verification.
The dirty original `main` worktree remains untouched and is reported as such at
handoff.

## Failure handling and stop conditions

- A deterministic test, safety check, lint, type check, import contract, build,
  or package smoke failure blocks push/merge until fixed.
- A live provider/configuration failure is recorded as `inconclusive` and does
  not become a quality score.
- Missing real-model evidence blocks the repository-rename closeout unless the
  user later relaxes that requirement.
- A GitHub rename failure does not require reverting the merged code; the old
  repository remains the recovery point and the rename is retried separately.
- No command may reset, clean, or overwrite the user's dirty original `main`
  worktree.

## Acceptance criteria

The work is complete only when all of the following are true:

- public branding consistently says Praxis and uses the approved tagline;
- GitHub repository slug is `praxis-agent-runtime`;
- distribution metadata is `praxis-agent-runtime` without a PyPI upload;
- `agent_runtime` is the actual runtime owner and `rag/agent` does not exist;
- active code, tests, scripts, and README contain no `rag.agent` path;
- the only Agent-to-RAG dependency is the explicit lazy knowledge adapter;
- Agent state uses `.praxis`; RAG state remains `.rag`; no old data was deleted;
- the MIT license exists with the approved copyright line;
- README contains the clearly labelled deterministic GIF;
- the one-page benchmark report and redacted Groq real-run record exist;
- the 30-task suite is described honestly without a current release-pass claim;
- full Ruff reports zero errors;
- full mypy, pytest, import-linter, build, installed CLI smoke, deterministic
  delivery checks, manifest validation, and macOS safety checks pass;
- the Groq 5-case x 3-trial gate has a current PASS, or the work remains open;
- the PR is merged and GitHub CI is green;
- the new repository URL, old redirect, origin URL, description, and topics are
  verified;
- the original dirty `main` worktree remains unmodified.
