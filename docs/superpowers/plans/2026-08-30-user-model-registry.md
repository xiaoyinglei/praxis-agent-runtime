# User Model Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe user-level model registration control plane and make every new Harness Turn resolve an authenticated, immutable model definition that survives later catalog edits.

**Architecture:** Keep `ModelCatalog` as an immutable runtime view and `ModelControlPlane` as the selection/policy facade. Add focused modules for crash-safe config I/O, the writable user registry, endpoint probing, and the user-owned binding trust domain; adapt the Harness binding protocol so thread/Turn identity exists before a model binding is signed and durably inserted.

**Tech Stack:** Python 3.12, Pydantic v2, Typer, PyYAML, `fcntl`, `hashlib`/`hmac`/`secrets`, SQLite RolloutStore, pytest, Ruff, mypy, uv/hatch.

---

## File map

### New production modules

- `agent_runtime/model_config_io.py` — exact-path validation, adjacent locking,
  content fingerprints, atomic no-replace/replace operations, and
  post-`os.replace` outcome reconciliation shared by registry, session, trust,
  and archive files.
- `agent_runtime/model_registry.py` — strict user-registry schema, path policy,
  compare-and-swap editor, built-in/user composition, alias origins, and
  normalized update semantics.
- `agent_runtime/model_definition.py` — canonical execution definition,
  effective generation/stage-budget capture, and content-derived definition
  revision shared by catalog and Turn trust code.
- `agent_runtime/model_trust.py` — explicit trust-domain initialization,
  content-addressed definition archive, authenticated per-Turn binding envelope,
  and verification over `model_definition` canonical bytes.
- `agent_runtime/model_probe.py` — provider/gateway probe service and structured
  probe evidence; no persistence or CLI rendering.

### Existing production modules

- `agent_runtime/core/llm_config.py` — retain the existing internal provider
  schema but replace arbitrary user defaults at the writable boundary with the
  typed normalized definition.
- `agent_runtime/core/llm_registry.py` — expose built-in/override parsing and
  construct `ResolvedModel` from an explicit frozen definition without alias
  lookup.
- `agent_runtime/models.py` — catalog origins, versioned session selection,
  catalog-independent binding policy, control-plane freeze/resolve methods.
- `agent_runtime/runtime/builder.py` — assemble the effective catalog, registry
  path, session store, trust domain, archive, and probe dependencies.
- `agent_runtime/agent.py` — pass the actual workspace and selection requester
  into control-plane construction, and resume directly from the frozen binding
  without reconstructing through a possibly removed alias.
- `agent_runtime/harness/protocol.py` — make binding capture identity-aware.
- `agent_runtime/harness/thread_manager.py` — allocate Turn ID before binding
  capture and pass the same ID to durable start.
- `agent_runtime/harness/session.py` — accept the preallocated Turn ID.
- `agent_runtime/harness/composition.py` — combine static tool/knowledge state
  with the identity-aware model binding without regenerating it during resume.
- `agent_runtime/harness/model_adapter.py` — replace whole-catalog revision
  checks with authenticated schema-v2 binding resolution; fail closed for
  incomplete schema-v1 provider resume.
- `agent_runtime/cli.py` — `model show/add/update/remove/probe/trust` commands;
  parsing/rendering only.
- `agent_runtime/workspace.py` — keep project session/checkpoint constants;
  expose no provider configuration path under the workspace.
- `README.md`, `docs/RUNBOOK.md` — integration requirements, registry, trust,
  probe, switching, recovery, and restart semantics only.

### New and expanded tests

- Create `tests/agent/test_model_config_io.py`.
- Create `tests/agent/test_user_model_registry.py`.
- Create `tests/agent/test_model_trust.py`.
- Create `tests/agent/test_model_probe.py`.
- Modify `tests/agent/test_model_control_plane.py`.
- Modify `tests/agent/test_cli_wiring.py`.
- Modify `tests/agent/harness/test_thread_manager.py`.
- Modify `tests/agent/harness/test_model_adapter.py`.
- Modify every small fake `snapshot()` implementation identified by
  `rg -n 'def snapshot\(self' tests/agent`.
- Modify `tests/repo/test_readme_presentation.py`.
- Modify `tests/agent/test_package_distribution.py` if the packaged catalog
  load path needs explicit registry isolation.

## Task 1: Crash-safe configuration I/O

**Files:**

- Create: `agent_runtime/model_config_io.py`
- Create: `tests/agent/test_model_config_io.py`

- [ ] **Step 1: Write RED tests for path containment and locking**

Cover absolute external paths, relative paths, direct workspace paths, paths
outside a nested workspace but still inside its Git worktree, symlinks resolving
into either boundary, and two processes contending on the same adjacent lock.
Tests must pass explicit workspace and worktree roots; production code must not
infer trust from the process CWD alone.

```python
def test_registry_path_rejects_symlink_into_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    external = tmp_path / "config"
    workspace.mkdir()
    external.mkdir()
    (workspace / "models.yaml").write_text("version: 1\nrevision: 0\nmodels: {}\n")
    link = external / "models.yaml"
    link.symlink_to(workspace / "models.yaml")

    with pytest.raises(UntrustedConfigPathError):
        validate_user_config_path(link, workspace=workspace, worktree=workspace)
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/agent/test_model_config_io.py`

Expected: import failure because `agent_runtime.model_config_io` does not exist.

- [ ] **Step 3: Implement the focused low-level API**

Implement only these responsibilities:

```python
@dataclass(frozen=True, slots=True)
class FileVersion:
    revision: int
    fingerprint: str

class ConfigVersionConflict(RuntimeError): ...
class CommitOutcomeUnknown(RuntimeError): ...
class UntrustedConfigPathError(ValueError): ...

def discover_git_worktree(workspace: Path) -> Path:
    """Return the enclosing Git worktree, or the workspace itself when non-Git."""
    ...
def validate_user_config_path(path: Path, *, workspace: Path, worktree: Path) -> Path: ...
def file_fingerprint(payload: bytes) -> str: ...

@contextmanager
def exclusive_config_lock(path: Path) -> Iterator[None]: ...

def atomic_replace_bytes(
    path: Path,
    payload: bytes,
    *,
    intended_fingerprint: str,
) -> None: ...

def atomic_install_bytes(path: Path, payload: bytes) -> Literal["created", "exists"]: ...
```

`discover_git_worktree` performs a read-only Git-root lookup and treats
"not a Git repository" as a normal non-Git workspace, returning the resolved
workspace root. Other discovery errors remain explicit.

Use adjacent temporary files opened with user-only permissions. Flush and
`fsync` the temporary file, make `os.replace` the visibility commit point, then
`fsync` the directory. When directory sync fails after visibility, re-read and
reconcile exact intended bytes; otherwise raise `CommitOutcomeUnknown`. Do not
attempt rollback after replacement. Atomic no-replace installation must let one
concurrent initializer win without overwriting a valid existing file.

- [ ] **Step 4: Add crash/concurrency RED tests, then GREEN**

Inject failures before replace and after replace. Prove pre-commit failure keeps
old bytes, post-commit failure reports unknown/reconciled state, and concurrent
no-replace creates one payload.

Run: `uv run pytest -q tests/agent/test_model_config_io.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_runtime/model_config_io.py tests/agent/test_model_config_io.py
git commit -m "feat(models): add crash-safe config writes"
```

## Task 2: Strict user registry and CAS editor

**Files:**

- Create: `agent_runtime/model_registry.py`
- Create: `tests/agent/test_user_model_registry.py`
- Modify: `agent_runtime/core/llm_config.py`

- [ ] **Step 1: Write schema RED tests**

Test the exact alias regex/reserved words, supported provider adapters, absolute
HTTP(S) base URLs, context/output constraints, credential environment names,
shell-free launch argv, strict unknown-field rejection, and typed defaults:

```python
class ModelGenerationDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    parallel_tool_calls: bool | None = None
    seed: int | None = Field(default=None, ge=-(2**63), le=2**63 - 1)
```

The normalized persisted mapping omits absent keys; explicit YAML `null` in
`defaults` is rejected before normalization.

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/agent/test_user_model_registry.py -k schema`

Expected: failure because registry schemas are missing.

- [ ] **Step 3: Implement registry document and user-entry models**

Implement strict Pydantic models:

```python
class UserModelDefinition(BaseModel): ...
class UserModelRegistryDocument(BaseModel):
    version: Literal[1] = 1
    revision: int = Field(ge=0)
    models: dict[str, UserModelDefinition] = Field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class RegistrySnapshot:
    document: UserModelRegistryDocument
    fingerprint: str
```

This layer validates and persists user declarations only. It does not publish
the runtime `definition_revision`, because that digest also covers effective
generation settings and stage budgets introduced during catalog composition in
Task 3. Identical-update detection compares normalized user declarations.

- [ ] **Step 4: Write editor RED tests**

Test add collision, built-in collision, patch update, complete `--from`
replacement, every allowed `--unset` path, invalid unset, identical no-op,
remove, retained empty document, stale revision/fingerprint, two-process lost
update prevention, active whole-catalog override rejection, and reconciliation
of the exact intended post-state after a durability-unknown result. The store
must return a mutation receipt containing base version, intended revision, and
intended fingerprint; `reconcile(receipt)` recognizes only that exact state.

- [ ] **Step 5: Implement `UserModelRegistryStore`**

Expose domain methods, not YAML mutation primitives:

```python
class UserModelRegistryStore:
    def read(self) -> RegistrySnapshot: ...
    def add(self, alias: str, definition: UserModelDefinition, *, expected: FileVersion) -> RegistryMutationResult: ...
    def update(self, alias: str, mutation: ModelDefinitionPatch, *, expected: FileVersion) -> RegistryMutationResult: ...
    def remove(self, alias: str, *, expected: FileVersion) -> RegistryMutationResult: ...
    def reconcile(self, receipt: MutationReceipt) -> RegistrySnapshot: ...
```

`RegistryMutationResult` contains the committed snapshot and its
`MutationReceipt`. If durability cannot be acknowledged, `CommitOutcomeUnknown`
carries that same receipt so CLI/service callers can report it and call
`reconcile(receipt)` without reconstructing intent.

All methods re-read under the lock, validate the complete effective result
against built-in aliases, increment once, retain an empty document, and use the
Task 1 commit semantics. The store never probes providers or changes session
state.

- [ ] **Step 6: Run GREEN and static checks**

Run:

```bash
uv run pytest -q tests/agent/test_user_model_registry.py
uv run ruff check agent_runtime/model_registry.py agent_runtime/model_config_io.py tests/agent/test_user_model_registry.py
uv run mypy agent_runtime/model_registry.py agent_runtime/model_config_io.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add agent_runtime/core/llm_config.py agent_runtime/model_registry.py tests/agent/test_user_model_registry.py
git commit -m "feat(models): add user registry editor"
```

## Task 3: Effective catalog layering and provenance

**Files:**

- Modify: `agent_runtime/core/llm_registry.py`
- Create: `agent_runtime/model_definition.py`
- Modify: `agent_runtime/models.py`
- Modify: `agent_runtime/runtime/builder.py`
- Modify: `tests/agent/test_model_control_plane.py`
- Modify: `tests/agent/test_llm_registry.py`
- Modify: `tests/agent/test_package_distribution.py`

- [ ] **Step 1: Write RED canonical-definition and layering tests**

First prove canonical JSON includes `None`, sorts keys, rejects NaN, and captures
the selected internal model plus effective generation settings and all stage
budgets. Fixed digest fixtures must change when any request-affecting field
changes and remain unchanged after unrelated alias edits.

Then prove normal load is built-ins plus user additions, origins are visible,
built-ins cannot be shadowed, malformed user files fail loudly, and existing
`RAG_AGENT_MODELS_PATH`/`RAG_AGENT_MODELS` keep whole-catalog replacement
semantics while disabling registry mutation.

```python
assert catalog.origin("groq_gpt_oss_120b") == "builtin"
assert catalog.origin("my_qwen") == "user"
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/agent/test_model_control_plane.py tests/agent/test_llm_registry.py -k 'definition or catalog or registry or override'`

Expected: user aliases are absent and origin API is missing.

- [ ] **Step 3: Implement immutable composition**

Implement `ModelExecutionDefinition` and its exact canonical digest in
`agent_runtime/model_definition.py`. Add a loader that returns one
`AgentModelsConfig` plus per-alias origin and the digest of the complete
effective definition. Do not mutate an already-created `ModelCatalog`; a newly
built control plane observes the latest committed registry. Keep default,
fallback, generation, and stage budgets owned by built-in/override config.

- [ ] **Step 4: Verify package isolation**

Ensure tests use `PRAXIS_MODEL_REGISTRY_PATH` pointed outside the test workspace
so the developer's real user registry cannot affect deterministic tests or an
installed wheel smoke.

Run:

```bash
uv run pytest -q tests/agent/test_model_control_plane.py tests/agent/test_llm_registry.py tests/agent/test_package_distribution.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_runtime/core/llm_registry.py agent_runtime/model_definition.py agent_runtime/models.py agent_runtime/runtime/builder.py tests/agent/test_model_control_plane.py tests/agent/test_llm_registry.py tests/agent/test_package_distribution.py
git commit -m "feat(models): compose effective model catalog"
```

## Task 4: Versioned session selection with requester safety

**Files:**

- Modify: `agent_runtime/agent.py`
- Modify: `agent_runtime/cli.py`
- Modify: `agent_runtime/models.py`
- Modify: `tests/agent/test_model_control_plane.py`
- Modify: `tests/agent/test_cli_wiring.py`

- [ ] **Step 1: Write session RED tests**

Cover schema migration from `{"current_model_id": ...}`, revision/fingerprint
CAS, concurrent switch conflict, atomic stale-alias repair, preservation of a
newer valid selection, one bounded retry for a newer invalid selection, and the
requester rule: project-persisted aliases always reload as `user`.

Add integration cases proving CLI `--model` and interactive `/model` freeze a
`user` selection, `Agent(model=...)` defaults to `system`, and an explicit
internal Agent switch freezes `agent`. The public `Agent` constructor gains a
private/internal selection-requester parameter used by CLI construction rather
than guessing from the same `model` argument.

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/agent/test_model_control_plane.py -k session`

Expected: current unversioned direct write behavior fails the new assertions.

- [ ] **Step 3: Implement `ModelSessionState` and store semantics**

Keep `current_model_id`, in-memory `selection_requester`, file revision, and
fingerprint. Reuse Task 1 locking/commit behavior. `switch_model()` policy-checks
first, then CAS-persists, then changes the in-memory selection; failure retains
the previous selection. Loading a legacy file assigns requester `user` and
upgrades only on the next successful write.

Thread the requester through `_create_agent_facade`, CLI `run/chat --model`,
`Agent._get_model_control_plane()`, and model-session reload. Do not persist a
privileged `agent`/`system` requester in project state; on a new process it
downgrades to `user`.

- [ ] **Step 4: Run GREEN**

Run:

```bash
uv run pytest -q tests/agent/test_model_control_plane.py tests/agent/test_cli_wiring.py
uv run mypy agent_runtime/models.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_runtime/agent.py agent_runtime/cli.py agent_runtime/models.py tests/agent/test_model_control_plane.py tests/agent/test_cli_wiring.py
git commit -m "feat(models): make session selection conflict-safe"
```

## Task 5: Explicit trust domain and definition archive

**Files:**

- Create: `agent_runtime/model_trust.py`
- Read/Reuse: `agent_runtime/model_definition.py`
- Create: `tests/agent/test_model_trust.py`

- [ ] **Step 1: Write trust initialization RED tests**

Cover explicit-only initialization, status redaction, `0700` parent and `0600`
file modes, strict JSON, key-ID verification, symlink/unsafe-mode rejection,
concurrent first use, crash before install, post-install outcome reconciliation,
and refusal to auto-create when a new Turn asks for a missing domain.

In the same RED batch, cover the archive trust anchor: concurrent identical
installs converge; a conflicting/malformed existing digest path fails without
replacement; canonical bytes and filename digest are both verified; files are
mode `0600`; workspace/worktree-contained or symlinked paths are rejected; and
a missing digest fails before provider I/O.

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/agent/test_model_trust.py`

Expected: import failure because trust module does not exist.

- [ ] **Step 3: Implement exact trust interfaces**

```python
@dataclass(frozen=True, slots=True)
class TrustDomainStatus:
    trust_domain_id: str
    signing_key_id: str

class ModelBindingTrustDomain:
    def initialize(self) -> TrustDomainStatus: ...
    def status(self) -> TrustDomainStatus: ...
    def sign(self, association: Mapping[str, JsonValue]) -> str: ...
    def verify(self, association: Mapping[str, JsonValue], signature: str) -> None: ...

class TrustedModelDefinitionArchive:
    def ensure(self, definition: ModelExecutionDefinition) -> str: ...
    def load(self, definition_revision: str) -> ModelExecutionDefinition: ...
```

Use HMAC-SHA-256 and `hmac.compare_digest`. Never log key bytes, HMAC, resolved
credentials, or the base64 key field. The archive verifies filename digest and
canonical contents and never replaces a conflicting existing digest path.

- [ ] **Step 4: Write binding-substitution RED tests, then implement**

The signed association must cover authentication schema, trust-domain/key IDs,
thread ID, Turn ID, selection requester, and the complete binding envelope.
Changing any field, swapping another valid archived definition, or copying a
binding across Turns must fail.

- [ ] **Step 5: Run GREEN and commit**

```bash
uv run pytest -q tests/agent/test_model_trust.py
uv run ruff check agent_runtime/model_trust.py tests/agent/test_model_trust.py
uv run mypy agent_runtime/model_trust.py
git add agent_runtime/model_trust.py tests/agent/test_model_trust.py
git commit -m "feat(models): authenticate frozen model definitions"
```

## Task 6: Frozen definition resolution and policy

**Files:**

- Modify: `agent_runtime/core/llm_registry.py`
- Modify: `agent_runtime/agent.py`
- Modify: `agent_runtime/models.py`
- Modify: `agent_runtime/runtime/builder.py`
- Modify: `tests/agent/test_model_control_plane.py`
- Modify: `tests/agent/test_llm_registry.py`

- [ ] **Step 1: Write RED definition-resolution tests**

Build a `ModelExecutionDefinition` with tokenizer, context/request/output
limits, timeout, typed defaults, capability flags, runtime argv, generation,
and all stage budgets. Resolve it without an alias lookup and assert the real
`ResolvedModel`/gateway uses those exact fields.

- [ ] **Step 2: Write RED policy tests**

Test `review_binding()` independently of the current catalog for all signed
requester domains, allowed provider kinds, remote host restrictions, local
launch permission, endpoint/location coherence, and credential environment
name validation. Rejection must happen before local launch, secret lookup, or
provider construction.

- [ ] **Step 3: Run RED**

Run: `uv run pytest -q tests/agent/test_model_control_plane.py tests/agent/test_llm_registry.py -k 'binding or definition or policy'`

- [ ] **Step 4: Implement freeze and resolve services**

`ModelControlPlane.freeze_model_binding(thread_id, turn_id)` must:

1. obtain current alias and in-memory requester;
2. normalize the complete execution definition;
3. run catalog-independent policy;
4. require an initialized trust domain;
5. ensure the definition archive entry;
6. build the exact binding envelope/digests;
7. HMAC-sign the Turn/thread association;
8. return the JSON-safe model-binding fields.

`resolve_frozen_binding()` must verify HMAC, archive, digest, policy, current
credential reference, and local readiness in that order, then call
`ModelRegistry.resolve_definition()`.

`build_model_control_plane()` and `Agent._get_model_control_plane()` must pass
the Agent's resolved workspace plus its Git worktree root into registry,
archive, and trust path validation. They must not substitute `Path.cwd()` when
`Agent(workspace_path=...)` points elsewhere.

- [ ] **Step 5: Run GREEN and commit**

```bash
uv run pytest -q tests/agent/test_model_control_plane.py tests/agent/test_llm_registry.py tests/agent/test_model_trust.py
uv run mypy agent_runtime/core/llm_registry.py agent_runtime/agent.py agent_runtime/models.py agent_runtime/runtime/builder.py
git add agent_runtime/core/llm_registry.py agent_runtime/agent.py agent_runtime/models.py agent_runtime/runtime/builder.py tests/agent/test_model_control_plane.py tests/agent/test_llm_registry.py
git commit -m "feat(models): resolve authenticated Turn bindings"
```

## Task 7: Bind real Thread/Turn identity before durable start

**Files:**

- Modify: `agent_runtime/harness/protocol.py`
- Modify: `agent_runtime/harness/thread_manager.py`
- Modify: `agent_runtime/harness/session.py`
- Modify: `agent_runtime/harness/composition.py`
- Modify: `agent_runtime/harness/model_adapter.py`
- Modify: `agent_runtime/agent.py`
- Modify: `agent_runtime/cli.py`
- Modify: `tests/agent/harness/test_thread_manager.py`
- Modify: `tests/agent/harness/test_model_adapter.py`
- Modify: all test fake `snapshot()` implementations returned by
  `rg -l 'def snapshot\(self' tests/agent`

- [ ] **Step 1: Write RED identity tests**

Assert that `ThreadManager` allocates one Turn ID, calls
`binding_provider.snapshot(thread_id=..., turn_id=...)`, and passes exactly the
same ID into `Session.run()`/`RolloutStore.start_turn()`. Cover ordinary,
follow-up/fork, and child Threads.

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/agent/harness/test_thread_manager.py`

Expected: current zero-argument snapshot and store-owned Turn allocation fail.

- [ ] **Step 3: Change the binding protocol and lifecycle**

```python
class BindingProvider(Protocol):
    def snapshot(self, *, thread_id: str, turn_id: str) -> Mapping[str, Any]: ...
```

Allocate `turn_<uuid>` in `ThreadManager` after final thread selection but
before snapshot. Extend `Session.run(..., turn_id: str)` and pass that value to
`start_turn`. `_CompositionBindingProvider` forwards IDs only to the model and
combines static tool/knowledge policy through a private static snapshot; resume
validation must never generate a new signed model binding.

- [ ] **Step 4: Write model-adapter RED tests**

Prove a schema-v2 binding is inserted before provider dispatch, unrelated alias
add/remove does not affect resume, updating/removing the selected alias keeps
old Turn resolution, project SQLite substitution fails HMAC, current policy may
block but not substitute, and replay performs no catalog/provider/trust lookup.

Add a schema-v1 regression proving history replay works while provider resume
fails with `legacy model binding is incomplete and cannot be resumed safely`.

Add a public `Agent.aresume()` and CLI `agent resume` regression where the user
alias has been removed from the current catalog. Resume must construct the
runtime from stored workspace/knowledge data without passing the frozen alias
through ordinary catalog initialization; the adapter resolves the authenticated
binding directly. CLI inspection/replay must also avoid alias lookup.

- [ ] **Step 5: Replace catalog revision gating**

`ControlPlaneHarnessModel.snapshot()` delegates to
`freeze_model_binding(thread_id, turn_id)`. `_resolve_frozen()` delegates to
`resolve_frozen_binding()` and verifies request IDs match the signed binding.
Remove `_catalog_revision` as a resume gate; retain whole-catalog revision only
if useful as non-authoritative audit metadata.

Refactor `_harness_agent_for_turn()` and CLI continuation into an explicit
frozen-resume construction path. Ordinary new/follow-up Turns still require a
current catalog alias; resume of the same nonterminal Turn does not.

- [ ] **Step 6: Update test fakes mechanically and run GREEN**

Each fake accepts keyword-only IDs and may ignore them only when the test does
not exercise authenticated model behavior.

Run:

```bash
uv run pytest -q tests/agent/harness/test_thread_manager.py tests/agent/harness/test_model_adapter.py tests/agent/harness
uv run mypy agent_runtime/harness
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add agent_runtime/harness agent_runtime/agent.py agent_runtime/cli.py tests/agent/harness tests/agent/test_canonical_streaming_protocol.py tests/agent/test_update_plan_surfaces.py tests/agent/test_agent_cli_resume.py
git commit -m "feat(harness): freeze authenticated model per Turn"
```

## Task 8: Provider capability probe service

**Files:**

- Create: `agent_runtime/model_probe.py`
- Create: `tests/agent/test_model_probe.py`
- Modify: `agent_runtime/core/llm_registry.py`

- [ ] **Step 1: Write probe RED tests using a real fake HTTP endpoint**

Use an in-process FastAPI/HTTP transport fixture rather than mocking the probe
service. Cover connectivity/model identity, at least one text delta plus
authoritative completion, a harmless forced tool-call schema without executing
the tool, structured output, auth failure, timeout, malformed stream, wrong
model, missing tool call, cancellation, and redacted errors.

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/agent/test_model_probe.py`

Expected: import failure because probe service does not exist.

- [ ] **Step 3: Implement probe levels and evidence**

```python
class ProbeLevel(StrEnum):
    CONNECTIVITY = "connectivity"
    STREAM = "stream"
    FULL = "full"

@dataclass(frozen=True, slots=True)
class ModelProbeEvidence:
    level: ProbeLevel
    connectivity_ok: bool
    text_delta_count: int
    completion_ok: bool
    tool_call_ok: bool | None
    structured_output_ok: bool | None

class ModelProbe:
    async def run(self, definition: ModelExecutionDefinition, *, level: ProbeLevel) -> ModelProbeEvidence: ...
```

Reuse the existing provider/gateway serialization path. Do not register, select,
or execute a returned tool call. Keep probe evidence ephemeral.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest -q tests/agent/test_model_probe.py
uv run ruff check agent_runtime/model_probe.py tests/agent/test_model_probe.py
uv run mypy agent_runtime/model_probe.py
git add agent_runtime/model_probe.py tests/agent/test_model_probe.py agent_runtime/core/llm_registry.py
git commit -m "feat(models): probe provider capabilities"
```

## Task 9: Model registration CLI

**Files:**

- Modify: `agent_runtime/cli.py`
- Modify: `agent_runtime/runtime/builder.py`
- Modify: `tests/agent/test_cli_wiring.py`
- Modify: `tests/agent/test_model_control_plane.py`

- [ ] **Step 1: Write CLI RED tests**

Cover:

- `model list --source` and `model show` perform no provider request;
- `model trust init/status` redact key material and are idempotent;
- `model add` flag and `--from` forms probe before commit by default;
- `--skip-probe` still validates and labels output unverified;
- probe failure/cancellation writes nothing;
- update patch, `--unset`, replacement, and normalized no-op;
- remove rejects built-ins and the addressed session's current alias;
- invalid alias/config/version/path/override/concurrency errors are actionable;
- no traceback or secret value reaches normal CLI output.

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/agent/test_cli_wiring.py -k model`

- [ ] **Step 3: Implement thin commands**

Add Typer subcommands and argument models, then delegate to registry, probe,
trust, catalog, and session services. The CLI must not parse YAML after the
single-definition input boundary, merge dictionaries, construct HTTP clients,
or write files directly.

Default add/update order:

```text
parse -> strict normalize -> expected registry snapshot -> full probe
      -> editor CAS commit -> print alias/source/definition/file revisions
```

No-op update skips probe. `--skip-probe` is explicit in output. Remove requires
a prior separate switch when current.

- [ ] **Step 4: Run GREEN and command help smoke**

```bash
uv run pytest -q tests/agent/test_cli_wiring.py tests/agent/test_model_control_plane.py
uv run agent model --help
uv run agent model add --help
uv run agent model update --help
uv run agent model trust --help
```

Expected: tests pass and help lists the exact public ACI.

- [ ] **Step 5: Commit**

```bash
git add agent_runtime/cli.py agent_runtime/runtime/builder.py tests/agent/test_cli_wiring.py tests/agent/test_model_control_plane.py
git commit -m "feat(cli): manage user model registrations"
```

## Task 10: README and runbook

**Files:**

- Modify: `README.md`
- Modify: `docs/RUNBOOK.md`
- Modify: `tests/repo/test_readme_presentation.py`

- [ ] **Step 1: Write documentation RED tests**

Require sections/commands for model interface requirements, trust init/status,
user registry path, local/cloud add examples with only credential environment
references, probe, list/show/switch/update/remove, built-in/user/project/Turn
ownership, process restart after edits, stale-session repair, override mode,
commit conflict/outcome unknown, archive/key backup, and legacy Turn limits.

Reject model-family marketing/comparisons and any example containing an API-key
literal.

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/repo/test_readme_presentation.py`

- [ ] **Step 3: Rewrite only the model integration sections**

Do not re-document Qwen, Kimi, DeepSeek, or model benchmarks. Explain what an
endpoint must support, how to register it, how declared capability probes map
to full Agent functionality, and how immutable per-Turn binding differs from
mutable session selection.

- [ ] **Step 4: Run docs tests and command-copy smoke**

Run:

```bash
uv run pytest -q tests/repo/test_readme_presentation.py
uv run agent model --help
git diff --check
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/RUNBOOK.md tests/repo/test_readme_presentation.py
git commit -m "docs(models): document model integration control plane"
```

## Task 11: End-to-end lifecycle, packaging, and regression gates

**Files:**

- Modify only files required by failures directly caused by Tasks 1-10.
- Do not weaken existing streaming, replay, cancellation, tool, security, or
  package tests.

- [ ] **Step 1: Run focused model/Harness suites**

```bash
uv run pytest -q \
  tests/agent/test_model_config_io.py \
  tests/agent/test_user_model_registry.py \
  tests/agent/test_model_trust.py \
  tests/agent/test_model_probe.py \
  tests/agent/test_model_control_plane.py \
  tests/agent/test_llm_registry.py \
  tests/agent/test_cli_wiring.py \
  tests/agent/harness/test_thread_manager.py \
  tests/agent/harness/test_model_adapter.py
```

Expected: PASS.

- [ ] **Step 2: Run a real isolated CLI lifecycle**

Use `mktemp -d` outside the repository and set
`PRAXIS_MODEL_REGISTRY_PATH=<temp>/models.yaml`. Pass
`--session-path <temp>/session.json` to `agent model` commands and
`--model-session-path <temp>/session.json` to `agent run/chat`. Do not print
any secret.

1. `agent model trust init` and `status`.
2. Register a local OpenAI-compatible test endpoint with `--skip-probe`.
3. List/show/switch it and start a fake-provider Turn under definition v1.
4. Update the alias to definition v2 and prove the old Turn still resolves v1.
5. Switch back to a built-in alias, remove the user alias, and prove the old
   Turn still resolves its archived v1 definition.
6. Tamper the project SQLite binding and prove resume performs zero provider
   I/O.

- [ ] **Step 3: Run the complete repository gates**

```bash
uv run ruff check .
uv run mypy
uv run pytest -q
uv run lint-imports
uv build
git diff --check
git status --short
```

Expected: all commands pass; only intended files are changed.

- [ ] **Step 4: Test the built wheel**

Install the wheel into a temporary environment and repeat:

```text
agent --help
agent model --help
agent model trust status
agent model list --source
```

Use isolated registry/trust/session paths. Confirm bundled models remain
read-only and the wheel never depends on the source checkout's CWD.

- [ ] **Step 5: Review security and lifecycle evidence**

Audit the final diff against every design acceptance criterion. Explicitly
verify:

- no project-level endpoint/credential/launch configuration;
- no model secret in files, Turn binding, archive, logs, errors, or test output;
- no unkeyed digest is treated as authorization;
- no provider I/O precedes HMAC/archive/policy validation;
- no catalog edit invalidates schema-v2 resume;
- schema-v1 replay works and provider continuation fails closed;
- canonical Turn/Item/Delta/completion/replay behavior is unchanged.

- [ ] **Step 6: Commit any direct gate fixes**

```bash
git add <only-directly-required-files>
git commit -m "test(models): verify registration lifecycle"
```

Skip this commit when the tree is already clean.

- [ ] **Step 7: Proceed to branch completion**

Invoke `superpowers:verification-before-completion`, then
`superpowers:finishing-a-development-branch`. The user already requested final
integration into `main`; still verify the exact current checkout, branch,
worktrees, and remote state before merge/push/synchronization. Do not delete any
other branch without renewed exact-target authorization.
