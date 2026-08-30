# User Model Registry and Frozen Turn Binding Design

## Status

Approved direction on 2026-08-30: Praxis gains a writable user-level model
registry. Repository configuration remains read-only product defaults, and a
project may select a model alias but may not define a provider, endpoint,
credential reference, or launch command.

This design replaces manual edits to `configs/models.yaml` as the normal model
onboarding path. It also repairs the existing Turn binding rule that makes an
unrelated catalog edit invalidate an older resumable Turn.

## Problem

Praxis currently has a clean selection boundary but no configuration control
plane:

- `ModelCatalog` is an immutable query view;
- `ModelSessionState` stores the selected alias;
- `ModelPolicy` reviews switches;
- `ModelControlPlane` selects and resolves models;
- `ModelRegistry` loads one static catalog from an environment override or the
  bundled `configs/models.yaml`;
- the CLI supports only `list`, `current`, and `switch`.

Consequently, adding a model requires editing YAML or replacing the complete
catalog through an environment variable. Neither path supplies ownership,
atomic mutation, collision handling, capability probing, provenance, or safe
concurrent writes.

The current Harness snapshot has a second defect. It records a hash of the
entire effective catalog and rejects resume when that hash changes. Adding an
unrelated alias can therefore make an otherwise unchanged Turn unavailable.
The binding also omits request-affecting fields present only in the internal
model configuration, so it is not a complete frozen execution definition.

## Design influences

The design takes boundary patterns, not type names, from mature agent systems:

- Codex composes an effective configuration from owned layers, records layer
  origins and versions, restricts writes to the user layer, and uses an
  expected version to reject stale writes.
- Codex treats repository-controlled provider and endpoint settings as
  security-sensitive and does not allow project configuration to redirect
  model traffic or credentials.
- Claude Code validates a custom model before changing the active selection
  and keeps its settings schema backward compatible.

Praxis will not copy Codex's generic configuration service or build a provider
plugin framework. The product needs one bounded model-registry writer and one
immutable effective catalog.

References:

- <https://github.com/openai/codex/blob/main/codex-rs/config/src/loader/README.md>
- <https://github.com/openai/codex/blob/main/codex-rs/app-server/src/config_manager_service.rs>
- <https://github.com/openai/codex/blob/main/codex-rs/config/src/loader/mod.rs>
- <https://github.com/openai/codex/blob/main/codex-rs/model-provider-info/src/lib.rs>
- `/Users/leixiaoying/PycharmProjects/brath-claude-code/src/commands/model/model.tsx`
- `/Users/leixiaoying/PycharmProjects/brath-claude-code/src/utils/settings/types.ts`

## Goals

1. Register, inspect, validate, update, probe, and remove user models through a
   documented CLI instead of editing repository YAML.
2. Keep built-in models version-controlled and immutable to the registry CLI.
3. Keep provider endpoints, credential references, and local launch commands
   outside repository-controlled project configuration.
4. Compose one immutable runtime catalog from built-ins plus user models and
   retain the source of every alias.
5. Reject stale or partially valid registry writes without corrupting the last
   valid file.
6. Validate a model before committing it by default, while allowing an
   explicit unverified registration for an offline endpoint.
7. Freeze every request-affecting model setting into each new Turn without
   persisting secret values.
8. Let schema-v2 Turns resume under their original frozen model definition even
   when aliases are later added, updated, or removed.
9. Preserve the current selection, policy, provider gateway, and canonical
   streaming ownership boundaries.
10. Update README and operational documentation around integration
    requirements and switching, without adding model marketing or comparisons.

## Non-goals

- No project-level model/provider registry.
- No web UI, interactive wizard, catalog marketplace, or remote registry sync.
- No arbitrary provider plugin loading. The supported adapter kinds remain
  `openai_compatible`, `mlx`, and `ollama` until a separate provider design is
  approved.
- No plaintext API keys, bearer tokens, or expanded environment values in
  config files, logs, errors, Turn bindings, or history.
- No automatic capability inference from model names.
- No silent repair of malformed registry files.
- No live file watcher. A running Agent keeps its immutable catalog; a new
  process or explicit future reload feature observes registry changes.
- No changes to embedding or reranking model configuration in this scope. The
  writable registry onboards chat models used by the Agent Harness.

## Alternatives considered

### 1. Add CRUD directly to `ModelCatalog`

This is rejected. It mixes runtime lookup, persistence, validation, locking,
and CLI ownership. It would also encourage an in-place mutable catalog while a
Turn is executing.

### 2. Keep one mutable repository `configs/models.yaml`

This is rejected. Project files are version-controlled, may be untrusted, and
cannot safely own credential-routing or executable launch configuration. It
also creates merge conflicts and makes package defaults indistinguishable from
user state.

### 3. Layered user registry plus immutable effective catalog

This is selected. It adds one write owner while preserving the existing
`ModelCatalog` and `ModelControlPlane` roles. It supplies the required safety
and lifecycle guarantees without a general configuration framework.

## Ownership and components

```text
read-only bundled models ─┐
                           ├─> ModelCatalogLoader ─> immutable ModelCatalog
writable user registry ──┘                              ├─> ModelControlPlane
UserModelRegistryEditor ─> writable user registry       │
TrustedModelDefinitionArchive <─ canonical definition ──└─> frozen Turn binding
TurnBindingAuthenticator ── HMAC(turn identity + binding) ─┘
```

### `UserModelRegistryEditor`

This is the only component allowed to mutate the user registry. It owns:

- parsing and schema validation;
- alias and built-in collision checks;
- compare-and-swap revision checks;
- process-level file locking;
- same-directory temporary writes, `fsync`, and atomic `os.replace`;
- add, update, and remove semantics;
- returning the new registry revision and normalized entry.

It does not construct provider clients, change session selection, mutate an
active catalog, or execute probes.

### `ModelCatalogLoader`

This component loads the read-only built-in catalog and the user registry,
normalizes both to the existing internal model schema, rejects invalid layers,
and creates one immutable effective catalog. It retains origin metadata for
each alias: `builtin`, `user`, or `override`.

`ModelCatalog` remains the runtime query API. It may expose origin and entry
revision queries, but it does not acquire locks or write files.

### `ModelProbe`

This component takes a normalized candidate definition and uses the existing
provider/gateway implementation to verify it. It never changes catalog or
session state. `add` and `update` call it before the registry editor commits;
`probe` calls it without any write.

### `TrustedModelDefinitionArchive`

This user-owned, content-addressed archive is the trust anchor for resuming a
Turn. Before a new Turn becomes dispatchable, the normalized execution
definition is atomically installed at:

```text
${XDG_CONFIG_HOME:-~/.config}/praxis/model-definitions/<definition-digest>.json
```

When `PRAXIS_MODEL_REGISTRY_PATH` selects an isolated administrative/test
registry, its trusted archive is the sibling `model-definitions/` directory so
catalog and binding trust state cannot accidentally cross environments.

Archive files are immutable, created with user-only permissions, and contain
no resolved secret. The Turn stores the same definition for auditability, but
resume executes it only after its canonical bytes match the trusted archived
definition. A digest in project-local SQLite is therefore not treated as an
authenticator. Alias update/removal never deletes archived definitions. Archive
garbage collection is outside this scope because Praxis cannot prove that no
other workspace still references a definition.

### `TurnBindingAuthenticator`

The archive proves that a definition came from a trusted catalog, but it does
not by itself prove that this was the definition selected for a particular
Turn. Praxis therefore uses one explicit user trust-domain file in the same
trusted configuration root as `model-definitions/`, at
`${XDG_CONFIG_HOME:-~/.config}/praxis/binding-trust.json`. It contains schema
version, a random trust-domain ID, a 256-bit random HMAC key, and the derived
key ID. The parent directory is mode `0700`; the regular, non-symlink file is
mode `0600`. Key material and HMAC values are never logged, and verification
uses constant-time comparison.

```json
{"version":1,"trust_domain_id":"uuid", "signing_key_id":"sha256:...", "hmac_key_base64":"..."}
```

The stored key ID must equal the digest derived from the decoded 32-byte key;
extra or malformed fields fail strict validation.

Praxis never creates or regenerates this file as a side effect of starting or
resuming a Turn. The user initializes the trust domain explicitly once:

```text
agent model trust init
agent model trust status
```

`init` writes a user-only temporary file, flushes and `fsync`s it, then installs
it with an atomic no-replace operation. Concurrent first-use attempts converge
on the one winning valid domain; they never overwrite it. A crash before
installation leaves no final file. A failure after installation follows the
same outcome-unknown reconciliation rule as other config writes. An existing
valid domain makes `init` idempotently report its ID; an existing malformed,
symlinked, or permission-unsafe file fails closed and is never replaced.

If the trust file is absent, a new schema-v2 Turn refuses to start and prints
the initialization command. This is deliberate: the runtime cannot know
whether the absence means first use or key loss. Running `init` after key loss
is an explicit creation of a new trust domain, and the command warns that Turns
signed by an earlier domain will remain unverifiable unless that old trust file
is restored. There is no implicit recovery or overwrite path.

Before dispatch, the authenticator signs the canonical binding association,
including `thread_id`, `turn_id`, alias, selection requester, complete binding
envelope, and definition digest. The Turn stores a key identifier derived from
the public hash of the key and an HMAC-SHA-256 signature. Resume verifies the
signature before trusting any project-local binding field. Substituting a
different valid archived definition, alias, requester, Turn ID, or envelope
therefore fails authentication.

With an isolated `PRAXIS_MODEL_REGISTRY_PATH`, the trust file is an adjacent
`binding-trust.json`, keeping tests and administrative registries separate from
the default user trust domain. A missing or replaced trust file makes resume
fail closed with the expected domain and key identifiers. Key backup, rotation,
and multi-device transfer are outside this scope and are documented operational
constraints.

### `ModelControlPlane`

The existing control plane remains the sole owner of selection, policy review,
local readiness, and resolution. It receives a fully composed immutable
catalog and resolver. Registry editing does not become a method on this class.

### CLI

Typer commands parse input and render results. They call the loader, probe, or
editor and contain no merge, persistence, collision, or provider-construction
logic.

## Configuration layers and precedence

Normal startup uses two layers:

1. packaged/repository `configs/models.yaml`, read-only;
2. the user registry, additions only.

The default user path is:

```text
${XDG_CONFIG_HOME:-~/.config}/praxis/models.yaml
```

`PRAXIS_MODEL_REGISTRY_PATH` may select a different user-registry file for
testing or administration. Registry mutation commands do not load this value
from a project `.env`. The value must be an absolute path, and its resolved
path must be outside the active workspace and its Git worktree. The CLI rejects
symlinked or relative targets that resolve inside those boundaries and reports
the exact accepted target before writing. Tests use an isolated temporary
directory outside their fixture workspace. The same containment and symlink
checks apply to the XDG-derived default registry and trusted definition archive;
an `XDG_CONFIG_HOME` inside the workspace is rejected rather than trusted.

The existing `RAG_AGENT_MODELS_PATH` and `RAG_AGENT_MODELS` variables remain
backward-compatible whole-catalog overrides for tests and controlled
deployment. When either is active:

- runtime loading preserves its current replacement semantics;
- the effective origin is `override`;
- every chat entry must normalize through the same strict, secret-free
  `ModelExecutionDefinition` validation before it may enter a Turn or trusted
  archive;
- registry mutation commands fail with an actionable message instead of
  editing a user file that the current runtime would ignore.

User entries may not shadow built-in aliases. Duplicate aliases and malformed
layers fail loudly. Built-in default, fallback, generation defaults, and stage
budgets continue to come from the built-in or explicit whole-catalog override;
the user registry adds chat model definitions only. Under an explicit
whole-catalog override, stale session recovery uses that effective catalog's
default rather than assuming a packaged built-in exists.

Project-local state such as `.praxis/model_session.json` may store only the
selected alias. Project files cannot supply provider type, raw model identity,
base URL, credential environment name, headers, or launch commands.

### Session selection concurrency

The session file becomes a versioned selection record:

```json
{"version":1,"revision":7,"current_model_id":"my_qwen"}
```

The mutable in-memory selection also carries the policy domain that authorized
it:

- CLI `--model`, `/model`, and `agent model switch` use `user`;
- an Agent-requested switch uses `agent`;
- configured defaults, fallback, and SDK construction use `system` unless the
  caller explicitly supplies another existing requester value.

The requester is deliberately not trusted from the project-local session file.
Any alias restored from that file is reviewed as `user`, even if an Agent or
system selection originally wrote it. This may become more restrictive across
a process restart but cannot escalate project-controlled state into an `agent`
or `system` policy domain. The authenticated per-Turn binding, not the session
file, retains the exact requester for resume.

Session reads return the revision and content fingerprint. Switch and stale
selection repair use an adjacent exclusive lock, expected revision/fingerprint,
same-directory temporary write, `fsync`, and atomic replacement. A stale
switch fails with a conflict rather than overwriting a newer selection. Stale
repair re-reads after a conflict: it preserves a newer valid selection, or
retries the same effective-default repair at most once before returning a
conflict. Each new Turn records and authenticates its in-memory selection
requester; resume re-applies that signed requester domain and never substitutes
the current session's requester.

## User registry schema

The user file is deliberately flat so one model definition is independently
versionable and free from hidden provider inheritance:

```yaml
version: 1
revision: 4
models:
  my_qwen:
    provider: openai_compatible
    model: Qwen/Qwen3.5-9B
    protocol: openai_compatible
    location: local
    base_url: http://127.0.0.1:8080/v1
    api_key_env: null
    context_window_tokens: 262144
    request_context_tokens: 131072
    max_tokens: 4096
    timeout_seconds: 120
    supports_tools: true
    supports_structured_output: true
    defaults: {}
    runtime:
      health_url: http://127.0.0.1:8080/v1/models
      expected_model_contains: Qwen3.5-9B
      launch_command: []
      startup_timeout_seconds: 120
      poll_interval_seconds: 1
```

Schema rules:

- aliases are 1-64 characters and match
  `^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$`; `list`, `current`, `switch`,
  `use`, `add`, `update`, `probe`, `remove`, `show`, `trust`, and `default` are
  reserved;
- `version: 1` is strict: unknown top-level, model, runtime, and generation
  fields are rejected. A client that encounters a newer version performs no
  write, so it cannot destroy fields it does not understand;
- `api_key_env` is an environment variable name, never a secret value;
- `defaults` is not an arbitrary provider request dictionary. It accepts only
  the typed generation options supported by Praxis; authentication, headers,
  query parameters, URLs, cookies, and other transport fields are rejected at
  every nesting level;
- `base_url` must be absolute HTTP(S); loopback endpoints are local and remote
  endpoints are cloud unless `location` makes the same classification
  explicit;
- launch commands are argv arrays executed without a shell and are permitted
  only in the user layer;
- context and output budgets must be positive and internally consistent;
- declared capability flags are explicit and are checked by the selected probe
  level rather than inferred from a model name.

The version-1 `defaults` keys and ranges are exact:

- `temperature`: finite float in `[0.0, 2.0]`;
- `top_p`: finite float in `(0.0, 1.0]`;
- `parallel_tool_calls`: boolean;
- `seed`: signed 64-bit integer.

An omitted key uses the canonical runtime default. Null is not accepted in
`defaults`; `--unset defaults.<key>` removes the override.

The exact version-1 nullable paths accepted by `--unset` are
`tokenizer_model`, `provider_name`, `base_url`, `api_key_env`,
`request_context_tokens`, `input_cost_per_1m`, `output_cost_per_1m`,
`cache_read_cost_per_1m`, `cache_write_cost_per_1m`, `runtime.health_url`, and
`runtime.expected_model_contains`, plus the four `defaults.<key>` removals
above. Clearing `base_url` must still satisfy the selected adapter's structural
rules. Collections such as `runtime.launch_command` are cleared by setting an
explicit empty list, not by `--unset`. Any other unset path is rejected before
probing or writing.

The first implementation supports `version: 1` and produces a precise error
for unsupported future versions. Schema migrations are explicit functions,
not scattered conditionals in the loader. Backward-compatible evolution adds
optional typed fields under a new reader; forward-incompatible files remain
untouched rather than being partially preserved by an older writer.

The file-wide integer `revision` is used only for compare-and-swap. A model's
`definition_revision` is not a counter stored in YAML: it is the
`sha256:<lowercase-hex>` digest of that model's normalized execution
definition. Unrelated edits therefore change the file revision but not the
definition revision. An update whose normalized definition digest is unchanged
is reported as a no-op and increments neither revision.

## Atomic mutation protocol

Each mutation follows one transaction-like sequence:

1. Resolve the exact user-registry path and reject override mode.
2. Open an adjacent lock file and acquire an exclusive lock.
3. Read and validate the current registry under the lock.
4. Compare its revision and content fingerprint with the caller's expected
   version.
5. Apply one normalized mutation in memory.
6. Validate the complete resulting registry and effective catalog, including
   built-in collisions.
7. Increment `revision` once.
8. Write a temporary file in the destination directory, flush and `fsync` it,
   atomically replace the destination, and `fsync` the directory.
9. Release the lock and return the committed revision.

The candidate probe happens before step 2. The editor revalidates the candidate
inside the lock but does not hold a filesystem lock during network I/O. A
concurrent change therefore produces a version conflict, never a lost update.
The CLI reloads and asks the user to retry; it does not silently overwrite.

`os.replace` is the visibility commit point. A failure before it guarantees the
previous registry remains byte-for-byte visible. Successful directory `fsync`
is the durability acknowledgement. If replacement succeeds but directory
`fsync` fails, rollback is not attempted and the command returns
`commit_outcome_unknown` with the intended revision and fingerprint. While
still holding the lock it re-reads the path: an exact intended post-state may
retry directory `fsync` and report success only if that succeeds; a different
or unreadable state remains unknown. Retrying the same mutation reconciles an
exact intended revision/fingerprint as already committed, otherwise it reports
a version conflict. The session writer and trusted archive use the same
commit-point and reconciliation semantics.

The registry file is retained even when `models` becomes empty so its
monotonically increasing revision and fingerprint cannot be reset through
delete/recreate ABA. Registry mutation never deletes the file.

## Registration and probe ACI

The public command family becomes:

```text
agent model list [--source]
agent model show <alias>
agent model add <alias> [connection and capability options]
agent model add <alias> --from <one-model-yaml>
agent model update <alias> [options or --from]
agent model probe <alias> [--level connectivity|stream|full]
agent model remove <alias>
agent model current
agent model switch <alias>
agent model trust init
agent model trust status
```

The flag form covers the common OpenAI-compatible, MLX, and Ollama paths. The
`--from` form carries advanced typed fields such as a shell-free launch argv;
it imports exactly one model definition and never treats a complete project
catalog as writable input.

`add` rejects an existing alias. `update` requires a user-owned alias and
creates a new content-derived definition revision for future Turns. For
`update`, explicitly supplied flags patch known fields and omitted flags retain
their prior values. Repeated `--unset <nullable-field>` options clear only
schema-declared nullable fields. `update --from` is a complete replacement of
the known model definition. Because version 1 rejects unknown fields, neither
path has ambiguous preservation behavior. A normalized no-op does not probe,
write, or increment the registry revision.

`remove` affects only a user-owned alias. Built-ins cannot be updated or
removed. Removing the alias selected in the addressed session is rejected; the
user must complete a separate `agent model switch <alias>` first. This avoids
pretending that session and registry files can be committed atomically as one
transaction.

By default, `add` and `update` run the `full` probe appropriate to the declared
capabilities:

1. construct the existing provider adapter without exposing a secret;
2. verify endpoint/model reachability;
3. receive at least one real text stream delta and one authoritative completed
   response;
4. when `supports_tools` is true, obtain and validate one harmless forced tool
   call without executing it;
5. when `supports_structured_output` is true, validate one minimal structured
   result through the supported gateway contract.

`--skip-probe` is an explicit escape hatch for an offline endpoint. The command
prints that the alias is unverified; structural validation still runs. Probe
results are evidence returned to the caller, not mutable catalog truth, and
are not used to guess or rewrite capability flags.

Probe failure performs no registry write. Logs and errors may mention the
environment variable name but never its value.

Other stale session files cannot be enumerated. On later startup, an
unavailable persisted alias falls back to the effective catalog default with
an explicit diagnostic and atomically repairs that session selection instead
of making all model commands unusable.

## Frozen per-Turn model binding

New Turns persist a versioned `FrozenModelBinding`, not merely an alias and a
whole-catalog hash. The binding contains every request-affecting normalized
field needed by the current resolver:

- binding schema version, alias, alias source, selection requester, and
  definition revision;
- provider adapter kind and display name;
- raw provider model, protocol, location, and base URL;
- credential environment variable name, never its resolved value;
- tokenizer identity, context/request/output limits, timeout, and defaults;
- tool and structured-output capabilities;
- normalized local runtime readiness/launch configuration;
- effective generation settings and LLM stage budgets;
- a definition digest and a binding-envelope digest;
- Turn/thread association, trust-domain and signing-key identifiers, and HMAC
  signature;
- the policy revision observed when the Turn began, as audit metadata.

Canonicalization is exact. A typed `ModelExecutionDefinition` is dumped with
`model_dump(mode="json", by_alias=True, exclude_none=False)`, encoded with
`json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False,
allow_nan=False).encode("utf-8")`, and hashed as
`sha256:<lowercase-hex>`. This definition excludes alias, catalog origin,
registry revision, policy revision, and all digest fields; it includes every
request-affecting field listed above. That hash is `definition_revision` and
the archive filename.

The binding envelope contains `schema_version`, alias, origin,
`definition_revision`, the complete definition, and the policy revision. Its
`binding_digest` uses the same canonicalization over the entire envelope with
only `binding_digest` itself omitted. Audit metadata is therefore covered, but
does not change the independently addressable execution definition.

The authenticated association payload is exactly
`{"authentication_schema_version":1,"trust_domain_id":...,
"signing_key_id":...,"thread_id":...,"turn_id":...,
"selection_requester":...,"binding":<complete envelope including
binding_digest>}` under the same canonical JSON rules.
Its signature is
`hmac-sha256:<lowercase-hex>`. `signing_key_id` is the ordinary
`sha256:<lowercase-hex>` digest of the user HMAC key and selects the expected
local key; neither identifier is an authentication substitute for the HMAC.

The Thread manager allocates `thread_id` and `turn_id` before inserting the
Turn. Before the Turn can dispatch, Praxis atomically ensures that the
canonical definition exists in `TrustedModelDefinitionArchive`, signs the
allocated identity and binding, then inserts the complete authenticated
association with the Turn-start transaction. Archive creation without a later
Turn insert is a harmless unreachable content-addressed entry; a Turn is never
inserted with an unsigned or partly written binding. Resolution verifies
the HMAC first. `ModelRegistry.resolve_binding()` then loads the archive entry
by definition digest, verifies its filename hash, requires its canonical bytes
to match the authenticated Turn copy, and only then constructs the
provider/gateway. An ordinary binding digest detects accidental corruption but
is never treated as authorization. Resume does not look up the current alias
definition and does not require the current whole-catalog hash to match.

`ModelPolicy.review_binding()` is catalog-independent. It receives the signed
`selection_requester`, reviews the frozen alias against that requester's
existing allowlist, confirms the provider
adapter is supported, revalidates endpoint/location and credential-reference
invariants, and applies explicit policy restrictions for allowed provider
kinds, remote hosts, and local launch commands. It is called before external
I/O on initial dispatch and resume. Policy may reject a formerly allowed
binding, but it may not substitute a different model. Credential values are
resolved from the current trusted environment at dispatch time. A launch argv
is executed only from the hash-verified user archive and only when current
policy permits local launch; the project-local Turn copy is never executed by
itself.

Catalog edits affect only future Turns:

```text
Turn A begins with alias x definition v1 -> stores binding v1
user updates alias x to definition v2
Turn B begins with alias x definition v2 -> stores binding v2
resume Turn A                         -> resolves stored binding v1
```

Durable history replay never requires a model provider or current catalog. It
projects committed records only. Model availability is checked only when new
I/O is required.

### Legacy Turn compatibility

Existing schema-v1 Turns do not contain a complete binding and cannot be made
fully independent retroactively. Their legacy digest omits tokenizer, defaults,
request-context limits, timeouts, generation settings, and stage budgets, so
matching it cannot prove execution equivalence. Praxis therefore supports
history inspection and replay for these Turns but fails closed before any new
provider I/O with `legacy model binding is incomplete and cannot be resumed
safely`. It does not pretend that a current catalog entry is the original
definition. An explicit operator-attested migration can be designed later if
real paused legacy Turns require it; silent migration is forbidden.

New schema-v2 Turns always use the complete frozen binding path. Existing
durable rows are not rewritten in place.

## Data flow

Registration:

```text
CLI input
  -> parse one candidate
  -> schema and security validation
  -> ModelProbe using existing gateway
  -> UserModelRegistryEditor(expected revision)
  -> atomic user registry commit
  -> print committed alias, source, revision, and probe evidence
```

New Turn:

```text
built-ins + user registry
  -> immutable effective ModelCatalog
  -> ModelControlPlane selects alias
  -> policy review
  -> atomically ensure trusted content-addressed definition
  -> authenticate Turn/thread/requester/binding association
  -> persist normalized authenticated FrozenModelBinding with Turn
  -> resolve_binding through trusted archive
  -> canonical model Item stream and durable completion
```

Resume:

```text
durable Turn + FrozenModelBinding
  -> authenticate Turn/thread/requester/binding association
  -> binding digest and trusted archive match
  -> catalog-independent current policy review in signed requester domain
  -> current credential lookup
  -> resolve exact frozen definition
  -> continue original Turn
```

## Error handling

- Invalid candidate: report field-level errors; write nothing.
- Missing credential environment variable: identify the variable; expose no
  value; write nothing unless `--skip-probe` was explicit.
- Probe connection/auth/model/capability failure: classify the phase and write
  nothing.
- Built-in or user alias collision: report owner and reject.
- Unsupported provider adapter or schema version: fail closed.
- Registry parse failure: report exact file and validation errors; do not
  overwrite the file.
- Concurrent write: return expected and actual revisions; do not retry a
  mutation against changed state automatically.
- Failure before `os.replace`: retain the prior destination and clean only the
  exact temporary file created by this operation.
- Failure after `os.replace` but before durability acknowledgement: return
  `commit_outcome_unknown` and reconcile the intended revision/fingerprint;
  never claim rollback.
- Stale session alias: diagnose, fall back to the effective catalog default,
  and repair the session state atomically.
- Frozen binding, archive filename hash, or Turn/archive content mismatch:
  treat durable data as untrusted and perform no provider request or launch.
- Missing/mismatched signing key, invalid HMAC, requester substitution, or
  cross-Turn binding substitution: fail authentication before policy/provider
  work.
- Missing trust domain on new Turn creation: refuse to start and direct the
  user to explicit `agent model trust init`; never generate a key implicitly.
- Missing trusted archived definition: leave the Turn resumable-but-blocked and
  report the exact digest; never execute the project-local snapshot alone.
- Current policy rejection on resume: keep the Turn paused/blocked with the
  original binding; never switch providers as a fallback.

## Testing strategy

Testing follows real ownership boundaries rather than test-count claims.

### Registry unit and filesystem tests

- empty user layer composes with built-ins;
- user aliases load with `user` origin;
- built-in collisions, malformed URLs, raw secrets, invalid budgets, and
  unsupported schema versions fail;
- unknown version-1 fields and arbitrary transport/request defaults fail
  without changing the file;
- failed validation leaves existing bytes unchanged;
- expected-revision mismatch rejects a stale writer;
- an identical normalized update is a no-op, while patch, explicit unset, and
  complete replacement have distinct tested semantics;
- two processes cannot lose one another's updates;
- injected failures before temporary flush and before replacement retain the
  last valid file;
- injected directory-`fsync` failure after replacement reports an unknown
  outcome and reconciles exact intended bytes without unsafe rollback;
- an empty registry retains its monotonic revision and rejects an ABA stale
  writer;
- relative, symlinked-into-workspace, and project-local registry targets are
  rejected; only the exact accepted user path can be created or replaced;
- active whole-catalog overrides make mutation commands fail clearly.

### Trusted definition archive tests

- first use atomically creates the canonical digest-named file with user-only
  permissions;
- concurrent installs of identical canonical bytes converge safely;
- an existing digest path with different, malformed, or non-canonical content
  fails closed and is never replaced;
- archive paths obey the same workspace-containment rules as the registry;
- alias update/removal retains older definitions, while a deliberately missing
  archive blocks resume without provider I/O.

### Binding trust-domain tests

- explicit first initialization installs one valid mode-`0600` trust file and
  status never displays key material;
- concurrent initializers converge on one domain without overwrite;
- interruption before installation leaves no final trust file, while a
  post-install durability failure is reconciled as outcome-unknown;
- a malformed, symlinked, or permission-unsafe existing trust file fails
  closed;
- new Turn creation with a missing trust file refuses to auto-initialize;
- after an initialized domain is lost, new Turn creation and old-Turn resume
  stay blocked until explicit operator action, and a newly initialized domain
  cannot validate old signatures.

### Probe tests

- fake OpenAI-compatible endpoints prove connectivity, text deltas, completed
  response, tool-call validation, and structured-output validation;
- auth, identity, timeout, malformed stream, missing tool call, and invalid
  structured result are distinguished;
- failure and cancellation perform no registry commit;
- `--skip-probe` still runs structural validation and records no secret.

### Catalog and control-plane tests

- list/show expose alias source without making provider requests;
- invalid switch retains the previous selection and makes no provider request;
- adding an unrelated alias does not mutate an already built catalog;
- a new control plane observes a committed registry revision;
- stale session recovery selects and persists the effective catalog default;
- removing the addressed session's current alias is rejected until a separate
  successful switch has completed.

### Session selection tests

- session switch and stale repair use revision/fingerprint CAS under a lock;
- concurrent switches cannot silently overwrite one another;
- stale repair preserves a newer valid selection and has a bounded retry for a
  newer invalid selection;
- user, agent, and system in-memory selection paths bind their distinct
  requester domains; project-session reload is always `user`, while resume
  uses the authenticated original Turn requester.

### Turn lifecycle tests

- a new Turn installs a trusted definition and persists a complete binding
  before dispatch;
- binding payload never contains resolved credentials;
- project-local binding edits cannot redirect an endpoint or execute a launch
  command because resume requires an exact user-archive match;
- replacing one signed Turn binding with another valid archived definition,
  recomputing ordinary hashes, changing its requester, or copying it across
  Turn IDs fails HMAC authentication;
- missing/replaced signing keys and archive entries fail closed before external
  I/O;
- adding or removing an unrelated alias does not affect resume;
- updating the selected alias gives the next Turn a new binding while the old
  Turn resumes the old binding;
- removing an alias still permits schema-v2 Turn resume from its snapshot;
- current policy can block resume but cannot substitute another model;
- schema-v1 history remains replayable but all new provider I/O fails closed
  because the incomplete legacy binding cannot prove equivalence;
- durable replay works with no catalog, credential, endpoint, or model server;
- canonical text/reasoning/plan/tool Item lifecycle remains unchanged.

### CLI and packaging tests

- `list --source`, `show`, `add`, `update`, `probe`, `remove`, `current`, and
  `switch`, plus `trust init/status`, exercise the same application services as
  SDK code;
- source and installed-wheel smoke tests use an isolated user registry and
  session path;
- README commands are copied into executable smoke tests where practical;
- full Ruff, mypy, pytest, import-linter, build, and diff checks run before
  integration.

## Documentation scope

README will describe only:

- the model interface requirements: supported provider adapter, canonical text
  streaming, context/output limits, and optional tool/structured capabilities;
- how to register a local or cloud endpoint without storing a secret;
- how to probe, list, show, switch, update, and remove aliases;
- how to initialize, inspect, back up, and restore the model-binding trust
  domain without exposing its HMAC key;
- the distinction between built-in defaults, the user registry, project
  session selection, and immutable per-Turn binding;
- when a new process is required to observe registry edits.

It will not introduce or compare Qwen, Kimi, DeepSeek, or other model families.
Runbook material will cover paths, overrides, recovery, concurrency conflicts,
probe diagnostics, and legacy Turn limitations.

## Acceptance criteria

- A user can onboard a supported local or cloud chat model without editing
  `configs/models.yaml`.
- The CLI never writes provider or credential-routing configuration inside the
  project workspace.
- Built-in aliases remain immutable and user aliases retain visible origin.
- Add/update probe before commit by default; failure changes no persistent
  state.
- User registry and session mutations are atomic and stale writers are
  rejected; failures after the visibility commit point report and reconcile an
  unknown outcome instead of claiming rollback.
- Praxis CLI and runtime accept only credential references and never copy a
  resolved credential value into registry files, logs, exceptions, Turn
  bindings, archives, or replay history; strict schemas reject plaintext
  transport/authentication fields supplied through `--from`.
- Invalid aliases and failed updates leave the prior session/catalog behavior
  unchanged and make no unintended provider request.
- New Turns install a trusted definition and persist complete schema-v2 frozen
  bindings before dispatch, authenticated to their Turn/thread identity and
  selection requester with a user-owned HMAC key.
- Trust-domain initialization is explicit, atomic, concurrent-safe, and never
  silently regenerates a missing key.
- Unrelated registry changes cannot break schema-v2 Turn resume.
- Updated or removed aliases do not change schema-v2 bindings already stored
  on Turns.
- Legacy Turns remain replayable but fail closed before provider I/O; they
  never silently resolve to a different selected definition.
- README and RUNBOOK accurately document integration requirements, registry
  ownership, switching, and recovery.
- Focused lifecycle/concurrency tests and all repository release gates pass.
