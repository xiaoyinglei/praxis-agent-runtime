# Turn context and prefix continuity

New Turns freeze `execution_context_policy.context_updates = append-v1`.
The policy applies to the shared Session used by the CLI and Python SDK.
Existing Turns retain their saved policy and request layout.

## Assembly and persistence

1. `RolloutContextManager` projects the visible conversation, including committed
   project and memory updates, completed tool groups and compaction replacements.
2. `TurnExecutor.capture_step_context` restores the Turn's frozen project rules
   and trusted scoped discoveries, compares them with the last visible snapshot,
   and appends changed snapshots. A new snapshot replaces the earlier snapshot
   of that kind, including removed project scopes.
3. Runtime clock, plan, acceptance and resource changes use the existing appended
   runtime state mechanism. Tool selection uses committed discovery evidence.
4. Preparation saves the exact messages, tools, bindings and clock in the Step
   snapshot. Context updates are committed with the model request, before dispatch.
   Speculative preparations never advance the visible baseline.
5. Recovery restores that Step snapshot. It does not reread project files,
   refresh memory, change tools or capture a new clock for the saved request.

The first project and memory snapshots follow the first user message in this
implementation. They remain context messages; the provider adapter owns their
protocol representation. Later changes remain at their effective point in the
conversation instead of rewriting the system message before old history.

Root rules and already discovered nested scopes refresh once for a new Turn,
within the existing permission, scope and byte limits. First discovery within a
Turn still stops the attempted action until the model has seen the rules.

## Memory boundaries

Navigation contains keys/titles and a digest of semantic entry content. Processing
cursors, timestamps and version-only commits do not produce a new navigation
message. Full version/provenance remains frozen in the Turn binding. `read_memory`
returns the version needed by `manage_memory`; writes still reject stale versions.

When the next Turn observes removed memory entries, new forgotten sources, or
memory opt-out against any earlier snapshot in its visible window, its binding
records `context_reset: {revision: v1, reason:
memory_revoked}`. The visible model and `read_context` window starts at that Turn.
Later Turns and forks inherit this boundary. This conservatively excludes mixed
summaries and assistant replies that may contain recalled data, at the cost of
losing the earlier conversational context. It is not a word-based redactor.

This boundary does not erase the audit ledger or rewrite historical frozen Turn
snapshots. The existing within-Turn frozen-memory contract is retained. A removal
is not a promise to purge a provider's previously cached or retained data.

## Tools, compaction and cache limits

Deferred activation requires a successful linked `find_tools` operation from the
visible conversation. Reopen and new Turns retain discovery; a fork cannot see
operations after its cutoff. Installed/discoverable names remain the ceiling.
Every invocation still passes the current execution permissions and tool origin
checks; discovery grants no permission.

The OpenAI-compatible Chat wire adds newly discovered tool schemas to `tools`.
That schema change is a legitimate cache boundary. Subsequent Turns keep the
expanded declarations. No Responses-only or Claude-only tool protocol is emitted.

Compaction keeps the original ledger, includes context updates in summary sources,
and rebuilds current project/memory/runtime baselines when their old events are
covered. The measured candidate and committed projection must agree. Subsequent
requests resume appending from that new window.

The design follows Codex's persisted baseline/context-update and compaction
approach, checked against commit
[`970111f6`](https://github.com/openai/codex/blob/970111f6cb62cc02fe36a82ea7d71b37a2889c97/codex-rs/core/src/session/mod.rs).
[OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
and [Claude prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
define provider-specific cache boundaries. This change does not implement Claude
Messages or rely on private Claude internals.

Matching serialized prefixes only establishes reuse eligibility. Actual hits and
cost come from provider usage. Cached input still counts toward context capacity;
the existing logical input, cache read/write and cost accounting remain separate.

## Verification

`tests/agent/harness/test_context_prefix_updates.py` checks actual serialized Chat
messages and tool declarations across Steps/Turns, root/nested rule changes and
removal, metadata-only and real memory changes, saved request replay, compaction,
reopen, fork cutoffs, and memory removal/opt-out boundaries. The memory management
test executes an initial read followed by a versioned explicit write through the
normal tool interface. Tests use deterministic model responses and require no
paid provider call.
