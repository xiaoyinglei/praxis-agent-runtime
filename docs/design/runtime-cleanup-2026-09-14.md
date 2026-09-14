# Runtime cleanup — 2026-09-14

Based on main after PR #37 (`36f25d53`). The cleanup removes retired writers,
unused APIs and aliases; it does not modify any user's database or configuration.

## Removed

- Legacy database importer, restore CLI, exports and importer-only RolloutStore
  methods. Existing databases are not imported or rewritten by this change.
- LegacyStreamProjectionSink and its old-wire conversion helpers. Consumers use
  canonical StreamEvent items directly. The adapter-only tests are retired with it.
- Unreferenced DurableTurnEvent module, BindingValidator protocol, positive-integer
  and remaining-token helpers, standalone skill-body loader, and unused model
  requires_api_key property.
- Unused transcript-rewrite verifier and its transcript revision helper. Active
  context compaction remains in place.
- Unused LLMGateway.agenerate_with_tools entrypoint. Current canonical model calls
  and streaming fallback retain their existing implementations.
- ExecutionStatus.RUNNING / UNKNOWN source aliases. Callers use STARTED /
  OUTCOME_UNKNOWN; persisted string values are unchanged.

## Retained after checking callers

- Historical event readers, projection rebuild and refusal to resume incompatible
  bindings. Historical replay fixtures live under tests instead of runtime APIs.
- Pre-reservation usage accounting, because removing it would undercount existing
  completed attempts. Versionless model selection reads still feed the current
  compare-and-swap store; these are live data readers, not orphaned code.
- Low-level stream helpers and terminal handling still used by existing consumers
  and tests. Removing the adapter does not silently change their event vocabulary.
- RAG/model provider adapters, file-manifest schemas, tool execution, approval,
  recovery and model trust. Their names or age alone do not make them dead code.
- Decorated CLI handlers, keyboard bindings and Pydantic validators. A textual
  single-reference scan is not proof that a registered callback is unused.

Historical design documents under docs/superpowers describe past implementations;
their migration commands and adapter references are superseded by this note.
Removed tracked source remains recoverable from Git history.
