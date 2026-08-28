# Praxis Harness Canonical Streaming Alignment Design

## Status

Approved direction: preserve the replacement Harness as the only execution
runtime and align its public/live protocol with the canonical streaming v2
contract already merged into `main`.

Recovery checkpoint:

- branch: `codex/praxis-harness-architecture`
- commit: `58e41c28dd3ea2417c4ef18294bae4126b28d793`
- message: `wip(agent): checkpoint praxis harness architecture`

That commit remains an ancestor of all alignment work. The integration uses a
merge from `origin/main`, not a rebase, so the exact WIP checkpoint stays
addressable.

## Goal

Make the Harness public execution path implement one complete protocol:

`Turn -> ItemStarted -> ItemDelta* -> ItemCompleted -> durable history`

The protocol covers model text, reasoning, plans, tools, commands, Turn
lifecycle, cancellation, bounded delivery, persistence, and replay without
restoring the deleted `AgentService`, `AgentLoop`, `LoopState`, or `TurnStore`.

## Non-negotiable ownership

There is one runtime and one durable truth:

```text
Agent public API
  -> RuntimeComposition
    -> ThreadManager
      -> Session
        -> model/tool operations
          -> RolloutStore transaction
            -> committed RolloutRecord
              -> live/replay StreamEvent projection
```

- `Session` owns one active Turn execution and its cancellation scope.
- `RolloutStore` owns canonical durable records and rebuildable projections.
- `StreamEvent` is a client envelope, never a second state store.
- live deltas are transient and are not written to SQLite.
- a durable event is published only after its `RolloutRecord` commits.
- CLI and SDK consume the same public protocol and do not infer lifecycle.

The merge must not reintroduce `AgentService -> AgentLoop -> TurnStore` as a
parallel path. Main's canonical protocol behavior is retained, while its old
runtime-specific persistence wiring is adapted to `RolloutStore`.

## Public protocol

`agent_runtime.streaming.events` keeps the exact v2 envelope and wire values
from `main`:

- Turn events: `turn_started`, `turn_paused`, `turn_resumed`,
  `turn_cancellation_requested`, `turn_completed`, `turn_aborted`.
- Item events: `item_started`, `item_delta`, `item_completed`.
- Item kinds: `agent_message`, `reasoning`, `plan`, `tool`, `command`,
  `reconciliation`, and `legacy_message`.
- Delta kinds: `text`, `reasoning`, `plan`, `tool_progress`,
  `command_stdout`, and `command_stderr`.
- Completion statuses: `success`, `failed`, `cancelled`, and
  `outcome_unknown`.

Every Item event carries a stable `turn_id`, `item_id`, and `item_kind`.
`ITEM_COMPLETED` carries the authoritative JSON-safe payload even when the live
consumer observed deltas. Legacy enum imports remain importable, but the
runtime emits v2 only. A consumer must opt into `LegacyStreamProjectionSink`
to receive legacy wire events.

## Durable records and public projection

Harness `RolloutRecord.thread_sequence` remains the only per-Thread durable
ordering key. Harness Items are richer than the seven public `TurnItemKind`
values, so the projection is deliberately not a blind one-to-one mapping.

Turn records project as follows:

| Rollout record | Canonical event |
| --- | --- |
| `turn_started` | `TURN_STARTED` |
| `turn_paused` | `TURN_PAUSED` |
| `turn_resumed` | `TURN_RESUMED` |
| `turn_cancellation_requested` | `TURN_CANCELLATION_REQUESTED` |
| `turn_completed` | `TURN_COMPLETED` with completed status |
| `turn_failed` | `TURN_COMPLETED` with failed status |
| `turn_cancelled` | `TURN_ABORTED` with cancelled reason |
| `turn_abandoned` | `TURN_ABORTED` with abandoned reason |
| `turn_interrupted` | nonterminal `TURN_PAUSED` with `data.status=interrupted` and `data.reason=outcome_unknown` |

Internal Item/operation facts project as follows:

| Harness fact | Public Item projection |
| --- | --- |
| `model_response` | one `agent_message` whose public ID is derived from the model attempt and stored on the response payload; payload contains `content` and validated `tool_calls` |
| `model_reasoning` | one `reasoning` Item keyed by model attempt and channel |
| `model_plan` | one `plan` Item keyed by model attempt and channel |
| accepted final `agent_message` | suppressed when it is only the completion projection of an already published `model_response`; used as `legacy_message` only for migrated history without a source response |
| `tool_operation_claimed` plus terminal `tool_result` | one operation-attempt `tool` Item; `tool_call` remains inside the parent `agent_message` payload |
| the same facts for `run_command` | one operation-attempt `command` Item; `command_execution` is an activity projection referencing the same operation ID, never a second outcome |
| `tool_reconciliation` | one `reconciliation` Item whose parent is the immutable outcome-unknown operation Item |
| successful `update_plan` ToolResult plus canonical PlanState revision | one completed `plan` Item containing the complete revision; proposed-plan deltas use that same Item lifecycle |
| `user_message`, `input_file`, `model_request`, `tool_call`, `final_proposal`, `completion_decision`, `completion_feedback`, `context_compaction`, approvals/interactions, verification, artifacts, and resource claims | retained as internal durable facts but suppressed from v2 unless a row above defines a semantic public projection |

Projection code is exhaustive: an unknown durable kind fails closed in replay
and tests, rather than being silently skipped. Public tool/command IDs derive
from `(turn_id, operation_id, attempt_generation)`. Every model channel ID
derives from `(turn_id, model_attempt_id, channel)` before dispatch and is
stored in the corresponding start/completion payload. The internal
`model_response` Item ID is not a public Item ID. An `update_plan` snapshot ID
derives from `(turn_id, canonical_plan_revision)` and is stored with that
revision. Live, retry, reopen, and replay call the same derivation helpers and
fail if persisted IDs do not match.

Publicly projectable completion records carry `status`, authoritative `payload`,
optional `error`, `iteration`, and optional `parent_item_id`. Existing successful
records are deterministically upcast to `status=success`; old records are not
rewritten in place.

Harness persists real `item_started` records. Replay projects only starts that
have a matching completion in the selected committed prefix, except when the
caller explicitly requests diagnostic raw rollout facts. This prevents an
ordinary history consumer from observing a permanently open Item after a crash.
Startup recovery closes every orphaned public Item before the Turn can resume:

- a committed operation result missing only its public completion is completed
  idempotently from the committed result;
- work proven not to have dispatched/executed is closed `cancelled`;
- dispatched model I/O or a side-effecting tool without a proven result is
  closed `outcome_unknown` and the Turn remains `interrupted`;
- a safely failed operation is closed `failed` with its durable error.

Recovery appends facts; it never mutates the orphaned start. Crash-injection
tests cover process death after start, after dispatch/claim, after result, and
before publish.

## Live event path

One Session-owned dispatcher fans one Turn's events into independent bounded
subscriber channels. `Agent.astream()` and the direct `event_sink` attached to
the executing call are controlling subscribers: closing/failing them requests
Turn cancellation. Read-only observers have independent cursors and may detach
without cancelling execution. Only controlling subscribers participate in
producer backpressure.

```text
provider/tool callback
  -> await TurnEventDispatcher.emit(ITEM_DELTA)
  -> bounded per-subscriber queue
  -> controlling stream/sink and optional observers
```

- An async producer awaits downstream capacity and therefore receives
  backpressure.
- A synchronous provider bridge uses a bounded standard-library queue, timed
  puts that poll a stop flag, and never calls event-loop APIs from the producer
  thread. Cancellation never joins a permanently blocked daemon producer.
- Each subscriber has `open`, `closing`, and `closed` states. `emit()` waits on
  either queue capacity or the subscriber close event; closing therefore wakes
  blocked emitters and getters even when the queue is full. Emitting after close
  raises `EventChannelClosed`.
- Passive observers receive best-effort transient deltas. If a passive queue is
  full, the dispatcher closes that observer with `ObserverLagged` and its last
  committed cursor; it never waits on passive capacity. The observer reconnects
  from durable history, which intentionally contains no deltas. Durable events
  are never silently dropped.
- Controlling close drops undelivered transient deltas, commits cancellation and
  closure facts through the durable path, and leaves them available for replay.
  A passive observer close affects only that observer.
- Sink delivery has a bounded cancellation grace period. A sink that raises or
  never returns is cancelled and treated as controlling-stream failure; runtime
  shutdown never waits indefinitely for it.
- A live consumer failure cannot roll back an already committed record.
- A delta sink failure cancels the affected producer and enters the same
  close/abort path as other cancellation.

`Agent.arun(event_sink=...)` and `Agent.astream()` use the same dispatcher. The
current unbounded `asyncio.Queue` and `put_nowait` record-listener bridges are
removed.

## Model Item lifecycle

Before provider streaming starts, the model attempt allocates stable IDs for
the response channels of one Step:

- `agent_message`: text plus validated tool calls;
- `reasoning`: provider-exposed reasoning only;
- `plan`: provider-exposed plan only.

The model ACI accepts an awaited `ProviderDeltaSink` carrying `text`,
`reasoning`, or `plan`. The first visible delta publishes `ITEM_STARTED`, then
each provider fragment publishes `ITEM_DELTA`. If an authoritative channel has
no visible delta, including a zero-text tool-only response, provider completion
publishes `ITEM_STARTED` immediately before committing and publishing
`ITEM_COMPLETED`. Provider completion commits one authoritative response Item
before its completion is delivered.

The first reasoning or plan delta also appends an `item_started` record with
kind `model_reasoning` or `model_plan`, `public_item_id`, `attempt_id`, channel,
and iteration. Model completion atomically appends their `item_completed`
records containing the full accumulated `content`, together with the
authoritative `model_response`, attempt usage/status, and logical operation
status. These records are the sole replay source for reasoning/plan completion;
deltas remain transient. Ordinary `update_plan` commits a separate
plan-revision Item with its derived snapshot ID and complete PlanState payload.

A tool-only response still produces a completed `agent_message` Item whose
payload contains empty content plus complete validated tool calls. A dedicated
ordering test requires start then completion with no fabricated delta. Partial
provider failure closes every started channel as `failed`. Cancellation before
dispatch, or after the provider acknowledges a definitive stop, closes it as
`cancelled`; cancellation after dispatch without a confirmed outcome closes it
as `outcome_unknown` and interrupts the Turn. Non-streaming providers may emit
one delta after obtaining the full response but the runtime does not fabricate
token timing by slicing text.

## Tool and command Item lifecycle

Permission and approval happen before execution Item start. Once a durable tool
attempt enters the executing state:

- normal tools use one `tool` Item;
- `run_command` uses one `command` Item;
- the optional streaming runner receives an awaited `ToolProgressSink`;
- progress maps to `tool_progress`;
- command pipes map to `command_stdout` and `command_stderr`;
- final completion commits the normalized ToolResult or authoritative error;
- uncertain side effects close as `outcome_unknown` and are never renamed or
  silently rerun.

Reconciliation creates a separate `reconciliation` Item whose
`parent_item_id` points at the immutable unknown-outcome Item.

Command streaming keeps independent bounds for live chunk size/count and final
retained stdout/stderr. Cancellation terminates and reaps the process group.

## Turn lifecycle and cancellation

A persisted Turn emits `TURN_STARTED` once. Resume emits `TURN_RESUMED`, not a
second start. Pause closes all active Items before committing `TURN_PAUSED`.

On stream close or task cancellation:

1. commit `turn_cancellation_requested` while the Turn still owns the Thread;
2. signal the Session cancellation scope and active model/tool operation;
3. close each started Item as `cancelled` or `outcome_unknown`;
4. if every operation has a known stopped outcome, commit terminal
   `turn_cancelled` and project `TURN_ABORTED`;
5. if any operation is `outcome_unknown`, commit `turn_interrupted`, retain the
   Thread active slot and claims, publish nonterminal `TURN_PAUSED` with
   interruption reason after `TURN_CANCELLATION_REQUESTED`, close the live
   stream, and require reconciliation before redispatch;
6. after reconciliation, resume the same Turn with `TURN_RESUMED`, or explicitly
   abandon/cancel it and then project terminal `TURN_ABORTED`.

`turn_interrupted` is represented by the existing nonterminal
`TURN_PAUSED` wire event rather than being misprojected as `TURN_ABORTED` or a
new Turn. No replay sequence may contain an Item after the final Turn event. Provider I/O
that cannot be interrupted may outlive the Turn only in an isolated daemon
producer; it cannot block event-loop or Session shutdown.

## Replay

`RolloutEventReader` and the public replay API project the same v2
`StreamEvent` envelope inside a separate `ReplayEvent` result containing
`event`, `cursor`, `record_id`, and `thread_sequence`. `StreamEvent.sequence` is
process-local for live delivery and per-replay-call projection order; it is
never a durable cursor. Live/delta `timestamp_ms` uses Unix epoch milliseconds
at creation. Replayed durable events use the committed record timestamp.

The opaque Thread cursor contains `version`, `thread_id`, `thread_sequence`,
`store_epoch`, and `schema_epoch`. Global maintenance tailing continues to use
`after_record_id`; those cursor types and APIs remain separate. Malformed,
cross-Thread, ahead-of-tail, store-epoch, and schema-epoch mismatches fail loud
with an actionable full-resync error and do not silently restart at zero.

Replay includes committed lifecycle and Item facts ordered by
`thread_sequence`. It excludes transient deltas. Reopening SQLite must reproduce
the same authoritative completed payloads, statuses, errors, parent links, and
terminal order.

## Atomic persistence boundaries

Every Rollout append, reducer projection, projection metadata update, and
post-commit notification is one SQLite transaction boundary. In addition:

- model completion atomically commits attempt status/usage/provider response
  identity, logical model-operation status, authoritative `model_response`, and
  its context-visible projection under the current attempt generation;
- proven tool completion atomically commits the canonical `tool_result`,
  terminal operation state, result link, claim/resource release, and public
  completion outcome under the fencing token. An `outcome_unknown` completion
  instead atomically retains its fencing claim/resource ownership, marks
  reconciliation required, and appends the public unknown outcome;
- pause/resume atomically validates the Thread active slot, transitions the
  Turn/interactions/approval state, and appends the lifecycle record;
- Turn completion atomically commits the accepted answer projection, terminal
  Turn state, final lifecycle record, and Thread active-slot release;
- no listener or public event runs until the outer transaction commits.

Fault injection between each logical substep must reopen the database and show
either the old committed prefix or the entire new transaction, never a mixed
projection. CAS/generation/fencing failures append no terminal outcome and
cannot release another worker's claim.

## Merge and migration strategy

1. Merge `origin/main` into the Harness branch so `58e41c28` remains an exact
   parent checkpoint.
2. Resolve runtime conflicts in favor of the Harness ownership chain.
3. Resolve public event-contract conflicts in favor of main's v2 envelope.
4. Adapt canonical persistence and queue behavior to `RolloutStore` and
   `Session`; do not restore `TurnStore` or the old Loop.
5. Register deterministic upcasters for pre-alignment Harness records.
6. Keep legacy wire projection opt-in only.
7. Preserve the offline union migration from legacy TurnStore/checkpoint data:
   acquire the maintenance lock, create a backup, perform a dry-run report,
   import Turns/messages/interactions/approvals/tool operations/attachments in
   one transaction, verify projection hashes, and support idempotent rerun or
   rollback. Runtime fallback to legacy storage remains forbidden.

## Verification contract

Tests must prove the real public path, not only isolated dataclasses:

1. stable Item identity across start, multiple deltas, and completion;
2. model text, reasoning, and plan deltas are native and not fabricated;
3. tool progress and command stdout/stderr arrive before completion;
4. every started Item closes exactly once on success, failure, cancellation,
   pause, process crash, and outcome-unknown reconciliation, including a
   zero-text tool-only response;
5. completed Items and Turn transitions replay identically after SQLite reopen;
6. deltas never appear in durable records;
7. bounded controlling queues exert measurable backpressure, while a full
   passive observer is detached with its last cursor and cannot block execution
   or change another observer's cursor;
8. closing a full stream, a non-reading sink, and a never-returning sink cannot
   hang; a blocked synchronous provider is signalled but never joined;
9. resume preserves the original Turn and emits `TURN_RESUMED` only;
10. CLI, `Agent.arun(event_sink=...)`, and `Agent.astream()` observe the same v2
    lifecycle;
11. `tests/agent/harness`, `test_canonical_streaming_protocol.py`, public facade,
    CLI, model gateway, tool executor, command cancellation, migration, and
    repository import-contract suites all pass; no suite may simply be removed;
12. deleted old-runtime modules remain deleted and import-boundary tests prevent
    their reintroduction;
13. real legacy SQLite fixtures prove both pre-v2 streaming upcast and the
    offline TurnStore/checkpoint-to-RolloutStore migration, including dry run,
    rollback, and idempotent rerun;
14. crash tests cover commit-before-publish, post-commit/pre-delivery restart,
    orphaned Item start recovery, independent subscribers, all cursor failures,
    and public import/signature compatibility.

## Intentional non-goals

- Persisting token or command deltas.
- Reintroducing the pre-Harness runtime as a fallback.
- Maintaining implicit v1 wire behavior.
- Adding a distributed broker or multi-tenant scheduler.
- Refactoring unrelated model, RAG, skill, or MCP behavior.
