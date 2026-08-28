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

## Durable record mapping

Harness `RolloutRecord.thread_sequence` remains the only per-Thread durable
ordering key. The public projection maps committed records as follows:

| Rollout record | Canonical event |
| --- | --- |
| `turn_started` | `TURN_STARTED` |
| `turn_paused` | `TURN_PAUSED` |
| `turn_resumed` | `TURN_RESUMED` |
| `turn_cancellation_requested` | `TURN_CANCELLATION_REQUESTED` |
| `turn_completed` | `TURN_COMPLETED` with completed status |
| `turn_failed` | `TURN_COMPLETED` with failed status |
| `turn_cancelled` | `TURN_ABORTED` |
| `item_started` | `ITEM_STARTED` |
| `item_completed` | `ITEM_COMPLETED` with authoritative outcome |

`item_completed` records gain explicit canonical outcome data where necessary:
`status`, `payload`, optional `error`, `iteration`, and optional
`parent_item_id`. Existing successful records are deterministically upcast to
`status=success`; old records are not rewritten in place.

Harness may persist a real `item_started` record because it is already part of
the canonical rollout truth. Replay projects facts that actually exist; it
never manufactures token chunks or original timing.

## Live event path

One Turn-scoped event channel sits between producers and the public consumer.
It has a finite capacity and a separate close signal.

```text
provider/tool callback
  -> await TurnEventChannel.emit(ITEM_DELTA)
  -> bounded queue
  -> Agent.astream() or event_sink
```

- An async producer awaits downstream capacity and therefore receives
  backpressure.
- A synchronous provider bridge blocks only on its bounded thread queue and
  never calls event-loop APIs from the producer thread.
- Closing uses an out-of-band event, not a sentinel that may block behind a full
  queue.
- A live consumer failure cannot roll back an already committed record.
- A delta sink failure cancels the affected producer and enters the same
  close/abort path as other cancellation.

`Agent.arun(event_sink=...)` and `Agent.astream()` use the same channel. The
current unbounded `asyncio.Queue` and `put_nowait` record-listener bridges are
removed.

## Model Item lifecycle

Before provider streaming starts, the model operation allocates stable IDs for
the response channels of one Step:

- `agent_message`: text plus validated tool calls;
- `reasoning`: provider-exposed reasoning only;
- `plan`: provider-exposed plan only.

The model ACI accepts an awaited `ProviderDeltaSink` carrying `text`,
`reasoning`, or `plan`. The first visible delta publishes `ITEM_STARTED`, then
each provider fragment publishes `ITEM_DELTA`. Provider completion commits one
authoritative response Item before `ITEM_COMPLETED` is delivered.

A tool-only response still produces a completed `agent_message` Item whose
payload contains empty content plus complete validated tool calls. Partial
provider failure closes every started channel as `failed`; cancellation closes
it as `cancelled`. Non-streaming providers may emit one delta after obtaining
the full response but the runtime does not fabricate token timing by slicing
text.

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
4. commit terminal `turn_cancelled`;
5. project `TURN_ABORTED` only after the terminal record commits.

No replay sequence may contain an Item after the final Turn event. Provider I/O
that cannot be interrupted may outlive the Turn only in an isolated daemon
producer; it cannot block event-loop or Session shutdown.

## Replay

`RolloutEventReader` and the public replay API project the same v2
`StreamEvent` envelope. Thread replay uses the versioned opaque cursor backed by
`thread_sequence`; global maintenance tailing continues to use `record_id`.
Those cursor types remain separate.

Replay includes committed lifecycle and Item facts ordered by
`thread_sequence`. It excludes transient deltas. Reopening SQLite must reproduce
the same authoritative completed payloads, statuses, errors, parent links, and
terminal order.

## Merge and migration strategy

1. Merge `origin/main` into the Harness branch so `58e41c28` remains an exact
   parent checkpoint.
2. Resolve runtime conflicts in favor of the Harness ownership chain.
3. Resolve public event-contract conflicts in favor of main's v2 envelope.
4. Adapt canonical persistence and queue behavior to `RolloutStore` and
   `Session`; do not restore `TurnStore` or the old Loop.
5. Register deterministic upcasters for pre-alignment Harness records.
6. Keep legacy wire projection opt-in only.

## Verification contract

Tests must prove the real public path, not only isolated dataclasses:

1. stable Item identity across start, multiple deltas, and completion;
2. model text, reasoning, and plan deltas are native and not fabricated;
3. tool progress and command stdout/stderr arrive before completion;
4. every started Item closes exactly once on success, failure, cancellation,
   pause, and outcome-unknown reconciliation;
5. completed Items and Turn transitions replay identically after SQLite reopen;
6. deltas never appear in durable records;
7. the bounded public queue exerts measurable backpressure;
8. closing a full stream cannot hang and durably aborts the Turn;
9. resume preserves the original Turn and emits `TURN_RESUMED` only;
10. CLI, `Agent.arun(event_sink=...)`, and `Agent.astream()` observe the same v2
    lifecycle;
11. old Harness focused tests remain green or are deliberately migrated to the
    new canonical contract;
12. deleted old-runtime modules remain deleted and import-boundary tests prevent
    their reintroduction.

## Intentional non-goals

- Persisting token or command deltas.
- Reintroducing the pre-Harness runtime as a fallback.
- Maintaining implicit v1 wire behavior.
- Adding a distributed broker or multi-tenant scheduler.
- Refactoring unrelated model, RAG, skill, or MCP behavior.
