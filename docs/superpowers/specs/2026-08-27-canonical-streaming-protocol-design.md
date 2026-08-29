# Canonical Streaming Protocol Design

## Goal

Replace the current collection of loosely related streaming callbacks with one
canonical execution protocol:

`Turn -> ItemStarted -> ItemDelta* -> ItemCompleted -> durable history`

The protocol must cover model text, reasoning, proposed plans, tool calls,
command stdout/stderr, terminal Turn state, cancellation, bounded transport, and
history reconstruction. It must use the existing `AgentService -> AgentLoop ->
ToolExecutor -> TurnStore/checkpoint` path rather than introduce a second
runtime or executor.

## Decisions

### One canonical event stream

`StreamEvent` remains the public envelope. It gains stable item identity and a
small set of canonical lifecycle event types:

- `TURN_STARTED`, `TURN_PAUSED`, `TURN_RESUMED`, `TURN_COMPLETED`,
  `TURN_CANCELLATION_REQUESTED`, `TURN_ABORTED`
- `ITEM_STARTED`, `ITEM_DELTA`, `ITEM_COMPLETED`

`TurnItemKind` identifies `agent_message`, `reasoning`, `plan`, `tool`,
`command`, `reconciliation`, and `legacy_message`. `ItemDeltaKind` identifies `text`, `reasoning`, `plan`,
`tool_progress`, `command_stdout`, and `command_stderr` payloads.

`ItemStatus` identifies `success`, `failed`, `cancelled`, and
`outcome_unknown`. Every `ITEM_COMPLETED` carries one of these statuses plus an
authoritative payload or error. Every started item is closed exactly once,
including partial provider failures and cancellation.

The canonical envelope has `protocol_version=2`. Legacy names such as
`TEXT_DELTA`, `THINKING_DELTA`, `TOOL_USE_START`, and `TOOL_USE_RESULT` remain
importable, but their old wire values are not emitted by the v2 runtime. One
`LegacyStreamProjectionSink` converts v2 events for explicitly opted-in legacy
consumers. The CLI migrates to v2 directly. This is a deliberate behavioral
migration, not a false claim that `text_delta` and `item_delta` are identical on
the wire.

Every item-scoped event carries the same non-empty `turn_id`, stable `item_id`,
and `item_kind`. A completed item carries the authoritative final content even
when the live stream contained deltas.

The exact v2 envelope is:

`protocol_version, type, turn_id, item_id, item_kind, delta_kind, status,
iteration, sequence, timestamp_ms, data, error, parent_item_id`.

Wire values and field rules are fixed:

| Event | Wire value | Required item fields | Forbidden item fields |
| --- | --- | --- | --- |
| Turn lifecycle | `turn_started`, `turn_paused`, `turn_resumed`, `turn_cancellation_requested`, `turn_completed`, `turn_aborted` | none; `data.status` and optional `data.reason` describe the transition | `item_id`, `item_kind`, `delta_kind`, `status`, `parent_item_id` |
| Item start | `item_started` | `item_id`, `item_kind`; optional `parent_item_id`; initial metadata in `data` | `delta_kind`, `status`, `error` |
| Item delta | `item_delta` | `item_id`, `item_kind`, `delta_kind`, and string `data.delta`; optional `parent_item_id` | `status`, `error` |
| Item completion | `item_completed` | `item_id`, `item_kind`, `status`, authoritative JSON-safe `data`; optional `parent_item_id` | `delta_kind`; `error` is required for `failed` and `outcome_unknown`, optional for `cancelled`, and forbidden for `success` |

`timestamp_ms` is Unix epoch milliseconds for display only. Live `sequence` is
process-local and is not a durable ordering key.

### Turn, resume, and model-iteration boundaries are separate

A persisted Turn spans its initial public request and zero or more pause/resume
executions under the same `turn_id`. It emits `TURN_STARTED` once,
`TURN_PAUSED`/`TURN_RESUMED` for each pause boundary, and exactly one final
`TURN_COMPLETED` or `TURN_ABORTED`. Resume never emits another `TURN_STARTED`.
Internal ReAct iterations are not called Turns; their iteration number remains
metadata on item events.

Closing a public stream or cancelling the running task durably marks the same
Turn with a nonterminal `TURN_CANCELLATION_REQUESTED` fact, stops live delivery,
and signals or isolates current work. The runtime then closes every started item
as `cancelled` or `outcome_unknown` and only afterward atomically commits final
`TURN_ABORTED`. Thus replay never contains an item after a final Turn event. A
failure emits `TURN_COMPLETED` with failed status; it is not misreported as a
user interrupt. Pausing closes any started item before `TURN_PAUSED`; a tool
waiting for approval has not started an execution item yet.

### Model items

Before consuming provider output, the model adapter allocates stable item IDs
from the Turn ID, iteration, and channel. The first visible delta starts the
item. One explicit provider delta ACI routes `text`, `reasoning`, and `plan`
channels; tool-call input fragments remain internal until they form a validated
call.

The provider ACI is:

```python
ProviderDeltaChannel = Literal["text", "reasoning", "plan"]

@dataclass(frozen=True, slots=True)
class ProviderDelta:
    channel: ProviderDeltaChannel
    content: str

ProviderDeltaSink = Callable[[ProviderDelta], Awaitable[None]]
```

The gateway awaits every callback, so async providers receive downstream
backpressure. A callback exception cancels the provider request and propagates
through the model turn. A synchronous producer blocks only on its bounded
thread queue, not on event-loop APIs.

When the provider response ends, it always produces one completed
`agent_message` item, even for a zero-text tool-only response. Its authoritative
payload contains both `content` and the complete validated `tool_calls`, so the
same transaction can project the exact assistant `ModelMessage`. A mixed text
plus tool-call response uses the same item. Every started reasoning/plan item is
also completed with its authoritative accumulated text. A provider error after partial output completes
started items as `failed`; cancellation completes them as `cancelled`. Providers
without a native stream may emit one delta only after the full response; the
protocol reports this honestly and does not manufacture token timing by slicing
completed text.

Plan updates use a `plan` item. Each canonical plan revision is completed and
durable. When a provider supplies proposed-plan deltas, they use the same plan
item lifecycle; ordinary `update_plan` snapshots are one-delta completed items.

### Tool and command items

Each approved execution attempt owns one item ID derived from its tool-call ID
and attempt count. Permission rejection and approval pause happen before
`ITEM_STARTED`. The item starts only after its durable execution record enters
`STARTED`. Progress is delivered through an explicit optional streaming runner
ACI on `Tool`, not through global state. `outcome_unknown` closes that attempt.
Checkpoint reconciliation never reruns or renames it as another execution
attempt; it creates a `reconciliation` decision item with `parent_item_id`
pointing to the immutable outcome-unknown item and records the durable decision.

The tool progress ACI is:

```python
ToolProgressKind = Literal["progress", "stdout", "stderr"]

@dataclass(frozen=True, slots=True)
class ToolProgress:
    kind: ToolProgressKind
    content: str
    percent: float | None = None

ToolProgressSink = Callable[[ToolProgress], Awaitable[None]]
StreamingToolRunner = Callable[
    [Mapping[str, JsonValue], ToolProgressSink], Awaitable[object]
]
```

`Tool.stream` is optional. `ToolExecutor` injects the sink only after permission
and approval pass and the execution record is durably `STARTED`. Each callback
is awaited and therefore backpressured. A callback exception cancels local
cancellable work and is handled by the same failure/outcome-unknown rules as a
runner exception.

The built-in `run_command` produces exactly one `command` item, not a tool parent
plus command child. Its streaming runner reads stdout and stderr concurrently,
emits bounded chunks while the process is alive, and retains at most the existing
per-stream output limit for the final structured result. Cancellation still
terminates and reaps the whole process group. The final item contains the
normalized ToolResult or the authoritative error. Other tools produce one
`tool` item per execution attempt.

### Durable history and replay

Live deltas are transient and are not written to SQLite. Durable state consists
of:

- the existing Turn row and current/final status;
- an append-only Turn lifecycle row for start, pause, resume, cancellation
  request, and final state;
- append-only completed item rows ordered within the Turn;
- item kind, authoritative payload, start/completion timestamps, and iteration.

Every durable lifecycle and completed item shares one Turn-local monotonic
`durable_ordinal`. It is allocated transactionally from the Turn row and is the
only replay ordering key. Idempotency comparison uses item ID, durable ordinal,
kind, iteration, status, authoritative payload, and error;
delivery timestamps are excluded. A divergent duplicate fails closed.

`begin_turn` atomically inserts the Turn row and its started lifecycle at ordinal
zero. Resume lease claim, running status, and resumed lifecycle are one
transaction. Unix timestamps are display facts only and never order replay.

`TurnStore.commit_completed_item` atomically appends the completed item and its
corresponding canonical model message, when one exists. A running Turn with
durable completed items is valid recoverable progress, not divergence.
`TurnStore.commit_turn_transition` atomically updates Turn status/terminal reason
and appends its lifecycle row before the terminal/pause event is delivered.
Therefore the database can never advertise a terminal event while the Turn row
is still running, and item replay cannot disagree with model-message history.

`TurnStore.replay_turn_events` merges lifecycle transitions and completed items
by `durable_ordinal`. It does not fabricate `ITEM_STARTED`, token chunks, or
original delta timing.

Schema migration is transactional. It adds `terminal_reason`,
`next_durable_ordinal`, lifecycle, and completed-item storage. For each pre-v2
Turn it deterministically backfills: started lifecycle at ordinal zero;
`legacy_message` completed items from canonical messages in message order; and
the current pause or final lifecycle from the Turn row. These items are marked
as legacy projections and do not claim original delta timing. A real pre-v2
SQLite fixture verifies migration and replay.

### Backpressure and cancellation

Both the public event queue and provider bridge are bounded. Native async
providers receive a cancellation signal directly. Synchronous providers run in
a dedicated daemon producer thread that talks only to a bounded standard-library
queue, never to the event loop. On cancellation the runtime signals an optional
provider `cancel_stream` ACI, sets the producer stop flag, and returns without
joining the thread. If provider I/O remains permanently blocked, that daemon may
remain until process exit but cannot block Turn cancellation or event-loop
shutdown. Timed producer puts observe the stop flag once provider iteration
returns.

Command delta size and count are bounded independently from retained final
output. The durable sink is upstream of the live queue. A live sink exception or
disconnect cannot roll back an already committed completed item or Turn
transition. Queue close uses a separate close signal and never waits to enqueue
a sentinel into a full queue. Live events may be abandoned after disconnect;
durable history remains authoritative.

## Data flow

1. `AgentService` atomically allocates the Turn and its started lifecycle in
   `TurnStore`.
2. The durable sink projects that committed `TURN_STARTED` live exactly once.
3. Model and tool producers emit item lifecycle events through the same sink.
4. The durable sink ignores deltas, atomically appends completed items plus their
   model-message projection, and then forwards the item event.
5. Pause/resume/final transitions update Turn status and lifecycle storage before
   forwarding their events. Existing transcript synchronization becomes an
   idempotent consistency check rather than a competing persistence path.
6. `replay_turn_events` projects the persisted Turn plus completed items into a
   deterministic history stream.

## Public compatibility

- `Agent.astream()` remains the public live API.
- `Agent.arun(..., event_sink=...)` remains supported and receives the same
  canonical events.
- Existing enum imports remain available, but v1 wire behavior requires explicit
  `LegacyStreamProjectionSink`; v2 consumers inspect `item_kind`, `delta_kind`,
  and `status`.
- `StreamEvent` JSON remains composed only of explicit JSON-safe fields.

## Verification

Tests must prove:

1. one durable Turn start and one final event per persisted Turn;
   pause/reopen/resume cycles do not duplicate either boundary;
2. stable item identity across start, deltas, and completion;
3. real reasoning and plan delta routing, without fabricated fallback chunks;
4. command output arrives before process completion and remains bounded;
5. completed items survive reopening SQLite and replay in order;
6. deltas are absent from durable storage;
7. bounded queues exert backpressure;
8. stream close cancels local work, records interruption, and cannot hang on a
   blocking provider producer or full live queue;
9. public API signatures and imports remain green; old wire behavior is tested
   only through `LegacyStreamProjectionSink`, while runtime event tests migrate
   to v2.
10. partial provider errors, cancel-after-delta, approval/resume,
    outcome-unknown/reconciliation, live sink failures, and event-loop shutdown
    close or durably preserve every lifecycle boundary.

## Non-goals

- Persisting every live token or command chunk.
- Inventing reasoning for providers that do not expose it.
- Adding a distributed event broker or a second execution engine.
- Replacing the existing checkpoint/reconciliation state machine.
