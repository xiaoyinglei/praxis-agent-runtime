# Praxis Harness Canonical Streaming Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge current `main` into the replacement Harness branch and make the Harness public path implement canonical streaming v2 from live delta through durable replay.

**Architecture:** Keep `RuntimeComposition -> ThreadManager -> Session -> RolloutStore` as the only runtime and durable truth. Reuse main's public v2 event envelope and provider/tool streaming ACIs, but adapt persistence, replay, bounded fan-out, cancellation, and migration to Harness Rollout records instead of restoring `AgentService`, `AgentLoop`, or `TurnStore`.

**Tech Stack:** Python 3.12, asyncio, dataclasses, SQLite, Pydantic, pytest, uv, Ruff, mypy, import-linter

---

### Task 1: Merge main without restoring the deleted runtime

**Files:**
- Merge: `origin/main`
- Keep deleted: `agent_runtime/service.py`
- Keep deleted: `agent_runtime/loop/`
- Keep deleted: `agent_runtime/turns.py`
- Keep deleted: `agent_runtime/core/checkpointing.py`
- Resolve: `agent_runtime/streaming/events.py`
- Resolve: `agent_runtime/streaming/sink.py`
- Resolve: `agent_runtime/modeling/gateway.py`
- Resolve: `agent_runtime/tools/tool.py`
- Resolve: `agent_runtime/tools/executor.py`
- Resolve: `agent_runtime/tools/builtins/shell.py`
- Resolve: `agent_runtime/agent.py`
- Resolve: `agent_runtime/cli.py`

- [ ] **Step 1: Record the integration boundary**

Run:

```bash
git status --short --branch
git merge-base HEAD origin/main
git log --reverse --oneline "$(git merge-base HEAD origin/main)"..origin/main
```

Expected: clean branch, WIP commit `58e41c28` remains in history, eight main commits are pending.

- [ ] **Step 2: Merge main**

Run:

```bash
git merge --no-ff origin/main
```

Expected: conflicts only in the precomputed shared files; no untracked work is lost.

- [ ] **Step 3: Resolve ownership conflicts**

Keep main's v2 `StreamEvent` contract, gateway delta ACI, Tool progress ACI, and canonical protocol tests. Keep Harness's public `Agent`, `RuntimeComposition`, `ThreadManager`, `Session`, and `RolloutStore`. Resolve delete/modify conflicts by keeping the old Runtime modules deleted. Do not copy `TurnStore`-bound durable sink behavior into the Harness.

- [ ] **Step 4: Prove one runtime remains**

Run:

```bash
uv run pytest -q tests/agent/harness/test_architecture_contract.py tests/agent/harness/test_public_agent_cutover.py
uv run lint-imports
```

Expected: public path reaches Harness only; deleted runtime imports remain forbidden. Streaming tests may still fail until later tasks.

- [ ] **Step 5: Commit the structural merge**

Commit the merge only after every conflict marker is gone and `git diff --check` is clean.

### Task 2: Canonical v2 envelope and exhaustive Rollout projection

**Files:**
- Modify: `agent_runtime/streaming/events.py`
- Modify: `agent_runtime/streaming/sink.py`
- Modify: `agent_runtime/harness/events.py`
- Modify: `agent_runtime/harness/rollout.py`
- Modify: `agent_runtime/harness/reducer.py`
- Modify: `agent_runtime/harness/__init__.py`
- Modify: `agent_runtime/__init__.py`
- Test: `tests/agent/test_canonical_streaming_protocol.py`
- Test: `tests/agent/harness/test_event_replay.py`
- Test: `tests/agent/harness/test_rollout_store.py`

- [ ] **Step 1: Write failing projection tests**

Add tests for exact v2 wire values and field validation; deterministic public IDs; exhaustive internal-kind handling; model/tool/command/reconciliation/update-plan projection; explicit suppression of internal facts; and unknown-kind fail-closed behavior.

- [ ] **Step 2: Run red tests**

Run:

```bash
uv run pytest -q tests/agent/test_canonical_streaming_protocol.py tests/agent/harness/test_event_replay.py tests/agent/harness/test_rollout_store.py
```

Expected: failures show the WIP still emits v1 events and lacks v2 projection metadata.

- [ ] **Step 3: Implement the minimal projection contract**

Use main's `protocol_version=2`, `EventType`, `TurnItemKind`, `ItemDeltaKind`, `ItemStatus`, and factories. Add one exhaustive Rollout-to-public projection in `harness/events.py`. Store/validate public IDs and completion outcome fields. Add deterministic upcasters for existing successful Harness items without rewriting old records.

- [ ] **Step 4: Add replay envelope and cursor validation**

Return `ReplayEvent(event, cursor, record_id, thread_sequence)`. Keep live `StreamEvent.sequence` process-local. Add `schema_epoch` to the opaque Thread cursor and fail loud for malformed, cross-Thread, ahead-of-tail, store-epoch, and schema-epoch mismatches.

- [ ] **Step 5: Run green tests and commit**

Run the Task 2 command again. Expected: all selected tests pass and no transient delta appears in Rollout records.

### Task 3: Bounded Session event dispatcher and public API wiring

**Files:**
- Modify: `agent_runtime/streaming/sink.py`
- Modify: `agent_runtime/harness/composition.py`
- Modify: `agent_runtime/harness/session.py`
- Modify: `agent_runtime/agent.py`
- Modify: `agent_runtime/cli.py`
- Test: `tests/agent/test_canonical_streaming_protocol.py`
- Test: `tests/agent/harness/test_public_agent_cutover.py`
- Test: `tests/agent/harness/test_event_replay.py`

- [ ] **Step 1: Write failing transport tests**

Cover controlling-subscriber backpressure, full-queue close, blocked emitter wakeup, post-close emit, raising sink, never-returning sink, passive observer lag/detach with last cursor, and two independent observer cursors.

- [ ] **Step 2: Verify red**

Run the three Task 3 test files. Expected: current unbounded `put_nowait` queues fail backpressure and bounded-close assertions.

- [ ] **Step 3: Implement one Session-owned dispatcher**

Give controlling subscribers awaited bounded queues and close events. Passive observers never block execution; on overflow close them with `ObserverLagged(last_cursor)`. Make close wake blocked put/get operations without enqueuing a sentinel. Apply a bounded grace period to sink shutdown.

- [ ] **Step 4: Route all public entry points through it**

Make `Agent.arun(event_sink=...)`, `Agent.astream()`, resume, CLI display, and post-commit Rollout notifications use the same dispatcher. Ensure a committed event is upstream of live delivery and cannot be rolled back by sink failure.

- [ ] **Step 5: Run green tests and commit**

Run Task 3 tests plus `tests/agent/test_cli_wiring.py`.

### Task 4: Native model text, reasoning, and plan Item lifecycles

**Files:**
- Modify: `agent_runtime/harness/protocol.py`
- Modify: `agent_runtime/harness/model_adapter.py`
- Modify: `agent_runtime/harness/session.py`
- Modify: `agent_runtime/harness/rollout.py`
- Modify: `agent_runtime/modeling/gateway.py`
- Modify: `agent_runtime/tools/builtins/planning.py` if present after merge
- Test: `tests/agent/test_canonical_streaming_protocol.py`
- Test: `tests/agent/harness/test_model_adapter.py`
- Test: `tests/agent/harness/test_session.py`
- Test: `tests/agent/test_update_plan_surfaces.py`

- [ ] **Step 1: Write failing model lifecycle tests**

Cover multiple native text/reasoning/plan deltas; stable IDs across live/replay/reopen; non-streaming one-delta fallback; zero-text tool-only start then completion; mixed text/tool call response; partial provider failure; acknowledged cancellation; dispatched outcome unknown; retry with a new attempt ID; and canonical `update_plan` snapshots.

- [ ] **Step 2: Verify red**

Run the four Task 4 suites. Expected: WIP model adapter returns only a completed response and cannot satisfy delta timing or reasoning/plan replay.

- [ ] **Step 3: Add the awaited provider delta ACI**

Allocate attempt/channel IDs before dispatch. Pass one `ProviderDeltaSink` through Harness model preparation/dispatch and the model gateway. Do not manufacture chunks. Accumulate authoritative text per channel.

- [ ] **Step 4: Commit model completion atomically**

In one Rollout transaction commit attempt usage/status/provider identity, logical operation status, `model_response`, `model_reasoning`, and `model_plan` completions. Publish completion only after commit. Persist full content, validated tool calls, iteration, status, and errors.

- [ ] **Step 5: Run green tests and commit**

Run Task 4 tests and `tests/provider/test_llm_gateway.py`.

### Task 5: Tool progress, command output, and reconciliation Items

**Files:**
- Modify: `agent_runtime/tools/tool.py`
- Modify: `agent_runtime/tools/executor.py`
- Modify: `agent_runtime/tools/builtins/shell.py`
- Modify: `agent_runtime/harness/tool_orchestrator.py`
- Modify: `agent_runtime/harness/rollout.py`
- Test: `tests/agent/test_single_tool_executor.py`
- Test: `tests/agent/test_builtin_coding_tools.py`
- Test: `tests/agent/harness/test_tool_orchestrator.py`
- Test: `tests/agent/harness/test_tool_claim_fencing.py`
- Test: `tests/agent/harness/test_turn_recovery.py`
- Test: `tests/agent/test_canonical_streaming_protocol.py`

- [ ] **Step 1: Write failing tool/command tests**

Cover approval-before-start, one Item per operation attempt, awaited progress, stdout before command completion, distinct stderr, bounded live/final output, process-group cancellation, failed/cancelled/outcome-unknown completion, and reconciliation parent identity.

- [ ] **Step 2: Verify red**

Run the Task 5 suites. Expected: WIP exposes only committed legacy tool events and no command deltas.

- [ ] **Step 3: Wire `ToolProgressSink` after durable claim**

Start the public operation Item only after permission, approval, and fenced claim succeed. Route normal progress and concurrent command pipe chunks through the awaited dispatcher. `run_command` emits one `command` Item, not a tool parent plus command child.

- [ ] **Step 4: Make terminal transaction semantics explicit**

For proven terminal outcomes atomically commit ToolResult, result link, operation state, public completion, and release the matching claim/resources. For `outcome_unknown`, retain the fencing claim and reconciliation-required state. Reconciliation appends a child Item and never mutates/reruns the unknown attempt.

- [ ] **Step 5: Run green tests and commit**

Run Task 5 suites, including the real subprocess cancellation test.

### Task 6: Turn cancellation, orphan recovery, and replay ordering

**Files:**
- Modify: `agent_runtime/harness/session.py`
- Modify: `agent_runtime/harness/thread_manager.py`
- Modify: `agent_runtime/harness/rollout.py`
- Modify: `agent_runtime/harness/reducer.py`
- Modify: `agent_runtime/harness/events.py`
- Modify: `agent_runtime/agent.py`
- Test: `tests/agent/harness/test_turn_recovery.py`
- Test: `tests/agent/harness/test_model_claim_recovery.py`
- Test: `tests/agent/harness/test_committed_response_recovery.py`
- Test: `tests/agent/harness/test_event_replay.py`
- Test: `tests/agent/test_canonical_streaming_protocol.py`

- [ ] **Step 1: Write failing crash/cancel tests**

Inject crashes after Item start, after provider dispatch/tool claim, after committed result, after completion commit but before publish, and while the controlling queue is full. Verify ordinary cancellation reaches terminal `TURN_ABORTED`, unknown outcome reaches `TURN_PAUSED(reason=outcome_unknown)`, and no Item follows a terminal Turn event.

- [ ] **Step 2: Verify red**

Run the Task 6 suites. Expected: current WIP lacks cancellation-request records, canonical closure, and orphan public-Item recovery.

- [ ] **Step 3: Implement cancellation state transitions**

Commit cancellation request first, signal model/tool work, close all public Items, then either cancel terminally or retain an interrupted Turn and fenced claims. Resume the original Turn only after reconciliation; abandonment produces terminal abort.

- [ ] **Step 4: Recover orphaned starts before resume**

Append idempotent success from committed results, cancelled for proven non-dispatch, failed for durable known failure, or outcome-unknown for unconfirmed dispatched work. Ordinary replay hides unmatched starts until recovery has appended closure.

- [ ] **Step 5: Run green tests and commit**

Run Task 6 suites and verify SQLite reopen produces identical event order and payloads.

### Task 7: Legacy union migration and compatibility boundary

**Files:**
- Modify: `agent_runtime/harness/migration.py`
- Modify: `scripts/migrate_agent_rollout.py`
- Modify: `scripts/verify_agent_rollout.py`
- Modify: `agent_runtime/streaming/sink.py`
- Test: `tests/agent/harness/test_legacy_migration.py`
- Test: `tests/agent/harness/test_architecture_contract.py`
- Test: `tests/agent/test_public_exports.py`
- Test: `tests/repo/test_distribution_contract.py`

- [ ] **Step 1: Write failing migration/compatibility tests**

Use real pre-v2 SQLite and legacy TurnStore/checkpoint fixtures. Cover dry run, maintenance lock, backup, one-transaction import of Turns/messages/interactions/approvals/tool operations/attachments, rollback, idempotent rerun, projection hash verification, and v1 opt-in projection.

- [ ] **Step 2: Verify red**

Run the Task 7 suites. Expected: missing v2 upcast/cursor fields or incomplete union migration causes explicit failures.

- [ ] **Step 3: Implement deterministic migration and adapter**

Upcast old Harness records by schema version; import legacy stores offline with no runtime fallback. Keep old enum imports public but emit old wire values only through `LegacyStreamProjectionSink`.

- [ ] **Step 4: Verify no legacy runtime returns**

Run import contracts and search production imports for deleted service/loop/TurnStore modules.

- [ ] **Step 5: Run green tests and commit**

Run Task 7 tests, migration CLI dry run, and rollout verification script.

### Task 8: Full repository verification and final alignment commit

**Files:**
- Modify only files required by failures caused by this merge.
- Verify: entire repository

- [ ] **Step 1: Run focused protocol and Harness suites**

```bash
uv run pytest -q tests/agent/harness tests/agent/test_canonical_streaming_protocol.py tests/agent/test_harness_acceptance.py tests/agent/test_public_path_benchmark.py tests/agent/test_completion_protocol_prompt.py
```

- [ ] **Step 2: Run static and architecture checks**

```bash
uv run ruff check .
uv run mypy
uv run lint-imports
```

- [ ] **Step 3: Run full tests and build**

```bash
uv run pytest -q
uv build
```

- [ ] **Step 4: Run product smoke and acceptance commands**

```bash
uv run python scripts/agent_cli_smoke.py
uv run python scripts/agent_delivery_smoke.py --fake-model --verbose
uv run python scripts/agent_harness_acceptance.py validate --schema evals/harness/acceptance_v1.json --contract docs/design/praxis_harness_architecture.md
uv run python scripts/agent_code_benchmark.py validate evals/code_agent/benchmark_v1.json --repository .
```

- [ ] **Step 5: Inspect Git evidence and commit**

```bash
git diff --check
git status --short --branch
git log --oneline --decorate --graph -15
```

Expected: no conflict markers or unrelated changes, WIP commit `58e41c28` remains an ancestor, and all alignment changes are committed on `codex/praxis-harness-architecture`.
