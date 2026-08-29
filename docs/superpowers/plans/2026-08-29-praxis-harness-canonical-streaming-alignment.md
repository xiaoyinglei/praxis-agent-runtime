# Praxis Harness Canonical Streaming Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge current `main` into the replacement Harness branch and make the Harness public path implement canonical streaming v2 from live delta through durable replay.

**Architecture:** Keep `RuntimeComposition -> ThreadManager -> Session -> RolloutStore` as the only runtime and durable truth. Reuse main's public v2 event envelope and provider/tool streaming ACIs, but adapt persistence, replay, bounded fan-out, cancellation, and migration to Harness Rollout records instead of restoring `AgentService`, `AgentLoop`, or `TurnStore`.

**Tech Stack:** Python 3.12, asyncio, dataclasses, SQLite, Pydantic, pytest, uv, Ruff, mypy, import-linter

---

## TDD execution rule

Tasks 2–7 are executed as the microcycles listed in each task. For each row:

1. add only that named test;
2. run its exact pytest node and observe the listed behavioral assertion failure
   (collection/import errors do not count as RED);
3. implement only enough production behavior to pass that node;
4. rerun the node and the named focused regression file;
5. stage only the test and production files used by that node and commit that
   microcycle before moving to the next node.

A list of nodes inside one numbered microcycle is an ordered list of independent
sub-microcycles, not a batch: complete RED -> GREEN -> focused regression ->
commit for the first node before writing the second node. Task-level regression
steps run after those commits and must leave the worktree clean; they do not
create an extra empty task commit.

If a named node unexpectedly passes before implementation, strengthen the test
to exercise the missing public path; do not change production code until a
valid RED is observed.

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
- Resolve: `scripts/agent_delivery_smoke.py`
- Resolve: `tests/agent/test_cli_wiring.py`
- Resolve: `tests/agent/test_single_tool_executor.py`
- Resolve: `tests/provider/test_llm_gateway.py`
- Port then remove old-runtime imports: `tests/agent/test_canonical_streaming_protocol.py`
- Keep deleted and port any still-required behavior to Harness tests:
  `tests/agent/test_agent_loop_runtime.py`,
  `tests/agent/test_agent_runtime_facade.py`,
  `tests/agent/test_agent_service_loop_boundary.py`,
  `tests/agent/test_llm_providers.py`,
  `tests/agent/test_public_turn_api.py`, and
  `tests/agent/test_turn_store.py`
- Retain and rewrite against Harness fixtures:
  `tests/agent/test_update_plan_surfaces.py`

- [ ] **Step 1: Record the integration boundary**

Run:

```bash
git status --short --branch
git merge-base HEAD origin/main
git log --reverse --oneline "$(git merge-base HEAD origin/main)"..origin/main
git rev-parse origin/main > /tmp/praxis-canonical-main-tip
test "$(cat /tmp/praxis-canonical-main-tip)" = "980d3dc1daa42b686800d290f484f61022707f5e"
```

Expected: clean branch, WIP commit `58e41c28` remains in history, eight main commits are pending.

Run the isolated-worktree baseline before merging:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy
uv run lint-imports
```

Expected: the WIP checkpoint is green on its own baseline. If it is not, record
the exact pre-existing failures and stop before mixing them with merge failures.

- [ ] **Step 2: Merge main**

Run:

```bash
git merge --no-ff origin/main
```

Expected: conflicts only in the precomputed shared files; no untracked work is lost.

- [ ] **Step 3: Resolve ownership conflicts**

Resolve every merge-tree conflict explicitly:

- retain old-runtime deletions for `core/llm_providers.py`, `loop/runtime.py`,
  `service.py`, `turns.py`, and the seven old-runtime test files listed above;
- keep main's v2 `streaming/events.py`, gateway delta ACI, Tool progress ACI,
  and shell command streaming behavior;
- retain only main's `StreamEventSink` and `LegacyStreamProjectionSink` portions
  of `streaming/sink.py`, without importing `TurnStore`; the new bounded
  dispatcher is deliberately deferred to Task 3's RED/GREEN cycles;
- combine Harness ACI/fencing behavior with main progress callbacks in
  `tools/executor.py` and its focused test;
- retain Harness public wiring while porting v2 rendering in `cli.py` and
  `test_cli_wiring.py`;
- retain Harness delivery smoke behavior and port its expected events to v2;
- port canonical protocol and update-plan assertions to Harness fixtures before
  running them, so pytest collection never depends on deleted modules.

- [ ] **Step 4: Prove one runtime remains**

Run:

```bash
uv run pytest -q tests/agent/harness/test_architecture_contract.py tests/agent/harness/test_public_agent_cutover.py
uv run lint-imports
uv run pytest --collect-only -q tests/agent/test_canonical_streaming_protocol.py tests/agent/test_update_plan_surfaces.py
```

Expected: public path reaches Harness only; deleted runtime imports remain forbidden. Streaming tests may still fail until later tasks.

- [ ] **Step 5: Commit the structural merge**

Run and require success:

```bash
git merge-base --is-ancestor 58e41c28 HEAD
git add agent_runtime/cli.py agent_runtime/streaming/events.py agent_runtime/streaming/sink.py agent_runtime/tools/builtins/shell.py agent_runtime/tools/executor.py scripts/agent_delivery_smoke.py tests/agent/test_canonical_streaming_protocol.py tests/agent/test_cli_wiring.py tests/agent/test_single_tool_executor.py tests/agent/test_update_plan_surfaces.py tests/provider/test_llm_gateway.py
git rm agent_runtime/core/llm_providers.py agent_runtime/loop/runtime.py agent_runtime/service.py agent_runtime/turns.py tests/agent/test_agent_loop_runtime.py tests/agent/test_agent_runtime_facade.py tests/agent/test_agent_service_loop_boundary.py tests/agent/test_llm_providers.py tests/agent/test_public_turn_api.py tests/agent/test_turn_store.py
test -z "$(git diff --name-only --diff-filter=U)"
git diff --cached --check
git commit
git merge-base --is-ancestor "$(cat /tmp/praxis-canonical-main-tip)" HEAD
```

The final `git commit` completes the in-progress merge with Git's prepared merge
message; it must not squash or amend `58e41c28`.

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
- Create: `tests/agent/harness/test_stream_projection.py`

- [ ] **Microcycle 2.1: Exhaustive projection**

RED node:
`tests/agent/harness/test_stream_projection.py::test_unknown_internal_item_kind_fails_closed`
with expected `DID NOT RAISE RuntimeError`. GREEN: add one exhaustive
Rollout-kind/public-kind match in `harness/events.py`; explicitly suppress only
the internal kinds named by the spec. Regression:
`tests/agent/harness/test_event_replay.py`.

- [ ] **Microcycle 2.2: Persisted deterministic IDs**

RED node:
`tests/agent/harness/test_stream_projection.py::test_public_item_ids_survive_reopen_and_mismatch_fails_closed`
with expected missing `public_item_id` or mismatch not rejected. GREEN: add and
validate persisted derivations for model channels, tool/command attempts, and
plan revisions in `harness/rollout.py`; retries use a new attempt ID. Regression:
`tests/agent/harness/test_rollout_store.py`.

- [ ] **Microcycle 2.3: Suppression and single outcomes**

RED nodes, executed separately:

```text
test_stream_projection.py::test_accepted_answer_does_not_duplicate_model_response
test_stream_projection.py::test_migrated_answer_projects_exactly_one_legacy_message
test_stream_projection.py::test_command_execution_does_not_create_a_second_command_outcome
```

Expected failures are duplicate `ITEM_COMPLETED` counts. GREEN: implement the
spec table exactly and keep `tool_call` inside the parent agent-message payload.

- [ ] **Microcycle 2.4: Replay envelope and cursors**

RED nodes, executed separately:

```text
test_event_replay.py::test_replay_returns_event_and_separate_thread_cursor
test_event_replay.py::test_thread_cursor_rejects_schema_epoch_mismatch
test_event_replay.py::test_thread_cursor_rejects_cross_thread_ahead_and_store_epoch_mismatch
test_event_replay.py::test_malformed_cursor_returns_actionable_full_resync_error
test_event_replay.py::test_global_tailer_accepts_only_after_record_id
```

Expected failures are absent `ReplayEvent`/`schema_epoch` fields or accepted
invalid cursors. GREEN: keep `StreamEvent.sequence` process-local and return
durable cursor metadata beside the event.

- [ ] **Microcycle 2.5: Timestamp and delta durability**

RED nodes:

```text
test_stream_projection.py::test_live_timestamp_is_unix_epoch_and_replay_uses_committed_timestamp
test_stream_projection.py::test_transient_deltas_never_enter_rollout_records
```

Expected failures show monotonic timestamps or durable delta rows. GREEN: add a
committed Unix-millisecond timestamp to `RolloutRecord`; do not append deltas.

- [ ] **Task 2 regression**

Run:

```bash
uv run pytest -q tests/agent/test_canonical_streaming_protocol.py tests/agent/harness/test_stream_projection.py tests/agent/harness/test_event_replay.py tests/agent/harness/test_rollout_store.py
test -z "$(git status --porcelain)"
```

### Task 3: Bounded Session event dispatcher and public API wiring

**Files:**
- Modify: `agent_runtime/streaming/sink.py`
- Modify: `agent_runtime/harness/composition.py`
- Modify: `agent_runtime/harness/session.py`
- Modify: `agent_runtime/harness/thread_manager.py`
- Modify: `agent_runtime/harness/tool_orchestrator.py`
- Modify: `agent_runtime/harness/rollout.py`
- Modify: `agent_runtime/agent.py`
- Modify: `agent_runtime/cli.py`
- Test: `tests/agent/test_canonical_streaming_protocol.py`
- Test: `tests/agent/harness/test_public_agent_cutover.py`
- Test: `tests/agent/harness/test_event_replay.py`

- [ ] **Microcycle 3.1: Controlling backpressure**

RED node:
`tests/agent/test_canonical_streaming_protocol.py::test_controlling_stream_blocks_producer_when_queue_is_full`
with expected producer already completed. GREEN: add a bounded controlling queue
whose awaited `emit()` blocks on capacity. Regression: the full canonical file.

- [ ] **Microcycle 3.2: Capacity-independent close**

RED nodes, separately:

```text
test_canonical_streaming_protocol.py::test_full_controlling_queue_close_wakes_blocked_emitter
test_canonical_streaming_protocol.py::test_emit_after_close_raises_event_channel_closed
```

Expected timeout/no exception. GREEN: model `open -> closing -> closed` with an
out-of-band close event watched by both putters and getters; do not enqueue a
sentinel.

- [ ] **Microcycle 3.3: Passive observer isolation**

Independent sub-microcycles:

```text
tests/agent/harness/test_event_replay.py::test_lagging_passive_observer_detaches_without_blocking_control
tests/agent/harness/test_event_replay.py::test_lagging_observer_does_not_change_another_observers_cursor
```

Expected controlling producer timeout or mutated peer cursor. GREEN: close only
the lagging observer with `ObserverLagged(last_cursor)` and preserve every other
subscriber's independent cursor state.

- [ ] **Microcycle 3.4: Sink failure and bounded shutdown**

RED nodes, separately:

```text
test_canonical_streaming_protocol.py::test_raising_sink_cannot_rollback_committed_event
test_canonical_streaming_protocol.py::test_never_returning_sink_is_cancelled_within_grace
test_canonical_streaming_protocol.py::test_delta_sink_failure_cancels_producer_and_durably_closes_item
```

Expected rollback, timeout, or open Item. GREEN: committed facts stay upstream
of delivery; cancel a stuck controlling sink after the configured grace period
and route delta-sink failure through the producer cancellation/durable closure
path.

- [ ] **Microcycle 3.5: Replace the synchronous listener bridge**

RED node:
`tests/agent/harness/test_public_agent_cutover.py::test_public_stream_awaits_post_commit_batch_without_record_listener_queue`
with expected `_record_listener`/`put_nowait` path observed. GREEN: each outer
`RolloutStore` transaction returns an immutable
`CommittedMutation[T](value, records)` containing only that transaction's
records. The active `Session` owns its dispatcher and awaits
`publish_committed_batch(mutation.records)` after receiving the return value;
`ThreadManager` routes mutations to the Session for that Thread, and
`ToolOrchestrator` returns its transaction-local batch to that same Session.
Remove production `_record_listener`, shared drain buffers, and all unbounded
Agent queues; no database callback awaits or touches an event loop.

- [ ] **Microcycle 3.6: Interleaved Session ownership**

RED node:
`tests/agent/harness/test_public_agent_cutover.py::test_interleaved_threads_publish_only_their_transaction_batches`
with expected cross-Thread event delivery. GREEN: keep dispatchers Session-owned
and transaction batches immutable/thread-scoped, including subagent Sessions.

- [ ] **Microcycle 3.7: Public entry-point parity**

RED node:
`tests/agent/harness/test_public_agent_cutover.py::test_cli_arun_sink_and_astream_share_identical_v2_lifecycle`
with expected differing event types/order/IDs. GREEN: route CLI,
`Agent.arun(event_sink=...)`, and `Agent.astream()` through the same Session
dispatcher and projection.

- [ ] **Task 3 regression**

```bash
uv run pytest -q tests/agent/test_canonical_streaming_protocol.py tests/agent/harness/test_public_agent_cutover.py tests/agent/harness/test_event_replay.py tests/agent/test_cli_wiring.py
test -z "$(git status --porcelain)"
```

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

- [ ] **Microcycle 4.1: Awaited native model channels**

RED nodes, separately:

```text
tests/agent/harness/test_model_adapter.py::test_native_text_deltas_are_awaited_and_keep_one_item_id
tests/agent/harness/test_model_adapter.py::test_native_reasoning_and_plan_deltas_use_distinct_items
```

Expected missing callback/delta events. GREEN: pass `ProviderDeltaSink` through
Harness prepare/dispatch and `modeling/gateway.py`; allocate and persist channel
IDs before dispatch, await each callback, and accumulate content without slicing.

- [ ] **Microcycle 4.2: Bounded synchronous provider bridge**

RED nodes, separately:

```text
tests/provider/test_llm_gateway.py::test_sync_provider_bridge_backpressures_with_bounded_standard_queue
tests/provider/test_llm_gateway.py::test_cancelled_sync_bridge_sets_stop_flag_and_never_joins_blocked_daemon
```

Expected unbounded production or event-loop hang. GREEN: use a bounded
`queue.Queue`, timed producer puts that poll a stop flag, a daemon producer
thread, and no cancellation join.

- [ ] **Microcycle 4.3: Zero-delta and non-streaming honesty**

RED nodes:

```text
tests/agent/harness/test_session.py::test_zero_text_tool_only_response_starts_then_completes_without_delta
tests/agent/harness/test_model_adapter.py::test_nonstreaming_provider_emits_one_full_delta_without_slicing
```

Expected missing start or fabricated chunk count. GREEN: at provider completion
start any authoritative zero-delta channel immediately before completion.

- [ ] **Microcycle 4.4: Failure and cancellation outcomes**

RED nodes, separately:

```text
tests/agent/harness/test_session.py::test_partial_provider_failure_closes_started_channels_failed
tests/agent/harness/test_session.py::test_acknowledged_provider_cancel_closes_started_channels_cancelled
tests/agent/harness/test_session.py::test_unconfirmed_dispatched_provider_cancel_closes_outcome_unknown
tests/agent/harness/test_session.py::test_model_retry_uses_new_attempt_and_public_item_ids
```

Expected open Item or wrong status/ID. GREEN: classify from durable dispatch and
provider acknowledgement, never from exception text.

- [ ] **Microcycle 4.5: Atomic model response/reasoning/plan commit**

RED nodes, separately:

```text
tests/agent/harness/test_session.py::test_model_completion_faults_are_atomic_at_every_substep
tests/agent/harness/test_session.py::test_stale_model_generation_appends_nothing
tests/agent/harness/test_event_replay.py::test_reasoning_and_plan_completed_content_replays_after_reopen
```

Expected partial attempt/Item/projection state or missing replay. GREEN: one
transaction commits usage, provider identity, attempt/logical-operation status,
model response, reasoning, plan, and context projection under generation CAS.

- [ ] **Microcycle 4.6: Canonical plan revision snapshots**

RED node:
`tests/agent/test_update_plan_surfaces.py::test_harness_update_plan_emits_one_persisted_plan_revision_item`
with expected missing plan completion. GREEN: derive ID from Turn and revision,
persist the complete PlanState, and reject a mismatched persisted ID.

- [ ] **Task 4 regression**

```bash
uv run pytest -q tests/agent/test_canonical_streaming_protocol.py tests/agent/harness/test_model_adapter.py tests/agent/harness/test_session.py tests/agent/harness/test_event_replay.py tests/agent/test_update_plan_surfaces.py tests/provider/test_llm_gateway.py
test -z "$(git status --porcelain)"
```

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

- [ ] **Microcycle 5.1: Approval before Item start**

RED node:
`tests/agent/harness/test_tool_orchestrator.py::test_public_tool_item_starts_only_after_approval_and_fenced_claim`
with expected premature start. GREEN: allocate the attempt ID early but publish
start only after permission, approval, and durable claim.

- [ ] **Microcycle 5.2: Awaited tool progress**

RED node:
`tests/agent/test_single_tool_executor.py::test_streaming_runner_progress_is_awaited_during_execution`
with expected no progress or runner finishing before sink release. GREEN: port
main's optional `ToolProgressSink` through the existing Harness executor and
orchestrator without bypassing ACI checks.

- [ ] **Microcycle 5.3: One command Item with live pipes**

RED nodes, separately:

```text
tests/agent/test_builtin_coding_tools.py::test_run_command_streams_stdout_before_exit_and_keeps_stderr_distinct
tests/agent/test_builtin_coding_tools.py::test_run_command_live_and_final_output_are_independently_bounded
tests/agent/harness/test_stream_projection.py::test_command_execution_does_not_duplicate_command_outcome
```

Expected no early stdout, merged stderr, unbounded chunks, or two outcomes.
GREEN: read pipes concurrently, bound chunks/count, retain existing final limits,
and use the operation attempt as the sole `command` Item.

- [ ] **Microcycle 5.4: Proven terminal tool transaction**

RED nodes:

```text
tests/agent/harness/test_tool_claim_fencing.py::test_tool_success_faults_are_atomic_at_every_substep
tests/agent/harness/test_tool_claim_fencing.py::test_stale_fencing_token_appends_nothing_and_releases_no_foreign_claim
```

Expected partial ToolResult/result-link/claim state. GREEN: one transaction
commits the result Item, operation terminal state, public completion, and only
the matching claim/resource release.

- [ ] **Microcycle 5.5: Unknown outcome and reconciliation**

RED nodes, separately:

```text
tests/agent/harness/test_tool_claim_fencing.py::test_outcome_unknown_retains_claim_and_reconciliation_state
tests/agent/harness/test_turn_recovery.py::test_reconciliation_item_parents_unknown_attempt_without_rerun
```

Expected released claim, mutated original Item, or runner call count two. GREEN:
retain fencing/resource ownership, append one child reconciliation Item, and
never rename or rerun the original attempt.

- [ ] **Microcycle 5.6: Command process-group cancellation**

RED node:
`tests/agent/test_builtin_coding_tools.py::test_run_command_cancellation_reaps_process_group_and_closes_item_once`
with expected surviving child or duplicate/missing completion. GREEN: terminate
and reap the entire group, then classify confirmed cancellation.

- [ ] **Task 5 regression**

```bash
uv run pytest -q tests/agent/test_single_tool_executor.py tests/agent/test_builtin_coding_tools.py tests/agent/harness/test_tool_orchestrator.py tests/agent/harness/test_tool_claim_fencing.py tests/agent/harness/test_turn_recovery.py tests/agent/harness/test_stream_projection.py tests/agent/test_canonical_streaming_protocol.py
test -z "$(git status --porcelain)"
```

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

- [ ] **Microcycle 6.1: Ordinary pause closes Items once**

RED node:
`tests/agent/harness/test_turn_recovery.py::test_approval_pause_closes_active_items_once_before_turn_paused`
with expected open/duplicate completion or wrong ordering. GREEN: atomically
transition interaction/approval/Turn state and append the pause lifecycle after
all active public Items close.

- [ ] **Microcycle 6.2: Ordinary cancellation terminal order**

RED node:
`tests/agent/test_canonical_streaming_protocol.py::test_confirmed_stream_cancel_commits_request_item_closures_then_turn_aborted`
with expected missing request, Item after final, or nonterminal Turn. GREEN:
commit request, signal work, close Items once, atomically release active slot and
append terminal cancellation.

- [ ] **Microcycle 6.3: Unknown outcome interruption**

RED node:
`tests/agent/harness/test_turn_recovery.py::test_unknown_outcome_projects_paused_reason_and_retains_claim`
with expected `TURN_ABORTED`, wrong paused status/reason, or released claim.
GREEN: keep durable Turn status `interrupted`, project
`TURN_PAUSED(status=paused, reason=outcome_unknown)`, and prohibit redispatch.

Then run a separate RED/GREEN/commit sub-microcycle for
`tests/agent/harness/test_turn_recovery.py::test_approval_pause_has_no_reconciliation_claim_but_interruption_retains_one`;
the initial failure must show the two pause reasons sharing the wrong claim
state, and GREEN must change only the reason-specific ownership transition.

- [ ] **Microcycle 6.4: Same-Turn reconciliation/resume**

RED node:
`tests/agent/harness/test_turn_recovery.py::test_reconciliation_closes_once_then_resumes_same_turn`
with expected new Turn ID, duplicate completion, or second `TURN_STARTED`. GREEN:
append reconciliation, release only reconciled claims, and emit `TURN_RESUMED`.

- [ ] **Microcycle 6.5: Orphaned start recovery matrix**

Run each new node RED before its matching branch:

```text
test_turn_recovery.py::test_orphan_before_dispatch_recovers_cancelled
test_turn_recovery.py::test_orphan_after_dispatch_recovers_outcome_unknown
test_committed_response_recovery.py::test_orphan_after_committed_result_recovers_success
test_turn_recovery.py::test_orphan_after_known_failure_recovers_failed
```

Expected unmatched start in ordinary replay or wrong status. GREEN: append only
the evidence-supported closure before resume; ordinary replay hides unmatched
starts until recovery.

- [ ] **Microcycle 6.6: Transaction fault/CAS matrix**

RED nodes:

```text
test_turn_recovery.py::test_pause_resume_faults_are_atomic_at_every_substep
test_turn_recovery.py::test_turn_completion_faults_are_atomic_at_every_substep
test_turn_recovery.py::test_active_slot_cas_failure_appends_nothing
test_model_claim_recovery.py::test_model_generation_failure_appends_nothing
```

Expected mixed projection or foreign release. GREEN: keep each lifecycle change,
reducer projection, metadata/hash update, and slot/claim CAS in one transaction.

- [ ] **Microcycle 6.7: Post-commit/pre-publish crash**

RED node:
`tests/agent/harness/test_event_replay.py::test_post_commit_pre_publish_crash_replays_identical_terminal_order`
with expected lost event or different payload after reopen. GREEN: notify only
after commit and reconstruct from Rollout order.

- [ ] **Task 6 regression**

```bash
uv run pytest -q tests/agent/harness/test_turn_recovery.py tests/agent/harness/test_model_claim_recovery.py tests/agent/harness/test_committed_response_recovery.py tests/agent/harness/test_event_replay.py tests/agent/test_canonical_streaming_protocol.py
test -z "$(git status --porcelain)"
```

### Task 7: Legacy union migration and compatibility boundary

**Files:**
- Modify: `agent_runtime/harness/migration.py`
- Modify: `scripts/migrate_agent_rollout.py`
- Modify: `scripts/verify_agent_rollout.py`
- Modify: `agent_runtime/streaming/sink.py`
- Create: `tests/agent/fixtures/build_legacy_rollout_fixtures.py`
- Create: `tests/agent/fixtures/legacy/pre_v2_harness.sqlite3`
- Create: `tests/agent/fixtures/legacy/legacy_turnstore_checkpoint.sqlite3`
- Create: `tests/agent/fixtures/legacy/manifest.json`
- Test: `tests/agent/harness/test_legacy_migration.py`
- Test: `tests/agent/harness/test_architecture_contract.py`
- Test: `tests/agent/test_public_exports.py`
- Test: `tests/repo/test_distribution_contract.py`

- [ ] **Microcycle 7.1: Reproducible real fixtures**

RED node:
`tests/agent/harness/test_legacy_migration.py::test_checked_in_legacy_sqlite_fixtures_match_manifest_hashes`
with expected missing files. GREEN: the builder creates both databases from
fixed SQL/data, writes their SHA-256 hashes and expected post-migration record
chain/projection hashes to `manifest.json`, then the test opens the real SQLite
files rather than mocking rows.

Creation command:

```bash
uv run python tests/agent/fixtures/build_legacy_rollout_fixtures.py --output tests/agent/fixtures/legacy
```

- [ ] **Microcycle 7.2: Pre-v2 Harness upcast**

RED node:
`tests/agent/harness/test_legacy_migration.py::test_pre_v2_harness_fixture_upcasts_to_v2_replay`
with expected unsupported schema or missing public outcome. GREEN: register the
deterministic payload upcaster; never rewrite old record bytes.

- [ ] **Microcycle 7.3: Offline union migration**

RED node:
`tests/agent/harness/test_legacy_migration.py::test_union_migration_imports_all_legacy_fact_kinds_atomically`
with expected missing interaction/approval/tool/attachment facts. GREEN: under
the maintenance lock import Turns, messages, interactions, approvals, tool
operations, and attachments in one transaction with no runtime fallback.

- [ ] **Microcycle 7.4: Dry run, backup, rollback, idempotence**

Run each node RED before implementation:

```text
test_legacy_migration.py::test_union_migration_dry_run_changes_no_bytes
test_legacy_migration.py::test_union_migration_backup_and_restore_round_trip
test_legacy_migration.py::test_union_migration_failure_rolls_back_every_fact
test_legacy_migration.py::test_union_migration_rerun_is_idempotent
```

GREEN: use explicit transaction/backup boundaries and compare the manifest's
record-chain/projection hashes after reopen.

- [ ] **Microcycle 7.5: Explicit legacy wire adapter**

RED node:
`tests/agent/test_canonical_streaming_protocol.py::test_runtime_emits_only_v2_and_legacy_sink_is_opt_in`
with expected duplicate v1 events or missing adapter. GREEN: retain enum imports
but project v1 only through `LegacyStreamProjectionSink`.

- [ ] **Microcycle 7.6: Public imports and signatures**

Independent RED/GREEN/commit nodes:

```text
tests/agent/test_public_exports.py::test_canonical_streaming_types_remain_public
tests/agent/test_public_exports.py::test_agent_arun_event_sink_and_astream_signatures_are_compatible
```

Expected missing exports or changed parameter kinds/defaults. GREEN: restore
only the documented public imports/signatures; do not expose Harness internals.

- [ ] **Run exact migration and verification commands**

```bash
migration_tmp="$(mktemp -d)"
cp tests/agent/fixtures/legacy/legacy_turnstore_checkpoint.sqlite3 "$migration_tmp/legacy.sqlite3"
uv run python scripts/migrate_agent_rollout.py "$migration_tmp/legacy.sqlite3" --dry-run --maintenance-confirmed
uv run python scripts/migrate_agent_rollout.py "$migration_tmp/legacy.sqlite3" --backup "$migration_tmp/backup.sqlite3" --maintenance-confirmed
uv run python scripts/verify_agent_rollout.py --database "$migration_tmp/legacy.sqlite3"
uv run python scripts/migrate_agent_rollout.py "$migration_tmp/legacy.sqlite3" --maintenance-confirmed
uv run python scripts/verify_agent_rollout.py --database "$migration_tmp/legacy.sqlite3"
uv run python scripts/migrate_agent_rollout.py "$migration_tmp/legacy.sqlite3" --restore "$migration_tmp/backup.sqlite3"
```

Expected: dry run reports the manifest Turn IDs without changing the copied DB;
first migration is valid and matches manifest hashes; rerun lists all Turn IDs
as skipped and preserves hashes; restore returns the copied database to the
fixture SHA-256.

- [ ] **Task 7 regression**

```bash
uv run pytest -q tests/agent/harness/test_legacy_migration.py tests/agent/harness/test_architecture_contract.py tests/agent/test_public_exports.py tests/repo/test_distribution_contract.py tests/agent/test_canonical_streaming_protocol.py
uv run lint-imports
test -z "$(git status --porcelain)"
```

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
test -z "$(git diff --name-only --diff-filter=U)"
git merge-base --is-ancestor 58e41c28 HEAD
git merge-base --is-ancestor "$(cat /tmp/praxis-canonical-main-tip)" HEAD
git status --short --branch
git log --oneline --decorate --graph -15
```

If verification repairs remain, stage them by exact path and run:

```bash
git diff --cached --check
git commit -m "fix(agent): close canonical harness verification gaps"
```

After any repair commit, rerun Steps 1–4 in full—focused tests, Ruff, mypy,
import contracts, full pytest, build, both smokes, Harness acceptance, and
benchmark validation. A repair commit is not considered verified by the command
that motivated it.

Then require:

```bash
test -z "$(git status --porcelain)"
git merge-base --is-ancestor 58e41c28 HEAD
git merge-base --is-ancestor "$(cat /tmp/praxis-canonical-main-tip)" HEAD
```

Expected: no conflict markers or unrelated changes, WIP commit `58e41c28` and
the pinned main tip remain ancestors, and every alignment change is committed on
`codex/praxis-harness-architecture`.
