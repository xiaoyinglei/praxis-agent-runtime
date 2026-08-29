# Canonical Streaming Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement one Codex-style `Turn -> Item -> Delta -> Completed Item -> durable history` protocol across the existing Agent runtime.

**Architecture:** Extend the existing `StreamEvent` envelope with canonical lifecycle and item identity, record ordered start/pause/resume/cancellation/final lifecycle plus completed items in `TurnStore`, and route model/tool/command producers through the existing AgentLoop sink. Preserve legacy imports and provide one explicit legacy wire projection without emitting duplicate events.

**Tech Stack:** Python 3.12, asyncio, dataclasses, SQLite, Pydantic, pytest

---

### Task 1: Canonical event contract

**Files:**
- Modify: `rag/agent/streaming/events.py`
- Modify: `agent_runtime/__init__.py`
- Test: `tests/agent/test_agent_runtime_facade.py`
- Test: `tests/agent/test_canonical_streaming_protocol.py`

- [ ] Write failing public-contract tests for `protocol_version=2`, `TurnItemKind`, `ItemDeltaKind`, `ItemStatus`, canonical lifecycle types, stable item fields, exact wire values, and JSON-safe serialization.
- [ ] Run the focused tests and verify failures are caused by missing protocol types.
- [ ] Add the typed v2 contract and the single explicit legacy projection adapter; do not dual-emit.
- [ ] Run the focused tests to green.

### Task 2: Durable completed-item history

**Files:**
- Modify: `rag/agent/turns.py`
- Create: `rag/agent/streaming/history.py`
- Test: `tests/agent/test_turn_store.py`
- Test: `tests/agent/test_canonical_streaming_protocol.py`

- [ ] Write failing tests for one shared durable ordinal across lifecycle and items, identical idempotent replay excluding delivery timestamps, divergent duplicate rejection, database reopen, and interleaved lifecycle/item replay.
- [ ] Add fault-injection tests proving completed item plus model-message projection are atomic and terminal/pause status is durable before event delivery.
- [ ] Add start-transaction and resume-claim fault tests plus a real pre-v2 SQLite migration fixture covering deterministic legacy projection and terminal reason.
- [ ] Run the tests and verify the completed-item schema/API is absent.
- [ ] Add transactional Turn columns, `agent_turn_items`, and `agent_turn_lifecycle` schemas, shared ordinal allocation, legacy backfill, and history projection.
- [ ] Run migration, reopen, and replay tests to green.

### Task 3: Turn and model-item lifecycle

**Files:**
- Modify: `rag/agent/loop/runtime.py`
- Modify: `rag/agent/core/llm_providers.py`
- Modify: `rag/providers/llm_gateway.py`
- Modify: `rag/agent/service.py`
- Modify: `rag/agent/streaming/sink.py`
- Test: `tests/agent/test_agent_loop_runtime.py`
- Test: `tests/agent/test_public_turn_api.py`
- Test: `tests/agent/test_canonical_streaming_protocol.py`

- [ ] Write failing tests for exactly one start/final lifecycle across pause -> reopen -> resume -> pause -> resume -> complete, stable text/reasoning/plan item IDs, completed authoritative text, honest single-delta fallback, and durable sink ordering.
- [ ] Add partial-model-error and cancel-after-delta tests requiring failed/cancelled item completion.
- [ ] Add tool-only, mixed text-plus-tool-call, and reopen-before-resume tests proving one authoritative agent-message item projects the exact assistant message.
- [ ] Run them red.
- [ ] Route text, reasoning, and proposed-plan channels through one provider delta ACI and remove fabricated 20-character fallback timing.
- [ ] Complete started model items and emit one terminal Turn event.
- [ ] Run focused model/Turn tests to green.

### Task 4: Plan-item lifecycle

**Files:**
- Modify: `rag/agent/loop/runtime.py`
- Modify: `rag/agent/core/llm_providers.py`
- Modify: `rag/providers/llm_gateway.py`
- Modify: `rag/agent/cli.py`
- Test: `tests/agent/test_update_plan_surfaces.py`
- Test: `tests/agent/test_cli_wiring.py`
- Test: `tests/agent/test_canonical_streaming_protocol.py`

- [ ] Write failing provider-to-CLI tests for native plan start/delta/completion identity and canonical update-plan snapshots.
- [ ] Run them red.
- [ ] Convert provider plan chunks and canonical plan revisions to plan items without duplicate legacy events.
- [ ] Run plan and CLI tests to green.

### Task 5: Tool progress ACI and command deltas

**Files:**
- Modify: `rag/agent/tools/tool.py`
- Modify: `rag/agent/tools/permissions.py`
- Modify: `rag/agent/tools/executor.py`
- Modify: `rag/agent/tools/builtins/shell.py`
- Modify: `rag/agent/loop/runtime.py`
- Test: `tests/agent/test_single_tool_contract.py`
- Test: `tests/agent/test_single_tool_executor.py`
- Test: `tests/agent/test_builtin_coding_tools.py`
- Test: `tests/agent/test_tool_process_cancellation.py`
- Test: `tests/agent/test_canonical_streaming_protocol.py`

- [ ] Write failing tests for the optional streaming-runner contract, progress delivery during execution, approval-before-start, and one item per execution attempt.
- [ ] Write a failing command test that observes stdout before process completion and distinguishes stderr.
- [ ] Run them red.
- [ ] Add the explicit progress callback ACI and concurrent bounded pipe readers.
- [ ] Preserve timeout, process-group cancellation, final output bounds, and outcome-unknown semantics; reconciliation creates a child decision item without rerunning the tool or mutating the original item.
- [ ] Assert one `run_command` call produces exactly one completed `command` item.
- [ ] Run tool, command, and cancellation tests to green.

### Task 6: End-to-end bounded transport and cancellation

**Files:**
- Modify: `rag/providers/llm_gateway.py`
- Modify: `rag/agent/streaming/sink.py`
- Modify: `rag/agent/service.py`
- Test: `tests/agent/test_canonical_streaming_protocol.py`
- Test: `tests/agent/test_public_turn_api.py`

- [ ] Write failing timeout-bounded tests for provider-to-async backpressure, a permanently blocked synchronous provider, a full queue followed by `astream.aclose()`, a raising live sink, event-loop shutdown, completion persistence before live forwarding, and interrupted replay with no item after the final Turn event.
- [ ] Run them red.
- [ ] Replace the unbounded provider bridge with a bounded daemon producer and optional provider cancellation ACI; never join a stuck synchronous producer on Turn cancellation.
- [ ] Put durable commits upstream of live delivery and make queue close independent of queue capacity.
- [ ] Run focused transport/cancellation tests to green.

### Task 7: Compatibility and verification

**Files:**
- Modify only files required by failing compatibility tests.
- Test: `tests/agent/`

- [ ] Run canonical protocol, facade, CLI, loop, TurnStore, tool, and checkpoint suites.
- [ ] Fix only regressions caused by the protocol migration.
- [ ] Run root test discovery and static checks used by the repository.
- [ ] Inspect `git diff --check`, `git diff --stat`, and `git status --short`.
- [ ] Confirm `configs/models.yaml` remains untouched.
