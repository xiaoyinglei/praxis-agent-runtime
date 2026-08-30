# Praxis Agent Naming Guide

This guide keeps agent names short, stable, and composable. Names should make
the local code readable; architecture docs explain the full pipeline.

## Core Rule

Use complete names for boundary types, short names for functions, and lifecycle
prefixes for risky values.

Good:

```python
raw_result = await run_tools_raw(...)
safe_update = compact_state(raw_update)
ref = memory.write(payload)
```

Avoid:

```python
checkpoint_safe_externalized_tool_result_output = ...
execute_observe_compact_checkpoint_safe_node(...)
```

## Lifecycle Prefixes

- `raw_`: not allowed in checkpoint state.
- `safe_`: sanitized enough for checkpoint state.
- `live_`: still referenced by current state.
- `stale_`: eligible for retention cleanup.
- `pinned_`: must not be evicted.
- `evicted_`: dropped with audit metadata.
- `missing_`: ref or data could not be resolved.

## Stable Suffixes

- `Ref`: external reference, not payload content.
- `Payload`: raw content stored in memory.
- `Record`: persisted store entry.
- `Snapshot`: bounded audit view at one point in time.
- `Policy`: limits and retention rules.
- `Guard`: invariant enforcement boundary.
- `Event`: recorded state transition.
- `Plan`: multi-step intent.
- `Step`: one plan item.
- `Call`: tool invocation request.
- `Result`: tool execution result.
- `Obs`: local short name for `StructuredObservation`.

## Function Verbs

- `build_*`: assemble a new object.
- `run_*`: execute tools or external work.
- `extract_*`: derive structured state from raw content.
- `compact_*`: summarize, externalize, or trim state.
- `cap_*`: enforce a per-channel limit.
- `advance_*`: move plan or loop state forward.
- `route_*`: choose the next graph node.
- `resolve_*`: fetch content by ref or id.
- `record_*`: append audit metadata.
- `sanitize_*`: remove raw or unsafe payloads.

## Main Loop Names

Preferred short names:

- `init_goal`
- `GoalBuilder`
- `control_turn`
- `run_tools_raw`
- `run_tools_guarded`
- `extract_obs`
- `extract_obs_legacy`
- `ObservationExtractor`
- `decide_next`
- `build_context`
- `build_answer`
- `MessageCompactor`
- `MemoryCompactor`
- `WorkingMemoryCompactor`
- `WorkingMemoryDraft`
- `RunRegistry`
- `GraphCompiler`
- `PlanTracker`

Legacy aliases may stay during migration when removing them would break tests,
imports, or checkpoint tooling.

## Avoid Generic Names

Do not use these alone across module boundaries:

- `manager`
- `handler`
- `processor`
- `runtime`
- `controller`
- `data`
- `info`
- `detail`
- `output`
- `result`
- `update`
- `context`

If the generic word is needed, qualify it:

```python
raw_error_detail
safe_error_detail
checkpoint_update
context_token_budget
memory_retention_policy
```

## Length Rule

If a function name needs more than four words, check the design first:

- The function may be doing too much.
- The invariant may need a type, `Guard`, or docstring.
- The full explanation may belong in architecture docs.

Keep graph labels stable unless there is a checkpoint migration plan. Prefer
renaming Python functions first.
