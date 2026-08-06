# Pause Reason Observability Design

## Status

Approved interactively on 2026-08-06. The user selected the additive,
end-to-end observability approach (Approach A).

## Context

The live DeepSeek model-quality report recorded all three
`approval_continue` trials as `paused` after the declared `apply_patch` had
been approved and the workspace assertion had passed. The report retained the
initial approval kind, but not the reason or request attached to the final
paused result.

The runtime already carries the missing information on the internal
`AgentRunResult.needs_user_input` field. The public `AgentResult` projects only
`human_input_request` into `pause`, so an untyped model pause becomes
`status="paused"` with `pause=None` and no public reason. The model-quality
gate consequently cannot distinguish:

- a second typed tool approval;
- an untyped model or stop-hook pause;
- a paused result with pending tools but incomplete public evidence.

## Goals

1. Preserve a paused Turn's reason on the public `AgentResult`.
2. Show that reason on CLI paused-result surfaces.
3. Record the final pause request kind, reason, and pending tool names in live
   model observations after any approval resume.
4. Render the same evidence in the human-readable approval run record.
5. Keep existing evaluator-v3 reports and baselines readable.

## Completeness requirement

This is an end-to-end contract repair, not an SDK-only patch. The change is
incomplete unless the same pause reason survives every public and evidence
boundary named in this specification: internal result, public SDK result,
shared CLI display, raw model-quality observation, report validation, and
rendered run record. YAGNI limits unrelated feature work; it must not be used
to omit one of these required surfaces or the typed, untyped, second-approval,
compatibility, bound, and redaction regressions.

## Non-goals

- Do not change pause, approval, checkpoint, resume, or completion semantics.
- Do not automatically continue an untyped pause.
- Do not approve a second tool call unless an existing caller explicitly does
  so.
- Do not change fixtures, model aliases, trial counts, thresholds, scoring, or
  evaluator version.
- Do not add checkpoint inspection or another reporting path.
- Do not rerun the live 5x3 model gate as part of this implementation.

## Chosen design

### Public result

Add an optional `needs_user_input: str | None = None` field to the end of the
public `AgentResult` dataclass and populate it from the existing internal
`AgentRunResult.needs_user_input` value.

Appending a defaulted field keeps existing direct constructors source
compatible. Typed approval callers continue to use `result.pause`; the new
field also exposes the human-readable reason for untyped pauses.

### CLI projection

When a paused result has no typed `pause` object but has
`needs_user_input`, `_display_agent_result` prints that reason exactly once.
The shared display function covers `agent run`, `agent resume`, and chat. The
existing typed-request prompt remains owned by `_handle_pause`; the new branch
must not duplicate it. Do not synthesize a decision, prompt for a new action,
or resume automatically.

### Model-quality observation

Extend `CaseObservation` with backward-compatible defaulted fields:

- `final_pause_request_kind: str | None`;
- `final_pause_reason: str | None`;
- `final_pause_tool_names: tuple[str, ...]`.

These describe the final result returned by `run_live_case`, not the initial
declared approval. Existing `approval_kind` and `approval_resumes` retain their
current meanings and continue to drive the unchanged evaluator.

For every final paused result, `final_pause_reason` comes from
`AgentResult.needs_user_input`. For a final typed pause, the request kind and
tool-name list come from `AgentResult.pause`. For an untyped pause, the request
kind remains `None` and the tool-name list is empty. Completed and failed
results store null/empty values.

The existing whole-artifact sanitizer remains the final write boundary for
environment secrets and absolute paths. The reason is also bounded before it
enters the observation to the first 2,000 Unicode code points. The pending
tool-name list is limited to the first 32 request entries; tool names already
come from the canonical runtime registry. These exact bounds make malformed
provider output unable to expand the report without limit and keep tests
deterministic.

### Report compatibility and rendering

The raw report remains schema version 1 and evaluator version 3 because this
change adds evidence without changing scoring semantics. Payload decoding
uses null/empty defaults when the three new fields are absent, so existing v3
baseline and run artifacts remain valid.

The renderer validates the optional fields when present and adds a "Final
pause" subsection to paused approval-trial records. It must use the existing
path and secret redaction boundary before writing Markdown.

## Data flow

```text
LoopPause.reason
    -> AgentRunResult.needs_user_input
    -> AgentResult.needs_user_input
       |-> CLI paused-result display
       `-> CaseObservation.final_pause_reason
            -> sanitized JSON report
            -> validated Markdown run record
```

Typed request metadata follows the existing parallel path:

```text
HumanInputRequest
    -> AgentResult.pause
    -> final_pause_request_kind + final_pause_tool_names
    -> sanitized JSON report
    -> validated Markdown run record
```

## Testing strategy

Use test-driven development in four focused slices:

1. Public result projection preserves an untyped pause reason.
2. CLI paused output displays the reason without resuming or approving.
3. `run_live_case` records typed and untyped final pause evidence after the
   one declared approval resume.
4. The report renderer accepts older v3 observations without the new fields,
   validates new field types, and renders final pause evidence safely.

Each production change begins only after its focused test fails for the
missing behavior. After focused tests pass, run changed-file Ruff and mypy,
the complete related test set, import contracts, `git diff --check`, and the
repository's full Task 9 verification gates. No live model call is required to
prove this deterministic observability change.

## Acceptance criteria

- Public callers can read the reason for typed and untyped paused results.
- CLI users see the reason for an untyped paused result.
- A post-resume second approval is distinguishable from an untyped pause in
  raw and rendered evidence.
- No new code path approves, continues, retries, or completes a paused Turn.
- Existing evaluator-v3 artifacts without the new fields still load and
  render.
- The evaluator version, metrics, thresholds, fixtures, and baseline remain
  byte-for-byte unchanged.
- Focused and full repository verification pass from a clean source tree.
