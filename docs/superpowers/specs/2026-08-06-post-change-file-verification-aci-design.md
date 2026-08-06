# Post-change File Verification ACI Design

## Status

Approach A was selected interactively on 2026-08-06. This written design is
pending independent specification review and explicit user approval before
implementation.

## Context

The pause-observability work made the final reason for every live-model pause
visible without changing approval or resume semantics. One unchanged
DeepSeek-v4-flash 5x3 gate was then run from clean source commit
`9abdaea28d3e7a2b618b16e54cc81006ea5d2737`.

The run was conclusive and isolated a different defect:

- every approval trial successfully applied the requested patch;
- every workspace assertion passed;
- the model then successfully read the changed file;
- the model attempted the same stable read again;
- the runtime correctly returned `repeated_inspection`;
- one trial requested another `apply_patch`, one eventually finished after two
  additional searches, and one escalated to approval-gated `run_command`.

The unchanged gate therefore failed on task success and excess tool/model
calls, not on patch application or approval continuation. The current generic
prompt says to run real verification after editing, while neither the working
state nor resident tool documentation distinguishes literal file-state proof
from behavioral code verification. The repeated-inspection feedback also
offers several new actions without making the safe terminal action explicit.

A historical, later-reverted change (`dc41b196`) only told the model not to
repeat an identical read. It did not add runtime-owned post-change evidence,
distinguish verification scopes, or prohibit a redundant second mutation or
command escalation. It is evidence of prior pressure at this boundary, not an
implementation to restore.

## Problem statement

After a real workspace change and a successful targeted content inspection,
the runtime knows that the file has been read in the new workspace state. It
does not currently project that temporal fact to the model. The model must
infer it from a growing transcript and can treat a literal content task as if
it still needs a behavioral test.

The repair must close the complete agent-computer-interface boundary:

1. project runtime-owned post-change file-inspection evidence dynamically;
2. define which evidence proves file contents and which proves code behavior;
3. make the generic prompt and resident tool descriptions teach the same
   verification ladder;
4. make a blocked repeated inspection direct the model toward completion when
   the existing result already satisfies the task;
5. prohibit a repeat write or process-execution request whose only purpose is
   reconfirming unchanged file content.

This is not permission to claim semantic success automatically. The runtime
can prove when and where an inspection happened, but only the model can assess
whether the returned content satisfies the user's literal requirement.

## Goals

1. Add a bounded dynamic `post_change_file_inspection` signal to the existing
   runtime evidence projected in `working_state`.
2. Bind that signal to the latest runtime-observed workspace change and a
   later successful inspection of the same concrete file.
3. Explicitly separate file-content inspection from behavioral command
   verification across the prompt, tool documentation, and runtime feedback.
4. Tell the model to finish immediately when existing post-change content
   evidence satisfies a literal file task and no distinct requirement remains.
5. Prevent redundant reconfirmation by repeated reads, repeated writes, or
   escalation to `run_command`, while preserving every approval boundary.
6. Prove the deterministic behavior with focused regressions, the complete
   Task 9 gates, and exactly one unchanged live 5x3 gate on the new clean
   commit.

## Non-goals

- Do not automatically mark a task complete or synthesize a final answer.
- Do not inspect file contents inside the runtime to decide semantic success.
- Do not approve, resume, retry, hide, or execute a requested tool
  automatically.
- Do not weaken `ToolPolicy`, checkpoint/resume behavior, stop hooks, or
  `verification_after_change` enforcement.
- Do not treat a content read or search as a behavioral test, lint, type-check,
  or build.
- Do not infer a precise changed file from an unscoped command that only
  exposes a whole-workspace tree hash.
- Do not add a second planner, verification manager, or persistent state
  machine.
- Do not change model aliases, fixtures, baseline, trial count, thresholds,
  evaluator version, or scoring.

## Invariants

### Runtime authority

- A workspace change exists only when `runtime_workspace_change(result)`
  accepts executor-owned before/after evidence.
- A post-change inspection exists only for a non-error `read_file` or
  `search_text` result after the latest accepted change.
- Canonical validated tool arguments and runtime-normalized result paths are
  the only path inputs. Model prose and plan claims never create this signal.
- An inspection before the latest change is stale and must not count.
- A failed inspection, `list_files`, `find_tools`, or an unrelated-file read
  must not count.

### Verification scope

- File-content evidence answers: "What bytes or text are now observable in
  this file?"
- Behavioral command evidence answers: "Does the changed code pass a
  recognized test, lint, type-check, or build command?"
- File-content evidence never satisfies a pending
  `verification_after_change` runtime constraint.
- A literal replacement, deletion, or formatting task may finish after one
  targeted post-change content inspection if that result demonstrates the
  requested state and no other requirement remains.
- A behavioral code change still requires the narrowest relevant real command
  whenever the task or runtime contract requires behavioral verification.

### Safety

- Dynamic guidance is advisory model context, not an execution shortcut.
- Approval-gated writes and commands remain approval-gated.
- The runtime must fail closed when it cannot bind a changed file precisely.
- Existing evidence must be reused; no new mutation or command is justified
  solely by a desire to reconfirm an unchanged successful content result.

## Chosen design

### 1. Keep one working-state projection

Extend the existing `_working_state_message` projection in
`agent_runtime/core/llm_providers.py`. Do not introduce a parallel memory or
completion subsystem. The new object lives under
`working_state.runtime_evidence`, next to workspace-change and command-
verification call IDs.

When the latest accepted change has a concrete file target, project:

```json
{
  "post_change_file_inspection": {
    "authority": "runtime",
    "latest_change_tool_call_id": "change-call-id",
    "changed_paths": ["relative/path.py"],
    "observation": "observed",
    "inspection_tool_call_ids": ["read-call-id"],
    "inspected_paths": ["relative/path.py"],
    "scope": "file_content",
    "semantic_target_satisfied": "not_evaluated"
  }
}
```

Before a related post-change inspection succeeds, use
`observation="pending"` with empty inspection IDs and paths. If no precise
changed path can be grounded, omit the object rather than imply file-level
knowledge.

The object describes observation history only. In particular,
`semantic_target_satisfied="not_evaluated"` prevents the field name from being
misread as a runtime completion verdict.

The projection is bounded to the latest change, at most eight inspection call
IDs, and at most eight normalized workspace-relative paths. It never repeats
file contents, diffs, command output, absolute paths, or model-authored prose.

### 2. Ground the changed file fail closed

Add one small runtime-evidence helper at the existing observation boundary,
using the accepted `ToolResult` and its canonical `ToolCall`:

1. Reject the result unless `runtime_workspace_change` confirms a change.
2. For the resident `apply_patch`, normalize its canonical `file_path`. The
   executor-owned workspace tree delta proves a real change, and the validated
   single-file call identifies its target.
3. Preserve a concrete legacy path returned by accepted protected metadata.
4. For a whole-workspace write with no precise target, return no changed file
   path. Do not guess from command text or later model claims.

This helper may be private or module-internal; it must not create a new public
SDK contract.

### 3. Detect only inspections after the latest change

Starting immediately after the latest change result, scan successful
`read_file` and `search_text` results in order:

- `read_file` relates to the change only when its normalized canonical `path`
  equals a grounded changed path.
- `search_text` relates when its normalized canonical file target equals a
  changed path or a runtime-returned match path equals a changed path.
- A search targeted exactly at the changed file counts even when it returns no
  matches, because absence can be the requested literal condition. A broad
  empty directory search does not establish a file binding.
- Partial reads still count as inspections, not as semantic success. The model
  must decide whether the observed range contains enough evidence.
- If a later real change occurs, discard every prior inspection for this
  projection and recompute from the new change boundary.

### 4. Add conditional completion guidance

When related post-change file-content evidence is observed and no required
`verification_after_change` constraint is pending, add a compact sibling
object under `runtime_evidence`:

```json
{
  "completion_guidance": {
    "authority": "runtime",
    "condition": "literal_file_task_and_existing_result_satisfies_target",
    "action": "finish",
    "prohibited_reconfirmation": [
      "repeat_inspection",
      "repeat_mutation",
      "run_command_only_to_reconfirm_file_content"
    ]
  }
}
```

This guidance is conditional rather than a completion decision. If a pending
behavioral verification constraint exists, omit it; the existing
`runtime_requirements` projection continues to identify the required command
evidence. If the task itself makes a behavioral claim even without an explicit
runtime constraint, the system prompt still requires the narrowest relevant
command.

### 5. Replace the ambiguous generic verification rule

Update `GENERIC_SYSTEM_PROMPT` with one consistent verification ladder:

1. For a literal file-content task, one targeted read or search after the
   successful write can be sufficient. If it shows the requested state and no
   distinct requirement remains, finish directly.
2. For a behavioral code change, run the narrowest relevant recognized test,
   lint, type-check, or build command.
3. A pending runtime requirement has authority over either default.
4. Reuse successful evidence from the unchanged workspace. Do not repeat the
   edit or escalate to command execution solely to confirm the same file
   content again.

This replaces the unconditional implication in "run the narrowest real
verification immediately". It does not lower behavioral verification
standards; it makes the kind of evidence proportional to the claim.

### 6. Teach the same rule at the tool boundary

Keep tool descriptions concise, but make the resident ACI self-contained:

- `apply_patch`: after a successful literal content edit, use at most one
  targeted content inspection; never reapply the patch merely to reconfirm it.
- `read_file`: a targeted post-change read can confirm literal file content;
  reuse the result while the workspace is unchanged.
- `search_text`: a targeted post-change search can confirm presence or absence
  of literal content; do not repeat it against unchanged state.

The descriptions must continue to document existing inputs, bounds, and
effects. No input/output schema, effect, approval classification, or execution
revision changes are needed because execution semantics do not change.

### 7. Make `repeated_inspection` terminal when appropriate

Replace the current open-ended error text with direct recovery guidance:

> This exact read-only inspection already succeeded in the current unchanged
> workspace state. Reuse that result. If it shows the requested state and no
> distinct requirement remains, finish now. Do not repeat the mutation or
> request `run_command` solely to reconfirm it. Use another tool only for a
> specific unmet requirement.

Also add bounded structured content so models that attend more reliably to
tool payloads receive the same ACI:

```json
{
  "repeated_inspection": true,
  "previous_tool_call_id": "prior-call-id",
  "recommended_action": "finish_if_existing_result_satisfies_task",
  "do_not_escalate_for_reconfirmation": true
}
```

The guard still blocks only the same stable inspection in the same delivery
cycle. It does not block narrowed arguments, a genuinely different file, or a
command required by a specific behavioral contract.

## Decision flow

```text
runtime-observed workspace change
    |
    +-- no precise changed path --> no file-level signal; existing rules apply
    |
    `-- precise changed path
          |
          +-- no later related successful read/search
          |      --> post_change_file_inspection = pending
          |
          `-- later related successful read/search
                 --> post_change_file_inspection = observed
                        |
                        +-- verification_after_change pending
                        |      --> command evidence remains required
                        |
                        `-- no pending command constraint
                               --> conditional finish guidance
                                      |
                                      +-- literal target shown --> finish
                                      `-- behavioral claim remains --> narrow command
```

## ACI consistency table

| Surface | File-content task | Behavioral code task | Redundant reconfirmation |
| --- | --- | --- | --- |
| Working state | Exposes post-change related inspection | Keeps command requirement separate | Emits conditional finish guidance only when safe |
| System prompt | One targeted post-change read/search may suffice | Requires narrow recognized command | Forbids repeat read/write or command-only reconfirmation |
| `apply_patch` docs | Directs one targeted content check | Does not claim behavior is verified | Forbids reapplying the patch to reconfirm |
| `read_file` / `search_text` docs | Identifies content-proof scope | Does not substitute for tests | Requires reuse on unchanged state |
| `repeated_inspection` | Says finish if existing result satisfies target | Allows a specifically required different verification | Forbids repeat mutation or process escalation solely to reconfirm |

## Testing strategy

Implementation uses test-driven development. Each production slice begins
with a focused failing regression.

### Runtime evidence tests

1. A successful `apply_patch` followed by a successful same-file `read_file`
   projects `observation="observed"`, the correct change and inspection IDs,
   normalized relative paths, file-content scope, and no semantic-success
   claim.
2. A same-file `search_text` after the change also projects observed evidence,
   including an exact-file zero-match absence check.
3. A pre-change read, failed read, unrelated-path read, `list_files`, broad
   empty search, and read made stale by a later change do not project observed
   evidence.
4. A whole-workspace change with no precise grounded target omits the
   file-level object.
5. IDs and paths are deterministically bounded; no content or absolute paths
   enter the new projection.

### Completion-guidance tests

6. Observed post-change content evidence with no pending command constraint
   produces the conditional finish/no-reconfirmation guidance.
7. A pending `verification_after_change` constraint suppresses that guidance
   and remains pending until a recognized successful post-change command.
8. File-content evidence never populates `verification_tool_call_ids` or marks
   behavioral verification observed.

### Prompt and tool-contract tests

9. The generic prompt explicitly distinguishes literal content proof from
   behavioral command verification and preserves runtime-requirement authority.
10. The prompt forbids repeated mutation and command escalation used only for
    unchanged-content reconfirmation.
11. `apply_patch`, `read_file`, and `search_text` descriptions carry bounded,
    consistent post-change guidance without changing schemas or effects.

### Runtime feedback and safety tests

12. The repeated-inspection error text and structured payload direct the model
    to finish when the existing result is sufficient and forbid redundant
    mutation/command escalation.
13. A genuinely different inspection remains executable.
14. Existing write and command approval, pause, checkpoint, and resume tests
    remain unchanged and green.

## Verification and rollout

After focused tests pass:

1. run changed-file Ruff and mypy plus the complete related test modules;
2. run import contracts and `git diff --check`;
3. run every full Task 9 gate from a clean implementation commit;
4. confirm protected fixtures, baseline, model configuration, trial count,
   evaluator version, and thresholds are unchanged;
5. run exactly one new DeepSeek-v4-flash 5x3 gate from that clean commit;
6. classify rate limiting as inconclusive and do not alter inputs or result-shop;
7. if the conclusive gate still fails, retain and inspect its evidence before
   designing any further change.

## Acceptance criteria

- The model receives runtime-owned, latest-change-bound file-inspection state
  without a fabricated semantic-success claim.
- Literal file tasks have a clear terminal path after one successful targeted
  post-change inspection.
- Behavioral changes and explicit `verification_after_change` constraints
  still require real command evidence.
- A repeated stable inspection explicitly directs completion and explicitly
  rejects repeated mutation or command-only reconfirmation.
- No new path approves, retries, resumes, executes, or auto-completes a tool or
  Turn.
- Approval/resume safety regressions and every full Task 9 gate pass.
- The unchanged live 5x3 gate passes on the new clean source, or its conclusive
  failure is reported without weakening any protected input.
