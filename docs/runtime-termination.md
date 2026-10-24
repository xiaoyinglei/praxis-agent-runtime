# Loop termination and durable continuation

## Design boundary

The model chooses actions and proposes an answer. The runtime owns input ordering,
tool outcomes, approvals, budgets and explicit completion contracts. A read count,
an advisory plan or answer wording cannot establish whether a task made progress.
Default `done` therefore continues to mean that the Turn ended, with independent
acceptance reported separately.

Reference implementations, inspected on 2026-10-24:

- [Codex turn loop, commit 2839ed763bd3f404398e1f0c95bf4e2d172213c0](https://github.com/openai/codex/blob/2839ed763bd3f404398e1f0c95bf4e2d172213c0/codex-rs/core/src/session/turn.rs#L591-L784):
  tool follow-up and pending input keep the loop running. Stop hooks run at the
  boundary where neither requires continuation.
- [Claude Code Stop protocol](https://code.claude.com/docs/en/hooks#stop-input):
  hooks receive continuation state; eight consecutive hook continuations are
  bounded, with a tool call resetting the count. This is a public protocol
  reference, not a claim to have inspected Claude Code's private core.

## Selected changes

1. Both fresh tool execution and committed-response recovery call
   `TurnExecutor._after_tool_batch`. It reads canonical results, respects pending
   interactions and superseding input, and applies the existing repeated-failure
   policy. Recovery executes only calls lacking a committed result. An accepted
   `tool_reconciliation.result` replaces its original unknown outcome according
   to the operation's result pointer; the original remains in the audit ledger.
   Parameter and approval-denial results without an operation result pointer
   still participate in failure counting.
2. Recovery feedback references the committed model response that triggered it.
   Reopening after that feedback does not consume the grant twice. Each batch
   receives one post-tool decision, including a partially recovered batch.
3. Final proposals reference their model response; feedback references its
   completion decision. Recovery reuses an existing proposal, decision and
   feedback instead of evaluating or counting the same response again.
4. New Turns freeze `completion_continuation_policy=bounded-v1`, allowing eight
   consecutive completion rejections without a tool exchange or new user input.
   A ninth rejection fails with `completion_continuation_limit`, preserving the
   actual check's validation status and scope. A ninth **acceptance** still passes.
   Unlike an advisory Stop hook, an explicit acceptance requirement must never be
   bypassed to end a loop. Every request shares the original step/token/cost limits.
5. Tool calls and new user input reset this protocol counter. This does not certify
   progress: repeated successful reads remain available, and the global budget
   still bounds tool loops. Old Turns without the new policy keep their frozen
   limits. No historical event is rewritten and no new execution loop is added.

## Alternatives rejected

- Enforcing a number of reads, edits or matching words would turn observations
  into a task-specific progress judge and would break valid investigation.
- Raising all budgets would postpone the failure without fixing replay or
  completion feedback.
- Automatically accepting after failed checks would violate the user's explicit
  contract. The bounded state is a failure with recorded evidence, not success.

## Verification plan

- Public Agent/CLI: normal execution, recovery after tool result and recovery
  grant, repeated Stop rejection, proposal/decision/feedback crash boundaries.
- Normal counterexamples: repeated successful reads; tool exchange and steering
  reset the count; acceptance at the boundary succeeds; old Turns keep their
  policy; a real committed write executes once after reopening.
- Existing approval, reconciliation, streaming, compaction, incomplete-response,
  completion and budget regressions; Ruff, mypy and import contracts.
- Scripted models verify the protocol and real filesystem effects. They do not
  establish model reasoning quality; paid model benchmarks are not part of this
  implementation run.

## Verified on macOS, 2026-10-24

- Final Harness suite: 776 passed, including 19 new termination cases.
- Main Agent/CLI regression: 3224 passed, 76 skipped, one old feedback-payload
  assertion failed because it omitted the new source link. That assertion now
  verifies the full proposal/decision/feedback chain; the complete Session suite
  (28 cases) and final Harness suite both passed after the update.
- Maintained source/test/script Ruff, runtime mypy (129 source files), all five
  import contracts and `git diff --check` passed. The CLI acceptance tests were
  rerun with native Seatbelt access; the product sandbox was not disabled.
- Real process exit after a committed recovery grant, a real file mutation with
  exactly one execution, mixed failure batches, steering, truncated tool JSON,
  immutable budgets and old-policy recovery are covered. No paid model call or
  Linux execution was performed in this run.
