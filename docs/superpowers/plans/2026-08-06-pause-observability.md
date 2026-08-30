# Pause Reason Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry every final pause reason through the public SDK, shared CLI display, raw model-quality evidence, and rendered run record without changing pause or approval behavior.

**Architecture:** Reuse the existing internal `AgentRunResult.needs_user_input` value as the single reason source. Project it additively onto `AgentResult`, capture final typed-request metadata at the model-quality boundary, and keep report decoding backward compatible. CLI and Markdown remain projections only; neither may resume, approve, retry, or complete a Turn.

**Tech Stack:** Python 3.12, frozen dataclasses, Pydantic internal results, Typer CLI, pytest, Ruff, mypy.

---

## File map

- `agent_runtime/result.py` — public immutable result projection.
- `agent_runtime/cli.py` — shared CLI result rendering for run, resume, and chat.
- `scripts/agent_model_quality_gate.py` — bounded final-pause observation and backward-compatible payload decoding.
- `scripts/render_model_quality_report.py` — report validation and safe Markdown rendering.
- `tests/agent/test_public_result_contract.py` — public DTO and projection regressions.
- `tests/agent/test_cli_wiring.py` — shared CLI display and no-duplication regressions.
- `tests/agent/test_model_quality_gate.py` — typed/untyped final pause, bounds, compatibility, and sanitization regressions.
- `tests/agent/test_model_quality_report.py` — old-artifact compatibility, field validation, and rendered evidence regressions.

The fixture, baseline, evaluator version, model catalog, loop, service,
checkpoint code, and approval policy are intentionally outside the authored
file set.

## Task 1: Preserve the internal pause reason on the public SDK result

**Files:**
- Modify: `tests/agent/test_public_result_contract.py`
- Modify: `agent_runtime/result.py`

- [ ] **Step 1: Write public-surface and untyped-pause projection tests**

Update the stable field assertion to append `needs_user_input`, then add a
focused projection test using the real internal DTO:

```python
def test_internal_projection_preserves_untyped_pause_reason() -> None:
    raw = AgentRunResult(
        turn_id="turn-untyped-pause",
        status="paused",
        needs_user_input="Choose a target branch before continuing.",
        human_input_request=None,
    )

    public = AgentResult._from_internal(raw)

    assert public.status == "paused"
    assert public.pause is None
    assert public.needs_user_input == "Choose a target branch before continuing."
```

Extend the existing typed approval projection test so it also proves the same
human-readable reason is retained while `public.pause` remains the canonical
typed request.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest -q \
  tests/agent/test_public_result_contract.py::test_public_result_dtos_are_frozen_and_have_the_stable_surface \
  tests/agent/test_public_result_contract.py::test_internal_projection_preserves_untyped_pause_reason
```

Expected: FAIL because `AgentResult` does not expose
`needs_user_input`.

- [ ] **Step 3: Add the backward-compatible public field and projection**

Append the field after all required dataclass fields:

```python
@dataclass(frozen=True, slots=True)
class AgentResult:
    # existing fields stay in their current order
    plan_events: tuple[PlanEvent, ...]
    needs_user_input: str | None = None
```

Populate it only from the existing internal contract:

```python
return cls(
    # existing projections
    plan_events=tuple(event.model_copy(deep=True) for event in result.plan_events),
    needs_user_input=result.needs_user_input,
)
```

Do not derive a reason from `answer`, diagnostics, or tool results.

- [ ] **Step 4: Run the public result contract tests and verify GREEN**

Run:

```bash
uv run pytest -q tests/agent/test_public_result_contract.py
```

Expected: all tests PASS.

- [ ] **Step 5: Commit the SDK contract**

```bash
git add agent_runtime/result.py tests/agent/test_public_result_contract.py
git commit -m "feat(agent): expose paused turn reason"
```

## Task 2: Show an untyped pause reason once on every CLI command surface

**Files:**
- Modify: `tests/agent/test_cli_wiring.py`
- Modify: `agent_runtime/cli.py`

- [ ] **Step 1: Extend the CLI result helper and write display regressions**

Allow `_result` to accept `pause` and `needs_user_input`. Add two tests:

```python
def test_cli_shows_untyped_pause_reason_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _display_agent_result(
        _result(
            status="paused",
            needs_user_input="Choose a target branch before continuing.",
        ),
        verbose=False,
    )

    output = capsys.readouterr().out
    assert output.count("Choose a target branch before continuing.") == 1


def test_cli_does_not_duplicate_typed_pause_question(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pause = AgentPause(
        request_id="request-1",
        kind="tool_approval",
        question="Allow apply_patch?",
    )
    _display_agent_result(
        _result(
            status="paused",
            pause=pause,
            needs_user_input="Allow apply_patch?",
        ),
        verbose=False,
    )

    assert "Allow apply_patch?" not in capsys.readouterr().out
```

The second assertion keeps the existing typed prompt owned by
`_handle_pause`; `_display_agent_result` must not duplicate it.

- [ ] **Step 2: Run the new CLI tests and verify RED**

Run:

```bash
uv run pytest -q \
  tests/agent/test_cli_wiring.py::test_cli_shows_untyped_pause_reason_once \
  tests/agent/test_cli_wiring.py::test_cli_does_not_duplicate_typed_pause_question
```

Expected: the untyped test FAILS because the reason is not displayed; the
typed no-duplication test may already pass and remains a guardrail.

- [ ] **Step 3: Add one shared projection-only CLI branch**

In `_display_agent_result`, render only untyped paused reasons:

```python
if (
    result.status == "paused"
    and result.pause is None
    and result.needs_user_input
):
    print(f"\n暂停原因: {result.needs_user_input}")
```

Do not modify `_handle_pause`, resume actions, checkpoint state, exit codes, or
interactive approval options.

- [ ] **Step 4: Run CLI wiring and resume tests and verify GREEN**

Run:

```bash
uv run pytest -q \
  tests/agent/test_cli_wiring.py \
  tests/agent/test_agent_cli_resume.py
```

Expected: all tests PASS and typed approval output remains unchanged.

- [ ] **Step 5: Commit the CLI projection**

```bash
git add agent_runtime/cli.py tests/agent/test_cli_wiring.py
git commit -m "feat(cli): display untyped pause reasons"
```

## Task 3: Capture bounded final-pause evidence in the live gate

**Files:**
- Modify: `tests/agent/test_model_quality_gate.py`
- Modify: `scripts/agent_model_quality_gate.py`

- [ ] **Step 1: Write final-pause extraction tests**

Add a helper-level test with a real public `AgentResult` or a shape-equivalent
`SimpleNamespace` that proves:

```python
kind, reason, tool_names = module._final_pause_evidence(result)

assert kind == "tool_approval"
assert reason == "x" * 2000
assert tool_names == tuple(f"tool_{index}" for index in range(32))
```

The input reason contains 2,001 code points and the typed request contains 33
tool summaries. Add a second case for `status="paused", pause=None` and assert
that kind is `None`, the reason survives, and tool names are empty. Add a done
case proving all three values are null/empty.

- [ ] **Step 2: Extend the real gate-runner fake tests**

Extend
`test_run_live_case_approves_only_the_declared_write_and_refuses_follow_up_tools`
so its post-resume typed pause exposes `needs_user_input` and asserts:

```python
assert observation.final_pause_request_kind == "tool_approval"
assert observation.final_pause_reason == "Allow run_command?"
assert observation.final_pause_tool_names == ("run_command",)
```

Add a sibling test whose first resume returns `status="paused"`, `pause=None`,
and a non-empty reason. Assert that only the declared `allow_once` resume was
performed and the final observation records an untyped pause without
synthesizing another action.

- [ ] **Step 3: Write payload compatibility and sanitization regressions**

Add tests proving:

- `_observation_from_payload` defaults missing new fields to `None`, `None`,
  and `()` for existing v3 artifacts;
- present fields round-trip through `CaseObservation.payload()`;
- the existing artifact writer redacts an environment secret and absolute path
  embedded in `final_pause_reason`;
- `EVALUATOR_VERSION` remains exactly `agent_model_quality_gate_v3`.

- [ ] **Step 4: Run the new gate tests and verify RED**

Run the exact new test nodes plus the existing evaluator-version test.

Expected: FAIL because the observation fields and extraction helper do not
exist.

- [ ] **Step 5: Implement bounded extraction and backward-compatible decoding**

Add constants:

```python
_MAX_FINAL_PAUSE_REASON_CHARS = 2000
_MAX_FINAL_PAUSE_TOOL_NAMES = 32
```

Append defaulted fields to `CaseObservation`:

```python
final_pause_request_kind: str | None = None
final_pause_reason: str | None = None
final_pause_tool_names: tuple[str, ...] = ()
```

Add one pure helper:

```python
def _final_pause_evidence(
    result: AgentResult,
) -> tuple[str | None, str | None, tuple[str, ...]]:
    if result.status != "paused":
        return None, None, ()
    reason = result.needs_user_input
    pause = result.pause
    return (
        None if pause is None else pause.kind,
        None if reason is None else reason[:_MAX_FINAL_PAUSE_REASON_CHARS],
        (
            ()
            if pause is None
            else tuple(
                item.tool_name
                for item in pause.tool_calls[:_MAX_FINAL_PAUSE_TOOL_NAMES]
            )
        ),
    )
```

Call it after the final `result` is known and pass the values into
`CaseObservation`. In `_observation_from_payload`, accept absent fields with
null/empty defaults, validate a present tool-name sequence with the existing
string-sequence helper, and do not change scoring.

- [ ] **Step 6: Run all model-quality gate tests and verify GREEN**

Run:

```bash
uv run pytest -q tests/agent/test_model_quality_gate.py
```

Expected: all tests PASS, including the existing one-resume security test.

- [ ] **Step 7: Commit the gate evidence contract**

```bash
git add scripts/agent_model_quality_gate.py tests/agent/test_model_quality_gate.py
git commit -m "feat(evals): record final pause evidence"
```

## Task 4: Validate and render the final pause without breaking old reports

**Files:**
- Modify: `tests/agent/test_model_quality_report.py`
- Modify: `scripts/render_model_quality_report.py`

- [ ] **Step 1: Write old-artifact and paused-evidence rendering tests**

Keep `_report_payload` without the new fields in one test and prove it still
renders. In a deep-copied payload, make the approval observation paused and add:

```python
observation.update(
    final_pause_request_kind="tool_approval",
    final_pause_reason="Allow run_command for verification?",
    final_pause_tool_names=["run_command"],
)
```

Update its score to a consistent failed result and assert the run record
contains a `### Final pause` section with the kind, safe reason, and pending
tool name.

- [ ] **Step 2: Write strict field-validation tests**

Parametrize invalid values for:

- non-text `final_pause_request_kind`;
- non-text `final_pause_reason`;
- a reason longer than 2,000 code points;
- a non-sequence or non-text entry in `final_pause_tool_names`;
- more than 32 tool names;
- non-empty final-pause evidence on a non-paused new observation.

Each test must assert the precise `ValueError` label so malformed evidence
cannot silently render.

- [ ] **Step 3: Write rendering redaction regression**

Put a POSIX and Windows absolute path in the final pause reason and assert both
are rendered as `[REDACTED_ABSOLUTE_PATH]`. Environment-secret replacement is
owned by the gate writer and remains covered in Task 3.

- [ ] **Step 4: Run the new renderer tests and verify RED**

Run the exact new test nodes.

Expected: FAIL because the renderer neither validates nor renders final pause
evidence.

- [ ] **Step 5: Add backward-compatible validation**

In `_validate_run_cases`, treat all three fields as optional for old v3
artifacts. When present, enforce the exact type and bounds from the gate. For a
new observation with any final-pause evidence, require
`observation_status == "paused"`. Do not require fields that are absent from
historical artifacts.

- [ ] **Step 6: Render one safe final-pause section**

For paused observations, append:

```markdown
### Final pause

- Request kind: `<value or none>`
- Reason: `<redacted safe text or none>`
- Pending tools: `<safe scalar list>`
```

Use `_safe_text`, `_scalar`, and the existing absolute-path redaction helpers.
Do not render raw context, request IDs, approval IDs, or arguments.

- [ ] **Step 7: Run all renderer tests and verify GREEN**

Run:

```bash
uv run pytest -q tests/agent/test_model_quality_report.py
```

Expected: all tests PASS, including historical v1/v2/v3 fixtures.

- [ ] **Step 8: Commit the renderer contract**

```bash
git add scripts/render_model_quality_report.py tests/agent/test_model_quality_report.py
git commit -m "feat(evals): render final pause evidence"
```

## Task 5: Verify the complete deterministic contract

**Files:** No new authored files expected.

- [ ] **Step 1: Run focused formatting, typing, and tests**

```bash
uv run ruff check \
  agent_runtime/result.py agent_runtime/cli.py \
  scripts/agent_model_quality_gate.py scripts/render_model_quality_report.py \
  tests/agent/test_public_result_contract.py tests/agent/test_cli_wiring.py \
  tests/agent/test_model_quality_gate.py tests/agent/test_model_quality_report.py
uv run mypy \
  agent_runtime/result.py agent_runtime/cli.py \
  scripts/agent_model_quality_gate.py scripts/render_model_quality_report.py
uv run pytest -q \
  tests/agent/test_public_result_contract.py \
  tests/agent/test_cli_wiring.py \
  tests/agent/test_agent_cli_resume.py \
  tests/agent/test_model_quality_gate.py \
  tests/agent/test_model_quality_report.py
```

Expected: all commands exit zero.

- [ ] **Step 2: Audit protected evidence inputs**

```bash
git diff origin/main -- \
  tests/agent/fixtures/model_quality_cases.json \
  evals/model_quality/baseline_v1.json \
  configs/models.yaml
rg -n "EVALUATOR_VERSION" scripts/agent_model_quality_gate.py
```

Expected: empty diff for fixture, baseline, and model catalog; evaluator remains
`agent_model_quality_gate_v3`.

- [ ] **Step 3: Run the full Task 9 gate**

```bash
uv run ruff check .
uv run mypy
uv run pytest -q
uv run lint-imports
uv build
uv run python scripts/agent_cli_smoke.py
uv run python scripts/agent_delivery_smoke.py --fake-model --verbose
uv run python scripts/agent_tool_aci_eval.py --fake-model --json
uv run python scripts/agent_code_benchmark.py validate \
  evals/code_agent/benchmark_v1.json --repository .
test -x /usr/bin/sandbox-exec
uv run pytest -q -rs \
  tests/agent/test_run_command_safety.py::test_run_command_default_blocks_ordinary_workspace_write \
  tests/agent/test_run_command_safety.py::test_run_command_workspace_write_never_writes_dot_git \
  tests/agent/test_run_command_safety.py::test_run_command_workspace_write_cannot_escape_workspace
uv run pytest -q -rs tests/agent/test_run_command_safety.py
git diff --check
```

Expected: every command exits zero; the three selected Seatbelt tests pass with
zero skips.

- [ ] **Step 4: Inspect the complete diff and commit any verification-only fixes**

```bash
git diff origin/main --stat
git diff origin/main --check
git status --short
```

Expected: only the approved SDK, CLI, gate, renderer, tests, spec, and plan are
changed. If verification required a code fix, repeat the affected RED/GREEN
cycle and Task 5 from Step 1 before committing.

## Task 6: Run one unchanged live DeepSeek diagnostic after the clean commit

**Files:** No repository artifact is authored by this diagnostic.

- [ ] **Step 1: Confirm clean source identity**

```bash
git status --porcelain=v1
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
```

Expected: empty status and recorded commit/tree identities.

- [ ] **Step 2: Run exactly one full 5-case x 3-trial gate to a temporary report**

```bash
uv run python scripts/agent_model_quality_gate.py gate \
  --env-file /Users/leixiaoying/LLM/RAG学习/.env \
  --model deepseek_v4_flash \
  --report /tmp/praxis-deepseek-pause-observability.json
```

Do not change the fixture, baseline, model, trial count, evaluator, or
thresholds. Do not repeatedly rerun to select a green result.

- [ ] **Step 3: Verify the observability acceptance**

For every final paused case, assert the temporary JSON contains:

- `final_pause_reason`;
- nullable `final_pause_request_kind`;
- bounded `final_pause_tool_names`.

If the gate passes, record that it is one diagnostic sample rather than a new
release benchmark. If the gate fails, use these fields to start the next
root-cause repair; do not call this observability task a model-quality pass and
do not leave the same unexplained pause unresolved.
