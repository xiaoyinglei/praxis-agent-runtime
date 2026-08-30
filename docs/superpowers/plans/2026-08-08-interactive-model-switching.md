# Interactive Model Switching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/model` a truthful in-chat model-selection ACI whose next linked Turn uses the selected alias while checkpoint resume remains bound to the original Turn model.

**Architecture:** Keep `configs/models.yaml` and the existing `ModelControlPlane` as the only catalog and mutable selection path. Treat model alias as an immutable per-Turn field that may change between linked Turns; keep workspace and knowledge bindings conversation-compatible, and restore exact persisted bindings for resume.

**Tech Stack:** Python 3.12, Typer, Pydantic, SQLite `TurnStore`, pytest, Ruff, mypy, uv/hatch build.

---

## File map

- `agent_runtime/cli.py` — interactive `/model` rendering and dispatch.
- `agent_runtime/agent.py` — explicit follow-up model selection and exact resume restoration.
- `agent_runtime/service.py` — candidate binding for a linked successor Turn.
- `agent_runtime/turns.py` — conversation resource compatibility contract.
- `tests/agent/test_cli_chat_commands.py` — interactive command and subsequent-Turn behavior.
- `tests/agent/test_turn_store.py` — persisted linked-Turn compatibility.
- `tests/agent/test_public_turn_api.py` or `tests/agent/test_agent_runtime_facade.py` — facade/provider selection and resume isolation.
- `README.md` — user-facing interactive chat path.
- `docs/RUNBOOK.md` — operational selection, persistence, and resume boundaries.

`configs/models.yaml`, checkpoint schemas, provider registries, and model policy
definitions are intentionally outside the authored file set.

## Task 1: Specify the interactive `/model` ACI

**Files:**
- Modify: `tests/agent/test_cli_chat_commands.py`
- Modify: `agent_runtime/cli.py`

- [ ] Add a failing test proving bare `/model` displays the current model,
  available aliases, and `/model <alias>` usage without calling `arun`.
- [ ] Add a failing test proving `/model model-b` is accepted after a completed
  Turn and the next message retains that Turn as `previous_turn_id`.
- [ ] Add a failing test proving an invalid alias reports valid aliases, keeps
  model A selected, and performs no `arun` or provider-resolution call.
- [ ] Run the exact new tests and confirm failures are caused by the existing
  frozen-switch guard and current-only bare output.
- [ ] Implement one bounded menu renderer, remove the switch guard, and route
  all switches through `Agent.switch_model()`.
- [ ] Re-run `tests/agent/test_cli_chat_commands.py` and verify GREEN.

## Task 2: Allow only model alias changes between linked Turns

**Files:**
- Modify: `tests/agent/test_turn_store.py`
- Modify: `agent_runtime/turns.py`
- Modify: `agent_runtime/service.py`

- [ ] Replace the existing broad mismatch test with focused RED tests: model A
  to model B succeeds; workspace drift fails; knowledge drift fails.
- [ ] Add a service-level RED test proving the successor Turn stores the
  service's selected alias while preserving predecessor history.
- [ ] Run the exact tests and confirm the alias case fails under full
  `RuntimeBinding` equality.
- [ ] Add one private compatibility predicate that ignores only `model_alias`.
- [ ] Make `_runtime_for_turn()` inherit predecessor resources and apply only
  the candidate model alias.
- [ ] Re-run the TurnStore and public Turn API test modules and verify GREEN.

## Task 3: Make the facade's next provider request use the explicit switch

**Files:**
- Modify: `tests/agent/test_agent_runtime_facade.py`
- Modify: `agent_runtime/agent.py`

- [ ] Add a RED integration test with a real `TurnStore` and model-control
  facade: complete model-A Turn, call `switch_model("model-b")`, run a linked
  Turn, and assert the model resolver receives `model-b` plus prior canonical
  messages.
- [ ] Add a RED test proving an invalid switch leaves the explicit follow-up
  selection at model A and causes no provider resolution.
- [ ] Record the explicit follow-up alias only after control-plane validation
  succeeds.
- [ ] Rebuild linked-Turn facades from predecessor workspace/knowledge plus the
  explicit model alias; leave ordinary follow-ups unchanged.
- [ ] Re-run the facade, public Turn, service, and chat command tests.

## Task 4: Prove checkpoint/resume and chat isolation

**Files:**
- Modify: `tests/agent/test_agent_runtime_facade.py`
- Modify: `tests/agent/test_agent_service_resume.py` if the lower boundary needs direct coverage.

- [ ] Add a RED/guard regression that pauses model-A Turn, changes mutable
  session selection to model B, resumes the Turn, and observes model A.
- [ ] Add a guard regression that two facades sharing the catalog do not share
  an in-memory explicit follow-up selection unless persistence is intentionally
  loaded by a newly constructed facade.
- [ ] Verify resume still uses `_agent_for_turn()` and the exact persisted
  `RuntimeBinding`; do not change checkpoint payloads.
- [ ] Run all resume, checkpoint, TurnStore, and model-control tests.

## Task 5: Document and smoke the public workflow

**Files:**
- Modify: `README.md`
- Modify: `docs/RUNBOOK.md`

- [ ] Document bare `/model`, `/model <alias>`, success/error output, same-chat
  context continuity, per-Turn alias binding, `/new`, and resume behavior.
- [ ] Run a source-checkout CLI smoke with an isolated model-session path and
  scripted TTY input; prove list, invalid alias, successful switch, and clean
  exit without credentials.
- [ ] Build the wheel, install it into a temporary isolated environment, and
  repeat `agent --help`, model listing/current, and interactive command smoke.
- [ ] Attempt one credential-redacted real provider CLI Turn. Record external
  authentication/network/quota failure as infrastructure evidence, never as a
  fabricated runtime success.

## Task 6: Release gates and delivery

- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run mypy`.
- [ ] Run `uv run pytest -q`.
- [ ] Run `uv run lint-imports`.
- [ ] Run `uv build`.
- [ ] Run the repository CLI/delivery/ACI/benchmark smoke commands required by
  `CLAUDE.md`.
- [ ] Run `git diff --check` and audit every changed/untracked file.
- [ ] Re-check the original checkout's `configs/models.yaml` hash and Git
  status against the recorded baseline.
- [ ] Review the complete diff against every acceptance criterion.
- [ ] Commit only intended files, push `codex/interactive-model-switching`, and
  create a pull request.
- [ ] Wait for every required PR check to pass, merge without force, then wait
  for and verify the resulting `main` workflow run.
