# Codex-Style Terminal Chat UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fragile bare terminal input and hard 180-character tool-result clipping with Unicode-safe editing and Codex-style bounded lifecycle rendering.

**Architecture:** Add one focused composer adapter and one focused terminal renderer. `agent_runtime.cli` remains the command/router layer and delegates input plus canonical `StreamEvent` projection; Harness events and durable payloads remain unchanged.

**Tech Stack:** Python 3.12, prompt_toolkit, regex, wcwidth, pytest/AnyIO, Typer

---

## File structure

- Create `agent_runtime/terminal_input.py`: bounded prompt history and `PromptSession` adapter.
- Create `agent_runtime/terminal_render.py`: Unicode-safe formatting, bounded command preview, and lifecycle renderer.
- Modify `agent_runtime/cli.py`: inject composer, delegate renderer, and connect `/verbose` for subsequent events.
- Modify `pyproject.toml` and `uv.lock`: declare the three terminal dependencies directly.
- Create `tests/agent/test_terminal_input.py`: composer/history behavior and PTY Unicode regression.
- Create `tests/agent/test_terminal_render.py`: rendering, budgets, lifecycle, and truncation semantics.
- Modify `tests/agent/test_cli_chat_commands.py` and `tests/agent/test_cli_wiring.py`: CLI wiring and compatibility expectations.

### Task 1: Unicode-safe bounded composer

- [ ] Add failing tests for oldest-first history eviction at 100 entries and 64 KiB, prompt delegation, and propagated EOF/interrupt.
- [ ] Add a PTY test that enters `你好`, sends two backspaces, submits, and asserts an empty valid Unicode value.
- [ ] Run `uv run pytest -q tests/agent/test_terminal_input.py` and confirm failures are caused by the missing adapter.
- [ ] Add direct `prompt-toolkit`, `regex`, and `wcwidth` dependencies with `uv add`.
- [ ] Implement `BoundedPromptHistory(History)` and `TerminalComposer(PromptSession)` in `agent_runtime/terminal_input.py` with constructor injection for tests.
- [ ] Run the focused tests and `uv run mypy agent_runtime/terminal_input.py`.
- [ ] Commit composer code, tests, and dependency lock changes.

### Task 2: Unicode-safe bounded formatting

- [ ] Add failing pure-function tests for JSON formatting, terminal-width wrapping, CJK, combining characters, ZWJ emoji, ANSI removal, and C0 sanitization.
- [ ] Add failing tests for eight-row head/tail retention and exact omitted-row counts.
- [ ] Run the focused tests and confirm expected failures.
- [ ] Implement grapheme iteration with `regex \X`, whole-cluster width with `wcwidth.wcswidth`, safe terminal text projection, and deterministic head/tail selection in `agent_runtime/terminal_render.py`.
- [ ] Run focused tests, Ruff, and mypy for the new module.
- [ ] Commit formatting code and tests.

### Task 3: Bounded command and progress streaming

- [ ] Add failing tests where arbitrary delta chunking reconstructs ordered stdout/stderr rows, equal adjacent deltas both survive, only six head and three tail rows render, and a 100 KiB no-newline delta leaves bounded state.
- [ ] Add failing tests that generic progress displays eight entries and reports the suppressed count at completion.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement item-scoped command accumulators with a 16 KiB partial-row budget, head/tail retention, omission counts, and completion cleanup.
- [ ] Ensure verbose mode streams supplied command deltas once and completion prints only metadata.
- [ ] Run focused tests and commit.

### Task 4: Canonical lifecycle renderer and CLI integration

- [ ] Add failing tests for best-effort start previews, fresh-process starts without previews, duplicate start/completion idempotence, 256-key lifecycle eviction, default structured previews, verbose full results, and distinct top-level ACI versus `run_command` truncation warnings.
- [ ] Add failing chat-loop tests proving composer injection and that `/verbose` changes subsequent Turn rendering without retroactive expansion.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement `TerminalToolEventDisplay` and retain `_CLIToolEventDisplay` as a CLI compatibility alias.
- [ ] Replace chat-loop `input()` with the composer, update `/verbose`, and leave approval prompts unchanged.
- [ ] Run `uv run pytest -q tests/agent/test_terminal_input.py tests/agent/test_terminal_render.py tests/agent/test_cli_chat_commands.py tests/agent/test_cli_wiring.py tests/agent/test_canonical_streaming_protocol.py`.
- [ ] Commit integration changes.

### Task 5: Documentation and release verification

- [ ] Update README chat instructions for Unicode editing, default tool summaries, `/verbose`, and explicit upstream-truncation warnings.
- [ ] Run focused CLI/Harness tests, then `uv run pytest -q` from a clean worktree.
- [ ] Run `uv run ruff check .`, `uv run mypy`, `uv run lint-imports`, `uv build`, and installed-wheel CLI smoke.
- [ ] Inspect `git diff --check`, commit documentation/fixes, merge fast-forward to local `main`, push `origin/main`, and verify both refs resolve to the same commit.
