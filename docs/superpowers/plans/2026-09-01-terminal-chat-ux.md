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

- [ ] Write `test_history_evicts_oldest_by_entry_limit` and run `uv run pytest -q tests/agent/test_terminal_input.py::test_history_evicts_oldest_by_entry_limit`; expect import failure because `BoundedPromptHistory` does not exist.
- [ ] Add direct dependencies with `uv add 'prompt-toolkit>=3.0.52' 'regex>=2025.7.34' 'wcwidth>=0.2.13'`, implement only the 100-entry FIFO boundary, and rerun the exact test; expect PASS.
- [ ] Write `test_history_evicts_oldest_by_utf8_byte_limit` and `test_empty_submission_is_not_recorded`; run both and expect FAIL on the missing byte/empty rules.
- [ ] Implement the 64 KiB UTF-8 boundary plus empty filtering and rerun both; expect PASS and no history file under `tmp_path`.
- [ ] Write prompt delegation and EOF/interrupt propagation tests; run them and expect FAIL because `TerminalComposer` is missing.
- [ ] Implement `TerminalComposer` with injectable session construction and rerun; expect PASS and reuse of one history instance for the process.
- [ ] Write and run a PTY subprocess regression that enters `你好`, sends two backspaces, and submits; expect the old bare-input control to return a surrogate-containing value and the new composer command to return empty valid Unicode.
- [ ] Run `uv run pytest -q tests/agent/test_terminal_input.py && uv run mypy agent_runtime/terminal_input.py`; expect PASS.
- [ ] Commit with `git add pyproject.toml uv.lock agent_runtime/terminal_input.py tests/agent/test_terminal_input.py && git commit -m 'feat(cli): add Unicode-safe chat composer'`.

### Task 2: Unicode-safe bounded formatting

- [ ] Write ANSI/C0 sanitization tests and run their node ids; expect import failure because `safe_terminal_text` is missing. Implement only sanitization and rerun; expect PASS.
- [ ] Write CJK, combining-mark, and ZWJ wrapping tests at injected widths and run their node ids; expect FAIL because `display_rows` is missing. Implement `regex \X` segmentation plus `wcwidth.wcswidth(cluster)` wrapping and rerun; expect PASS without split clusters.
- [ ] Write JSON formatting and eight-row head/tail tests asserting the exact `… +N lines (/verbose 查看完整结果)` row; run and expect FAIL because `bounded_result_lines` is missing. Implement only formatting and deterministic middle omission, then rerun; expect PASS.
- [ ] Run `uv run pytest -q tests/agent/test_terminal_render.py -k 'sanitize or unicode or bounded_result' && uv run ruff check agent_runtime/terminal_render.py tests/agent/test_terminal_render.py && uv run mypy agent_runtime/terminal_render.py`; expect PASS.
- [ ] Commit with `git add agent_runtime/terminal_render.py tests/agent/test_terminal_render.py && git commit -m 'feat(cli): format bounded terminal results'`.

### Task 3: Bounded command and progress streaming

- [ ] Write an arbitrary-chunk test containing equal adjacent deltas and interleaved stdout/stderr; run it and expect FAIL because `BoundedCommandPreview` is missing. Implement ordered line reconstruction without content deduplication and rerun; expect PASS.
- [ ] Write the six-head/three-tail test with an exact omitted-row marker and assert command completion does not repeat stdout/stderr; run and expect FAIL. Implement suppression, tail flush, and metadata-only completion; rerun and expect PASS.
- [ ] Write a 100 KiB no-newline test asserting retained head and tail, exact omitted-character count, and internal retained bytes no greater than 16 KiB; run and expect FAIL. Implement bounded partial-row head/tail storage and rerun; expect PASS.
- [ ] Write generic-progress tests asserting eight visible messages, an exact suppressed-count completion row, and cleanup of command buffer, tool-name, and progress-count state; run and expect FAIL. Implement the counters and complete-item cleanup, then rerun; expect PASS.
- [ ] Write a verbose command test asserting each supplied delta appears exactly once and completion prints no output duplicate; run and expect FAIL, connect verbose bypass, rerun and expect PASS.
- [ ] Run `uv run pytest -q tests/agent/test_terminal_render.py -k 'command or progress'`; expect PASS.
- [ ] Commit with `git add agent_runtime/terminal_render.py tests/agent/test_terminal_render.py && git commit -m 'feat(cli): bound live tool output'`.

### Task 4: Canonical lifecycle renderer and CLI integration

- [ ] Write start tests for preview-present, preview-absent resume, same item id in different turns, and duplicate `(turn_id,item_id,ITEM_STARTED)`; run and expect FAIL because `TerminalToolEventDisplay` is missing. Implement only start projection and rerun; expect PASS.
- [ ] Write completion tests for duplicate completion and total 256-key LRU capacity across start/completion keys; run and expect FAIL. Implement the exact lifecycle key plus bounded LRU and rerun; expect PASS.
- [ ] Write top-level `result.truncated` and `run_command` `structured_content.truncated` tests with distinct exact warnings; run and expect FAIL. Implement non-recursive checks and rerun; expect PASS.
- [ ] Write a mutation regression that deep-copies `StreamEvent.data`, renders in default and verbose modes, and asserts equality afterward; add a `RolloutStore` reopen assertion that the persisted tool-result payload is identical. Run and expect FAIL until renderer integration exists; implement without mutating event mappings and rerun; expect PASS.
- [ ] Write chat-loop composer injection and `/verbose`-before-next-Turn tests; run and expect FAIL because the loop still calls `input()`. Inject one composer instance, call `event_display.set_verbose(verbose)`, retain `_CLIToolEventDisplay` as a compatibility alias, and rerun; expect PASS.
- [ ] Run `uv run pytest -q tests/agent/test_terminal_input.py tests/agent/test_terminal_render.py tests/agent/test_cli_chat_commands.py tests/agent/test_cli_wiring.py tests/agent/test_canonical_streaming_protocol.py`; expect PASS.
- [ ] Commit with `git add agent_runtime/cli.py agent_runtime/terminal_render.py tests/agent/test_terminal_render.py tests/agent/test_cli_chat_commands.py tests/agent/test_cli_wiring.py && git commit -m 'feat(cli): render Codex-style tool lifecycle'`.

### Task 5: Documentation and release verification

- [ ] Update README chat instructions for Unicode editing, default tool summaries, `/verbose`, and explicit upstream-truncation warnings; run `uv run pytest -q tests/repo/test_readme_presentation.py` and commit the documentation.
- [ ] Run exact focused regression: `uv run pytest -q tests/agent/test_terminal_input.py tests/agent/test_terminal_render.py tests/agent/test_cli_chat_commands.py tests/agent/test_cli_wiring.py tests/agent/test_canonical_streaming_protocol.py tests/agent/harness/test_tool_orchestrator.py`.
- [ ] Create a clean detached worktree with `verify_tree="$(mktemp -d)/repo"; git worktree add --detach "$verify_tree" HEAD`, run `UV_PROJECT_ENVIRONMENT='/Users/leixiaoying/.codex/worktrees/b1e9/RAG学习/.venv' uv run pytest -q` from it, then remove only that created worktree with `git worktree remove "$verify_tree"`; expect the full suite to pass.
- [ ] From the feature worktree run `uv run ruff check . && uv run mypy && uv run lint-imports && uv build && uv run python scripts/agent_cli_smoke.py && uv run python scripts/agent_delivery_smoke.py --fake-model --verbose`; expect all commands to succeed.
- [ ] Reproduce the CI wheel smoke: create a `mktemp -d` root, assert exactly one `dist/*.whl`, run `uv venv --no-project --python 3.12 "$smoke_venv"`, `uv pip install --python "$smoke_venv/bin/python" "$wheel"`, then execute installed `agent --help`, `agent model --help`, and `rag --help`; expect exit 0.
- [ ] Run `git diff --check && git status --short --branch`; commit only scoped remaining files and require a clean feature worktree.
- [ ] Verify `/Users/leixiaoying/LLM/RAG学习` is a clean `main` checkout and `main` is not ahead/behind `origin/main`; then run `git -C /Users/leixiaoying/LLM/RAG学习 merge --ff-only codex/canonical-streaming-protocol` and `git -C /Users/leixiaoying/LLM/RAG学习 push origin main`.
- [ ] Verify `git -C /Users/leixiaoying/LLM/RAG学习 rev-parse main`, `git rev-parse codex/canonical-streaming-protocol`, and `git rev-parse origin/main` are identical. Do not delete the feature worktree or any other branch without separate authorization.
