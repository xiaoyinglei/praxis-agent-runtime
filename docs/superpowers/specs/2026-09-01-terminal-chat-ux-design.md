# Codex-Style Terminal Chat UX Design

## Goal

Make `agent chat` behave like a mature coding-agent terminal client: Unicode-safe editing and readable, lifecycle-aware tool output without changing the canonical streaming or durable-history protocol.

## Current Failures

`agent chat` calls bare `input()` without loading a line editor. On the supported macOS runtime, a PTY reproduction shows that two backspaces after `你好` remove two UTF-8 bytes rather than two user-visible characters and leave a surrogate-containing value.

`_CLIToolEventDisplay` also flattens structured tool results into one line and clips them at 180 Python characters. The chat `/verbose` state is not connected to this renderer, so users cannot expand the live result. This is a UI projection defect; the canonical Item and durable result remain intact.

## Design

### Input composer

Add a small terminal-input adapter backed by `prompt_toolkit.PromptSession`. It owns only interactive text entry and persistent in-process history. The chat loop depends on its `prompt()` method instead of `builtins.input`. EOF and interrupt behavior remains unchanged.

`prompt_toolkit` is a direct runtime dependency because Unicode-aware cursor movement, deletion, paste handling, and history are product behavior rather than incidental terminal behavior. Tests inject a lightweight prompt callable, so normal chat-loop tests do not require a real terminal.

### Tool Item renderer

Move terminal-only formatting into a focused renderer module while keeping `_CLIToolEventDisplay` as the stream-event consumer for compatibility. Each tool or command is correlated by canonical `item_id`/`tool_id` and projected as a lifecycle:

- start: tool name plus a readable argument preview;
- progress: incremental status or command output;
- completion: success/failure plus structured result;
- replay: duplicate lifecycle events remain idempotent.

The default view wraps structured output as terminal lines and applies a display-row budget. If output exceeds the budget, it retains both the head and tail and inserts an explicit `… +N lines (/verbose 查看完整结果)` marker. It never cuts a grapheme or silently replaces durable data.

Verbose mode renders the complete result held by the event, formatted as JSON when possible. A `truncated=true` value originating from the tool ACI is shown separately as an upstream truncation warning; verbose mode cannot claim to recover bytes the tool never retained.

### Mode changes

`/verbose` updates the live renderer immediately. The renderer mode is presentation state only and is not persisted in the Turn binding.

## Boundaries

- No full-screen TUI, alternate screen, mouse support, or new streaming event types.
- No changes to provider adapters, tool execution, ACI output budgets, Turn identity, Item persistence, or replay semantics.
- No tool-specific formatting registry in this iteration; JSON-aware generic formatting is sufficient.

## Verification

- PTY integration test: insert and erase Chinese text without leaving invalid bytes.
- Composer unit tests: submission, EOF/interrupt propagation, and history reuse.
- Renderer tests: short result, wrapped result, Unicode, head/tail retention, exact omitted-line marker, verbose expansion, upstream truncation warning, and duplicate replay.
- Existing canonical streaming, CLI, Harness, static, packaging, and full-suite checks remain green.
