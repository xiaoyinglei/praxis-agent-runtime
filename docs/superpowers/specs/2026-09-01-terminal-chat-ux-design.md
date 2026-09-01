# Codex-Style Terminal Chat UX Design

## Goal

Make `agent chat` behave like a mature coding-agent terminal client: Unicode-safe editing and readable, lifecycle-aware tool output without changing the canonical streaming or durable-history protocol.

## Current Failures

`agent chat` calls bare `input()` without loading a line editor. On the supported macOS runtime, a PTY reproduction shows that two backspaces after `你好` remove two UTF-8 bytes rather than two user-visible characters and leave a surrogate-containing value.

`_CLIToolEventDisplay` also flattens structured tool results into one line and clips them at 180 Python characters. The chat `/verbose` state is not connected to this renderer, so users cannot expand the live result. This is a UI projection defect; the canonical Item and durable result remain intact.

## Design

### Input composer

Add a small terminal-input adapter backed by `prompt_toolkit.PromptSession`. It owns only interactive text entry and bounded in-process history. History retains the newest 100 non-empty submissions within a 64 KiB UTF-8 budget and evicts oldest entries first; it is never written to disk. The chat loop depends on its `prompt()` method instead of `builtins.input`. EOF and interrupt behavior remains unchanged.

`prompt_toolkit`, `regex`, and `wcwidth` are direct runtime dependencies because Unicode-aware editing, grapheme segmentation, and terminal-cell measurement are product behavior rather than incidental transitive dependencies. Tests inject a lightweight prompt callable, so normal chat-loop tests do not require a real terminal.

### Tool Item renderer

Move terminal-only formatting into a focused renderer module while keeping `_CLIToolEventDisplay` as the stream-event consumer for compatibility. Each tool or command is correlated by canonical `item_id`/`tool_id` and projected as a lifecycle:

- start: tool name plus a readable argument preview when the event already contains one; durable resume may show only the tool name and does not change the canonical schema to reconstruct arguments;
- progress: incremental status or a bounded command-output preview;
- completion: success/failure plus structured result;
- replay: start and completion are idempotent by `(turn_id, item_id, event type)`; deltas are ordered, at-most-once live events and are never deduplicated by their content.

The default structured-result view uses an eight-display-row budget. It formats JSON with indentation, wraps at the terminal width from `shutil.get_terminal_size(fallback=(100, 24))`, retains both head and tail rows, and inserts an explicit `… +N lines (/verbose 查看完整结果)` marker. ANSI escape sequences and unsafe C0 controls are removed from terminal projection only. Wrapping operates on Unicode grapheme clusters (`regex` `\X`) and measures each whole cluster with `wcwidth.wcswidth`, so CJK, combining marks, and ZWJ emoji are neither split nor over-counted.

Command stdout/stderr deltas share one item-scoped bounded accumulator in event order. Default mode prints the first six completed display rows as they arrive, retains only the last three suppressed rows plus a bounded partial row, and prints the omitted-row count and retained tail at completion. A no-newline row keeps at most 16 KiB split between its head and tail and reports omitted characters. Verbose mode streams every delta supplied by the already-bounded command ACI and completion prints metadata rather than duplicating stdout/stderr. Generic tool progress prints at most eight progress messages per item and reports additional suppressed messages at completion. Item-local buffers are released on completion.

Verbose mode renders the complete structured result held by the completion event, formatted as JSON when possible. It is still subject to upstream data retention. Two upstream truncation signals are handled explicitly without recursive key searching: top-level `result.truncated=true` means ToolExecutor/ACI externalization truncation; for `run_command`, `result.structured_content.truncated=true` means the command tool bounded stdout/stderr. Each gets distinct wording, and UI folding never mutates either durable value.

Lifecycle seen-state is bounded to the most recent 256 start/completion keys. This preserves normal live/resume idempotence without unbounded memory. Durable replay cursors remain owned by the existing replay layer; the renderer does not invent delta identities.

### Mode changes

Because the current chat loop is serial, `/verbose` changes the renderer mode for subsequent Turn events. It does not retroactively expand an already completed item and cannot be entered while a Turn is running. The mode is presentation state only and is not persisted in the Turn binding.

## Boundaries

- No full-screen TUI, alternate screen, mouse support, or new streaming event types.
- No changes to provider adapters, tool execution, canonical event payloads, ACI output budgets, Turn identity, Item persistence, or replay semantics.
- No tool-specific formatting registry in this iteration; JSON-aware generic formatting is sufficient.

## Verification

- PTY integration test: insert and erase Chinese text without leaving invalid bytes.
- Composer unit tests: submission, EOF/interrupt propagation, history reuse, and oldest-first eviction by both entry and UTF-8 byte budgets.
- Renderer tests: short result, terminal-width wrapping, combining/CJK/ZWJ Unicode, ANSI/control sanitization, head/tail retention, bounded no-newline output, exact omitted-row marker, verbose expansion, the two upstream truncation paths, and bounded lifecycle state.
- Protocol regressions: fresh-process resume may omit argument preview; duplicate start/completion render once; two equal consecutive deltas both survive; stdout/stderr preserve event order and are not repeated at completion; canonical event and durable payload tests remain byte-for-byte unchanged.
- Existing canonical streaming, CLI, Harness, static, packaging, and full-suite checks remain green.
