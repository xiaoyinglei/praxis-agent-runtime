# Interactive Model Switching Design

## Status

Implementation authorized on 2026-08-08. The task requires an end-to-end
implementation, documentation, release gates, pull request, merge, and
post-merge CI verification.

## Context

Praxis already has one model control plane and a partial interactive command:

- `ModelCatalog` loads chat aliases from `configs/models.yaml`;
- `ModelSessionState` persists the selected alias separately from YAML;
- `ModelPolicy` validates switch authority;
- `ModelControlPlane` lists, selects, and resolves providers;
- `agent model ...` exposes the same state through non-chat CLI commands;
- `agent chat` recognizes `/model` but currently freezes the alias after the
  first Turn.

The visible CLI restriction matches deeper runtime behavior. A follow-up Turn
is rebuilt from its predecessor's full `RuntimeBinding`, `AgentService`
discards the newly selected binding whenever `previous_turn_id` is present,
and `TurnStore` rejects any follow-up whose full runtime differs. Removing only
the CLI guard would therefore display a switch without changing the provider.

## Goals

1. Bare `/model` shows the current model, every available chat alias, and the
   direct `/model <alias>` usage.
2. `/model <alias>` and `/model switch <alias>` select through the existing
   `ModelControlPlane` and persist through the existing model-session file.
3. The next Turn in the same interactive chat keeps its conversation history
   but resolves the newly selected model.
4. An invalid or policy-rejected alias reports the error and available aliases,
   leaves the previous selection unchanged, and performs no provider request.
5. Every Turn retains an immutable model binding. Resuming a paused or
   interrupted Turn always rebuilds the model recorded on that Turn, even if
   the session selection has changed since the checkpoint was written.
6. Linked Turn history may cross a model switch, but workspace and knowledge
   bindings may not change across that link.

## Non-goals

- Do not add a second model registry, session store, provider router, or chat
  state machine.
- Do not edit `configs/models.yaml`; it remains the only alias catalog.
- Do not mutate a completed predecessor Turn when a later Turn changes model.
- Do not change checkpoint payloads or replay a paused Turn under a new model.
- Do not add an interactive picker dependency; the explicit
  `/model <alias>` path is the selection ACI.
- Do not change standalone `agent model list/current/switch` semantics.

## Chosen design

### CLI projection

One helper renders the current model plus `format_model_rows(...)` output and a
short switch hint. Bare `/model` calls it. Invalid aliases use the same helper
after printing the validation error, so users can recover without leaving
chat. Listing and validation read the catalog only; they never call
`ModelControlPlane.resolve()` or a provider gateway.

The existing post-first-Turn switch guard is removed. The chat loop continues
to track `current_turn_id`; it does not clear conversation context when the
model changes.

### Agent selection state

`Agent.switch_model()` remains the only SDK/CLI mutation path and delegates to
`ModelControlPlane.switch_model()`. After validation succeeds, the facade
records that the user explicitly selected a follow-up model. Invalid switches
never update this marker.

For an ordinary follow-up, the facade restores the predecessor's full runtime
as before. After an explicit switch, it creates the next Turn runtime by taking
the predecessor's workspace and knowledge binding and replacing only
`model_alias` with the selected alias. The marker remains active so later Turns
in that chat continue on the new alias.

### Turn persistence

`TurnStore.begin_turn()` continues to require a terminal predecessor. Runtime
compatibility is narrowed to the properties that define the conversation's
resource boundary: `workspace_path` and `knowledge`. `model_alias` is allowed
to differ because it is an immutable per-Turn choice, not a shared-session
resource.

`AgentService._runtime_for_turn()` uses the service's candidate per-Turn model
binding while inheriting the predecessor's workspace and knowledge. It rejects
any attempted non-model drift through the existing TurnStore gate.

### Resume boundary

`Agent.aresume()` continues to call `_agent_for_turn()`, which reconstructs the
facade from the exact `RuntimeBinding` persisted on the paused Turn. It does not
consult the mutable model-session selection. The existing checkpoint and
canonical transcript are resumed unchanged.

## Data flow

```text
/model <alias>
    -> Agent.switch_model
    -> ModelControlPlane.switch_model
    -> ModelPolicy + ModelCatalog validation
    -> ModelSessionState persistence
    -> explicit follow-up model selection

next user message
    -> previous_turn_id keeps canonical history
    -> new Turn RuntimeBinding(model_alias=<selected>)
    -> ModelControlPlane.resolve_for_node
    -> selected provider/model
```

Resume stays on a separate immutable path:

```text
agent resume <paused-turn>
    -> paused Turn RuntimeBinding
    -> paused Turn checkpoint
    -> original model provider
```

## Testing strategy

Use TDD across four layers:

1. CLI command behavior: bare display, successful switch after a Turn, invalid
   alias recovery, and no agent/provider call for slash commands.
2. TurnStore contract: alias changes are allowed while workspace and knowledge
   drift remain rejected.
3. Agent/service integration: a linked follow-up retains real canonical history
   and resolves the newly selected alias.
4. Resume isolation: changing the session selection after a pause does not
   change the model used to continue the paused Turn; independent chat facades
   do not leak an in-memory selection into one another.

After focused GREEN tests, run repository Ruff, mypy, full pytest,
`lint-imports`, build, source CLI smoke, installed-wheel smoke,
`git diff --check`, and a credential-redacted real CLI attempt. Provider
infrastructure failure remains inconclusive evidence rather than a code pass.

## Acceptance criteria

- Bare `/model` shows current and available aliases.
- `/model <alias>` works before or after prior completed Turns without clearing
  chat history.
- The first subsequent model request resolves the selected alias.
- Invalid aliases leave the current alias unchanged, list valid aliases, and
  make zero provider requests.
- The predecessor and successor Turns persist their own different aliases.
- Paused/interrupted Turn resume uses its original alias and checkpoint.
- Workspace/knowledge drift across `previous_turn_id` is still rejected.
- README and RUNBOOK explain the behavior and boundaries.
- All requested local/release/CI gates pass before merge and on main.
