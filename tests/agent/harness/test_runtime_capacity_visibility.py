"""Provider context capacity and host resource facts have separate projections."""

from dataclasses import replace

from agent_runtime.harness import HarnessModelRequest, ModelMessage, RuntimeStore
from agent_runtime.modeling.contracts import ContextCapacityPolicy, LLMStageBudget
from agent_runtime.modeling.gateway import model_request_input_text
from tests.agent.harness.test_affordable_tool_continuation import file_runner

POLICY = {"revision": "working-context-v2", "cache_continuity": "append-v1"}


def test_output_reservation_changes_metadata_without_changing_input(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "run.db") as store:
        runner, _, gateway = file_runner(store, tmp_path)

        def budget(stage, *, kwargs=None):
            output = (kwargs or {}).get("max_tokens") or 65_536
            return LLMStageBudget(max_input_tokens=1_048_576 - output - 512, max_output_tokens=output)

        gateway.effective_stage_budget = budget
        request = HarnessModelRequest(
            thread_id="thread",
            turn_id="turn",
            messages=(ModelMessage(role="user", body="Inspect data"),),
            binding_manifest={
                "execution_context_policy": POLICY,
                "context_capacity_policy": ContextCapacityPolicy().model_dump(),
            },
        )
        first = runner._model.prepare(request)
        capped = runner._model.prepare(replace(request, output_token_limit=8192))
        assert (
            first.request_ref["context_projection"]["effective_input_limit"]
            != (capped.request_ref["context_projection"]["effective_input_limit"])
        )
        assert first.dispatch_payload.request.messages == capped.dispatch_payload.request.messages
        assert first.context_hash == capped.context_hash
        first_input = model_request_input_text(
            first.dispatch_payload.request, provider="openai_compatible", supports_native_tools=True
        )
        capped_input = model_request_input_text(
            capped.dispatch_payload.request, provider="openai_compatible", supports_native_tools=True
        )
        assert "Runtime input limits" not in capped_input
        # The conservative counter includes wire settings as well as messages.
        assert first.resource_request.input_tokens == len(first_input)
        assert capped.resource_request.input_tokens == len(capped_input)
        assert "runtime_input_update" not in capped.request_ref
        assert not gateway.requests


def test_runtime_reports_exact_changed_limits_without_completion_instructions(tmp_path):
    from agent_runtime.harness.control import ContextSnapshot

    with RuntimeStore.open_sync(tmp_path / "run.db") as store:
        runner, turn, _ = file_runner(store, tmp_path)
        context = runner.restore_turn_context(turn.turn_id)
        messages = (ModelMessage(role="user", body="Complete and verify the task"),)
        clock = {"date": "2026-10-20"}
        messages, initial = runner._requests._append_runtime_state(context, messages, step=1, clock=clock)
        assert initial is not None
        messages, same = runner._requests._append_runtime_state(context, messages, step=1, clock=clock)
        assert same is None
        messages, update = runner._requests._append_runtime_state(context, messages, step=7, clock=clock)
        facts = ContextSnapshot.from_messages(messages).runtime_state["Resource limits (runtime ledger)"]
        assert facts["agent_requests_remaining_including_this"] == 2
        assert update is not None
        assert "required actions and verification" not in update.content
        assert "Prefer a final answer" not in update.content
        _, update = runner._requests._append_runtime_state(context, messages, step=8, clock=clock)
        assert (
            update.runtime_metadata["state"]["Resource limits (runtime ledger)"][
                "agent_requests_remaining_including_this"
            ]
            == 1
        )
