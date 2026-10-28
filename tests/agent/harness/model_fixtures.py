"""Well-formed model event streams for Runtime integration tests."""

from dataclasses import replace

from agent_runtime.harness.protocol import HarnessModelResponse
from agent_runtime.modeling.events import (
    ModelEventSink,
    ModelOutput,
    ModelOutputCompleted,
    ModelOutputDelta,
    ModelOutputStarted,
    ModelResponseCompleted,
    ModelResponseStarted,
    emit_model_event,
)


async def emit_partial_outputs(sink: ModelEventSink, outputs: tuple[ModelOutput, ...]) -> None:
    await emit_model_event(sink, ModelResponseStarted())
    for output in outputs:
        await emit_model_event(sink, ModelOutputStarted(replace(output, content="", arguments=None)))
        await emit_model_event(
            sink,
            ModelOutputDelta(output.output_id, output.delta_kind, output.content),
        )


async def emit_response(sink: ModelEventSink, response: HarnessModelResponse) -> None:
    await emit_model_event(sink, ModelResponseStarted(response.provider_response_id))
    for output in response.outputs:
        await emit_model_event(sink, ModelOutputStarted(replace(output, content="", arguments=None)))
        await emit_model_event(
            sink,
            ModelOutputDelta(output.output_id, output.delta_kind, output.content),
        )
        await emit_model_event(sink, ModelOutputCompleted(output))
    await emit_model_event(
        sink,
        ModelResponseCompleted(
            usage=response.usage,
            provider_response_id=response.provider_response_id,
            stop_reason=response.stop_reason or "stop",
            status=response.status,
            incomplete_reason=response.incomplete_reason,
        ),
    )
