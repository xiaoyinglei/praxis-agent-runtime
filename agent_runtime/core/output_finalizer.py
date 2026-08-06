from __future__ import annotations

import json
from collections.abc import Awaitable, Mapping
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from agent_runtime.core.context import TurnRegistry
from agent_runtime.core.llm_context import AgentLLMContextAssembler
from agent_runtime.core.llm_registry import ModelResolver
from agent_runtime.core.output_models import (
    ValidatedFinalOutput,
    output_model_path,
)
from agent_runtime.modeling.contracts import DEFAULT_LLM_STAGE_BUDGETS, LLMCallStage
from agent_runtime.modeling.gateway import LLMGateway, TokenAccounting
from agent_runtime.modeling.tokenization import TokenAccountingService, TokenizerContract

if TYPE_CHECKING:
    from agent_runtime.core.definition import AgentRuntimePolicy
    from agent_runtime.loop.state import LoopState


class OutputValidationExhaustedError(RuntimeError):
    def __init__(
        self,
        *,
        attempts: int,
        validation_errors: list[dict[str, object]],
    ) -> None:
        super().__init__(f"Structured output validation failed after {attempts} attempts")
        self.attempts = attempts
        self.validation_errors = validation_errors


class StructuredOutputFinalizer(Protocol):
    def finalize(
        self,
        *,
        definition: AgentRuntimePolicy,
        state: LoopState,
        candidate_text: str,
    ) -> BaseModel | Awaitable[BaseModel]: ...


class ModelStructuredOutputFinalizer:
    def __init__(
        self,
        *,
        gateway: LLMGateway,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self._gateway = gateway
        self._kwargs = dict(kwargs or {})
        self._assembler = AgentLLMContextAssembler(
            token_accounting=gateway.token_accounting,
            stage_budgets={
                LLMCallStage.FINAL_SYNTHESIS: gateway.effective_stage_budget(
                    LLMCallStage.FINAL_SYNTHESIS,
                    kwargs=self._kwargs,
                )
            },
        )

    @property
    def token_accounting(self) -> TokenAccounting:
        return self._gateway.token_accounting

    async def finalize(
        self,
        *,
        definition: AgentRuntimePolicy,
        state: LoopState,
        candidate_text: str,
    ) -> BaseModel:
        output_model = definition.output_model
        if output_model is None:
            raise ValueError("AgentRuntimePolicy.output_model is not configured")
        try:
            handles = TurnRegistry.get(state["run_config"].turn_id)
        except KeyError as exc:
            raise RuntimeError(f"Runtime handles missing for turn_id={state['run_config'].turn_id}") from exc

        feedback: str | None = None
        last_errors: list[dict[str, object]] = []
        max_attempts = 1 + definition.output_validation_max_retries
        for attempt_index in range(max_attempts):
            assembled = self._assembler.assemble_final_output(
                definition=definition,
                state=state,
                candidate_text=candidate_text,
                validation_feedback=feedback,
                output_schema=output_model,
            )
            try:
                result = await self._gateway.agenerate_structured(
                    stage=LLMCallStage.FINAL_SYNTHESIS,
                    prompt=assembled.prompt,
                    schema=output_model,
                    ledger=handles.llm_budget_ledger,
                    lease_id=(f"{state['run_config'].turn_id}:final_output:{attempt_index}:{uuid4().hex}"),
                    kwargs=self._kwargs,
                )
            except Exception as exc:
                validation_error = _find_validation_error(exc)
                if validation_error is None:
                    raise
                last_errors = _validation_error_details(validation_error)
                if attempt_index + 1 >= max_attempts:
                    raise OutputValidationExhaustedError(
                        attempts=max_attempts,
                        validation_errors=last_errors,
                    ) from exc
                feedback = self._bounded_feedback(last_errors)
                continue
            return result.value

        raise OutputValidationExhaustedError(
            attempts=max_attempts,
            validation_errors=last_errors,
        )

    def _bounded_feedback(
        self,
        errors: list[dict[str, object]],
    ) -> str:
        raw = json.dumps(errors, ensure_ascii=False, sort_keys=True)
        return self._gateway.token_accounting.clip(
            raw,
            512,
            add_ellipsis=True,
        )


def validated_final_output(output: BaseModel) -> ValidatedFinalOutput:
    return ValidatedFinalOutput(
        model_path=output_model_path(type(output)),
        data=output.model_dump(mode="json"),
    )


def final_answer_from_output(output: BaseModel) -> str:
    for field_name in (
        "text",
        "answer",
        "final_answer",
        "content",
        "summary",
    ):
        value = getattr(output, field_name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return output.model_dump_json(exclude_none=True)


def create_model_structured_output_finalizer(
    registry: ModelResolver,
) -> ModelStructuredOutputFinalizer:
    resolved = registry.resolve_for_node(
        node_model=None,
        node_name="final_output",
    )
    gateway = resolved.gateway
    if gateway is None:
        token_accounting = resolved.token_accounting or TokenAccountingService(
            TokenizerContract(
                embedding_model_name="agent-final-output",
                tokenizer_model_name="agent-final-output",
                chunking_tokenizer_model_name="agent-final-output",
                tokenizer_backend="simple",
                max_context_tokens=resolved.context_window_tokens,
                prompt_reserved_tokens=512,
            )
        )
        gateway = LLMGateway(
            generator=resolved.generator,
            token_accounting=token_accounting,
            model_context_tokens=resolved.context_window_tokens,
            stage_budgets=DEFAULT_LLM_STAGE_BUDGETS,
        )
    return ModelStructuredOutputFinalizer(
        gateway=gateway,
        kwargs=resolved.kwargs,
    )


def _find_validation_error(exc: BaseException) -> ValidationError | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ValidationError):
            return current
        current = current.__cause__ or current.__context__
    return None


def _validation_error_details(
    error: ValidationError,
) -> list[dict[str, object]]:
    return [
        {
            "location": [str(part) for part in item.get("loc", ())],
            "message": str(item.get("msg", "validation failed")),
            "type": str(item.get("type", "value_error")),
        }
        for item in error.errors(
            include_url=False,
            include_input=False,
        )
    ]


__all__ = [
    "ModelStructuredOutputFinalizer",
    "OutputValidationExhaustedError",
    "StructuredOutputFinalizer",
    "create_model_structured_output_finalizer",
    "final_answer_from_output",
    "validated_final_output",
]
