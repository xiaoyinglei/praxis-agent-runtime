from __future__ import annotations

from pydantic import ValidationError

from agent_runtime.loop.state import (
    LoopState,
    ModelTurn,
    ModelTurnDraft,
    materialize_model_turn,
)


class FinishCandidateBuildError(RuntimeError):
    pass


class FinishCandidateBuilder:
    """Normalize finish intent before strict ModelTurn validation."""

    async def build(
        self,
        draft: ModelTurnDraft,
        *,
        state: LoopState,
    ) -> ModelTurn:
        del state
        try:
            return materialize_model_turn(draft)
        except ValidationError as exc:
            raise FinishCandidateBuildError(
                f"model turn is incomplete: {exc}"
            ) from exc

__all__ = [
    "FinishCandidateBuildError",
    "FinishCandidateBuilder",
]
