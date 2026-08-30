"""Migration-time SDK facade that reaches only the replacement Harness."""

from __future__ import annotations

import asyncio

from agent_runtime.harness.protocol import TurnResult
from agent_runtime.harness.thread_manager import ThreadManager


class HarnessAgent:
    def __init__(self, thread_manager: ThreadManager) -> None:
        self._thread_manager = thread_manager

    def run(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        previous_turn_id: str | None = None,
    ) -> TurnResult:
        return asyncio.run(
            self.arun(
                message,
                thread_id=thread_id,
                previous_turn_id=previous_turn_id,
            )
        )

    async def arun(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        previous_turn_id: str | None = None,
    ) -> TurnResult:
        return await self._thread_manager.run(
            user_message=message,
            thread_id=thread_id,
            previous_turn_id=previous_turn_id,
        )

    def resume(self, turn_id: str, decision: str) -> TurnResult:
        return asyncio.run(self.aresume(turn_id, decision))

    async def aresume(self, turn_id: str, decision: str) -> TurnResult:
        return await self._thread_manager.resume(
            turn_id=turn_id,
            decision=decision,
        )

    def respond_interaction(
        self,
        turn_id: str,
        request_id: str,
        response: str,
    ) -> TurnResult:
        return asyncio.run(self.arespond_interaction(turn_id, request_id, response))

    async def arespond_interaction(
        self,
        turn_id: str,
        request_id: str,
        response: str,
    ) -> TurnResult:
        return await self._thread_manager.respond_interaction(
            turn_id=turn_id,
            request_id=request_id,
            response=response,
        )

    def retry_unknown_model(self, turn_id: str) -> TurnResult:
        return asyncio.run(self.aretry_unknown_model(turn_id))

    async def aretry_unknown_model(self, turn_id: str) -> TurnResult:
        return await self._thread_manager.retry_unknown_model(turn_id=turn_id)

    def recover_committed_model_response(self, turn_id: str) -> TurnResult:
        return asyncio.run(self.arecover_committed_model_response(turn_id))

    async def arecover_committed_model_response(self, turn_id: str) -> TurnResult:
        return await self._thread_manager.recover_committed_model_response(turn_id=turn_id)

    def read_result(self, turn_id: str) -> TurnResult:
        return self._thread_manager.read_result(turn_id=turn_id)
