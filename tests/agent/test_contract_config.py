from __future__ import annotations

import asyncio

import pytest

from rag.agent.builtin.generic import GENERIC_AGENT, GENERIC_SYSTEM_PROMPT
from rag.agent.core.context import (
    AgentRunConfig,
    AgentRuntimeHandles,
    LLMBudgetLedger,
    TurnRegistry,
)
from rag.agent.core.definition import AgentRuntimePolicy, ModelSelectionPolicy, ToolPolicy


class TestLLMBudgetLedger:
    @pytest.mark.anyio
    async def test_reserve_commit(self) -> None:
        ledger = LLMBudgetLedger(total=1000)
        ok = await ledger.reserve("lease-1", 300)
        assert ok is True
        assert await ledger.remaining() == 700
        overrun = await ledger.commit("lease-1", 250)
        assert overrun == 0
        assert await ledger.remaining() == 750

    @pytest.mark.anyio
    async def test_reserve_rejects_over_budget(self) -> None:
        ledger = LLMBudgetLedger(total=500)
        ok = await ledger.reserve("lease-1", 600)
        assert ok is False

    @pytest.mark.anyio
    async def test_refund_returns_tokens(self) -> None:
        ledger = LLMBudgetLedger(total=1000)
        await ledger.reserve("lease-1", 300)
        refunded = await ledger.refund("lease-1")
        assert refunded == 300
        assert await ledger.remaining() == 1000

    @pytest.mark.anyio
    async def test_commit_records_overrun(self) -> None:
        ledger = LLMBudgetLedger(total=1000)
        await ledger.reserve("lease-1", 200)
        overrun = await ledger.commit("lease-1", 350)
        assert overrun == 150
        assert await ledger.remaining() == 650

    @pytest.mark.anyio
    async def test_concurrent_reserve(self) -> None:
        ledger = LLMBudgetLedger(total=500)

        async def reserve_300() -> bool:
            return await ledger.reserve("a", 300)

        async def reserve_300_b() -> bool:
            return await ledger.reserve("b", 300)

        results = await asyncio.gather(reserve_300(), reserve_300_b())
        assert sum(1 for result in results if result) == 1

    @pytest.mark.anyio
    async def test_exposes_committed_and_reserved_totals(self) -> None:
        ledger = LLMBudgetLedger(total=500)
        assert await ledger.reserve("call-1", 200) is True
        assert await ledger.commit("call-1", 125) == 0
        assert await ledger.reserve("call-2", 50) is True

        assert await ledger.committed() == 125
        assert await ledger.reserved() == 50


class TestAgentRunConfig:
    def test_minimal_config(self) -> None:
        cfg = AgentRunConfig(
            turn_id="r1",
            llm_budget_total=10000,
        )
        assert cfg.turn_id == "r1"
        for removed in (
            "run_id",
            "thread_id",
            "max_depth",
            "access_policy",
            "agent_type",
            "parent_run_id",
            "source_scope",
            "deadline_iso",
            "trace_parent_id",
        ):
            assert not hasattr(cfg, removed)

    def test_config_defaults(self) -> None:
        cfg = AgentRunConfig(
            turn_id="r1",
            llm_budget_total=5000,
        )
        assert cfg.llm_budget_total == 5000
        assert isinstance(cfg.tool_policy, ToolPolicy)

    @pytest.mark.parametrize("max_turns", [0, -1, True])
    def test_config_rejects_invalid_max_turns(self, max_turns: object) -> None:
        with pytest.raises((TypeError, ValueError)):
            AgentRunConfig(
                turn_id="invalid-turn-limit",
                max_turns=max_turns,  # type: ignore[arg-type]
            )


class TestTurnRegistry:
    def test_get_or_create_initializes_handles(self) -> None:
        cfg = AgentRunConfig(
            turn_id="reg-test",
            llm_budget_total=8000,
        )
        TurnRegistry.remove(cfg.turn_id)
        handles = TurnRegistry.get_or_create(cfg)
        assert isinstance(handles, AgentRuntimeHandles)
        assert isinstance(handles.llm_budget_ledger, LLMBudgetLedger)
        assert isinstance(handles.cancellation, asyncio.Event)

    def test_get_or_create_returns_same_handles(self) -> None:
        cfg = AgentRunConfig(
            turn_id="reg-test-2",
            llm_budget_total=8000,
        )
        TurnRegistry.remove(cfg.turn_id)
        h1 = TurnRegistry.get_or_create(cfg)
        h2 = TurnRegistry.get_or_create(cfg)
        assert h1 is h2

    def test_remove_cleans_up(self) -> None:
        cfg = AgentRunConfig(
            turn_id="reg-test-3",
            llm_budget_total=8000,
        )
        TurnRegistry.remove(cfg.turn_id)
        TurnRegistry.get_or_create(cfg)
        TurnRegistry.remove("reg-test-3")
        h_new = TurnRegistry.get_or_create(cfg)
        assert h_new is not None


class TestAgentRuntimePolicy:
    def test_minimal_definition(self) -> None:
        ad = AgentRuntimePolicy.test_factory(
            system_prompt="You are a research agent.",
            allowed_tools=["vector_search", "grounding"],
        )
        assert ad.configured_tool_names == ("vector_search", "grounding")
        assert ad.max_iterations == 10
        assert not hasattr(ad, "token_budget")
        assert not hasattr(ad, "agent_type")
        assert not hasattr(ad, "description")
        assert not hasattr(ad, "allowed_tools")
        assert ad.output_validation_max_retries == 2

    def test_generic_coding_agent_has_delivery_budget_and_aci_guidance(self) -> None:
        assert GENERIC_AGENT.max_iterations == 50
        assert GENERIC_AGENT.model_selection.tool_decision_max_tokens == 4_096
        assert "read_file.start_line" in GENERIC_SYSTEM_PROMPT
        assert "first concrete edit" in GENERIC_SYSTEM_PROMPT
        assert "twelve inspection" in GENERIC_SYSTEM_PROMPT
        normalized_prompt = " ".join(GENERIC_SYSTEM_PROMPT.split())
        assert "complete coherent change across every affected layer" in (
            normalized_prompt
        )
        assert "smallest coherent change" not in normalized_prompt

    def test_definition_rejects_negative_output_validation_retries(self) -> None:
        with pytest.raises(
            ValueError,
            match="output_validation_max_retries must be non-negative",
        ):
            AgentRuntimePolicy.test_factory(
                system_prompt="Invalid",
                allowed_tools=[],
                output_validation_max_retries=-1,
            )

    def test_tool_policy_defaults(self) -> None:
        tp = ToolPolicy()
        assert tp.max_parallel_calls == 4
        assert len(tp.require_confirmation_for) == 0
        assert len(tp.deny_tools) == 0

    def test_tool_policy_custom(self) -> None:
        tp = ToolPolicy(
            max_parallel_calls=2,
            require_confirmation_for=frozenset({"kg_upsert"}),
            deny_tools=frozenset({"web_search"}),
        )
        assert "kg_upsert" in tp.require_confirmation_for
        assert "web_search" in tp.deny_tools
        assert tp.max_parallel_calls == 2

    @pytest.mark.parametrize("value", [0, -1, True])
    def test_tool_policy_rejects_invalid_parallel_limit(
        self,
        value: object,
    ) -> None:
        with pytest.raises((TypeError, ValueError)):
            ToolPolicy(max_parallel_calls=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("require_confirmation_for", frozenset({""})),
            ("deny_tools", frozenset({1})),
        ],
    )
    def test_tool_policy_rejects_invalid_tool_names(
        self,
        field_name: str,
        value: object,
    ) -> None:
        with pytest.raises((TypeError, ValueError)):
            ToolPolicy(**{field_name: value})

    def test_tool_policy_rejects_non_boolean_sandbox_approval(self) -> None:
        with pytest.raises(TypeError, match="auto_approve_sandboxed"):
            ToolPolicy(auto_approve_sandboxed=1)  # type: ignore[arg-type]

    def test_model_selection_defaults(self) -> None:
        ms = ModelSelectionPolicy()
        assert ms.tool_decision_model is None
        assert ms.tool_decision_temperature == 0.0
        assert ms.tool_decision_max_tokens is None
