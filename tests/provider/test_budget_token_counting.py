from agent_runtime.modeling.tokenization import TokenAccountingService, TokenizerContract


def test_simple_text_units_are_not_used_for_request_reservations():
    accounting = TokenAccountingService(TokenizerContract(
        embedding_model_name="unused", tokenizer_model_name="unused",
        chunking_tokenizer_model_name="unused", tokenizer_backend="simple",
    ))
    text = '{"id":900719925474099312345,"value":"\\n\\n中文"}'
    assert accounting.count_for_budget(text) == len(text.encode("utf-8"))
    assert accounting.count(text) < accounting.count_for_budget(text)
    assert accounting.budget_count_source() == "utf8_bytes_conservative"


def test_bundled_deepseek_tokenizer_is_offline_and_preserves_long_numbers():
    accounting = TokenAccountingService(TokenizerContract(
        embedding_model_name="unused", tokenizer_model_name="deepseek-v4-official",
        chunking_tokenizer_model_name="unused",
    ))
    text = '{"value":9007199254740993}'
    assert accounting.count_for_budget(text) == 10
    assert accounting.budget_count_source() == "tokenizers:deepseek-v4-official"
    assert accounting.count(accounting.clip(text, 5)) <= 5


def test_available_tokenizer_is_used_without_changing_chunk_counting():
    class Encoding:
        def encode(self, text):
            return [1, 2, 3]

    accounting = TokenAccountingService(TokenizerContract(
        embedding_model_name="unused", tokenizer_model_name="unused",
        chunking_tokenizer_model_name="unused",
    ), _backend_kind="tiktoken", _backend=Encoding())
    assert accounting.count_for_budget("example") == accounting.count("example") == 3
    assert accounting.budget_count_source() == "tiktoken:unused"


def test_adapter_and_gateway_reserve_the_same_serialized_input():
    from agent_runtime.harness import GatewayHarnessModel, HarnessMessage, HarnessModelRequest
    from agent_runtime.modeling.contracts import LLMCallStage, LLMStageBudget
    from agent_runtime.modeling.gateway import LLMGateway, model_request_input_text
    from tests.agent.harness.test_model_adapter import _resolved_model

    accounting = TokenAccountingService(TokenizerContract(
        embedding_model_name="unused", tokenizer_model_name="unused",
        chunking_tokenizer_model_name="unused", tokenizer_backend="simple",
    ))
    gateway = LLMGateway(generator=object(), token_accounting=accounting,
                         model_context_tokens=100000,
                         stage_budgets={LLMCallStage.AGENT_STEP: LLMStageBudget(
                             max_input_tokens=64000, max_output_tokens=256, safety_margin_tokens=0,
                         )})
    resolved = _resolved_model(gateway=gateway, token_accounting=accounting)
    model = GatewayHarnessModel(model_id="test", resolved=resolved, instructions=("Answer.",))
    prepared = model.prepare(HarnessModelRequest(
        thread_id="thread", turn_id="turn", binding_manifest={},
        messages=(HarnessMessage(role="user", content='{"中文":1234567890123456789}'),),
    ))
    wire = model_request_input_text(prepared.dispatch_payload.request,
                                    provider=resolved.provider, supports_native_tools=True)
    _, _, counted, reserved = gateway._prepare_call(
        stage=LLMCallStage.AGENT_STEP, prompt=wire, kwargs=None,
    )
    assert counted == prepared.resource_request.input_tokens == len(wire.encode("utf-8"))
    assert reserved == prepared.resource_request.total_tokens
    assert prepared.request_ref["context_projection"]["count_source"] == "utf8_bytes_conservative"


def test_context_compaction_can_consume_agent_history_without_generic_summary_splitting():
    """A valid agent history must not inherit the smaller generic summarize window."""
    from dataclasses import replace

    from agent_runtime.harness import GatewayHarnessModel, HarnessMessage, HarnessModelRequest
    from agent_runtime.modeling.contracts import LLMCallStage, LLMStageBudget
    from agent_runtime.modeling.gateway import LLMGateway
    from tests.agent.harness.test_model_adapter import _resolved_model

    accounting = TokenAccountingService(TokenizerContract(
        embedding_model_name="unused", tokenizer_model_name="unused",
        chunking_tokenizer_model_name="unused", tokenizer_backend="simple",
    ))
    gateway = LLMGateway(generator=object(), token_accounting=accounting,
                         model_context_tokens=100000, stage_budgets={
                             LLMCallStage.AGENT_STEP: LLMStageBudget(
                                 max_input_tokens=64000, max_output_tokens=4096),
                             LLMCallStage.LLM_SUMMARIZE: LLMStageBudget(
                                 max_input_tokens=16000, max_output_tokens=2048),
                         })
    resolved = _resolved_model(gateway=gateway, token_accounting=accounting)
    resolved = replace(resolved, capabilities=replace(resolved.capabilities, max_output_tokens=4096))
    model = GatewayHarnessModel(model_id="test", resolved=resolved, instructions=("Answer.",))
    prepared = model.prepare(HarnessModelRequest(
        thread_id="thread", turn_id="turn", binding_manifest={}, purpose="context_summary",
        messages=(HarnessMessage(role="user", content="history " * 3000),),
    ))
    assert 16000 < prepared.resource_request.input_tokens < 64000
    assert prepared.resource_request.output_tokens == 2048
    assert prepared.request_ref["context_projection"]["max_input_tokens"] == 64000
    # Ordinary summarization keeps its configured policy.
    assert gateway.effective_stage_budget(LLMCallStage.LLM_SUMMARIZE).max_input_tokens == 16000
    stage = prepared.dispatch_payload.stage
    _, _, counted, reserved = gateway._prepare_call(
        stage=stage, prompt="history " * 3000, kwargs={"max_tokens": 2048},
    )
    assert counted == 24000
    assert reserved == 24000 + 2048


def test_compaction_policy_still_reserves_output_and_honors_explicit_limit():
    from agent_runtime.modeling.contracts import LLMCallStage, LLMStageBudget
    from agent_runtime.modeling.gateway import LLMGateway

    policies = {
        LLMCallStage.AGENT_STEP: LLMStageBudget(max_input_tokens=64000, max_output_tokens=4096,
                                              safety_margin_tokens=800),
        LLMCallStage.LLM_SUMMARIZE: LLMStageBudget(max_input_tokens=16000, max_output_tokens=2048,
                                                 safety_margin_tokens=512),
    }
    gateway = LLMGateway(generator=object(), token_accounting=object(), model_context_tokens=20000,
                         stage_budgets=policies)
    budget = gateway.effective_stage_budget(LLMCallStage.CONTEXT_COMPACTION)
    assert budget.max_input_tokens == 20000 - 2048 - 800
    policies[LLMCallStage.CONTEXT_COMPACTION] = LLMStageBudget(
        max_input_tokens=5000, max_output_tokens=1000, safety_margin_tokens=600,
    )
    gateway = LLMGateway(generator=object(), token_accounting=object(), model_context_tokens=20000,
                         stage_budgets=policies)
    assert gateway.effective_stage_budget(LLMCallStage.CONTEXT_COMPACTION) == policies[LLMCallStage.CONTEXT_COMPACTION]
