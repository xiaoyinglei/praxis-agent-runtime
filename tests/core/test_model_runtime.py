from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from pydantic import BaseModel

from agent_runtime.modeling.config import ModelCapability, ModelRuntimeConfig
from agent_runtime.modeling.contracts import LLMCallStage
from rag.assembly.models import ProviderConfig
from rag.assembly.support import _OpenAICompatibleChatGenerator, build_provider
from rag.models.assembly_adapter import to_assembly_overrides
from rag.models.catalog import ModelCatalog
from rag.models.guard import EmbeddingSpaceMismatchError, assert_embedding_space_compatible
from rag.models.runtime import RuntimeOverrides, resolve_runtime_config
from rag.runtime import _generator_bindings_from_chat_bindings


class _StructuredPayload(BaseModel):
    answer: str


CATALOG_YAML = """
models:
  mlx-community/Qwen3-14B-4bit:
    capability: chat
    provider: openai_compatible
    base_url: http://127.0.0.1:8080/v1
    context_window_tokens: 32768

  deepseek-chat:
    capability: chat
    provider: openai_compatible
    base_url: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY
    context_window_tokens: 65536

  mlx-community/Qwen3-Embedding-8B-4bit-DWQ:
    capability: embedding
    provider: mlx_embedding
    embedding_space: mlx/Qwen3-Embedding-8B-4bit-DWQ

  Qwen/Qwen3-Reranker-4B:
    capability: reranker
    provider: sentence_transformers

defaults:
  primary_model: mlx-community/Qwen3-14B-4bit
  embedding_model: mlx-community/Qwen3-Embedding-8B-4bit-DWQ
  reranker_model: Qwen/Qwen3-Reranker-4B

llm_budgets:
  tool_decision:
    max_input_tokens: 12000
    max_output_tokens: 2048
    safety_margin_tokens: 512
"""


@pytest.fixture
def catalog_path(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(CATALOG_YAML, encoding="utf-8")
    return path


@pytest.fixture
def catalog(catalog_path: Path) -> ModelCatalog:
    return ModelCatalog.from_yaml(str(catalog_path))


# ── catalog ──


def test_catalog_loads_models(catalog: ModelCatalog) -> None:
    assert catalog.get_model("mlx-community/Qwen3-14B-4bit").capability == ModelCapability.CHAT
    assert catalog.get_model("deepseek-chat").capability == ModelCapability.CHAT
    embedding = catalog.get_model("mlx-community/Qwen3-Embedding-8B-4bit-DWQ")
    reranker = catalog.get_model("Qwen/Qwen3-Reranker-4B")
    assert embedding.capability == ModelCapability.EMBEDDING
    assert embedding.id == "mlx-community/Qwen3-Embedding-8B-4bit-DWQ"
    assert reranker.capability == ModelCapability.RERANKER
    assert reranker.id == "Qwen/Qwen3-Reranker-4B"
    assert not hasattr(embedding, "alias")
    assert not hasattr(embedding, "model")
    assert catalog.get_model("mlx-community/Qwen3-14B-4bit").context_window_tokens == 32768


def test_catalog_supports_provider_section_schema(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        """
providers:
  groq:
    protocol: openai_compatible
    location: cloud
    base_url: https://api.groq.com/openai/v1
    api_key_env: GROQ_API_KEY
  local_mlx_embedding:
    protocol: mlx_embedding
    location: local
  local_sentence_transformers:
    protocol: sentence_transformers
    location: local

models:
  openai/gpt-oss-120b:
    capability: chat
    provider: groq
    context_window_tokens: 131072
  mlx-community/Qwen3-Embedding-8B-4bit-DWQ:
    capability: embedding
    provider: local_mlx_embedding
    embedding_space: mlx/Qwen3-Embedding-8B-4bit-DWQ
  Qwen/Qwen3-Reranker-4B:
    capability: reranker
    provider: local_sentence_transformers

defaults:
  primary_model: openai/gpt-oss-120b
  embedding_model: mlx-community/Qwen3-Embedding-8B-4bit-DWQ
  reranker_model: Qwen/Qwen3-Reranker-4B
""",
        encoding="utf-8",
    )

    catalog = ModelCatalog.from_yaml(str(path))
    config = resolve_runtime_config(catalog=catalog)
    overrides = to_assembly_overrides(config)

    assert config.primary_model.provider == "openai_compatible"
    assert config.primary_model.base_url == "https://api.groq.com/openai/v1"
    assert config.primary_model.api_key_env == "GROQ_API_KEY"
    assert config.embedding_model.provider == "mlx_embedding"
    assert config.reranker_model is not None
    assert config.reranker_model.provider == "sentence_transformers"
    assert overrides.chat is not None
    assert overrides.chat.provider_kind == "openai-compatible"
    assert overrides.chat.chat_model == "openai/gpt-oss-120b"
    assert overrides.embedding is not None
    assert overrides.embedding.provider_kind == "mlx-embedding"
    assert overrides.rerank is not None
    assert overrides.rerank.provider_kind == "local-bge"
    assert overrides.tokenizer is not None
    assert overrides.tokenizer.max_context_tokens == 131_072


def test_catalog_loads_llm_stage_budgets(catalog: ModelCatalog) -> None:
    budget = catalog.llm_stage_budgets[LLMCallStage.TOOL_DECISION]
    assert budget.max_input_tokens == 12000
    assert budget.max_output_tokens == 2048
    assert budget.safety_margin_tokens == 512


def test_catalog_defaults(catalog: ModelCatalog) -> None:
    assert catalog.get_default_primary().id == "mlx-community/Qwen3-14B-4bit"
    assert catalog.get_default_embedding().id == "mlx-community/Qwen3-Embedding-8B-4bit-DWQ"
    assert catalog.get_default_reranker().id == "Qwen/Qwen3-Reranker-4B"


def test_catalog_list_models(catalog: ModelCatalog) -> None:
    chat_models = catalog.list_models(ModelCapability.CHAT)
    assert len(chat_models) == 2
    assert {m.id for m in chat_models} == {
        "deepseek-chat",
        "mlx-community/Qwen3-14B-4bit",
    }


@pytest.mark.parametrize(
    "forbidden_field, forbidden_value",
    [
        ("model", "shadow/model"),
        ("max_context_window_tokens", 65536),
    ],
)
def test_catalog_rejects_redundant_chat_identity_and_context_fields(
    tmp_path: Path,
    forbidden_field: str,
    forbidden_value: object,
) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "openai/gpt-oss-120b": {
                        "capability": "chat",
                        "provider": "openai_compatible",
                        "context_window_tokens": 131072,
                        forbidden_field: forbidden_value,
                    }
                },
                "defaults": {"primary_model": "openai/gpt-oss-120b"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=forbidden_field):
        ModelCatalog.from_yaml(str(path))


@pytest.mark.parametrize("capability", ["chat", "embedding", "reranker"])
def test_catalog_rejects_nested_model_for_every_capability(
    tmp_path: Path,
    capability: str,
) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "exact/model-id": {
                        "capability": capability,
                        "provider": "test-provider",
                        "model": "shadow/model-id",
                    }
                },
                "defaults": {
                    "primary_model": "exact/model-id",
                    "embedding_model": "exact/model-id",
                    "reranker_model": "exact/model-id",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model"):
        ModelCatalog.from_yaml(str(path))


@pytest.mark.parametrize("capability", ["chat", "embedding", "reranker"])
@pytest.mark.parametrize("model_id", ["", " exact/model-id "])
def test_catalog_rejects_non_trimmed_or_empty_model_ids(
    tmp_path: Path,
    capability: str,
    model_id: str,
) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    model_id: {
                        "capability": capability,
                        "provider": "openai_compatible",
                    }
                },
                "defaults": {"primary_model": model_id},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty trimmed IDs"):
        ModelCatalog.from_yaml(str(path))


def test_catalog_unknown_model_raises(catalog: ModelCatalog) -> None:
    with pytest.raises(KeyError, match="Unknown model ID"):
        catalog.get_model("nonexistent")


# ── runtime resolution ──


def test_runtime_default_primary_model(catalog: ModelCatalog) -> None:
    config = resolve_runtime_config(RuntimeOverrides(), catalog=catalog)
    assert config.primary_model.id == "mlx-community/Qwen3-14B-4bit"
    assert config.llm_stage_budgets[LLMCallStage.TOOL_DECISION].max_input_tokens == 12000


def test_runtime_generator_bindings_attach_budget_gateway(catalog: ModelCatalog) -> None:
    class _Tokens:
        def count(self, text: str) -> int:
            return len(text.split())

    binding = type(
        "ChatBinding",
        (),
        {
            "backend": object(),
            "provider_name": "test",
            "model_name": "test-model",
            "location": "local",
            "chat": lambda self, prompt, **kwargs: "answer",
        },
    )()

    [generator_binding] = _generator_bindings_from_chat_bindings(
        [binding],
        token_accounting=_Tokens(),
        model_context_tokens=32_768,
        stage_budgets=catalog.llm_stage_budgets,
    )

    assert generator_binding.gateway is not None


def test_runtime_override_primary_model(catalog: ModelCatalog) -> None:
    config = resolve_runtime_config(
        RuntimeOverrides(model_id="deepseek-chat"),
        catalog=catalog,
    )
    assert config.primary_model.id == "deepseek-chat"
    assert config.primary_model.base_url == "https://api.deepseek.com/v1"


def test_runtime_rejects_capability_mismatch(catalog: ModelCatalog) -> None:
    with pytest.raises(ValueError, match="capability 'embedding'.*expected 'chat'"):
        resolve_runtime_config(
            RuntimeOverrides(model_id="mlx-community/Qwen3-Embedding-8B-4bit-DWQ"),
            catalog=catalog,
        )


def test_runtime_rejects_unknown_model(catalog: ModelCatalog) -> None:
    with pytest.raises(KeyError, match="Unknown model ID"):
        resolve_runtime_config(
            RuntimeOverrides(model_id="gpt4"),
            catalog=catalog,
        )


def test_runtime_disabled_reranker(catalog: ModelCatalog) -> None:
    for model_id in ("none", "null", "off", "false"):
        config = resolve_runtime_config(
            RuntimeOverrides(reranker_model_id=model_id),
            catalog=catalog,
        )
        assert config.reranker_model is None, f"reranker should be None for model_id={model_id!r}"


# ── embedding space guard ──


def test_embedding_space_match_passes() -> None:
    assert_embedding_space_compatible(
        "mlx/Qwen3-Embedding-8B-4bit-DWQ",
        "mlx/Qwen3-Embedding-8B-4bit-DWQ",
    )


def test_embedding_space_mismatch_raises() -> None:
    with pytest.raises(EmbeddingSpaceMismatchError) as exc:
        assert_embedding_space_compatible(
            "mlx/Qwen3-Embedding-8B-4bit-DWQ",
            "BGE-V3/default",
        )
    assert "mlx/Qwen3-Embedding-8B-4bit-DWQ" in str(exc.value)
    assert "BGE-V3/default" in str(exc.value)


# ── assembly adapter ──


def test_assembly_adapter_produces_chat_provider_config(catalog: ModelCatalog) -> None:
    spec = catalog.get_model("mlx-community/Qwen3-14B-4bit")
    config = ModelRuntimeConfig(
        primary_model=spec,
        embedding_model=catalog.get_model("mlx-community/Qwen3-Embedding-8B-4bit-DWQ"),
        reranker_model=catalog.get_model("Qwen/Qwen3-Reranker-4B"),
    )
    overrides = to_assembly_overrides(config)

    assert overrides.chat is not None
    assert overrides.chat.provider_kind == "openai-compatible"
    assert overrides.chat.chat_model == "mlx-community/Qwen3-14B-4bit"
    assert overrides.chat.base_url == "http://127.0.0.1:8080/v1"
    assert overrides.chat.api_key is None
    assert overrides.tokenizer is not None
    assert overrides.tokenizer.max_context_tokens == 32_768


def test_assembly_adapter_embedding_provider_config(catalog: ModelCatalog) -> None:
    spec = catalog.get_model("mlx-community/Qwen3-Embedding-8B-4bit-DWQ")
    config = ModelRuntimeConfig(
        primary_model=catalog.get_model("mlx-community/Qwen3-14B-4bit"),
        embedding_model=spec,
    )
    overrides = to_assembly_overrides(config)

    assert overrides.embedding is not None
    assert overrides.embedding.provider_kind == "mlx-embedding"
    assert overrides.embedding.embedding_model == "mlx-community/Qwen3-Embedding-8B-4bit-DWQ"


def test_assembly_adapter_reranker_maps_to_local_bge(catalog: ModelCatalog) -> None:
    spec = catalog.get_model("Qwen/Qwen3-Reranker-4B")
    config = ModelRuntimeConfig(
        primary_model=catalog.get_model("mlx-community/Qwen3-14B-4bit"),
        embedding_model=catalog.get_model("mlx-community/Qwen3-Embedding-8B-4bit-DWQ"),
        reranker_model=spec,
    )
    overrides = to_assembly_overrides(config)

    assert overrides.rerank is not None
    assert overrides.rerank.provider_kind == "local-bge"
    assert overrides.rerank.rerank_model == "Qwen/Qwen3-Reranker-4B"


def test_exact_non_chat_ids_reach_backend_constructors_unchanged(
    catalog: ModelCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rag.assembly.support as assembly_support

    captured: dict[str, str] = {}

    class _CapturingEmbedder:
        def __init__(self, model_name_or_path: str, **_: object) -> None:
            captured["embedding"] = model_name_or_path
            self.embedding_model_name = model_name_or_path

    class _CapturingReranker:
        def __init__(self, model_name_or_path: str, **_: object) -> None:
            captured["reranker"] = model_name_or_path
            self.rerank_model_name = model_name_or_path

    monkeypatch.setattr(assembly_support, "MLXEmbedder", _CapturingEmbedder)
    monkeypatch.setattr(assembly_support, "FlagEmbeddingReranker", _CapturingReranker)

    config = resolve_runtime_config(catalog=catalog)
    overrides = to_assembly_overrides(config)
    assert overrides.embedding is not None
    assert overrides.rerank is not None

    assembly_support.build_provider(overrides.embedding)
    assembly_support.build_provider(overrides.rerank)

    assert captured == {
        "embedding": "mlx-community/Qwen3-Embedding-8B-4bit-DWQ",
        "reranker": "Qwen/Qwen3-Reranker-4B",
    }


def test_assembly_adapter_none_reranker(catalog: ModelCatalog) -> None:
    config = ModelRuntimeConfig(
        primary_model=catalog.get_model("mlx-community/Qwen3-14B-4bit"),
        embedding_model=catalog.get_model("mlx-community/Qwen3-Embedding-8B-4bit-DWQ"),
        reranker_model=None,
    )
    overrides = to_assembly_overrides(config)
    assert overrides.rerank is None


# ── openai-compatible provider ──


def test_build_provider_openai_compatible_no_longer_unavailable() -> None:
    provider = build_provider(
        ProviderConfig(
            provider_kind="openai-compatible",
            chat_model="mlx-community/Qwen3-14B-4bit",
            base_url="http://127.0.0.1:8080/v1",
        )
    )
    assert hasattr(provider, "generate_text")
    assert callable(provider.generate_text)


def test_build_provider_openai_compatible_without_api_key() -> None:
    """Local MLX server does not require api_key."""
    provider = build_provider(
        ProviderConfig(
            provider_kind="openai-compatible",
            chat_model="mlx-community/Qwen3-14B-4bit",
            base_url="http://127.0.0.1:8080/v1",
        )
    )
    assert provider.is_chat_configured


def test_build_provider_missing_chat_model_returns_unavailable() -> None:
    provider = build_provider(
        ProviderConfig(
            provider_kind="openai-compatible",
            base_url="http://127.0.0.1:8080/v1",
        )
    )
    assert not provider.is_chat_configured


def test_build_provider_missing_base_url_returns_unavailable() -> None:
    provider = build_provider(
        ProviderConfig(
            provider_kind="openai-compatible",
            chat_model="deepseek-chat",
        )
    )
    assert not provider.is_chat_configured


def test_openai_compatible_generator_repr_no_api_key() -> None:
    gen = _OpenAICompatibleChatGenerator(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key="sk-secret-key-12345",
    )
    rep = repr(gen)
    assert "deepseek-chat" in rep
    assert "api.deepseek.com" in rep
    assert "sk-secret" not in rep


def test_openai_compatible_generator_system_prompt() -> None:
    gen = _OpenAICompatibleChatGenerator(
        model="test-model",
        base_url="http://127.0.0.1:8080/v1",
    )

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "response text"
    mock_response.usage = None

    original_create = gen._client.chat.completions.create

    def fake_create(*, model, messages, **kwargs):  # noqa: ARG001
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "hello"
        return mock_response

    gen._client.chat.completions.create = fake_create
    try:
        result = gen.generate_text(prompt="hello", system_prompt="You are helpful.")
        assert result == "response text"
    finally:
        gen._client.chat.completions.create = original_create


def test_openai_compatible_generator_null_content() -> None:
    gen = _OpenAICompatibleChatGenerator(
        model="test-model",
        base_url="http://127.0.0.1:8080/v1",
    )
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = None
    mock_response.usage = None

    original_create = gen._client.chat.completions.create
    gen._client.chat.completions.create = lambda **kw: mock_response
    try:
        result = gen.generate_text(prompt="hello")
        assert result == ""
    finally:
        gen._client.chat.completions.create = original_create


def test_openai_compatible_generator_structured_fallback_parses_fenced_json() -> None:
    gen = _OpenAICompatibleChatGenerator(
        model="test-model",
        base_url="http://127.0.0.1:8080/v1",
    )
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '```json\n{"answer": "ok"}\n```'
    mock_response.usage = None

    original_create = gen._client.chat.completions.create
    gen._client.chat.completions.create = lambda **kw: mock_response
    try:
        result = gen.generate_structured(prompt="return json", schema=_StructuredPayload)
    finally:
        gen._client.chat.completions.create = original_create

    assert result == _StructuredPayload(answer="ok")


def test_openai_compatible_generator_structured_fallback_includes_schema() -> None:
    gen = _OpenAICompatibleChatGenerator(
        model="test-model",
        base_url="http://127.0.0.1:8080/v1",
    )
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '说明文字\n{"answer": "ok"}'
    mock_response.usage = None

    original_create = gen._client.chat.completions.create

    def fake_create(*, model, messages, **kwargs):  # noqa: ARG001
        prompt = messages[-1]["content"]
        assert "Return ONLY valid JSON matching this schema." in prompt
        assert "JSON schema:" in prompt
        assert '"answer"' in prompt
        assert "User task:" in prompt
        return mock_response

    gen._client.chat.completions.create = fake_create
    try:
        result = gen.generate_structured(prompt="return json", schema=_StructuredPayload)
    finally:
        gen._client.chat.completions.create = original_create

    assert result == _StructuredPayload(answer="ok")


# ── 端到端集成测试：模拟数据跑完整 RAG 链路 ──


def test_e2e_model_runtime_driven_ingest_and_query(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """从 models.yaml → AssemblyOverrides → Runtime → ingest → query 全链路验证。

    用 FakeProvider 模拟 chat/embedding/reranker，测试：
    1. catalog → runtime_config → assembly_overrides 链路正确
    2. 模型信息通过 runtime_config 控制，不硬编码
    3. ingest 和 query 正常执行
    4. --model deepseek-chat 可以覆盖默认 primary_model
    """
    from rag import (
        AssemblyRequest,
        CapabilityAssemblyService,
        CapabilityRequirements,
        RAGRuntime,
        StorageConfig,
    )
    from rag.assembly.support import _CompositeProvider
    from rag.models.assembly_adapter import to_assembly_overrides
    from rag.models.runtime import RuntimeOverrides, resolve_runtime_config

    # ── 1. 构造测试 catalog ──
    catalog_path = tmp_path / "models.yaml"
    catalog_path.write_text(CATALOG_YAML, encoding="utf-8")

    # ── 2. 构造 FakeProvider（模拟本地 MLX chat + embedding + reranker）──
    class FakeReranker:
        rerank_model_name = "test-reranker"

        def rerank(self, query: str, documents: list[str], **kwargs: object) -> list[float]:
            return [1.0 - i * 0.1 for i in range(len(documents))]

    def make_fake_provider(config: ProviderConfig) -> _CompositeProvider:
        return _CompositeProvider(
            provider_name="_fake_test_provider",
            generator=_FakeChat(model=config.chat_model or "test-chat"),
            embedder=_FakeEmbedder(model=config.embedding_model or "test-embed"),
            reranker=FakeReranker() if config.rerank_model else None,
        )

    # ── 3. 解析 runtime config（默认 mlx-community/Qwen3-14B-4bit）──
    runtime_config = resolve_runtime_config(
        RuntimeOverrides(),
        catalog_path=str(catalog_path),
    )
    assert runtime_config.primary_model.id == "mlx-community/Qwen3-14B-4bit"
    assert runtime_config.embedding_model.id == "mlx-community/Qwen3-Embedding-8B-4bit-DWQ"
    assert runtime_config.reranker_model.id == "Qwen/Qwen3-Reranker-4B"

    assembly_overrides = to_assembly_overrides(runtime_config)

    # 验证 assembly_overrides 不包含硬编码模型名
    assert assembly_overrides.chat.chat_model == "mlx-community/Qwen3-14B-4bit"
    assert assembly_overrides.embedding.embedding_model == "mlx-community/Qwen3-Embedding-8B-4bit-DWQ"
    assert assembly_overrides.rerank.rerank_model == "Qwen/Qwen3-Reranker-4B"

    # ── 4. 构建 RAGRuntime ──
    service = CapabilityAssemblyService(env_path=".env.test-unused")
    monkeypatch.setattr(service, "_load_env", lambda: None)
    monkeypatch.setattr(service, "_build_provider", make_fake_provider)

    runtime = RAGRuntime.from_request(
        storage=StorageConfig.in_memory(),
        request=AssemblyRequest(
            requirements=CapabilityRequirements(
                require_chat=True,
                require_embedding=True,
                require_rerank=True,
            ),
            overrides=assembly_overrides,
        ),
        assembly_service=service,
    )

    try:
        # 验证 binding 上的模型名来自 runtime_config
        chat_binding = runtime.capability_bundle.chat_bindings[0]
        assert chat_binding.model_name == "mlx-community/Qwen3-14B-4bit"

        embedding_binding = runtime.capability_bundle.embedding_bindings[0]
        assert embedding_binding.model_name == "mlx-community/Qwen3-Embedding-8B-4bit-DWQ"

        # ── 5. Ingest 测试文档 ──
        documents = [
            {
                "title": "请假制度 V3",
                "content": "正式员工每年享有年假 10 天。病假需出具医院证明。事假每年累计不超过 15 天。",
            },
            {
                "title": "报销流程 2024",
                "content": "差旅报销需在返回后 7 个工作日内提交。住宿标准一线城市不超过 500 元/晚。",
            },
            {"title": "绩效考核办法", "content": "考核周期为季度考核。考核结果分 ABCD 四档。连续两次 D 档启动 PIP。"},
            {
                "title": "数据安全管理条例",
                "content": "敏感数据必须加密存储。数据导出需主管审批。违规操作记入安全审计日志。",
            },
            {
                "title": "远程办公指南",
                "content": "每周可申请远程办公 2 天。远程办公期间需保持 IM 在线。核心会议要求线下参加。",
            },
        ]

        for doc_meta in documents:
            result = runtime.insert(
                location=f"test://docs/{doc_meta['title']}",
                source_type="plain_text",
                owner="test",
                title=doc_meta["title"],
                content_text=doc_meta["content"],
            )
            assert result.doc_id > 0, f"Ingest failed for {doc_meta['title']}"

        # ── 6. Query 检索 ──
        r1 = runtime.query_public("年假有多少天")
        assert r1.answer.answer_text
        evidence_texts_1 = " ".join(e.text for e in r1.context.evidence if e.text)
        assert "年假" in evidence_texts_1, f"Expected '年假' in evidence, got: {evidence_texts_1[:200]}"

        r2 = runtime.query_public("如何报销差旅费")
        evidence_texts_2 = " ".join(e.text for e in r2.context.evidence if e.text)
        assert "报销" in evidence_texts_2, f"Expected '报销' in evidence, got: {evidence_texts_2[:200]}"

        r3 = runtime.query_public("绩效考核怎么评")
        assert any("考核" in e.text for e in r3.context.evidence if e.text), "Expected '考核' in evidence"

        # 验证 generation provider 来自我们的模型
        assert r1.generation_model is not None

    finally:
        runtime.close()

    # ── 7. --model deepseek-chat 覆盖测试 ──
    ds_config = resolve_runtime_config(
        RuntimeOverrides(model_id="deepseek-chat"),
        catalog_path=str(catalog_path),
    )
    ds_overrides = to_assembly_overrides(ds_config)
    assert ds_overrides.chat.chat_model == "deepseek-chat"
    assert ds_overrides.chat.base_url == "https://api.deepseek.com/v1"

    # embedding 应保持默认
    assert ds_overrides.embedding.embedding_model == "mlx-community/Qwen3-Embedding-8B-4bit-DWQ"


# ── FakeProvider helpers ──


class _FakeChat:
    def __init__(self, model: str = "test-chat") -> None:
        self.chat_model_name = model

    def generate_text(self, *, prompt: str, **kwargs: object) -> str:
        return f"[chat response for: {prompt[:50]}...]"


class _FakeEmbedder:
    def __init__(self, model: str = "test-embed") -> None:
        self.embedding_model_name = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        result: list[list[float]] = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            result.append([float((seed >> i) & 0xFF) / 255.0 for i in range(0, 16)])
        return result


# ── override priority: --model > compatibility env ──


def test_override_priority_model_beats_compat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """--model deepseek-chat must override compatible environment chat config."""
    from rag.assembly import (
        AssemblyConfig,
        AssemblyOverrides,
        AssemblyRequest,
        CapabilityRequirements,
        ProviderConfig,
    )
    from tests.core.test_capability_assembly import _isolated_service

    compatibility_config = AssemblyConfig(
        profiles=(
            ProviderConfig(
                profile_id="compat-default",
                provider_kind="openai-compatible",
                location="local",
                chat_model="compat-default-model",
                base_url="http://compat-url/v1",
            ),
        ),
    )

    service = _isolated_service(monkeypatch, compatibility_config=compatibility_config)

    # CLI 传入 --model deepseek-chat 对应的 overrides
    override_chat = ProviderConfig(
        provider_kind="openai-compatible",
        chat_model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test",
    )

    bundle = service.assemble_request(
        AssemblyRequest(
            requirements=CapabilityRequirements(require_chat=True),
            overrides=AssemblyOverrides(chat=override_chat),
        )
    )

    assert bundle.chat_bindings
    assert bundle.chat_bindings[0].model_name == "deepseek-chat"


# ── generation config ──────────────────────────────────────────

_GENERATION_CATALOG_YAML = (
    CATALOG_YAML
    + """
generation:
  summary:
    max_tokens: 8192
    temperature: 0.3

  answer:
    max_tokens: 4096

  planner:
    max_tokens: 4096
    temperature: 0.3

  synthesize:
    max_tokens: 8192

  factcheck:
    max_tokens: 2048
    temperature: 0.1
"""
)


@pytest.fixture
def gen_catalog_path(tmp_path: Path) -> Path:
    path = tmp_path / "models_gen.yaml"
    path.write_text(_GENERATION_CATALOG_YAML, encoding="utf-8")
    return path


@pytest.fixture
def gen_catalog(gen_catalog_path: Path) -> ModelCatalog:
    return ModelCatalog.from_yaml(str(gen_catalog_path))


def test_generation_config_parsing(gen_catalog: ModelCatalog) -> None:
    """models.yaml 中 generation.summary 能正确解析"""
    gen = gen_catalog.generation

    assert gen.summary.max_tokens == 8192
    assert gen.summary.temperature == 0.3

    assert gen.answer.max_tokens == 4096
    assert gen.answer.temperature is None  # YAML 未配置 temperature

    assert gen.planner.temperature == 0.3
    assert gen.synthesize.max_tokens == 8192
    assert gen.factcheck.max_tokens == 2048
    assert gen.factcheck.temperature == 0.1


def test_generation_config_defaults_when_missing(catalog: ModelCatalog) -> None:
    """无 generation section 时全部字段为 None"""
    gen = catalog.generation
    assert gen.summary.max_tokens is None
    assert gen.summary.temperature is None
    assert not hasattr(gen.answer, "model")


def test_resolve_runtime_config_includes_generation(gen_catalog: ModelCatalog) -> None:
    """resolve_runtime_config 返回的 ModelRuntimeConfig 包含 generation"""
    config = resolve_runtime_config(catalog=gen_catalog)
    assert config.generation.summary.max_tokens == 8192


def test_generation_task_rejects_nested_model_selector(tmp_path: Path) -> None:
    path = tmp_path / "nested-model.yaml"
    path.write_text(
        _GENERATION_CATALOG_YAML.replace(
            "summary:\n    max_tokens:",
            "summary:\n    model: deepseek-chat\n    max_tokens:",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="generation.summary.*model"):
        ModelCatalog.from_yaml(str(path))


def test_summarizer_receives_max_tokens(gen_catalog: ModelCatalog) -> None:
    """验证 summarizer 构造时 max_tokens 来自 generation.summary"""
    from rag.ingest.retrievalsummarizer import RetrievalSummaryConfig

    gen_summary = gen_catalog.generation.summary
    max_tokens = gen_summary.max_tokens or 4096

    config = RetrievalSummaryConfig(
        max_output_tokens=max_tokens,
        temperature=gen_summary.temperature,
    )
    assert config.max_output_tokens == 8192
    assert config.temperature == 0.3


def test_chat_model_requires_context_window_tokens(tmp_path: Path) -> None:
    path = tmp_path / "missing-context.yaml"
    path.write_text(
        CATALOG_YAML.replace("    context_window_tokens: 65536\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="deepseek-chat.*context_window_tokens"):
        ModelCatalog.from_yaml(str(path))


def test_tokenizer_cannot_override_model_context_window(tmp_path: Path) -> None:
    path = tmp_path / "global-context.yaml"
    path.write_text(
        CATALOG_YAML + "\ntokenizer:\n  max_context_tokens: 999\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tokenizer.max_context_tokens is unsupported"):
        ModelCatalog.from_yaml(str(path))
