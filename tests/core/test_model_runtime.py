from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from agent_runtime.modeling.config import GenerationTaskConfig, ModelCapability, ModelRuntimeConfig
from agent_runtime.modeling.contracts import LLMCallStage
from rag.assembly.models import ProviderConfig
from rag.assembly.support import _OpenAICompatibleChatGenerator, build_provider
from rag.models.assembly_adapter import resolve_task_model, to_assembly_overrides
from rag.models.catalog import ModelCatalog
from rag.models.guard import EmbeddingSpaceMismatchError, assert_embedding_space_compatible
from rag.models.runtime import RuntimeOverrides, resolve_runtime_config
from rag.runtime import _generator_bindings_from_chat_bindings


class _StructuredPayload(BaseModel):
    answer: str


CATALOG_YAML = """
models:
  qwen_local:
    capability: chat
    provider: openai_compatible
    model: mlx-community/Qwen3-14B-4bit
    base_url: http://127.0.0.1:8080/v1
    context_window_tokens: 32768

  deepseek:
    capability: chat
    provider: openai_compatible
    model: deepseek-chat
    base_url: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY

  qwen_embedding_mlx:
    capability: embedding
    provider: mlx_embedding
    model: mlx-community/Qwen3-Embedding-8B-4bit-DWQ
    embedding_space: mlx/Qwen3-Embedding-8B-4bit-DWQ

  qwen3_reranker:
    capability: reranker
    provider: sentence_transformers
    model: Qwen/Qwen3-Reranker-4B

defaults:
  primary_model: qwen_local
  embedding_model: qwen_embedding_mlx
  reranker_model: qwen3_reranker

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
    assert catalog.get_model("qwen_local").capability == ModelCapability.CHAT
    assert catalog.get_model("deepseek").capability == ModelCapability.CHAT
    assert catalog.get_model("qwen_embedding_mlx").capability == ModelCapability.EMBEDDING
    assert catalog.get_model("qwen3_reranker").capability == ModelCapability.RERANKER
    assert catalog.get_model("qwen_local").context_window_tokens == 32768


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
  groq_gpt_oss_120b:
    capability: chat
    provider: groq
    model: openai/gpt-oss-120b
    context_window_tokens: 131072
  qwen_embedding_mlx:
    capability: embedding
    provider: local_mlx_embedding
    model: mlx-community/Qwen3-Embedding-8B-4bit-DWQ
    embedding_space: mlx/Qwen3-Embedding-8B-4bit-DWQ
  qwen3_reranker:
    capability: reranker
    provider: local_sentence_transformers
    model: Qwen/Qwen3-Reranker-4B

defaults:
  primary_model: groq_gpt_oss_120b
  embedding_model: qwen_embedding_mlx
  reranker_model: qwen3_reranker
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


def test_catalog_loads_llm_stage_budgets(catalog: ModelCatalog) -> None:
    budget = catalog.llm_stage_budgets[LLMCallStage.TOOL_DECISION]
    assert budget.max_input_tokens == 12000
    assert budget.max_output_tokens == 2048
    assert budget.safety_margin_tokens == 512


def test_catalog_defaults(catalog: ModelCatalog) -> None:
    assert catalog.get_default_primary().alias == "qwen_local"
    assert catalog.get_default_embedding().alias == "qwen_embedding_mlx"
    assert catalog.get_default_reranker().alias == "qwen3_reranker"


def test_catalog_list_models(catalog: ModelCatalog) -> None:
    chat_models = catalog.list_models(ModelCapability.CHAT)
    assert len(chat_models) == 2
    assert {m.alias for m in chat_models} == {"deepseek", "qwen_local"}


def test_catalog_unknown_model_raises(catalog: ModelCatalog) -> None:
    with pytest.raises(KeyError, match="Unknown model alias"):
        catalog.get_model("nonexistent")


# ── runtime resolution ──


def test_runtime_default_primary_model(catalog: ModelCatalog) -> None:
    config = resolve_runtime_config(RuntimeOverrides(), catalog=catalog)
    assert config.primary_model.alias == "qwen_local"
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
        RuntimeOverrides(model_alias="deepseek"),
        catalog=catalog,
    )
    assert config.primary_model.alias == "deepseek"
    assert config.primary_model.base_url == "https://api.deepseek.com/v1"


def test_runtime_rejects_capability_mismatch(catalog: ModelCatalog) -> None:
    with pytest.raises(ValueError, match="capability 'embedding'.*expected 'chat'"):
        resolve_runtime_config(
            RuntimeOverrides(model_alias="qwen_embedding_mlx"),
            catalog=catalog,
        )


def test_runtime_rejects_unknown_model(catalog: ModelCatalog) -> None:
    with pytest.raises(KeyError, match="Unknown model alias"):
        resolve_runtime_config(
            RuntimeOverrides(model_alias="gpt4"),
            catalog=catalog,
        )


def test_runtime_disabled_reranker(catalog: ModelCatalog) -> None:
    for alias in ("none", "null", "off", "false"):
        config = resolve_runtime_config(
            RuntimeOverrides(reranker_model_alias=alias),
            catalog=catalog,
        )
        assert config.reranker_model is None, f"reranker should be None for alias={alias!r}"


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
    spec = catalog.get_model("qwen_local")
    config = ModelRuntimeConfig(
        primary_model=spec,
        embedding_model=catalog.get_model("qwen_embedding_mlx"),
        reranker_model=catalog.get_model("qwen3_reranker"),
    )
    overrides = to_assembly_overrides(config)

    assert overrides.chat is not None
    assert overrides.chat.provider_kind == "openai-compatible"
    assert overrides.chat.chat_model == "mlx-community/Qwen3-14B-4bit"
    assert overrides.chat.base_url == "http://127.0.0.1:8080/v1"
    assert overrides.chat.api_key is None


def test_assembly_adapter_embedding_provider_config(catalog: ModelCatalog) -> None:
    spec = catalog.get_model("qwen_embedding_mlx")
    config = ModelRuntimeConfig(
        primary_model=catalog.get_model("qwen_local"),
        embedding_model=spec,
    )
    overrides = to_assembly_overrides(config)

    assert overrides.embedding is not None
    assert overrides.embedding.provider_kind == "mlx-embedding"
    assert overrides.embedding.embedding_model == "mlx-community/Qwen3-Embedding-8B-4bit-DWQ"


def test_assembly_adapter_reranker_maps_to_local_bge(catalog: ModelCatalog) -> None:
    spec = catalog.get_model("qwen3_reranker")
    config = ModelRuntimeConfig(
        primary_model=catalog.get_model("qwen_local"),
        embedding_model=catalog.get_model("qwen_embedding_mlx"),
        reranker_model=spec,
    )
    overrides = to_assembly_overrides(config)

    assert overrides.rerank is not None
    assert overrides.rerank.provider_kind == "local-bge"
    assert overrides.rerank.rerank_model == "Qwen/Qwen3-Reranker-4B"


def test_assembly_adapter_none_reranker(catalog: ModelCatalog) -> None:
    config = ModelRuntimeConfig(
        primary_model=catalog.get_model("qwen_local"),
        embedding_model=catalog.get_model("qwen_embedding_mlx"),
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
    4. --model deepseek 可以覆盖默认 primary_model
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

    # ── 3. 解析 runtime config（默认 qwen_local）──
    runtime_config = resolve_runtime_config(
        RuntimeOverrides(),
        catalog_path=str(catalog_path),
    )
    assert runtime_config.primary_model.alias == "qwen_local"
    assert runtime_config.embedding_model.alias == "qwen_embedding_mlx"
    assert runtime_config.reranker_model.alias == "qwen3_reranker"

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

    # ── 7. --model deepseek 覆盖测试 ──
    ds_config = resolve_runtime_config(
        RuntimeOverrides(model_alias="deepseek"),
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
    """--model deepseek must override compatible environment chat config."""
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

    # CLI 传入 --model deepseek 对应的 overrides
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
    model: qwen_local
    max_tokens: 8192
    temperature: 0.3

  answer:
    model: deepseek
    max_tokens: 4096

  planner:
    model: qwen_local
    max_tokens: 4096
    temperature: 0.3

  synthesize:
    model: qwen_local
    max_tokens: 8192

  factcheck:
    model: qwen_local
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

    assert gen.summary.model == "qwen_local"
    assert gen.summary.max_tokens == 8192
    assert gen.summary.temperature == 0.3

    assert gen.answer.model == "deepseek"
    assert gen.answer.max_tokens == 4096
    assert gen.answer.temperature is None  # YAML 未配置 temperature

    assert gen.planner.temperature == 0.3
    assert gen.synthesize.max_tokens == 8192
    assert gen.factcheck.max_tokens == 2048
    assert gen.factcheck.temperature == 0.1


def test_generation_config_defaults_when_missing(catalog: ModelCatalog) -> None:
    """无 generation section 时全部字段为 None"""
    gen = catalog.generation
    assert gen.summary.model is None
    assert gen.summary.max_tokens is None
    assert gen.summary.temperature is None
    assert gen.answer.model is None


def test_resolve_runtime_config_includes_generation(gen_catalog: ModelCatalog) -> None:
    """resolve_runtime_config 返回的 ModelRuntimeConfig 包含 generation"""
    config = resolve_runtime_config(catalog=gen_catalog)
    assert config.generation.summary.model == "qwen_local"
    assert config.generation.summary.max_tokens == 8192


def test_resolve_task_model_uses_explicit_model(gen_catalog: ModelCatalog) -> None:
    """task_config.model 有值时直接使用"""
    spec = resolve_task_model(gen_catalog.generation.answer, gen_catalog)
    assert spec.alias == "deepseek"
    assert spec.model == "deepseek-chat"


def test_resolve_task_model_falls_back_to_default(gen_catalog: ModelCatalog) -> None:
    """task_config.model 为 None 时 fallback 到 defaults.primary_model"""
    task = GenerationTaskConfig(max_tokens=4096)  # model=None
    spec = resolve_task_model(task, gen_catalog)
    assert spec.alias == "qwen_local"


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


def test_switch_summary_model(gen_catalog: ModelCatalog) -> None:
    """只改 generation.summary.model，不改业务代码即可切换到其他模型"""
    # 用 qwen_local（默认）
    spec1 = resolve_task_model(gen_catalog.generation.summary, gen_catalog)
    assert spec1.alias == "qwen_local"

    # 构造一个新的 task config，切换到 deepseek（模型池中存在）
    switched = GenerationTaskConfig(
        model="deepseek",
        max_tokens=gen_catalog.generation.summary.max_tokens,
    )
    spec2 = resolve_task_model(switched, gen_catalog)
    assert spec2.alias == "deepseek"
    assert spec2.model == "deepseek-chat"
    assert spec2.base_url == "https://api.deepseek.com/v1"
    # 业务逻辑不变：只需 resolve_task_model(task_config, catalog)
