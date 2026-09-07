"""
验证摘要生成：通过 resolve_runtime_config() 从 configs/models.yaml 读取全部配置。
不手写 model / base_url / max_tokens，全由 generation.summary 驱动。
使用 RecordingLLM 捕获真实调用参数，确认 max_tokens 透传正确。
"""

from __future__ import annotations

import os
import re

import pytest

from rag.assembly.bindings import ChatCapabilityBinding
from rag.assembly.support import _CompositeProvider, _OpenAICompatibleChatGenerator
from rag.ingest.retrievalsummarizer import RetrievalSummarizer, RetrievalSummaryConfig
from rag.models import resolve_runtime_config
from rag.models.assembly_adapter import resolve_task_model
from rag.models.catalog import ModelCatalog
from rag.runtime import _ChatGeneratorAdapter
from rag.schema.core import ParsedSection

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MLX_SUMMARY_VERIFY") != "1",
    reason="manual MLX summary verification; set RUN_MLX_SUMMARY_VERIFY=1",
)


# ── adapter factory ────────────────────────────────────────────


def _make_chat_adapter(model_spec):
    """从 ModelSpec 构建 _ChatGeneratorAdapter（不手写 model/base_url）"""
    generator = _OpenAICompatibleChatGenerator(
        model=model_spec.id,
        base_url=model_spec.base_url or "http://127.0.0.1:8080/v1",
        api_key="not-needed",
    )
    composite = _CompositeProvider(provider_name=model_spec.id, generator=generator)
    binding = ChatCapabilityBinding(backend=composite, location="local")
    return _ChatGeneratorAdapter(binding)


# ── RecordingLLM wrapper ───────────────────────────────────────


class RecordingLLM:
    """只捕获内部真实 LLM 调用，不做额外调用。"""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[dict] = []

    @property
    def provider_name(self):
        return getattr(self.inner, "provider_name", None)

    @property
    def model_name(self):
        return getattr(self.inner, "model_name", None)

    def generate_text(self, *, prompt: str, **kwargs):
        output = self.inner.generate_text(prompt=prompt, **kwargs)
        self.calls.append(
            {
                "method": "generate_text",
                "prompt": prompt,
                "kwargs": kwargs,
                "output": output,
            }
        )
        return output

    def chat(self, prompt: str, **kwargs):
        output = self.inner.chat(prompt, **kwargs)
        self.calls.append(
            {
                "method": "chat",
                "prompt": prompt,
                "kwargs": kwargs,
                "output": output,
            }
        )
        return output


# ── helpers ────────────────────────────────────────────────────


def _print_recording(recording: RecordingLLM, label: str):
    if not recording.calls:
        print(f"\n[{label}] 没有 LLM 调用（可能走 direct/empty 路径）")
        return
    call = recording.calls[-1]
    print(f"\n[{label}] method:  {call['method']}")
    print(f"[{label}] kwargs:  {call['kwargs']}")
    has_max_tokens = "max_tokens" in call["kwargs"]
    print(f"[{label}] max_tokens in kwargs: {'YES' if has_max_tokens else 'NO — 需要修复！'}")
    output = call["output"]
    print(f"[{label}] output:  {len(output)} chars")
    if output:
        print(f"[{label}] content preview:\n{output[:400]}")
    else:
        print(f"[{label}] content: (empty)")


def _evaluate(summary: str, original: str):
    """简单质量评估"""
    print("\n--- 质量评估 ---")
    lines = summary.strip().splitlines()
    fields = {}
    for line in lines:
        for prefix in ("Semantic Core:", "Fact Anchors:", "Retrieval Keywords:"):
            if line.startswith(prefix):
                fields[prefix.rstrip(":")] = line[len(prefix) :].strip()

    issues = []

    for field in ("Semantic Core", "Fact Anchors", "Retrieval Keywords"):
        val = fields.get(field, "")
        if val and val.lower() != "none":
            print(f"  ✅ {field}: 存在")
        else:
            print(f"  ❌ {field}: 缺失或为 none")
            issues.append(field)

    core = fields.get("Semantic Core", "")
    bad = ["this section", "the text", "the section", "本节", "本文", "该部分"]
    if any(b in core.lower() for b in bad):
        print("  ❌ Semantic Core 含废话前缀")
        issues.append("废话前缀")
    else:
        print("  ✅ Semantic Core 无废话前缀")

    anchors = fields.get("Fact Anchors", "")
    nums_in_original = set(re.findall(r"\d+", original))
    nums_in_summary = set(re.findall(r"\d+", anchors))
    hallucinated = nums_in_summary - nums_in_original
    key_nums = {n for n in nums_in_original if len(n) >= 3}
    missed_key = key_nums - nums_in_summary

    if not missed_key:
        print("  ✅ 关键数字均已保留")
    else:
        print(f"  ⚠️  缺失关键数字: {missed_key}")
    if not hallucinated:
        print("  ✅ 无幻觉数字")
    else:
        print(f"  ⚠️  可能出现幻觉数字: {hallucinated}")

    keywords = fields.get("Retrieval Keywords", "")
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
    print(f"  ℹ️  关键词数量: {len(kw_list)}")

    if not issues:
        print("\n  🎉 摘要质量合格！")
    else:
        print(f"\n  ⚠️  存在问题: {', '.join(issues)}")


# ── pytest fixtures ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def catalog():
    return ModelCatalog.from_yaml()


@pytest.fixture(scope="module")
def gen_config(catalog):
    return catalog.generation


@pytest.fixture(scope="module")
def model_spec(catalog, gen_config):
    return resolve_task_model(gen_config.summary, catalog)


# ── tests ──────────────────────────────────────────────────────


def test_section_summary(model_spec, gen_config, catalog):
    adapter = _make_chat_adapter(model_spec)
    recording = RecordingLLM(adapter)

    summarizer = RetrievalSummarizer(
        llm_client=recording,
        config=RetrievalSummaryConfig(
            max_output_tokens=gen_config.summary.max_tokens or 4096,
            temperature=gen_config.summary.temperature,
        ),
    )

    print(f"provider: {summarizer._provider_name()}")
    print(f"model:    {summarizer._model_name()}")

    section = ParsedSection(
        toc_path=("财务管理", "差旅报销", "审批流程"),
        heading_level=3,
        page_range=(12, 14),
        order_index=0,
        text=(
            "第五条 差旅报销审批流程\n\n"
            "5.1 单次差旅费用在人民币 5000 元以下的，由部门负责人审批。\n"
            "5.2 单次差旅费用在人民币 5000 元至 20000 元之间的，需经部门负责人初审后，"
            "报财务总监复审。\n"
            "5.3 单次差旅费用超过人民币 20000 元的，须经总经理最终批准。\n"
            "5.4 国际差旅无论金额大小，须提前 5 个工作日提交出差申请，"
            "并附详细行程计划和预算表。\n"
            "5.5 紧急差旅可在事后 3 个工作日内补办审批手续，但须事先获得部门负责人口头批准。\n"
            "5.6 住宿费标准：一线城市不超过 800 元/晚，二线城市不超过 500 元/晚，"
            "其他城市不超过 350 元/晚。\n"
            "5.7 餐饮补助每人每天 150 元，半天按 75 元计算。"
        ),
        char_range_start=0,
        char_range_end=680,
    )

    result = summarizer.summarize_section_with_metadata(section, "XX公司差旅报销管理制度（2025版）")

    _print_recording(recording, "Section")
    print(f"method:          {result.method}")
    print(f"fallback_reason: {result.fallback_reason}")

    if result.method == "llm":
        print("\n✅ LLM 摘要生成成功！\n")
        print("=" * 60)
        print(result.text)
        print("=" * 60)
        _evaluate(result.text, section.text)
    else:
        print(f"❌ 未走 LLM 路径，method={result.method}，fallback={result.fallback_reason}")

    # 验证 max_tokens 透传
    if recording.calls:
        assert "max_tokens" in recording.calls[-1]["kwargs"], "max_tokens should be in kwargs!"
        expected = gen_config.summary.max_tokens or 4096
        actual = recording.calls[-1]["kwargs"]["max_tokens"]
        assert actual == expected, f"Expected max_tokens={expected}, got {actual}"


def test_asset_summary(model_spec, gen_config, catalog):
    adapter = _make_chat_adapter(model_spec)
    recording = RecordingLLM(adapter)

    summarizer = RetrievalSummarizer(
        llm_client=recording,
        config=RetrievalSummaryConfig(
            max_output_tokens=gen_config.summary.max_tokens or 4096,
            temperature=gen_config.summary.temperature,
        ),
    )

    result = summarizer.summarize_asset_with_metadata(
        asset_type="table",
        asset_text=(
            "| 城市等级 | 住宿标准(元/晚) | 餐饮补助(元/天) |\n"
            "|---|---|---|\n"
            "| 一线 | 800 | 150 |\n"
            "| 二线 | 500 | 150 |\n"
            "| 其他 | 350 | 150 |"
        ),
        document_title="XX公司差旅报销管理制度（2025版）",
        toc_path=("财务管理", "差旅标准"),
        caption="表1: 差旅费用标准明细",
    )

    _print_recording(recording, "Asset")
    print(f"method:          {result.method}")
    print(f"fallback_reason: {result.fallback_reason}")

    if result.method == "llm":
        print("\n✅ Asset 摘要生成成功！\n")
        print("=" * 60)
        print(result.text)
        print("=" * 60)


def test_doc_summary(model_spec, gen_config, catalog):
    adapter = _make_chat_adapter(model_spec)
    recording = RecordingLLM(adapter)

    summarizer = RetrievalSummarizer(
        llm_client=recording,
        config=RetrievalSummaryConfig(
            max_output_tokens=gen_config.summary.max_tokens or 4096,
            temperature=gen_config.summary.temperature,
        ),
    )

    result = summarizer.summarize_doc_with_metadata(
        document_title="XX公司差旅报销管理制度（2025版）",
        section_summaries=[
            "Semantic Core: 差旅审批权限按金额分级。\n"
            "Fact Anchors: 5000, 20000, 部门负责人, 财务总监, 总经理\n"
            "Retrieval Keywords: 审批, 差旅, 报销, 金额阈值",
            "Semantic Core: 差旅费用住宿标准按城市等级划分。\n"
            "Fact Anchors: 800元, 500元, 350元, 150元/天, 75元/半天\n"
            "Retrieval Keywords: 住宿, 餐饮补助, 城市等级, 费用标准",
        ],
    )

    _print_recording(recording, "Doc")
    print(f"method:          {result.method}")
    print(f"fallback_reason: {result.fallback_reason}")

    if result.method == "llm_doc_reduce":
        print("\n✅ 文档级摘要生成成功！\n")
        print("=" * 60)
        print(result.text)
        print("=" * 60)


# ── model switching test ───────────────────────────────────────


def test_switch_summary_model(catalog):
    """验证只改 generation.summary.model 即可切换模型，不改业务代码"""
    task_config = catalog.generation.summary
    assert task_config.model is not None, "generation.summary.model should be set"

    # 默认模型
    spec1 = resolve_task_model(task_config, catalog)
    print(f"\n[Switch] 当前 summary model: {spec1.id}")

    # 切换到小型 Qwen（不改业务逻辑，只解析不同的 task config）
    small_task = type(task_config)(
        model="mlx-community/Qwen3-0.6B-4bit",
        max_tokens=task_config.max_tokens,
    )
    spec2 = resolve_task_model(small_task, catalog)
    print(f"[Switch] 切换后 summary model: {spec2.id}")
    assert spec2.id == "mlx-community/Qwen3-0.6B-4bit"

    print("[Switch] ✅ 只改 model ID 即可切换，业务代码零改动")


if __name__ == "__main__":
    # All config from models.yaml, no hardcoded values
    resolve_runtime_config()
    catalog = ModelCatalog.from_yaml()
    gen_config = catalog.generation

    # Resolve summary task model (generation.summary.model or fallback to defaults.primary_model)
    summary_model_spec = resolve_task_model(gen_config.summary, catalog)

    print("=" * 60)
    print("摘要生成验证")
    print(f"  模型: {summary_model_spec.id}")
    print(f"  地址: {summary_model_spec.base_url}")
    print(f"  summary.max_tokens: {gen_config.summary.max_tokens}")
    print(f"  summary.temperature: {gen_config.summary.temperature}")
    print("=" * 60)

    test_section_summary(summary_model_spec, gen_config, catalog)
    test_asset_summary(summary_model_spec, gen_config, catalog)
    test_doc_summary(summary_model_spec, gen_config, catalog)
    test_switch_summary_model(catalog)

    print("\n🎉 全部测试完成！")
