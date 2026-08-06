from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import random
import statistics
import time
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
from datasets import load_dataset
from tqdm.auto import tqdm

from agent_runtime.text import load_env_file
from rag import AssemblyRequest, CapabilityRequirements, RAGRuntime, StorageComponentConfig, StorageConfig
from rag.assembly import AssemblyConfig, AssemblyOverrides, CapabilityAssemblyService, ProviderConfig, TokenizerConfig
from rag.ingest.pipeline import IngestRequest
from rag.models.assembly_adapter import to_assembly_overrides
from rag.models.runtime import RuntimeOverrides, resolve_runtime_config
from rag.retrieval import QueryOptions
from rag.schema.core import SourceType
from rag.schema.runtime import (
    AccessPolicy,
    DataContractMetadataRepo,
    RuntimeMode,
)

if TYPE_CHECKING:
    from typing import Literal

    class _EmbeddingBackend:
        def set_embedding_batch_size(self, batch_size: int) -> None: ...
        def set_device_preference(self, device: str) -> None: ...
        def set_embedding_call_logging(self, enabled: bool) -> None: ...
        def set_backend_progress_enabled(self, enabled: bool) -> None: ...
        def reset_embedding_stats(self) -> None: ...
        def embedding_runtime_info(self) -> dict[str, object]: ...
        def embedding_stats(self) -> dict[str, object]: ...


FIQA_DATASET = "fiqa"
MEDICAL_RETRIEVAL_DATASET = "medical_retrieval"
DEFAULT_VECTOR_BACKEND = "milvus"
DEFAULT_MILVUS_DSN = "http://127.0.0.1:19530"
FIQA_BEIR_ZIP_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"


@dataclass(frozen=True, slots=True)
class BenchmarkPaths:
    dataset_dir: Path
    raw_dir: Path
    prepared_root: Path
    index_root: Path
    eval_root: Path
    subsets_root: Path

    @property
    def prepared_dir(self) -> Path:
        return self.prepared_variant_dir("full")

    @property
    def index_dir(self) -> Path:
        return self.index_variant_dir("full")

    @property
    def eval_dir(self) -> Path:
        return self.eval_variant_dir("retrieval", "full")

    def prepared_variant_dir(self, variant: str) -> Path:
        return self.prepared_root / variant

    def index_variant_dir(self, variant: str) -> Path:
        return self.index_root / variant

    def eval_variant_dir(self, task: str, variant: str) -> Path:
        return self.eval_root / task / variant

    def subset_dir(self, name: str) -> Path:
        return self.subsets_root / name


def ensure_benchmark_layout(
    paths: BenchmarkPaths,
    *,
    variants: Sequence[str] = ("full", "mini"),
    tasks: Sequence[str] = ("retrieval",),
) -> BenchmarkPaths:
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.subsets_root.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        paths.prepared_variant_dir(variant).mkdir(parents=True, exist_ok=True)
        paths.index_variant_dir(variant).mkdir(parents=True, exist_ok=True)
        for task in tasks:
            paths.eval_variant_dir(task, variant).mkdir(parents=True, exist_ok=True)
    return paths


@dataclass(frozen=True, slots=True)
class BenchmarkDatasetSpec:
    dataset: str
    display_name: str
    default_split: str
    mini_query_count: int
    mini_target_doc_count: int
    beir_zip_url: str | None = None
    hf_dataset: str | None = None
    hf_qrels_dataset: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkDownloadResult:
    dataset: str
    archive_path: Path
    corpus_path: Path
    queries_path: Path
    qrels_path: Path


@dataclass(frozen=True, slots=True)
class BenchmarkPrepareResult:
    dataset: str
    split: str
    document_count: int
    query_count: int
    qrel_count: int
    documents_path: Path
    queries_path: Path
    qrels_path: Path


DATASET_SPECS: dict[str, BenchmarkDatasetSpec] = {
    FIQA_DATASET: BenchmarkDatasetSpec(
        dataset=FIQA_DATASET,
        display_name="BEIR FiQA",
        default_split="test",
        mini_query_count=200,
        mini_target_doc_count=5000,
        beir_zip_url=FIQA_BEIR_ZIP_URL,
    ),
    MEDICAL_RETRIEVAL_DATASET: BenchmarkDatasetSpec(
        dataset=MEDICAL_RETRIEVAL_DATASET,
        display_name="C-MTEB MedicalRetrieval",
        default_split="dev",
        mini_query_count=300,
        mini_target_doc_count=10000,
        hf_dataset="C-MTEB/MedicalRetrieval",
        hf_qrels_dataset="C-MTEB/MedicalRetrieval-qrels",
    ),
}


def benchmark_dataset_spec(dataset: str) -> BenchmarkDatasetSpec:
    try:
        return DATASET_SPECS[dataset]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unsupported benchmark dataset: {dataset}. Supported: {supported}") from exc


@dataclass(frozen=True, slots=True)
class BenchmarkIngestResult:
    dataset: str
    request_count: int
    success_count: int
    duplicate_count: int
    failure_count: int
    indexed_object_count: int = 0
    elapsed_ms: float | None = None

    @property
    def docs_per_second(self) -> float:
        if not self.elapsed_ms or self.elapsed_ms <= 0:
            return 0.0
        return self.request_count / (self.elapsed_ms / 1000.0)

    @property
    def indexed_objects_per_second(self) -> float:
        if not self.elapsed_ms or self.elapsed_ms <= 0:
            return 0.0
        return self.indexed_object_count / (self.elapsed_ms / 1000.0)

    def as_json(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "request_count": self.request_count,
            "success_count": self.success_count,
            "duplicate_count": self.duplicate_count,
            "failure_count": self.failure_count,
            "indexed_object_count": self.indexed_object_count,
            "elapsed_ms": None if self.elapsed_ms is None else round(self.elapsed_ms, 3),
            "docs_per_second": round(self.docs_per_second, 3),
            "indexed_objects_per_second": round(self.indexed_objects_per_second, 3),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkEmbeddingRuntimeInfo:
    provider: str
    model_name: str | None
    device: str
    encode_batch_size: int


@dataclass(frozen=True, slots=True)
class BenchmarkIngestProfileResult:
    dataset: str
    doc_count: int
    indexed_object_count: int
    ingest_strategy: str
    ingest_batch_size: int
    encode_batch_size: int
    embedding_model: str | None
    device: str
    total_elapsed_ms: float
    embedding_elapsed_ms: float

    @property
    def docs_per_second(self) -> float:
        if self.total_elapsed_ms <= 0:
            return 0.0
        return self.doc_count / (self.total_elapsed_ms / 1000.0)

    @property
    def indexed_objects_per_second(self) -> float:
        if self.total_elapsed_ms <= 0:
            return 0.0
        return self.indexed_object_count / (self.total_elapsed_ms / 1000.0)

    def as_json(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "doc_count": self.doc_count,
            "indexed_object_count": self.indexed_object_count,
            "ingest_strategy": self.ingest_strategy,
            "ingest_batch_size": self.ingest_batch_size,
            "encode_batch_size": self.encode_batch_size,
            "embedding_model": self.embedding_model,
            "device": self.device,
            "total_elapsed_ms": round(self.total_elapsed_ms, 3),
            "embedding_elapsed_ms": round(self.embedding_elapsed_ms, 3),
            "docs_per_second": round(self.docs_per_second, 3),
            "indexed_objects_per_second": round(self.indexed_objects_per_second, 3),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkQueryRecord:
    query_id: str
    query_text: str


@dataclass(frozen=True, slots=True)
class BenchmarkQrel:
    query_id: str
    doc_id: str
    relevance: int


@dataclass(frozen=True, slots=True)
class BenchmarkPerQueryResult:
    run_id: str
    dataset: str
    query_id: str
    query_text: str
    predicted_doc_ids: list[str]
    gold_doc_ids: list[str]
    hit_at_10: int
    recall_at_10: float
    reciprocal_rank: float
    ndcg: float
    latency_ms: float

    def as_json(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "query_id": self.query_id,
            "query_text": self.query_text,
            "predicted_doc_ids": self.predicted_doc_ids,
            "gold_doc_ids": self.gold_doc_ids,
            "hit_at_10": self.hit_at_10,
            "recall_at_10": round(self.recall_at_10, 6),
            "reciprocal_rank": round(self.reciprocal_rank, 6),
            "ndcg": round(self.ndcg, 6),
            "latency_ms": round(self.latency_ms, 3),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkRunSummary:
    run_id: str
    dataset: str
    split: str
    query_count: int
    top_k: int
    evidence_top_k: int
    embedding_model: str | None
    retrieval_profile: str
    rerank_enabled: bool
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    avg_latency_ms: float
    p95_latency_ms: float

    @property
    def queries_per_second(self) -> float:
        if self.avg_latency_ms <= 0:
            return 0.0
        return 1000.0 / self.avg_latency_ms

    def baseline_row(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "query_count": self.query_count,
            "top_k": self.top_k,
            "embedding_model": self.embedding_model or "",
            "retrieval_profile": self.retrieval_profile,
            "rerank_enabled": self.rerank_enabled,
            "Recall@10": round(self.recall_at_10, 6),
            "MRR@10": round(self.mrr_at_10, 6),
            "NDCG@10": round(self.ndcg_at_10, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "queries_per_second": round(self.queries_per_second, 3),
        }

    def as_json(self) -> dict[str, object]:
        payload = dict(self.baseline_row())
        payload.update(
            {
                "split": self.split,
                "evidence_top_k": self.evidence_top_k,
            }
        )
        return payload


class RetrievalBenchmarkEvaluator:
    def __init__(
        self,
        *,
        runtime: RAGRuntime,
        dataset: str,
        split: str,
        retrieval_profile: str,
        top_k: int,
        evidence_top_k: int,
        retrieval_pool_k: int | None = None,
        rerank_enabled: bool,
        rerank_pool_k: int | None = None,
        access_policy: AccessPolicy | None = None,
    ) -> None:
        self.runtime = runtime
        self.dataset = dataset
        self.split = split
        self.retrieval_profile = retrieval_profile
        self.top_k = top_k
        self.evidence_top_k = evidence_top_k
        self.retrieval_pool_k = retrieval_pool_k
        self.rerank_enabled = rerank_enabled
        self.rerank_pool_k = rerank_pool_k
        self.access_policy = access_policy or benchmark_access_policy()

    def evaluate(
        self,
        *,
        queries_path: Path,
        qrels_path: Path,
        eval_dir: Path,
        query_limit: int | None = None,
    ) -> BenchmarkRunSummary:
        self._validate_retrieval_index()
        queries = load_queries(queries_path)
        if query_limit is not None:
            queries = queries[: max(query_limit, 0)]
        qrels = load_qrels(qrels_path)
        gold_by_query = group_qrels_by_query(qrels)
        eval_dir.mkdir(parents=True, exist_ok=True)
        run_id = benchmark_run_id(self.dataset)
        run_dir = eval_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        per_query_path = run_dir / "per_query.jsonl"
        cumulative_per_query_path = eval_dir / "per_query.jsonl"
        per_query_results: list[BenchmarkPerQueryResult] = []
        latencies: list[float] = []

        with per_query_path.open("w", encoding="utf-8") as handle:
            for query_record in _progress(
                queries,
                total=len(queries),
                desc=f"Evaluating {self.dataset} retrieval",
                unit="query",
            ):
                if query_record.query_id not in gold_by_query:
                    continue
                started = time.perf_counter()
                retrieval_result = self.runtime.retrieval_service.retrieve(
                    query_record.query_text,
                    access_policy=self.access_policy,
                    query_options=QueryOptions(
                        retrieval_profile=cast(
                            "Literal['bypass', 'fast', 'auto', 'deep', 'asset']",
                            self.retrieval_profile,
                        ),
                        top_k=self.top_k,
                        evidence_top_k=self.evidence_top_k,
                        retrieval_pool_k=self.retrieval_pool_k,
                        enable_rerank=self.rerank_enabled,
                        rerank_pool_k=self.rerank_pool_k,
                        max_context_tokens=self.runtime.token_contract.max_context_tokens,
                    ),
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                predicted_doc_ids = retrieval_result.reranked_benchmark_doc_ids[: self.top_k]
                gold = gold_by_query[query_record.query_id]
                metrics = compute_doc_ranking_metrics(
                    predicted_doc_ids=predicted_doc_ids,
                    gold_relevances=gold,
                    top_k=self.top_k,
                )
                result = BenchmarkPerQueryResult(
                    run_id=run_id,
                    dataset=self.dataset,
                    query_id=query_record.query_id,
                    query_text=query_record.query_text,
                    predicted_doc_ids=predicted_doc_ids,
                    gold_doc_ids=sorted(gold),
                    hit_at_10=cast(int, metrics["hit_at_10"]),
                    recall_at_10=metrics["recall_at_10"],
                    reciprocal_rank=metrics["mrr_at_10"],
                    ndcg=metrics["ndcg_at_10"],
                    latency_ms=latency_ms,
                )
                handle.write(json.dumps(result.as_json(), ensure_ascii=False) + "\n")
                per_query_results.append(result)
                latencies.append(latency_ms)

        summary = BenchmarkRunSummary(
            run_id=run_id,
            dataset=self.dataset,
            split=self.split,
            query_count=len(per_query_results),
            top_k=self.top_k,
            evidence_top_k=self.evidence_top_k,
            embedding_model=_embedding_model_name(self.runtime),
            retrieval_profile=self.retrieval_profile,
            rerank_enabled=self.rerank_enabled,
            recall_at_10=_mean(result.recall_at_10 for result in per_query_results),
            mrr_at_10=_mean(result.reciprocal_rank for result in per_query_results),
            ndcg_at_10=_mean(result.ndcg for result in per_query_results),
            avg_latency_ms=_mean(latencies),
            p95_latency_ms=_p95(latencies),
        )
        append_baseline_row(eval_dir / "baseline.csv", summary)
        summary_payload = summary.as_json()
        write_json(eval_dir / "run_summary.json", summary_payload)
        write_json(run_dir / "run_summary.json", summary_payload)
        append_jsonl(eval_dir / "run_history.jsonl", summary_payload)
        for result in per_query_results:
            append_jsonl(cumulative_per_query_path, result.as_json())
        return summary

    def _validate_retrieval_index(self) -> None:
        metadata = cast(DataContractMetadataRepo, self.runtime.stores.metadata_repo)
        documents = metadata.list_documents()
        document_count = len(documents)
        section_count = sum(len(metadata.list_sections(doc_id=document.doc_id)) for document in documents)
        asset_count = sum(len(metadata.list_assets(doc_id=document.doc_id)) for document in documents)
        vector_count = sum(
            self.runtime.stores.vector_repo.count_vectors(item_kind=item_kind)
            for item_kind in ("doc_summary", "section_summary", "asset_summary")
        )
        if document_count <= 0 or section_count <= 0:
            raise RuntimeError(
                "benchmark index is empty: "
                f"storage_root={self.runtime.stores.root} documents={document_count} "
                f"sections={section_count} assets={asset_count}. "
                "Run benchmark ingest before evaluation."
            )
        vector_required_profiles = {"fast", "auto", "deep", "asset"}
        if self.retrieval_profile in vector_required_profiles and vector_count <= 0:
            raise RuntimeError(
                "benchmark vector index is empty: "
                f"storage_root={self.runtime.stores.root} vectors={vector_count} "
                f"retrieval_profile={self.retrieval_profile}. "
                "Run benchmark ingest before evaluation."
            )


def default_benchmark_paths(dataset: str) -> BenchmarkPaths:
    dataset_dir = Path("data") / "benchmarks" / dataset
    return BenchmarkPaths(
        dataset_dir=dataset_dir,
        raw_dir=dataset_dir / "raw",
        prepared_root=dataset_dir / "prepared",
        index_root=dataset_dir / "index",
        eval_root=dataset_dir / "eval",
        subsets_root=dataset_dir / "subsets",
    )


def download_fiqa(
    raw_dir: Path,
    *,
    force: bool = False,
    timeout_seconds: float = 120.0,
) -> BenchmarkDownloadResult:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "fiqa.zip"
    _ensure_valid_fiqa_archive(
        archive_path,
        force=force,
        timeout_seconds=timeout_seconds,
    )

    with zipfile.ZipFile(archive_path) as zf:
        corpus_member = _required_zip_member(zf, "corpus.jsonl")
        queries_member = _required_zip_member(zf, "queries.jsonl")
        qrels_member = _required_zip_member(zf, "qrels/test.tsv")
        corpus_path = raw_dir / "corpus.jsonl"
        queries_path = raw_dir / "queries.jsonl"
        qrels_path = raw_dir / "qrels" / "test.tsv"
        qrels_path.parent.mkdir(parents=True, exist_ok=True)
        _extract_zip_member(zf, corpus_member, corpus_path)
        _extract_zip_member(zf, queries_member, queries_path)
        _extract_zip_member(zf, qrels_member, qrels_path)

    return BenchmarkDownloadResult(
        dataset=FIQA_DATASET,
        archive_path=archive_path,
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
    )


def download_medical_retrieval(
    raw_dir: Path,
    *,
    force: bool = False,
) -> BenchmarkDownloadResult:
    spec = benchmark_dataset_spec(MEDICAL_RETRIEVAL_DATASET)
    if spec.hf_dataset is None or spec.hf_qrels_dataset is None:
        raise RuntimeError("Medical retrieval dataset spec is missing Hugging Face sources.")
    raw_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = raw_dir / "corpus.jsonl"
    queries_path = raw_dir / "queries.jsonl"
    qrels_path = raw_dir / "qrels" / f"{spec.default_split}.jsonl"
    qrels_path.parent.mkdir(parents=True, exist_ok=True)

    if force:
        corpus_path.unlink(missing_ok=True)
        queries_path.unlink(missing_ok=True)
        qrels_path.unlink(missing_ok=True)

    if not corpus_path.exists():
        corpus_ds = load_dataset(spec.hf_dataset, split="corpus")
        with corpus_path.open("w", encoding="utf-8") as handle:
            for row in _progress(
                corpus_ds,
                total=len(corpus_ds),
                desc="Downloading MedicalRetrieval corpus",
                unit="doc",
            ):
                handle.write(json.dumps({"id": row["id"], "text": row["text"]}, ensure_ascii=False) + "\n")

    if not queries_path.exists():
        queries_ds = load_dataset(spec.hf_dataset, split="queries")
        with queries_path.open("w", encoding="utf-8") as handle:
            for row in _progress(
                queries_ds,
                total=len(queries_ds),
                desc="Downloading MedicalRetrieval queries",
                unit="query",
            ):
                handle.write(json.dumps({"id": row["id"], "text": row["text"]}, ensure_ascii=False) + "\n")

    if not qrels_path.exists():
        qrels_ds = load_dataset(spec.hf_qrels_dataset, split=spec.default_split)
        with qrels_path.open("w", encoding="utf-8") as handle:
            for row in _progress(
                qrels_ds,
                total=len(qrels_ds),
                desc="Downloading MedicalRetrieval qrels",
                unit="qrel",
            ):
                handle.write(
                    json.dumps(
                        {"qid": row["qid"], "pid": row["pid"], "score": int(row["score"])},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    return BenchmarkDownloadResult(
        dataset=MEDICAL_RETRIEVAL_DATASET,
        archive_path=raw_dir,
        corpus_path=corpus_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
    )


def download_public_benchmark(
    dataset: str,
    raw_dir: Path,
    *,
    force: bool = False,
    timeout_seconds: float = 120.0,
) -> BenchmarkDownloadResult:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if dataset == FIQA_DATASET:
        return download_fiqa(raw_dir, force=force, timeout_seconds=timeout_seconds)
    if dataset == MEDICAL_RETRIEVAL_DATASET:
        return download_medical_retrieval(raw_dir, force=force)
    spec = benchmark_dataset_spec(dataset)
    raise ValueError(f"Download is not implemented for dataset: {spec.dataset}")


def prepare_fiqa(
    raw_dir: Path,
    prepared_dir: Path,
    *,
    split: str = "test",
) -> BenchmarkPrepareResult:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = raw_dir / "corpus.jsonl"
    queries_path = raw_dir / "queries.jsonl"
    qrels_path = raw_dir / "qrels" / f"{split}.tsv"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing raw corpus: {corpus_path}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing raw queries: {queries_path}")
    if not qrels_path.exists():
        raise FileNotFoundError(f"Missing raw qrels: {qrels_path}")

    qrels = load_qrels_tsv(qrels_path)
    positive_qrels = [qrel for qrel in qrels if qrel.relevance > 0]
    query_ids = {qrel.query_id for qrel in positive_qrels}
    corpus_line_count = _count_lines(corpus_path)
    query_line_count = _count_lines(queries_path)

    documents_out = prepared_dir / "documents.jsonl"
    queries_out = prepared_dir / "queries.jsonl"
    qrels_out = prepared_dir / "qrels.jsonl"

    document_count = 0
    with documents_out.open("w", encoding="utf-8") as handle:
        for record in _progress(
            iter_jsonl(corpus_path),
            total=corpus_line_count,
            desc="Preparing FiQA documents",
            unit="doc",
        ):
            doc_id = _coerce_required_str(record.get("_id") or record.get("doc_id"), field_name="doc_id")
            title = _coerce_optional_str(record.get("title")) or doc_id
            text = _coerce_optional_str(record.get("text")) or ""
            payload = {
                "doc_id": doc_id,
                "title": title,
                "text": text,
                "source_type": SourceType.PLAIN_TEXT.value,
                "metadata": {
                    "dataset": FIQA_DATASET,
                    "benchmark": True,
                    "benchmark_dataset": FIQA_DATASET,
                    "benchmark_doc_id": doc_id,
                    "parent_doc_id": doc_id,
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            document_count += 1

    query_count = 0
    with queries_out.open("w", encoding="utf-8") as handle:
        for record in _progress(
            iter_jsonl(queries_path),
            total=query_line_count,
            desc="Preparing FiQA queries",
            unit="query",
        ):
            query_id = _coerce_required_str(record.get("_id") or record.get("query_id"), field_name="query_id")
            if query_id not in query_ids:
                continue
            text = _coerce_required_str(record.get("text") or record.get("query"), field_name="query_text")
            payload = {
                "query_id": query_id,
                "query_text": text,
                "metadata": {"dataset": FIQA_DATASET, "split": split},
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            query_count += 1

    with qrels_out.open("w", encoding="utf-8") as handle:
        for qrel in _progress(
            positive_qrels,
            total=len(positive_qrels),
            desc="Preparing FiQA qrels",
            unit="qrel",
        ):
            handle.write(
                json.dumps(
                    {
                        "query_id": qrel.query_id,
                        "doc_id": qrel.doc_id,
                        "relevance": qrel.relevance,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return BenchmarkPrepareResult(
        dataset=FIQA_DATASET,
        split=split,
        document_count=document_count,
        query_count=query_count,
        qrel_count=len(positive_qrels),
        documents_path=documents_out,
        queries_path=queries_out,
        qrels_path=qrels_out,
    )


def prepare_medical_retrieval(
    raw_dir: Path,
    prepared_dir: Path,
    *,
    split: str = "dev",
) -> BenchmarkPrepareResult:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = raw_dir / "corpus.jsonl"
    queries_path = raw_dir / "queries.jsonl"
    qrels_path = raw_dir / "qrels" / f"{split}.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing raw corpus: {corpus_path}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing raw queries: {queries_path}")
    if not qrels_path.exists():
        raise FileNotFoundError(f"Missing raw qrels: {qrels_path}")

    qrels = load_qrels_jsonl(qrels_path)
    positive_qrels = [qrel for qrel in qrels if qrel.relevance > 0]
    query_ids = {qrel.query_id for qrel in positive_qrels}

    documents_out = prepared_dir / "documents.jsonl"
    queries_out = prepared_dir / "queries.jsonl"
    qrels_out = prepared_dir / "qrels.jsonl"

    document_count = 0
    corpus_line_count = _count_lines(corpus_path)
    with documents_out.open("w", encoding="utf-8") as handle:
        for record in _progress(
            iter_jsonl(corpus_path),
            total=corpus_line_count,
            desc="Preparing MedicalRetrieval documents",
            unit="doc",
        ):
            doc_id = _coerce_required_str(record.get("id") or record.get("doc_id"), field_name="doc_id")
            title = _coerce_optional_str(record.get("title")) or doc_id
            text = _coerce_optional_str(record.get("text")) or ""
            handle.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "text": text,
                        "source_type": SourceType.PLAIN_TEXT.value,
                        "metadata": {
                            "dataset": MEDICAL_RETRIEVAL_DATASET,
                            "benchmark": True,
                            "benchmark_dataset": MEDICAL_RETRIEVAL_DATASET,
                            "benchmark_doc_id": doc_id,
                            "parent_doc_id": doc_id,
                            "language": "zh",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            document_count += 1

    query_count = 0
    query_line_count = _count_lines(queries_path)
    with queries_out.open("w", encoding="utf-8") as handle:
        for record in _progress(
            iter_jsonl(queries_path),
            total=query_line_count,
            desc="Preparing MedicalRetrieval queries",
            unit="query",
        ):
            query_id = _coerce_required_str(record.get("id") or record.get("query_id"), field_name="query_id")
            if query_id not in query_ids:
                continue
            text = _coerce_required_str(record.get("text") or record.get("query"), field_name="query_text")
            handle.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "query_text": text,
                        "metadata": {
                            "dataset": MEDICAL_RETRIEVAL_DATASET,
                            "split": split,
                            "language": "zh",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            query_count += 1

    with qrels_out.open("w", encoding="utf-8") as handle:
        for qrel in _progress(
            positive_qrels,
            total=len(positive_qrels),
            desc="Preparing MedicalRetrieval qrels",
            unit="qrel",
        ):
            handle.write(
                json.dumps(
                    {
                        "query_id": qrel.query_id,
                        "doc_id": qrel.doc_id,
                        "relevance": qrel.relevance,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return BenchmarkPrepareResult(
        dataset=MEDICAL_RETRIEVAL_DATASET,
        split=split,
        document_count=document_count,
        query_count=query_count,
        qrel_count=len(positive_qrels),
        documents_path=documents_out,
        queries_path=queries_out,
        qrels_path=qrels_out,
    )


def build_prepared_mini_subset(
    *,
    dataset: str,
    source_prepared_dir: Path,
    target_prepared_dir: Path,
    query_count: int,
    target_doc_count: int,
    seed: int = 42,
) -> BenchmarkPrepareResult:
    target_prepared_dir.mkdir(parents=True, exist_ok=True)
    full_queries = load_queries(source_prepared_dir / "queries.jsonl")
    full_qrels = load_qrels(source_prepared_dir / "qrels.jsonl")
    gold_by_query = group_qrels_by_query(full_qrels)
    eligible_query_ids = [record.query_id for record in full_queries if record.query_id in gold_by_query]
    rng = random.Random(seed)
    selected_query_ids = set(rng.sample(eligible_query_ids, min(query_count, len(eligible_query_ids))))

    selected_qrels = [qrel for qrel in full_qrels if qrel.query_id in selected_query_ids and qrel.relevance > 0]
    positive_doc_ids = {qrel.doc_id for qrel in selected_qrels}
    selected_queries = [record for record in full_queries if record.query_id in selected_query_ids]

    documents_out = target_prepared_dir / "documents.jsonl"
    queries_out = target_prepared_dir / "queries.jsonl"
    qrels_out = target_prepared_dir / "qrels.jsonl"

    selected_documents: dict[str, dict[str, object]] = {}
    all_documents: list[dict[str, object]] = []
    for record in iter_jsonl(source_prepared_dir / "documents.jsonl"):
        doc_id = _coerce_required_str(record.get("doc_id"), field_name="doc_id")
        all_documents.append(record)
        if doc_id in positive_doc_ids:
            selected_documents[doc_id] = record

    remaining = [
        record
        for record in all_documents
        if _coerce_required_str(record.get("doc_id"), field_name="doc_id") not in selected_documents
    ]
    rng.shuffle(remaining)
    target_size = max(target_doc_count, len(selected_documents))
    for record in remaining:
        if len(selected_documents) >= target_size:
            break
        doc_id = _coerce_required_str(record.get("doc_id"), field_name="doc_id")
        selected_documents[doc_id] = record

    with documents_out.open("w", encoding="utf-8") as handle:
        for doc_id in sorted(selected_documents):
            record = selected_documents[doc_id]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with queries_out.open("w", encoding="utf-8") as handle:
        for query_record in selected_queries:
            handle.write(
                json.dumps(
                    {
                        "query_id": query_record.query_id,
                        "query_text": query_record.query_text,
                        "metadata": {
                            "dataset": dataset,
                            "subset": "mini",
                            "split": benchmark_dataset_spec(dataset).default_split,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with qrels_out.open("w", encoding="utf-8") as handle:
        for qrel in selected_qrels:
            handle.write(
                json.dumps(
                    {"query_id": qrel.query_id, "doc_id": qrel.doc_id, "relevance": qrel.relevance},
                    ensure_ascii=False,
                )
                + "\n"
            )

    split = benchmark_dataset_spec(dataset).default_split
    return BenchmarkPrepareResult(
        dataset=dataset,
        split=split,
        document_count=len(selected_documents),
        query_count=len(selected_queries),
        qrel_count=len(selected_qrels),
        documents_path=documents_out,
        queries_path=queries_out,
        qrels_path=qrels_out,
    )


def build_prepared_query_slice_subset(
    *,
    dataset: str,
    source_prepared_dir: Path,
    target_prepared_dir: Path,
    query_limit: int,
    target_doc_count: int,
    seed: int = 42,
) -> BenchmarkPrepareResult:
    target_prepared_dir.mkdir(parents=True, exist_ok=True)
    full_queries = load_queries(source_prepared_dir / "queries.jsonl")
    full_qrels = load_qrels(source_prepared_dir / "qrels.jsonl")
    selected_queries = full_queries[: max(query_limit, 0)]
    selected_query_ids = {record.query_id for record in selected_queries}
    selected_qrels = [qrel for qrel in full_qrels if qrel.query_id in selected_query_ids and qrel.relevance > 0]
    positive_doc_ids = {qrel.doc_id for qrel in selected_qrels}

    documents_out = target_prepared_dir / "documents.jsonl"
    queries_out = target_prepared_dir / "queries.jsonl"
    qrels_out = target_prepared_dir / "qrels.jsonl"

    rng = random.Random(seed)
    selected_documents: dict[str, dict[str, object]] = {}
    all_documents: list[dict[str, object]] = []
    for record in iter_jsonl(source_prepared_dir / "documents.jsonl"):
        doc_id = _coerce_required_str(record.get("doc_id"), field_name="doc_id")
        all_documents.append(record)
        if doc_id in positive_doc_ids:
            selected_documents[doc_id] = record

    remaining = [
        record
        for record in all_documents
        if _coerce_required_str(record.get("doc_id"), field_name="doc_id") not in selected_documents
    ]
    rng.shuffle(remaining)
    target_size = max(target_doc_count, len(selected_documents))
    for record in remaining:
        if len(selected_documents) >= target_size:
            break
        doc_id = _coerce_required_str(record.get("doc_id"), field_name="doc_id")
        selected_documents[doc_id] = record

    with documents_out.open("w", encoding="utf-8") as handle:
        for doc_id in sorted(selected_documents):
            handle.write(json.dumps(selected_documents[doc_id], ensure_ascii=False) + "\n")

    split = benchmark_dataset_spec(dataset).default_split
    with queries_out.open("w", encoding="utf-8") as handle:
        for query_record in selected_queries:
            handle.write(
                json.dumps(
                    {
                        "query_id": query_record.query_id,
                        "query_text": query_record.query_text,
                        "metadata": {
                            "dataset": dataset,
                            "subset": "query_slice",
                            "split": split,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with qrels_out.open("w", encoding="utf-8") as handle:
        for qrel in selected_qrels:
            handle.write(
                json.dumps(
                    {"query_id": qrel.query_id, "doc_id": qrel.doc_id, "relevance": qrel.relevance},
                    ensure_ascii=False,
                )
                + "\n"
            )

    return BenchmarkPrepareResult(
        dataset=dataset,
        split=split,
        document_count=len(selected_documents),
        query_count=len(selected_queries),
        qrel_count=len(selected_qrels),
        documents_path=documents_out,
        queries_path=queries_out,
        qrels_path=qrels_out,
    )


def prepare_public_benchmark(
    dataset: str,
    raw_dir: Path,
    prepared_root: Path,
    *,
    split: str | None = None,
    build_mini: bool = True,
    mini_query_count: int | None = None,
    mini_target_doc_count: int | None = None,
) -> dict[str, BenchmarkPrepareResult]:
    spec = benchmark_dataset_spec(dataset)
    resolved_split = split or spec.default_split
    prepared_root.mkdir(parents=True, exist_ok=True)
    full_dir = prepared_root / "full"
    if dataset == FIQA_DATASET:
        full_result = prepare_fiqa(raw_dir, full_dir, split=resolved_split)
    elif dataset == MEDICAL_RETRIEVAL_DATASET:
        full_result = prepare_medical_retrieval(raw_dir, full_dir, split=resolved_split)
    else:
        raise ValueError(f"Prepare is not implemented for dataset: {dataset}")

    results: dict[str, BenchmarkPrepareResult] = {"full": full_result}
    if build_mini:
        mini_result = build_prepared_mini_subset(
            dataset=dataset,
            source_prepared_dir=full_dir,
            target_prepared_dir=prepared_root / "mini",
            query_count=mini_query_count or spec.mini_query_count,
            target_doc_count=mini_target_doc_count or spec.mini_target_doc_count,
        )
        results["mini"] = mini_result
    return results


def ingest_prepared_documents(
    runtime: RAGRuntime,
    *,
    dataset: str,
    documents_path: Path,
    batch_size: int = 64,
    continue_on_error: bool = False,
    streaming: bool = True,
) -> BenchmarkIngestResult:
    if streaming:
        total_requests = _count_lines(documents_path)
        return ingest_prepared_request_stream(
            runtime,
            dataset=dataset,
            requests=iter_prepared_benchmark_requests(
                documents_path=documents_path,
                dataset=dataset,
                show_progress=False,
            ),
            total_requests=total_requests,
            batch_size=batch_size,
            continue_on_error=continue_on_error,
        )
    return ingest_prepared_requests(
        runtime,
        dataset=dataset,
        requests=load_prepared_benchmark_requests(
            documents_path=documents_path,
            dataset=dataset,
            show_progress=False,
        ),
        batch_size=batch_size,
        continue_on_error=continue_on_error,
    )


def load_prepared_benchmark_requests(
    *,
    documents_path: Path,
    dataset: str,
    limit: int | None = None,
    show_progress: bool = True,
) -> list[IngestRequest]:
    return list(
        iter_prepared_benchmark_requests(
            documents_path=documents_path,
            dataset=dataset,
            limit=limit,
            show_progress=show_progress,
        )
    )


def iter_prepared_benchmark_requests(
    *,
    documents_path: Path,
    dataset: str,
    limit: int | None = None,
    show_progress: bool = True,
) -> Iterator[IngestRequest]:
    records: Iterable[dict[str, object]] = iter_jsonl(documents_path)
    if show_progress:
        document_count = _count_lines(documents_path)
        total = document_count if limit is None else min(document_count, limit)
        records = _progress(
            records,
            total=total,
            desc=f"Loading {dataset} documents",
            unit="doc",
        )
    for index, record in enumerate(records, start=1):
        yield prepared_document_to_ingest_request(record, dataset=dataset)
        if limit is not None and index >= limit:
            break


def ingest_prepared_requests(
    runtime: RAGRuntime,
    *,
    dataset: str,
    requests: Sequence[IngestRequest],
    batch_size: int = 64,
    continue_on_error: bool = False,
) -> BenchmarkIngestResult:
    success_count = 0
    duplicate_count = 0
    failure_count = 0
    indexed_object_count = 0
    total_requests = len(requests)
    total_batches = max((total_requests + batch_size - 1) // batch_size, 1) if total_requests else 0
    started = time.perf_counter()
    with tqdm(
        total=total_requests,
        desc=f"Ingesting {dataset} documents",
        unit="doc",
    ) as progress:
        for batch_index, start in enumerate(range(0, total_requests, batch_size), start=1):
            batch = requests[start : start + batch_size]
            with _suppress_process_output():
                result = runtime.insert_many(list(batch), continue_on_error=continue_on_error)
            success_count += result.success_count
            failure_count += result.failure_count
            indexed_object_count += result.indexed_object_count
            processed = result.success_count + result.failure_count
            for _ in range(processed):
                progress.update(1)
            if processed < len(batch):
                progress.update(len(batch) - processed)
            progress.set_postfix(
                batch=f"{batch_index}/{total_batches}",
                success=success_count,
                duplicate=duplicate_count,
                failure=failure_count,
            )
    return BenchmarkIngestResult(
        dataset=dataset,
        request_count=total_requests,
        success_count=success_count,
        duplicate_count=duplicate_count,
        failure_count=failure_count,
        indexed_object_count=indexed_object_count,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def ingest_prepared_request_stream(
    runtime: RAGRuntime,
    *,
    dataset: str,
    requests: Iterator[IngestRequest],
    total_requests: int,
    batch_size: int = 64,
    continue_on_error: bool = False,
) -> BenchmarkIngestResult:
    success_count = 0
    duplicate_count = 0
    failure_count = 0
    indexed_object_count = 0
    total_batches = max((total_requests + batch_size - 1) // batch_size, 1) if total_requests else 0
    started = time.perf_counter()
    batch: list[IngestRequest] = []
    batch_index = 0
    with tqdm(
        total=total_requests,
        desc=f"Ingesting {dataset} documents",
        unit="doc",
    ) as progress:
        for request in requests:
            batch.append(request)
            if len(batch) < batch_size:
                continue
            batch_index += 1
            success_count, duplicate_count, failure_count, indexed_object_count = _ingest_batch(
                runtime=runtime,
                batch=batch,
                continue_on_error=continue_on_error,
                progress=progress,
                success_count=success_count,
                duplicate_count=duplicate_count,
                failure_count=failure_count,
                indexed_object_count=indexed_object_count,
                batch_index=batch_index,
                total_batches=total_batches,
            )
            batch = []
        if batch:
            batch_index += 1
            success_count, duplicate_count, failure_count, indexed_object_count = _ingest_batch(
                runtime=runtime,
                batch=batch,
                continue_on_error=continue_on_error,
                progress=progress,
                success_count=success_count,
                duplicate_count=duplicate_count,
                failure_count=failure_count,
                indexed_object_count=indexed_object_count,
                batch_index=batch_index,
                total_batches=total_batches,
            )
    return BenchmarkIngestResult(
        dataset=dataset,
        request_count=total_requests,
        success_count=success_count,
        duplicate_count=duplicate_count,
        failure_count=failure_count,
        indexed_object_count=indexed_object_count,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def _ingest_batch(
    *,
    runtime: RAGRuntime,
    batch: Sequence[IngestRequest],
    continue_on_error: bool,
    progress: tqdm,
    success_count: int,
    duplicate_count: int,
    failure_count: int,
    indexed_object_count: int,
    batch_index: int,
    total_batches: int,
) -> tuple[int, int, int, int]:
    with _suppress_process_output():
        result = runtime.insert_many(list(batch), continue_on_error=continue_on_error)
    success_count += result.success_count
    failure_count += result.failure_count
    indexed_object_count += result.indexed_object_count
    processed = result.success_count + result.failure_count
    if processed < len(batch):
        progress.update(len(batch))
    else:
        progress.update(processed)
    progress.set_postfix(
        batch=f"{batch_index}/{total_batches}",
        success=success_count,
        duplicate=duplicate_count,
        failure=failure_count,
    )
    return success_count, duplicate_count, failure_count, indexed_object_count


def prepared_document_to_ingest_request(record: Mapping[str, object], *, dataset: str) -> IngestRequest:
    doc_id = _coerce_required_str(record.get("doc_id"), field_name="doc_id")
    title = _coerce_optional_str(record.get("title")) or doc_id
    text = _coerce_optional_str(record.get("text")) or ""
    source_type = SourceType(_coerce_optional_str(record.get("source_type")) or SourceType.PLAIN_TEXT.value)
    metadata = _stringify_metadata(record.get("metadata"))
    metadata.setdefault("dataset", dataset)
    metadata.setdefault("benchmark", "true")
    metadata.setdefault("benchmark_dataset", dataset)
    metadata.setdefault("benchmark_doc_id", doc_id)
    metadata.setdefault("parent_doc_id", doc_id)
    visible_text = text or title
    language = metadata.get("language", "en")
    metadata.setdefault("language", language)
    return IngestRequest(
        location=f"benchmark://{dataset}/documents/{doc_id}",
        source_type=source_type,
        owner="benchmark",
        title=title,
        content_text=visible_text,
        metadata=metadata,
    )


def load_queries(path: Path) -> list[BenchmarkQueryRecord]:
    return [
        BenchmarkQueryRecord(
            query_id=_coerce_required_str(record.get("query_id"), field_name="query_id"),
            query_text=_coerce_required_str(record.get("query_text"), field_name="query_text"),
        )
        for record in iter_jsonl(path)
    ]


def load_qrels(path: Path) -> list[BenchmarkQrel]:
    return [
        BenchmarkQrel(
            query_id=_coerce_required_str(record.get("query_id"), field_name="query_id"),
            doc_id=_coerce_required_str(record.get("doc_id"), field_name="doc_id"),
            relevance=int(cast("int | str", record.get("relevance", 0))),
        )
        for record in iter_jsonl(path)
    ]


def load_qrels_jsonl(path: Path) -> list[BenchmarkQrel]:
    return [
        BenchmarkQrel(
            query_id=_coerce_required_str(record.get("qid") or record.get("query_id"), field_name="query_id"),
            doc_id=_coerce_required_str(record.get("pid") or record.get("doc_id"), field_name="doc_id"),
            relevance=int(cast("int | str", record.get("score") or record.get("relevance") or 0)),
        )
        for record in iter_jsonl(path)
    ]


def load_qrels_tsv(path: Path) -> list[BenchmarkQrel]:
    results: list[BenchmarkQrel] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            query_id = _coerce_required_str(row.get("query-id") or row.get("query_id"), field_name="query_id")
            doc_id = _coerce_required_str(row.get("corpus-id") or row.get("doc_id"), field_name="doc_id")
            results.append(
                BenchmarkQrel(
                    query_id=query_id,
                    doc_id=doc_id,
                    relevance=int(row.get("score") or row.get("relevance") or 0),
                )
            )
    return results


def group_qrels_by_query(qrels: Iterable[BenchmarkQrel]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in qrels:
        if qrel.relevance <= 0:
            continue
        grouped[qrel.query_id][qrel.doc_id] = qrel.relevance
    return dict(grouped)


def compute_doc_ranking_metrics(
    *,
    predicted_doc_ids: list[str],
    gold_relevances: Mapping[str, int],
    top_k: int,
) -> dict[str, float | int]:
    ranked = predicted_doc_ids[:top_k]
    gold_ids = {doc_id for doc_id, relevance in gold_relevances.items() if relevance > 0}
    hits = [doc_id for doc_id in ranked if doc_id in gold_ids]
    recall = 0.0 if not gold_ids else len(set(hits)) / len(gold_ids)

    reciprocal_rank = 0.0
    for index, doc_id in enumerate(ranked, start=1):
        if doc_id in gold_ids:
            reciprocal_rank = 1.0 / index
            break

    dcg = 0.0
    for index, doc_id in enumerate(ranked, start=1):
        relevance = gold_relevances.get(doc_id, 0)
        if relevance <= 0:
            continue
        dcg += (2**relevance - 1) / math.log2(index + 1)

    ideal_relevances = sorted(
        (relevance for relevance in gold_relevances.values() if relevance > 0),
        reverse=True,
    )[:top_k]
    ideal_dcg = sum(
        (2**relevance - 1) / math.log2(index + 1) for index, relevance in enumerate(ideal_relevances, start=1)
    )
    ndcg = 0.0 if ideal_dcg == 0.0 else dcg / ideal_dcg

    return {
        "hit_at_10": int(bool(hits)),
        "recall_at_10": recall,
        "mrr_at_10": reciprocal_rank,
        "ndcg_at_10": ndcg,
    }


def append_baseline_row(path: Path, summary: BenchmarkRunSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = summary.baseline_row()
    fieldnames = [
        "run_id",
        "dataset",
        "query_count",
        "top_k",
        "embedding_model",
        "retrieval_profile",
        "rerank_enabled",
        "Recall@10",
        "MRR@10",
        "NDCG@10",
        "avg_latency_ms",
        "p95_latency_ms",
        "queries_per_second",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    existing_rows: list[dict[str, str]] = []
    if not write_header:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_header = reader.fieldnames or []
            if existing_header != fieldnames:
                existing_rows = [dict(item) for item in reader]
                write_header = True
    if write_header and existing_rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for existing in existing_rows:
                writer.writerow({name: existing.get(name, "") for name in fieldnames})
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header and not existing_rows:
            writer.writeheader()
        writer.writerow(row)


def load_baseline_rows(path: Path) -> list[dict[str, object]]:
    def _coerce(value: str) -> object:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if value == "":
            return ""
        for caster in (int, float):
            try:
                return caster(value)
            except ValueError:
                continue
        return value

    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: _coerce(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def write_dataset_baseline_summary(
    path: Path,
    *,
    dataset: str,
    baselines: Sequence[Mapping[str, object]],
    reference_run_id: str | None = None,
    latency_priority_run_id: str | None = None,
    quality_priority_run_id: str | None = None,
) -> None:
    def _queries_per_second(record: Mapping[str, object]) -> float:
        raw = record.get("queries_per_second", "")
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str) and raw.strip():
            return float(raw)
        avg_latency_ms = float(cast("int | float | str", record.get("avg_latency_ms", 0.0)))
        if avg_latency_ms <= 0:
            return 0.0
        return 1000.0 / avg_latency_ms

    def _to_float(value: object, default: float = 0.0) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        return default

    ordered = sorted(
        (dict(record) for record in baselines),
        key=lambda item: (
            _to_float(item.get("Recall@10", 0.0)),
            _to_float(item.get("MRR@10", 0.0)),
            -_to_float(item.get("avg_latency_ms", 0.0)),
        ),
        reverse=True,
    )
    index = {str(record["run_id"]): record for record in ordered if record.get("run_id")}

    def _comparison(target_run_id: str | None) -> dict[str, object] | None:
        if not reference_run_id or not target_run_id:
            return None
        reference = index.get(reference_run_id)
        target = index.get(target_run_id)
        if reference is None or target is None:
            return None
        return {
            "from_run_id": reference_run_id,
            "to_run_id": target_run_id,
            "Recall@10_delta": round(
                _to_float(target.get("Recall@10", 0.0)) - _to_float(reference.get("Recall@10", 0.0)), 6
            ),
            "MRR@10_delta": round(_to_float(target.get("MRR@10", 0.0)) - _to_float(reference.get("MRR@10", 0.0)), 6),
            "NDCG@10_delta": round(_to_float(target.get("NDCG@10", 0.0)) - _to_float(reference.get("NDCG@10", 0.0)), 6),
            "avg_latency_ms_delta": round(
                _to_float(target.get("avg_latency_ms", 0.0)) - _to_float(reference.get("avg_latency_ms", 0.0)), 3
            ),
            "queries_per_second_delta": round(
                _queries_per_second(target) - _queries_per_second(reference),
                3,
            ),
        }

    payload = {
        "dataset": dataset,
        "updated_at": datetime.now(UTC).isoformat(),
        "reference_run_id": reference_run_id,
        "latency_priority_run_id": latency_priority_run_id,
        "quality_priority_run_id": quality_priority_run_id,
        "baseline_count": len(ordered),
        "baselines": ordered,
        "comparison": {
            "latency_priority_vs_reference": _comparison(latency_priority_run_id),
            "quality_priority_vs_reference": _comparison(quality_priority_run_id),
        },
    }
    write_json(path, payload)


def append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
            yield payload


def build_runtime_for_benchmark(
    *,
    storage_root: Path,
    require_chat: bool,
    require_rerank: bool,
    skip_graph_extraction: bool = False,
    embedding_batch_size: int | None = None,
    embedding_device: str | None = None,
    log_embedding_calls: bool = False,
    show_backend_progress: bool = False,
    chunk_token_size: int | None = None,
    chunk_overlap_tokens: int | None = None,
    embedding_provider_kind: str | None = None,
    embedding_model: str | None = None,
    embedding_model_path: str | None = None,
    rerank_provider_kind: str | None = None,
    rerank_model: str | None = None,
    rerank_model_path: str | None = None,
    chat_provider_kind: str | None = None,
    chat_model: str | None = None,
    chat_model_path: str | None = None,
    chat_backend: str | None = None,
    summary_provider_kind: str | None = None,
    summary_model: str | None = None,
    summary_model_path: str | None = None,
    summary_backend: str | None = None,
    strict_summary_generation: bool = False,
    vector_backend: str | None = None,
    vector_dsn: str | None = None,
    vector_namespace: str | None = None,
    vector_collection_prefix: str | None = None,
) -> RAGRuntime:
    storage_root.mkdir(parents=True, exist_ok=True)
    load_env_file()
    assembly_service = CapabilityAssemblyService()
    tokenizer_override = (
        TokenizerConfig(
            chunk_token_size=chunk_token_size,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )
        if chunk_token_size is not None or chunk_overlap_tokens is not None
        else None
    )
    embedding_override = (
        ProviderConfig(
            provider_kind=embedding_provider_kind,
            location="local",
            embedding_model=embedding_model,
            embedding_model_path=embedding_model_path,
            embedding_space=embedding_model or embedding_model_path,
            embedding_batch_size=embedding_batch_size,
            device=embedding_device,
        )
        if embedding_provider_kind is not None
        else None
    )
    rerank_override = (
        ProviderConfig(
            provider_kind=rerank_provider_kind or "local-bge",
            location="local",
            rerank_model=rerank_model,
            rerank_model_path=rerank_model_path,
        )
        if rerank_model is not None or rerank_model_path is not None
        else None
    )
    runtime_config = resolve_runtime_config(RuntimeOverrides())
    model_config = to_assembly_overrides(runtime_config)
    chat_override = (
        ProviderConfig(
            provider_kind=chat_provider_kind or "ollama",
            location="local",
            chat_model=chat_model,
            chat_model_path=chat_model_path,
            chat_backend=chat_backend,
        )
        if chat_model is not None or chat_model_path is not None or chat_provider_kind is not None
        else None
    )
    config = AssemblyConfig(
        embedding=model_config.embedding if embedding_override is None else None,
        chat=model_config.chat if require_chat and chat_override is None else None,
        rerank=model_config.rerank if require_rerank and rerank_override is None else None,
    )

    # ── service URL env vars → pre-built HTTP providers (env > YAML) ──
    _embedding_service_url = os.environ.get("RAG_EMBEDDING_SERVICE_URL", "").strip()
    _rerank_service_url = os.environ.get("RAG_RERANK_SERVICE_URL", "").strip()
    _http_embedding_provider: object | None = None
    _http_rerank_provider: object | None = None

    if _embedding_service_url:
        from rag.assembly.support import _CompositeProvider  # noqa: F811
        from rag.providers.embedding_http import EmbeddingHttpClient

        _http_embedding_provider = _CompositeProvider(
            provider_name="embedding-http",
            embedder=EmbeddingHttpClient(
                base_url=_embedding_service_url,
                batch_size=embedding_batch_size,
            ),
        )
    if require_rerank and _rerank_service_url:
        from rag.assembly.support import _CompositeProvider  # noqa: F811
        from rag.providers.rerank_http import RerankHttpClient

        _http_rerank_provider = _CompositeProvider(
            provider_name="rerank-http",
            reranker=RerankHttpClient(base_url=_rerank_service_url),
        )

    _has_override = (
        embedding_override is not None
        or rerank_override is not None
        or chat_override is not None
        or tokenizer_override is not None
        or _http_embedding_provider is not None
        or _http_rerank_provider is not None
    )
    request = AssemblyRequest(
        requirements=CapabilityRequirements(
            require_embedding=True,
            require_chat=require_chat,
            require_rerank=require_rerank,
        ),
        config=config,
        overrides=(
            AssemblyOverrides(
                embedding=embedding_override,
                rerank=rerank_override,
                chat=chat_override,
                tokenizer=tokenizer_override,
                embedding_provider=_http_embedding_provider,
                rerank_provider=_http_rerank_provider,
            )
            if _has_override
            else None
        ),
    )
    storage = _benchmark_storage_config(
        root=storage_root,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
    )
    runtime = RAGRuntime.from_request(
        storage=storage,
        request=request,
        assembly_service=assembly_service,
    )
    runtime.generation_config = runtime_config.generation
    if summary_provider_kind == "none":
        runtime.configure_default_summary_generator(raw_text_mode=True, strict_generation=False)
    elif summary_model is not None or summary_model_path is not None or summary_provider_kind is not None:
        if strict_summary_generation:
            runtime.configure_summary_generator(
                provider_kind=summary_provider_kind or "ollama",
                model=summary_model,
                model_path=summary_model_path,
                backend=summary_backend,
                strict_generation=True,
            )
        else:
            runtime.configure_summary_generator(
                provider_kind=summary_provider_kind or "ollama",
                model=summary_model,
                model_path=summary_model_path,
                backend=summary_backend,
            )
    elif strict_summary_generation:
        runtime.configure_default_summary_generator(strict_generation=True)
    if skip_graph_extraction:
        runtime.ingest_pipeline.extractor = None  # type: ignore[attr-defined]
        runtime.ingest_pipeline.merger = None  # type: ignore[attr-defined]
    configure_runtime_embedding(
        runtime,
        encode_batch_size=embedding_batch_size,
        device=embedding_device,
        log_embedding_calls=log_embedding_calls,
        show_backend_progress=show_backend_progress,
    )
    return runtime


def _benchmark_storage_config(
    *,
    root: Path,
    vector_backend: str | None = None,
    vector_dsn: str | None = None,
    vector_namespace: str | None = None,
    vector_collection_prefix: str | None = None,
) -> StorageConfig:
    backend = (vector_backend or DEFAULT_VECTOR_BACKEND).strip().lower()
    vector_dsn = _normalize_optional_string(vector_dsn)
    vector_namespace = _normalize_optional_string(vector_namespace)
    vector_collection_prefix = _normalize_optional_string(vector_collection_prefix)
    if backend in {"", "sqlite", "in_memory"}:
        return StorageConfig(root=root)
    if backend == "milvus" and vector_dsn is None:
        vector_dsn = DEFAULT_MILVUS_DSN
    return StorageConfig(
        root=root,
        vectors=StorageComponentConfig(
            backend=backend,
            dsn=vector_dsn,
            namespace=vector_namespace,
            collection=vector_collection_prefix,
        ),
    )


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def configure_runtime_embedding(
    runtime: RAGRuntime,
    *,
    encode_batch_size: int | None = None,
    device: str | None = None,
    log_embedding_calls: bool = False,
    show_backend_progress: bool = False,
) -> BenchmarkEmbeddingRuntimeInfo | None:
    provider = _benchmark_embedding_provider(runtime)
    if provider is None:
        return None
    if encode_batch_size is not None:
        provider.set_embedding_batch_size(encode_batch_size)
    if device is not None:
        provider.set_device_preference(device)
    provider.set_embedding_call_logging(log_embedding_calls)
    provider.set_backend_progress_enabled(show_backend_progress)
    provider.reset_embedding_stats()
    info = provider.embedding_runtime_info()
    return BenchmarkEmbeddingRuntimeInfo(
        provider=str(info["provider"]),
        model_name=_coerce_str(info.get("model_name")),
        device=str(info["device"]),
        encode_batch_size=int(cast("int | str", info["encode_batch_size"])),
    )


def runtime_embedding_stats(runtime: RAGRuntime) -> dict[str, object] | None:
    provider = _benchmark_embedding_provider(runtime)
    if provider is None:
        return None
    return provider.embedding_stats()


def profile_benchmark_ingest(
    *,
    dataset: str,
    documents_path: Path,
    storage_root: Path,
    doc_limit: int,
    ingest_batch_size: int,
    encode_batch_size: int,
    ingest_strategy: str = "stream",
    embedding_device: str | None = None,
    embedding_provider_kind: str | None = None,
    embedding_model: str | None = None,
    embedding_model_path: str | None = None,
    chunk_token_size: int | None = None,
    chunk_overlap_tokens: int | None = None,
    skip_graph_extraction: bool = True,
    log_embedding_calls: bool = False,
    show_backend_progress: bool = False,
) -> BenchmarkIngestProfileResult:
    runtime = build_runtime_for_benchmark(
        storage_root=storage_root,
        require_chat=False,
        require_rerank=False,
        skip_graph_extraction=skip_graph_extraction,
        embedding_batch_size=encode_batch_size,
        embedding_device=embedding_device,
        log_embedding_calls=log_embedding_calls,
        show_backend_progress=show_backend_progress,
        chunk_token_size=chunk_token_size,
        chunk_overlap_tokens=chunk_overlap_tokens,
        embedding_provider_kind=embedding_provider_kind,
        embedding_model=embedding_model,
        embedding_model_path=embedding_model_path,
    )
    try:
        runtime_info = configure_runtime_embedding(
            runtime,
            encode_batch_size=encode_batch_size,
            device=embedding_device,
            log_embedding_calls=log_embedding_calls,
            show_backend_progress=show_backend_progress,
        )
        if runtime_info is None:
            raise RuntimeError("No local embedding provider is available for benchmark profiling.")
        tqdm.write(
            json.dumps(
                {
                    "event": "embedding_runtime",
                    "provider": runtime_info.provider,
                    "model_name": runtime_info.model_name,
                    "device": runtime_info.device,
                    "encode_batch_size": runtime_info.encode_batch_size,
                    "ingest_batch_size": ingest_batch_size,
                },
                ensure_ascii=False,
            )
        )
        if ingest_strategy == "preload":
            result = ingest_prepared_requests(
                runtime,
                dataset=dataset,
                requests=load_prepared_benchmark_requests(
                    documents_path=documents_path,
                    dataset=dataset,
                    limit=doc_limit,
                    show_progress=False,
                ),
                batch_size=ingest_batch_size,
                continue_on_error=False,
            )
        else:
            total_requests = min(_count_lines(documents_path), doc_limit)
            result = ingest_prepared_request_stream(
                runtime,
                dataset=dataset,
                requests=iter_prepared_benchmark_requests(
                    documents_path=documents_path,
                    dataset=dataset,
                    limit=doc_limit,
                    show_progress=False,
                ),
                total_requests=total_requests,
                batch_size=ingest_batch_size,
                continue_on_error=False,
            )
        embedding_stats = runtime_embedding_stats(runtime) or {}
        total_elapsed_ms = float(result.elapsed_ms or 0.0)
        embedding_elapsed_ms = float(_coerce_float(embedding_stats.get("total_duration_ms")) or 0.0)
        return BenchmarkIngestProfileResult(
            dataset=dataset,
            doc_count=result.success_count,
            indexed_object_count=result.indexed_object_count,
            ingest_strategy=ingest_strategy,
            ingest_batch_size=ingest_batch_size,
            encode_batch_size=encode_batch_size,
            embedding_model=runtime_info.model_name,
            device=runtime_info.device,
            total_elapsed_ms=total_elapsed_ms,
            embedding_elapsed_ms=embedding_elapsed_ms,
        )
    finally:
        runtime.close()


def benchmark_access_policy() -> AccessPolicy:
    return AccessPolicy(
        allowed_runtimes=frozenset({RuntimeMode.FAST, RuntimeMode.DEEP}),
    )


def benchmark_run_id(dataset: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{dataset}-{timestamp}"


def _ensure_valid_fiqa_archive(
    archive_path: Path,
    *,
    force: bool,
    timeout_seconds: float,
) -> None:
    if not force and _is_valid_zip(archive_path):
        return
    archive_path.unlink(missing_ok=True)
    _download_file_with_progress(
        FIQA_BEIR_ZIP_URL,
        archive_path,
        timeout_seconds=timeout_seconds,
        desc="Downloading FiQA archive",
    )
    if not _is_valid_zip(archive_path):
        archive_path.unlink(missing_ok=True)
        raise zipfile.BadZipFile(f"Downloaded archive is invalid: {archive_path}")


def _download_file_with_progress(
    url: str,
    destination: Path,
    *,
    timeout_seconds: float,
    desc: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".download")
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        total = _coerce_content_length(response.headers.get("Content-Length"))
        with (
            temporary_path.open("wb") as handle,
            tqdm(
                total=total,
                desc=desc,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress,
        ):
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))
    temporary_path.replace(destination)


def _is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            zf.testzip()
    except zipfile.BadZipFile:
        return False
    return True


def _required_zip_member(zf: zipfile.ZipFile, suffix: str) -> str:
    for name in zf.namelist():
        normalized = name.rstrip("/")
        if normalized.endswith(suffix):
            return normalized
    raise FileNotFoundError(f"Archive does not contain required member ending with {suffix!r}")


def _extract_zip_member(zf: zipfile.ZipFile, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = zf.read(member)
    if member.endswith(".gz"):
        payload = gzip.decompress(payload)
    with destination.open("wb") as handle:
        handle.write(payload)


def _coerce_required_str(value: object, *, field_name: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"Missing required string field: {field_name}")


def _coerce_optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _stringify_metadata(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    metadata: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        if item is None:
            continue
        if isinstance(item, bool):
            metadata[key] = "true" if item else "false"
        else:
            metadata[key] = str(item)
    return metadata


def _embedding_model_name(runtime: RAGRuntime) -> str | None:
    bindings = runtime.capability_bundle.embedding_bindings
    if not bindings:
        return None
    return cast("str | None", bindings[0].model_name or bindings[0].provider_name)


def _benchmark_embedding_provider(runtime: RAGRuntime) -> _EmbeddingBackend | None:
    bindings = runtime.capability_bundle.embedding_bindings
    if not bindings:
        return None
    backend = getattr(bindings[0], "backend", None)
    if backend is None:
        return None
    required = (
        "set_embedding_batch_size",
        "set_embedding_call_logging",
        "set_backend_progress_enabled",
        "reset_embedding_stats",
        "embedding_runtime_info",
        "embedding_stats",
    )
    if not all(hasattr(backend, attr) for attr in required):
        return None
    return cast(_EmbeddingBackend, backend)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return 0.0 if not items else statistics.fmean(items)


def _p95(values: Iterable[float]) -> float:
    items = sorted(values)
    if not items:
        return 0.0
    if len(items) == 1:
        return items[0]
    index = math.ceil(0.95 * len(items)) - 1
    index = min(max(index, 0), len(items) - 1)
    return items[index]


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _coerce_content_length(value: str | None) -> int | None:
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _count_lines(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _suppress_process_output() -> ExitStack:
    sink = io.StringIO()
    stack = ExitStack()
    stack.enter_context(redirect_stdout(sink))
    stack.enter_context(redirect_stderr(sink))
    return stack


def _progress[T](
    iterable: Iterable[T],
    *,
    total: int | None,
    desc: str,
    unit: str,
) -> Iterable[T]:
    return cast(Iterable[T], tqdm(iterable, total=total, desc=desc, unit=unit))


__all__ = [
    "BenchmarkDatasetSpec",
    "BenchmarkDownloadResult",
    "BenchmarkEmbeddingRuntimeInfo",
    "BenchmarkIngestResult",
    "BenchmarkIngestProfileResult",
    "BenchmarkPaths",
    "BenchmarkPerQueryResult",
    "BenchmarkPrepareResult",
    "BenchmarkQueryRecord",
    "BenchmarkQrel",
    "BenchmarkRunSummary",
    "FIQA_BEIR_ZIP_URL",
    "FIQA_DATASET",
    "MEDICAL_RETRIEVAL_DATASET",
    "RetrievalBenchmarkEvaluator",
    "append_baseline_row",
    "benchmark_access_policy",
    "benchmark_dataset_spec",
    "benchmark_run_id",
    "build_runtime_for_benchmark",
    "build_prepared_mini_subset",
    "build_prepared_query_slice_subset",
    "configure_runtime_embedding",
    "compute_doc_ranking_metrics",
    "default_benchmark_paths",
    "download_medical_retrieval",
    "download_public_benchmark",
    "download_fiqa",
    "ensure_benchmark_layout",
    "group_qrels_by_query",
    "ingest_prepared_documents",
    "ingest_prepared_requests",
    "iter_jsonl",
    "load_baseline_rows",
    "load_qrels",
    "load_qrels_jsonl",
    "load_qrels_tsv",
    "load_queries",
    "load_prepared_benchmark_requests",
    "prepare_medical_retrieval",
    "prepare_public_benchmark",
    "prepare_fiqa",
    "profile_benchmark_ingest",
    "prepared_document_to_ingest_request",
    "runtime_embedding_stats",
    "write_dataset_baseline_summary",
    "write_json",
]
