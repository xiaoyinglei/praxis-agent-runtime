from __future__ import annotations

import json
import os
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Annotated

import click
import typer
from pydantic import BaseModel

from agent_runtime.text import load_env_file
from rag import AssemblyRequest, CapabilityRequirements, RAGRuntime, StorageConfig
from rag.models.assembly_adapter import to_assembly_overrides
from rag.models.runtime import RuntimeOverrides, resolve_runtime_config
from rag.retrieval import QueryOptions, RetrievalProfile
from rag.schema.core import SourceType
from rag.storage.runtime_config import DEFAULT_VECTOR_BACKEND, runtime_storage_config

app = typer.Typer(add_completion=False, no_args_is_help=True)
DEFAULT_STORAGE_ROOT = Path(".rag")
FIQA_DATASET = "fiqa"
MEDICAL_RETRIEVAL_DATASET = "medical_retrieval"
STORAGE_ROOT_OPTION = typer.Option("--storage-root")
SOURCE_TYPE_OPTION = typer.Option("--source-type")
LOCATION_OPTION = typer.Option("--location")
CONTENT_OPTION = typer.Option("--content")
TITLE_OPTION = typer.Option("--title")
OWNER_OPTION = typer.Option("--owner")
QUERY_OPTION = typer.Option("--query")
RETRIEVAL_PROFILE_OPTION = typer.Option(
    "--retrieval-profile",
    help="New retrieval profile: auto, fast, deep, or bypass.",
)
JSON_OPTION = typer.Option("--json")
DOC_ID_OPTION = typer.Option("--doc-id")
SOURCE_ID_OPTION = typer.Option("--source-id")
MODEL_OPTION = typer.Option("--model", help="Chat model alias from configs/models.yaml.")
EMBEDDING_MODEL_OPTION = typer.Option("--embedding-model", help="Embedding model alias from configs/models.yaml.")
RERANKER_MODEL_OPTION = typer.Option("--reranker-model", help="Reranker model alias from configs/models.yaml.")
DATASET_OPTION = typer.Option("--dataset", help="Public benchmark dataset.")
VECTOR_BACKEND_OPTION = typer.Option("--vector-backend", help="Vector backend: milvus or sqlite.")
VECTOR_DSN_OPTION = typer.Option("--vector-dsn", help="Vector backend DSN, for example Milvus URI.")
VECTOR_NAMESPACE_OPTION = typer.Option("--vector-namespace", help="Vector backend namespace/database.")
VECTOR_COLLECTION_PREFIX_OPTION = typer.Option(
    "--vector-collection-prefix",
    help="Milvus collection prefix. Must match the prefix used at ingest time.",
)


def _runtime(
    storage_root: Path,
    *,
    require_chat: bool = False,
    require_rerank: bool = False,
    model: str | None = None,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    vector_backend: str = DEFAULT_VECTOR_BACKEND,
    vector_dsn: str | None = None,
    vector_namespace: str | None = None,
    vector_collection_prefix: str | None = None,
) -> RAGRuntime:
    # ── service URL env vars (higher priority than YAML) ──
    embedding_service_url = os.environ.get("RAG_EMBEDDING_SERVICE_URL", "").strip()
    rerank_service_url = os.environ.get("RAG_RERANK_SERVICE_URL", "").strip()

    # ── conflict detection: env + CLI explicit → error ──
    if embedding_service_url and embedding_model is not None:
        raise typer.BadParameter(
            "RAG_EMBEDDING_SERVICE_URL is set but --embedding-model was also specified. "
            "Unset RAG_EMBEDDING_SERVICE_URL or remove --embedding-model."
        )
    if rerank_service_url and reranker_model is not None:
        raise typer.BadParameter(
            "RAG_RERANK_SERVICE_URL is set but --reranker-model was also specified. "
            "Unset RAG_RERANK_SERVICE_URL or remove --reranker-model."
        )

    # ── resolve YAML config (CLI override > YAML default) ──
    load_env_file()
    runtime_config = resolve_runtime_config(
        RuntimeOverrides(
            model_alias=model,
            embedding_model_alias=embedding_model,
            reranker_model_alias=reranker_model,
        )
    )
    overrides = to_assembly_overrides(runtime_config)

    # ── service URL env → pre-built HTTP providers (env > YAML) ──
    if embedding_service_url or rerank_service_url:
        from rag.assembly.support import _CompositeProvider  # noqa: F811

    if embedding_service_url:
        from rag.providers.embedding_http import EmbeddingHttpClient

        embedding_client = EmbeddingHttpClient(base_url=embedding_service_url)
        overrides = replace(
            overrides,
            embedding_provider=_CompositeProvider(
                provider_name="embedding-http",
                embedder=embedding_client,
            ),
        )
    if rerank_service_url:
        from rag.providers.rerank_http import RerankHttpClient

        rerank_client = RerankHttpClient(base_url=rerank_service_url)
        overrides = replace(
            overrides,
            rerank_provider=_CompositeProvider(
                provider_name="rerank-http",
                reranker=rerank_client,
            ),
        )

    # ── strip reranker when not required (prevents loading during ingest) ──
    if not require_rerank:
        overrides = replace(overrides, rerank=None, rerank_provider=None)

    request = CapabilityRequirements(
        require_chat=require_chat,
        require_rerank=require_rerank,
        default_context_tokens=QueryOptions().max_context_tokens,
    )
    return RAGRuntime.from_request(
        storage=_default_storage_config(
            storage_root,
            vector_backend=vector_backend,
            vector_dsn=vector_dsn,
            vector_namespace=vector_namespace,
            vector_collection_prefix=vector_collection_prefix,
        ),
        request=AssemblyRequest(
            requirements=request,
            overrides=overrides,
        ),
        generation_config=runtime_config.generation,
        chat_context_window_tokens=(runtime_config.primary_model.context_window_tokens or 32_768),
        llm_stage_budgets=runtime_config.llm_stage_budgets,
    )


def _default_storage_config(
    storage_root: Path,
    *,
    vector_backend: str = DEFAULT_VECTOR_BACKEND,
    vector_dsn: str | None = None,
    vector_namespace: str | None = None,
    vector_collection_prefix: str | None = None,
) -> StorageConfig:
    return runtime_storage_config(
        storage_root,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
    )


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        payload = {field.name: getattr(value, field.name) for field in fields(value)}
        for computed_name in ("indexed_object_count", "success_count", "failure_count"):
            computed_value = getattr(value, computed_name, None)
            if computed_value is not None:
                payload[computed_name] = computed_value
        return payload
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=True, default=_json_default))


def _requires_content(source_type: SourceType) -> bool:
    return source_type in {SourceType.PLAIN_TEXT, SourceType.PASTED_TEXT, SourceType.BROWSER_CLIP}


@app.command()
def ingest(
    storage_root: Annotated[Path, STORAGE_ROOT_OPTION] = DEFAULT_STORAGE_ROOT,
    model: Annotated[str | None, MODEL_OPTION] = None,
    embedding_model: Annotated[str | None, EMBEDDING_MODEL_OPTION] = None,
    reranker_model: Annotated[str | None, RERANKER_MODEL_OPTION] = None,
    vector_backend: Annotated[str, VECTOR_BACKEND_OPTION] = DEFAULT_VECTOR_BACKEND,
    vector_dsn: Annotated[str | None, VECTOR_DSN_OPTION] = None,
    vector_namespace: Annotated[str | None, VECTOR_NAMESPACE_OPTION] = None,
    vector_collection_prefix: Annotated[str | None, VECTOR_COLLECTION_PREFIX_OPTION] = None,
    source_type: Annotated[SourceType | None, SOURCE_TYPE_OPTION] = None,
    location: Annotated[str | None, LOCATION_OPTION] = None,
    content: Annotated[str | None, CONTENT_OPTION] = None,
    title: Annotated[str | None, TITLE_OPTION] = None,
    owner: Annotated[str, OWNER_OPTION] = "user",
) -> None:
    if source_type is None:
        raise typer.BadParameter("--source-type is required")
    if location is None or not location.strip():
        raise typer.BadParameter("--location is required")
    if _requires_content(source_type) and content is None:
        raise typer.BadParameter("--content is required for text-based ingest")

    with _runtime(
        storage_root,
        require_chat=False,
        require_rerank=False,
        model=model,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
    ) as runtime:
        result = runtime.insert(
            source_type=source_type.value,
            location=location,
            owner=owner,
            title=title,
            content_text=content,
        )
    _echo_json(result)


@app.command()
def query(
    storage_root: Annotated[Path, STORAGE_ROOT_OPTION] = DEFAULT_STORAGE_ROOT,
    model: Annotated[str | None, MODEL_OPTION] = None,
    embedding_model: Annotated[str | None, EMBEDDING_MODEL_OPTION] = None,
    reranker_model: Annotated[str | None, RERANKER_MODEL_OPTION] = None,
    vector_backend: Annotated[str, VECTOR_BACKEND_OPTION] = DEFAULT_VECTOR_BACKEND,
    vector_dsn: Annotated[str | None, VECTOR_DSN_OPTION] = None,
    vector_namespace: Annotated[str | None, VECTOR_NAMESPACE_OPTION] = None,
    vector_collection_prefix: Annotated[str | None, VECTOR_COLLECTION_PREFIX_OPTION] = None,
    query: Annotated[str | None, QUERY_OPTION] = None,
    retrieval_profile: Annotated[RetrievalProfile, RETRIEVAL_PROFILE_OPTION] = RetrievalProfile.AUTO,
    json_output: Annotated[bool, JSON_OPTION] = False,
) -> None:
    if query is None or not query.strip():
        raise typer.BadParameter("--query is required")
    with _runtime(
        storage_root,
        require_chat=False,
        require_rerank=True,
        model=model,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
    ) as runtime:
        result = runtime.query_public(query, options=QueryOptions(retrieval_profile=retrieval_profile.value))
    if json_output:
        _echo_json(result)
        return
    typer.echo(result.answer.answer_text)
    typer.echo(result.answer.answer_text)


@app.command()
def delete(
    storage_root: Annotated[Path, STORAGE_ROOT_OPTION] = DEFAULT_STORAGE_ROOT,
    model: Annotated[str | None, MODEL_OPTION] = None,
    embedding_model: Annotated[str | None, EMBEDDING_MODEL_OPTION] = None,
    reranker_model: Annotated[str | None, RERANKER_MODEL_OPTION] = None,
    vector_backend: Annotated[str, VECTOR_BACKEND_OPTION] = DEFAULT_VECTOR_BACKEND,
    vector_dsn: Annotated[str | None, VECTOR_DSN_OPTION] = None,
    vector_namespace: Annotated[str | None, VECTOR_NAMESPACE_OPTION] = None,
    vector_collection_prefix: Annotated[str | None, VECTOR_COLLECTION_PREFIX_OPTION] = None,
    doc_id: Annotated[str | None, DOC_ID_OPTION] = None,
    source_id: Annotated[str | None, SOURCE_ID_OPTION] = None,
    location: Annotated[str | None, LOCATION_OPTION] = None,
) -> None:
    if doc_id is None and source_id is None and (location is None or not location.strip()):
        raise typer.BadParameter("--doc-id, --source-id, or --location is required")
    with _runtime(
        storage_root,
        require_chat=False,
        require_rerank=False,
        model=model,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
    ) as runtime:
        result = runtime.delete(doc_id=doc_id, source_id=source_id, location=location)
    _echo_json(result)


@app.command()
def rebuild(
    storage_root: Annotated[Path, STORAGE_ROOT_OPTION] = DEFAULT_STORAGE_ROOT,
    model: Annotated[str | None, MODEL_OPTION] = None,
    embedding_model: Annotated[str | None, EMBEDDING_MODEL_OPTION] = None,
    reranker_model: Annotated[str | None, RERANKER_MODEL_OPTION] = None,
    vector_backend: Annotated[str, VECTOR_BACKEND_OPTION] = DEFAULT_VECTOR_BACKEND,
    vector_dsn: Annotated[str | None, VECTOR_DSN_OPTION] = None,
    vector_namespace: Annotated[str | None, VECTOR_NAMESPACE_OPTION] = None,
    vector_collection_prefix: Annotated[str | None, VECTOR_COLLECTION_PREFIX_OPTION] = None,
    doc_id: Annotated[str | None, DOC_ID_OPTION] = None,
    source_id: Annotated[str | None, SOURCE_ID_OPTION] = None,
    location: Annotated[str | None, LOCATION_OPTION] = None,
) -> None:
    if doc_id is None and source_id is None and (location is None or not location.strip()):
        raise typer.BadParameter("--doc-id, --source-id, or --location is required")
    try:
        with _runtime(
            storage_root,
            require_chat=False,
            require_rerank=False,
            model=model,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            vector_backend=vector_backend,
            vector_dsn=vector_dsn,
            vector_namespace=vector_namespace,
            vector_collection_prefix=vector_collection_prefix,
        ) as runtime:
            result = runtime.rebuild(doc_id=doc_id, source_id=source_id, location=location)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    _echo_json(result)


@app.command("benchmark-download")
def benchmark_download(
    dataset: Annotated[str, DATASET_OPTION] = FIQA_DATASET,
    raw_dir: Annotated[Path | None, typer.Option("--raw-dir")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    from rag.benchmarks import default_benchmark_paths, download_public_benchmark, ensure_benchmark_layout

    paths = ensure_benchmark_layout(default_benchmark_paths(dataset))
    result = download_public_benchmark(dataset, paths.raw_dir if raw_dir is None else raw_dir, force=force)
    _echo_json(result)


@app.command("benchmark-prepare")
def benchmark_prepare(
    dataset: Annotated[str, DATASET_OPTION] = FIQA_DATASET,
    raw_dir: Annotated[Path | None, typer.Option("--raw-dir")] = None,
    prepared_dir: Annotated[Path | None, typer.Option("--prepared-dir")] = None,
    split: Annotated[str | None, typer.Option("--split")] = None,
    no_mini: Annotated[bool, typer.Option("--no-mini")] = False,
    mini_query_count: Annotated[int | None, typer.Option("--mini-query-count")] = None,
    mini_doc_count: Annotated[int | None, typer.Option("--mini-doc-count")] = None,
) -> None:
    from rag.benchmarks import (
        benchmark_dataset_spec,
        default_benchmark_paths,
        ensure_benchmark_layout,
        prepare_public_benchmark,
    )

    paths = ensure_benchmark_layout(default_benchmark_paths(dataset))
    spec = benchmark_dataset_spec(dataset)
    result = prepare_public_benchmark(
        dataset,
        paths.raw_dir if raw_dir is None else raw_dir,
        paths.prepared_root if prepared_dir is None else prepared_dir,
        split=split or spec.default_split,
        build_mini=not no_mini,
        mini_query_count=mini_query_count,
        mini_target_doc_count=mini_doc_count,
    )
    _echo_json(result)


@app.command("benchmark-ingest")
def benchmark_ingest(
    dataset: Annotated[str, DATASET_OPTION] = FIQA_DATASET,
    variant: Annotated[str, typer.Option("--variant")] = "full",
    storage_root: Annotated[Path | None, STORAGE_ROOT_OPTION] = None,
    documents_path: Annotated[Path | None, typer.Option("--documents-path")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 64,
    embedding_provider: Annotated[str | None, typer.Option("--embedding-provider")] = None,
    embedding_model: Annotated[str | None, typer.Option("--embedding-model")] = None,
    embedding_model_path: Annotated[str | None, typer.Option("--embedding-model-path")] = None,
    embedding_batch_size: Annotated[int | None, typer.Option("--embedding-batch-size")] = None,
    chat_provider: Annotated[str | None, typer.Option("--chat-provider")] = None,
    chat_model: Annotated[str | None, typer.Option("--chat-model")] = None,
    chat_model_path: Annotated[str | None, typer.Option("--chat-model-path")] = None,
    chat_backend: Annotated[str | None, typer.Option("--chat-backend")] = None,
    summary_provider: Annotated[str | None, typer.Option("--summary-provider")] = None,
    summary_model: Annotated[str | None, typer.Option("--summary-model")] = None,
    summary_model_path: Annotated[str | None, typer.Option("--summary-model-path")] = None,
    summary_backend: Annotated[str | None, typer.Option("--summary-backend")] = None,
    vector_backend: Annotated[str, typer.Option("--vector-backend")] = "milvus",
    vector_dsn: Annotated[str | None, typer.Option("--vector-dsn")] = None,
    vector_namespace: Annotated[str | None, typer.Option("--vector-namespace")] = None,
    vector_collection_prefix: Annotated[str | None, typer.Option("--vector-collection-prefix")] = None,
    continue_on_error: Annotated[bool, typer.Option("--continue-on-error")] = False,
    skip_graph_extraction: Annotated[bool, typer.Option("--skip-graph-extraction/--with-graph-extraction")] = True,
) -> None:
    from rag.benchmarks import (
        build_runtime_for_benchmark,
        default_benchmark_paths,
        ensure_benchmark_layout,
        ingest_prepared_documents,
    )

    if variant not in {"full", "mini"}:
        raise typer.BadParameter("variant must be one of: full, mini")
    paths = ensure_benchmark_layout(default_benchmark_paths(dataset))
    runtime = build_runtime_for_benchmark(
        storage_root=storage_root or paths.index_variant_dir(variant),
        require_chat=not skip_graph_extraction,
        require_rerank=False,
        skip_graph_extraction=skip_graph_extraction,
        embedding_provider_kind=embedding_provider,
        embedding_model=embedding_model,
        embedding_model_path=embedding_model_path,
        embedding_batch_size=embedding_batch_size,
        chat_provider_kind=chat_provider,
        chat_model=chat_model,
        chat_model_path=chat_model_path,
        chat_backend=chat_backend,
        summary_provider_kind=summary_provider,
        summary_model=summary_model,
        summary_model_path=summary_model_path,
        summary_backend=summary_backend,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
    )
    try:
        result = ingest_prepared_documents(
            runtime,
            dataset=dataset,
            documents_path=documents_path or (paths.prepared_variant_dir(variant) / "documents.jsonl"),
            batch_size=max(batch_size, 1),
            continue_on_error=continue_on_error,
        )
        _echo_json(result)
    finally:
        runtime.close()


@app.command("benchmark-evaluate")
def benchmark_evaluate(
    dataset: Annotated[str, DATASET_OPTION] = FIQA_DATASET,
    variant: Annotated[str, typer.Option("--variant")] = "full",
    storage_root: Annotated[Path | None, STORAGE_ROOT_OPTION] = None,
    queries_path: Annotated[Path | None, typer.Option("--queries-path")] = None,
    qrels_path: Annotated[Path | None, typer.Option("--qrels-path")] = None,
    eval_dir: Annotated[Path | None, typer.Option("--eval-dir")] = None,
    retrieval_profile: Annotated[RetrievalProfile, RETRIEVAL_PROFILE_OPTION] = RetrievalProfile.AUTO,
    top_k: Annotated[int, typer.Option("--top-k")] = 10,
    evidence_top_k: Annotated[int | None, typer.Option("--evidence-top-k")] = None,
    vector_backend: Annotated[str, typer.Option("--vector-backend")] = "milvus",
    vector_dsn: Annotated[str | None, typer.Option("--vector-dsn")] = None,
    vector_namespace: Annotated[str | None, typer.Option("--vector-namespace")] = None,
    vector_collection_prefix: Annotated[str | None, typer.Option("--vector-collection-prefix")] = None,
    rerank_enabled: Annotated[bool, typer.Option("--rerank/--no-rerank")] = True,
    embedding_provider: Annotated[str | None, typer.Option("--embedding-provider")] = None,
    embedding_model: Annotated[str | None, typer.Option("--embedding-model")] = None,
    embedding_model_path: Annotated[str | None, typer.Option("--embedding-model-path")] = None,
    rerank_provider: Annotated[str | None, typer.Option("--rerank-provider")] = None,
    rerank_model: Annotated[str | None, typer.Option("--rerank-model")] = None,
    rerank_model_path: Annotated[str | None, typer.Option("--rerank-model-path")] = None,
    chat_provider: Annotated[str | None, typer.Option("--chat-provider")] = None,
    chat_model: Annotated[str | None, typer.Option("--chat-model")] = None,
    chat_model_path: Annotated[str | None, typer.Option("--chat-model-path")] = None,
    chat_backend: Annotated[str | None, typer.Option("--chat-backend")] = None,
    split: Annotated[str | None, typer.Option("--split")] = None,
) -> None:
    from rag.benchmarks import (
        RetrievalBenchmarkEvaluator,
        benchmark_access_policy,
        benchmark_dataset_spec,
        build_runtime_for_benchmark,
        default_benchmark_paths,
        ensure_benchmark_layout,
    )

    if variant not in {"full", "mini"}:
        raise typer.BadParameter("variant must be one of: full, mini")
    paths = ensure_benchmark_layout(default_benchmark_paths(dataset))
    spec = benchmark_dataset_spec(dataset)
    top_k = max(top_k, 1)
    evidence_top_k = max(evidence_top_k or max(top_k * 4, 40), top_k)
    runtime = build_runtime_for_benchmark(
        storage_root=storage_root or paths.index_variant_dir(variant),
        require_chat=False,
        require_rerank=rerank_enabled,
        embedding_provider_kind=embedding_provider,
        embedding_model=embedding_model,
        embedding_model_path=embedding_model_path,
        rerank_provider_kind=rerank_provider,
        rerank_model=rerank_model,
        rerank_model_path=rerank_model_path,
        chat_provider_kind=chat_provider,
        chat_model=chat_model,
        chat_model_path=chat_model_path,
        chat_backend=chat_backend,
        vector_backend=vector_backend,
        vector_dsn=vector_dsn,
        vector_namespace=vector_namespace,
        vector_collection_prefix=vector_collection_prefix,
    )
    try:
        summary = RetrievalBenchmarkEvaluator(
            runtime=runtime,
            dataset=dataset,
            split=split or spec.default_split,
            retrieval_profile=retrieval_profile.value,
            top_k=top_k,
            evidence_top_k=evidence_top_k,
            rerank_enabled=rerank_enabled,
            access_policy=benchmark_access_policy(),
        ).evaluate(
            queries_path=queries_path or (paths.prepared_variant_dir(variant) / "queries.jsonl"),
            qrels_path=qrels_path or (paths.prepared_variant_dir(variant) / "qrels.jsonl"),
            eval_dir=eval_dir or paths.eval_variant_dir("retrieval", variant),
        )
        payload = summary.as_json()
        payload["variant"] = variant
        _echo_json(payload)
    finally:
        runtime.close()


@app.command("embedding-service")
def embedding_service(
    model: Annotated[
        str,
        typer.Option("--model", help="MLX embedding model name or path"),
    ] = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
    port: Annotated[int, typer.Option("--port", help="HTTP port")] = 9090,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Embedding batch size")] = 8,
    host: Annotated[str, typer.Option("--host", help="Bind address")] = "127.0.0.1",
) -> None:
    """Start a long-lived embedding HTTP service with MLX embedder."""
    import uvicorn

    from rag.embedding_service import create_app

    app_instance = create_app(model_name_or_path=model, batch_size=batch_size)
    typer.echo(f"Embedding service starting on http://{host}:{port} (model={model})")
    uvicorn.run(app_instance, host=host, port=port, log_level="info")


@app.command("rerank-service")
def rerank_service(
    model: Annotated[str, typer.Option("--model", help="Reranker model name or path")] = "Qwen/Qwen3-Reranker-4B",
    port: Annotated[int, typer.Option("--port", help="HTTP port")] = 9091,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Rerank batch size")] = 8,
    max_length: Annotated[int, typer.Option("--max-length", help="Max token length per document")] = 1024,
    host: Annotated[str, typer.Option("--host", help="Bind address")] = "127.0.0.1",
) -> None:
    """Start a long-lived rerank HTTP service with FlagEmbedding reranker."""
    import uvicorn

    from rag.rerank_service import create_app

    app_instance = create_app(
        model_name_or_path=model,
        batch_size=batch_size,
        max_length=max_length,
    )
    typer.echo(f"Rerank service starting on http://{host}:{port} (model={model})")
    uvicorn.run(app_instance, host=host, port=port, log_level="info")


__all__ = ["app", "main"]


def main() -> None:
    app()


if __name__ == "__main__":
    main()
