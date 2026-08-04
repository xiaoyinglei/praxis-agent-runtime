from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_runtime.modeling.config import ModelCapability
from rag.models.catalog import ModelCatalog

_MAX_BATCH_SIZE = 32
_MAX_REQUEST_BYTES = 10 * 1024 * 1024  # 10 MB
_REQUEST_BODY_LIMIT = _MAX_REQUEST_BYTES


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(max_length=_MAX_BATCH_SIZE, min_length=1)
    mode: Literal["query", "document"] = "document"
    batch_size: int | None = Field(default=None, ge=1, le=_MAX_BATCH_SIZE)


class EmbeddingResponse(BaseModel):
    vectors: list[list[float]]
    dimension: int


class HealthResponse(BaseModel):
    model: str
    embedding_space: str
    dimension: int


app = FastAPI(title="rag-embedding-service")
_embedder: Any = None
_dimension: int = 0
_model_name: str = ""
_embedding_space: str = "default"
_semaphore = asyncio.Semaphore(1)


@app.on_event("startup")
async def _warmup() -> None:  # not called when loaded as module (uvicorn calls it)
    pass


def create_app(
    model_name_or_path: str,
    *,
    batch_size: int = 8,
    pooling: Literal["last_token", "mean"] = "last_token",
    normalize: bool = True,
    query_prefix: str = "",
    document_prefix: str = "",
    tokenizer_config: dict[str, Any] | None = None,
) -> FastAPI:
    """Create and warmup the FastAPI app with an MLX embedder."""
    from rag.providers.mlx.embedder import MLXEmbedder

    global _embedder, _dimension, _model_name, _embedding_space

    _embedder = MLXEmbedder(
        model_name_or_path=model_name_or_path,
        batch_size=batch_size,
        pooling=pooling,
        normalize=normalize,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
        tokenizer_config=tokenizer_config,
    )

    # warmup to determine dimension
    warmup_vecs = _embedder.embed(["warmup"])
    if not warmup_vecs or not warmup_vecs[0]:
        raise RuntimeError("MLX embedder warmup failed: empty result")

    _dimension = len(warmup_vecs[0])
    _model_name = _embedder.embedding_model_name
    _embedding_space = embedding_space_for_model(model_name_or_path)

    return app


def embedding_space_for_model(model_name_or_path: str) -> str:
    """Resolve the stable vector-space identifier for an embedding model."""
    model_name_or_path = model_name_or_path.strip()
    if not model_name_or_path:
        return "default"
    try:
        catalog = ModelCatalog.from_yaml()
    except Exception:
        return model_name_or_path
    for spec in catalog.list_models(ModelCapability.EMBEDDING):
        if spec.model == model_name_or_path:
            return spec.embedding_space or spec.model
    return model_name_or_path


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if _embedder is None:
        raise HTTPException(status_code=503, detail="embedder not loaded")
    return HealthResponse(
        model=_model_name,
        embedding_space=_embedding_space,
        dimension=_dimension,
    )


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest) -> EmbeddingResponse:
    if _embedder is None:
        raise HTTPException(status_code=503, detail="embedder not loaded")

    async with _semaphore:
        loop = asyncio.get_running_loop()
        try:
            vectors: list[list[float]] = await loop.run_in_executor(
                None,
                lambda: _embed(request.texts, mode=request.mode, batch_size=request.batch_size),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if len(vectors) != len(request.texts):
        raise HTTPException(
            status_code=500,
            detail=f"count mismatch: expected {len(request.texts)}, got {len(vectors)}",
        )

    return EmbeddingResponse(vectors=vectors, dimension=_dimension)


def _embed(texts: Sequence[str], *, mode: str, batch_size: int | None) -> list[list[float]]:
    kwargs: dict[str, Any] = {"mode": mode}
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    result = _embedder.embed(texts, **kwargs)
    return [list(vec) for vec in result]
