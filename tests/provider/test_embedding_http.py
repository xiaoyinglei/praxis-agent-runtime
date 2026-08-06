from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from rag.embedding_service import embedding_space_for_model
from rag.providers.embedding_http import EmbeddingHttpClient


class _FakeTransport(httpx.BaseTransport):
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self._handler = handler

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._handler(request)


def _json_response(status_code: int, data: object) -> httpx.Response:
    return httpx.Response(status_code, json=data)


def _make_client(handler: Callable[[httpx.Request], httpx.Response]) -> EmbeddingHttpClient:
    return EmbeddingHttpClient(
        "http://127.0.0.1:9090",
        client=httpx.Client(transport=_FakeTransport(handler)),
    )


# ── Health check failures ──


def test_health_check_fails_on_non_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "internal error"})

    with pytest.raises(RuntimeError, match="health check failed"):
        _make_client(handler)


def test_health_check_fails_on_missing_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, {"embedding_space": "default", "dimension": 768})

    with pytest.raises(RuntimeError, match="health check"):
        _make_client(handler)


def test_health_check_fails_on_missing_dimension() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, {"model": "test-model", "embedding_space": "default"})

    with pytest.raises(RuntimeError, match="health check"):
        _make_client(handler)


# ── Successful construction ──


def test_health_check_success_stores_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, {"model": "bge-m3", "embedding_space": "default", "dimension": 1024})

    client = _make_client(handler)
    assert client.embedding_model_name == "bge-m3"
    assert client.embedding_space == "default"
    assert client.dimension == 1024


def test_embedding_service_resolves_catalog_embedding_space_for_known_model() -> None:
    assert (
        embedding_space_for_model("mlx-community/Qwen3-Embedding-4B-4bit-DWQ")
        == "mlx/Qwen3-Embedding-4B-4bit-DWQ"
    )


def test_embedding_service_uses_model_name_as_space_for_unknown_model() -> None:
    assert embedding_space_for_model("custom/local-embedder") == "custom/local-embedder"


# ── embed ──


def test_embed_empty_texts_returns_empty() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if request.url.path == "/health":
            call_count += 1
            return _json_response(200, {"model": "m", "embedding_space": "d", "dimension": 4})
        raise AssertionError("unexpected request")

    client = _make_client(handler)
    assert client.embed([]) == []


def test_embed_returns_correct_count() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(200, {"model": "m", "embedding_space": "d", "dimension": 4})
        if request.url.path == "/v1/embeddings":
            body = json.loads(request.content)
            n = len(body["texts"])
            return _json_response(200, {"vectors": [[0.1] * 4] * n, "dimension": 4})
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = _make_client(handler)
    result = client.embed(["hello", "world"])
    assert len(result) == 2
    assert len(result[0]) == 4


def test_embed_splits_requests_by_call_batch_size() -> None:
    request_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(200, {"model": "m", "embedding_space": "d", "dimension": 4})
        if request.url.path == "/v1/embeddings":
            body = json.loads(request.content)
            request_sizes.append(len(body["texts"]))
            assert body["batch_size"] == 2
            return _json_response(200, {"vectors": [[0.1] * 4] * len(body["texts"]), "dimension": 4})
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = _make_client(handler)
    result = client.embed(["a", "b", "c", "d", "e"], batch_size=2)

    assert len(result) == 5
    assert request_sizes == [2, 2, 1]


def test_embed_splits_requests_by_default_batch_size() -> None:
    request_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(200, {"model": "m", "embedding_space": "d", "dimension": 4})
        if request.url.path == "/v1/embeddings":
            body = json.loads(request.content)
            request_sizes.append(len(body["texts"]))
            assert body["batch_size"] == 2
            return _json_response(200, {"vectors": [[0.1] * 4] * len(body["texts"]), "dimension": 4})
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = EmbeddingHttpClient(
        "http://127.0.0.1:9090",
        batch_size=2,
        client=httpx.Client(transport=_FakeTransport(handler)),
    )
    result = client.embed(["a", "b", "c", "d", "e"])

    assert len(result) == 5
    assert request_sizes == [2, 2, 1]


def test_embed_count_mismatch_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(200, {"model": "m", "embedding_space": "d", "dimension": 4})
        if request.url.path == "/v1/embeddings":
            return _json_response(200, {"vectors": [[0.1, 0.2, 0.3, 0.4]], "dimension": 4})
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = _make_client(handler)
    with pytest.raises(RuntimeError, match="count mismatch"):
        client.embed(["hello", "world"])


def test_embed_dimension_mismatch_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(200, {"model": "m", "embedding_space": "d", "dimension": 4})
        if request.url.path == "/v1/embeddings":
            return _json_response(200, {"vectors": [[0.1] * 8], "dimension": 8})
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = _make_client(handler)
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        client.embed(["hello"])


def test_embed_request_body_has_expected_shape() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(200, {"model": "m", "embedding_space": "d", "dimension": 3})
        if request.url.path == "/v1/embeddings":
            nonlocal captured
            captured = json.loads(request.content)
            return _json_response(200, {"vectors": [[0.1, 0.2, 0.3]], "dimension": 3})
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = _make_client(handler)
    client.embed(["test text"], mode="query")
    assert captured["texts"] == ["test text"]
    assert captured.get("mode") == "query"


def test_embed_service_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(200, {"model": "m", "embedding_space": "d", "dimension": 4})
        if request.url.path == "/v1/embeddings":
            return _json_response(422, {"detail": "batch too large"})
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = _make_client(handler)
    with pytest.raises(RuntimeError, match="Embedding service error"):
        client.embed(["hello"])


def test_embed_preserves_query_prefix_mode() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(200, {"model": "m", "embedding_space": "d", "dimension": 3})
        if request.url.path == "/v1/embeddings":
            nonlocal captured
            captured = json.loads(request.content)
            return _json_response(200, {"vectors": [[0.1, 0.2, 0.3]], "dimension": 3})
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = _make_client(handler)
    client.embed(["hello"], mode="document")
    assert captured.get("mode") == "document"
