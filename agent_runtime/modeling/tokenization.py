from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec
from typing import Any

from agent_runtime.text import (
    DEFAULT_TOKENIZER_FALLBACK_MODEL,
    _token_unit_spans,
    text_unit_count,
)


@dataclass(frozen=True, slots=True)
class TokenizerContract:
    embedding_model_name: str
    tokenizer_model_name: str
    chunking_tokenizer_model_name: str
    tokenizer_backend: str = "auto"
    chunk_token_size: int = 480
    chunk_overlap_tokens: int = 64
    max_context_tokens: int = 1024
    prompt_reserved_tokens: int = 256
    local_files_only: bool = True

    def normalized_chunk_overlap_tokens(self) -> int:
        return min(self.chunk_overlap_tokens, max(self.chunk_token_size - 1, 0))


@dataclass(slots=True)
class TokenAccountingService:
    contract: TokenizerContract
    _backend_kind: str | None = None
    _backend: Any | None = None

    def count(self, text: str) -> int:
        normalized = text.strip()
        if not normalized:
            return 0
        encoded = self._encode(normalized)
        return len(encoded) if encoded is not None else text_unit_count(normalized)

    def clip(self, text: str, token_budget: int, *, add_ellipsis: bool = False) -> str:
        normalized_budget = max(token_budget, 1)
        normalized = text.strip()
        if not normalized:
            return ""
        spans = self._offset_spans(normalized)
        if spans is not None:
            clipped = self._clip_with_spans(normalized, spans, normalized_budget)
        else:
            encoded = self._encode(normalized)
            if encoded is None:
                clipped = self._clip_with_units(normalized, normalized_budget)
            elif len(encoded) <= normalized_budget:
                clipped = normalized
            else:
                clipped = self._decode(encoded[:normalized_budget]).strip()
        if not clipped:
            return ""
        if add_ellipsis and self.count(clipped) < self.count(normalized):
            return f"{clipped} ..."
        return clipped

    def tail(self, text: str, token_budget: int) -> str:
        normalized_budget = max(token_budget, 0)
        normalized = text.strip()
        if normalized_budget <= 0 or not normalized:
            return ""
        spans = self._offset_spans(normalized)
        if spans is not None:
            return self._tail_with_spans(normalized, spans, normalized_budget)
        encoded = self._encode(normalized)
        if encoded is None:
            unit_spans = _token_unit_spans(normalized)
            if not unit_spans:
                return ""
            start = unit_spans[max(len(unit_spans) - normalized_budget, 0)][0]
            return normalized[start:].strip()
        if len(encoded) <= normalized_budget:
            return normalized
        return self._decode(encoded[-normalized_budget:]).strip()

    def chunk_text(
        self,
        text: str,
        *,
        chunk_token_size: int | None = None,
        chunk_overlap_tokens: int | None = None,
    ) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        size = max(chunk_token_size or self.contract.chunk_token_size, 1)
        resolved_overlap = (
            chunk_overlap_tokens
            if chunk_overlap_tokens is not None
            else self.contract.normalized_chunk_overlap_tokens()
        )
        overlap = min(max(resolved_overlap, 0), max(size - 1, 0))
        spans = self._offset_spans(normalized)
        if spans is not None:
            if not spans:
                return []
            step = max(size - overlap, 1)
            chunks: list[str] = []
            for start_index in range(0, len(spans), step):
                span_window = spans[start_index : start_index + size]
                if not span_window:
                    continue
                chunks.append(normalized[span_window[0][0] : span_window[-1][1]].strip())
                if start_index + size >= len(spans):
                    break
            return [chunk for chunk in chunks if chunk]
        encoded = self._encode(normalized)
        if encoded is None:
            spans = _token_unit_spans(normalized)
            if not spans:
                return []
            step = max(size - overlap, 1)
            chunks = []
            for start_index in range(0, len(spans), step):
                span_window = spans[start_index : start_index + size]
                if not span_window:
                    continue
                chunks.append(normalized[span_window[0][0] : span_window[-1][1]].strip())
                if start_index + size >= len(spans):
                    break
            return [chunk for chunk in chunks if chunk]
        step = max(size - overlap, 1)
        chunks = []
        for start in range(0, len(encoded), step):
            token_window = encoded[start : start + size]
            if not token_window:
                continue
            chunk = self._decode(token_window).strip()
            if chunk:
                chunks.append(chunk)
            if start + size >= len(encoded):
                break
        return chunks

    def prompt_budget(self, total_budget: int | None = None) -> int:
        resolved_budget = total_budget or self.contract.max_context_tokens
        return max(resolved_budget - self.contract.prompt_reserved_tokens, 1)

    def backend_descriptor(self) -> tuple[str, str]:
        self._ensure_backend()
        return self._backend_kind or "simple", self.contract.tokenizer_model_name

    def _ensure_backend(self) -> None:
        if self._backend is not None or self._backend_kind is not None:
            return
        backend_kind, backend = _build_tokenizer_backend(
            self.contract.tokenizer_model_name,
            backend=self.contract.tokenizer_backend,
            local_files_only=self.contract.local_files_only,
        )
        self._backend_kind = backend_kind
        self._backend = backend

    def _encode(self, text: str) -> list[int] | None:
        self._ensure_backend()
        if self._backend_kind == "tiktoken":
            backend = self._backend
            return list(backend.encode(text)) if backend is not None else None
        if self._backend_kind == "transformers":
            backend = self._backend
            return list(backend.encode(text, add_special_tokens=False)) if backend is not None else None
        return None

    def _decode(self, tokens: Sequence[int]) -> str:
        self._ensure_backend()
        if self._backend_kind == "tiktoken":
            backend = self._backend
            return "" if backend is None else str(backend.decode(list(tokens)))
        if self._backend_kind == "transformers":
            backend = self._backend
            if backend is None:
                return ""
            return str(
                backend.decode(
                    list(tokens),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        return ""

    @staticmethod
    def _clip_with_units(text: str, token_budget: int) -> str:
        spans = _token_unit_spans(text)
        if len(spans) <= token_budget:
            return text
        return text[: spans[token_budget - 1][1]].strip()

    def _offset_spans(self, text: str) -> list[tuple[int, int]] | None:
        self._ensure_backend()
        if self._backend_kind == "simple":
            return _token_unit_spans(text)
        if self._backend_kind != "transformers":
            return None
        backend = self._backend
        if backend is None:
            return None
        try:
            payload = backend(text, add_special_tokens=False, return_offsets_mapping=True)
        except Exception:
            return None
        offsets = getattr(payload, "offset_mapping", None)
        if offsets is None and isinstance(payload, dict):
            offsets = payload.get("offset_mapping")
        if not isinstance(offsets, list):
            return None
        return [
            (int(start), int(end))
            for start, end in offsets
            if isinstance(start, int) and isinstance(end, int) and end > start
        ]

    @staticmethod
    def _clip_with_spans(text: str, spans: Sequence[tuple[int, int]], token_budget: int) -> str:
        if len(spans) <= token_budget:
            return text
        return text[: spans[token_budget - 1][1]].strip()

    @staticmethod
    def _tail_with_spans(text: str, spans: Sequence[tuple[int, int]], token_budget: int) -> str:
        if len(spans) <= token_budget:
            return text
        start = spans[max(len(spans) - token_budget, 0)][0]
        return text[start:].strip()


@lru_cache(maxsize=12)
def _build_tokenizer_backend(
    model_name: str,
    *,
    backend: str,
    local_files_only: bool,
) -> tuple[str, Any | None]:
    preferred = _preferred_backend(backend=backend)
    for candidate in preferred:
        built = _try_build_tokenizer(model_name=model_name, backend=candidate, local_files_only=local_files_only)
        if built is not None:
            return candidate, built
    return "simple", None


def _preferred_backend(*, backend: str) -> tuple[str, ...]:
    normalized_backend = backend.strip().lower()
    if normalized_backend == "heuristic":
        normalized_backend = "simple"
    if normalized_backend in {"tiktoken", "transformers", "simple"}:
        return (normalized_backend,)
    return ("transformers", "tiktoken", "simple")


def _try_build_tokenizer(*, model_name: str, backend: str, local_files_only: bool) -> Any | None:
    if backend == "simple":
        return None
    if backend == "tiktoken":
        spec = find_spec("tiktoken")
        if spec is None:
            return None
        import tiktoken

        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            return None
    if backend == "transformers":
        spec = find_spec("transformers")
        if spec is None:
            return None
        from transformers import AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                use_fast=True,
                trust_remote_code=True,
            )
        except Exception:
            return None
    return None


__all__ = [
    "DEFAULT_TOKENIZER_FALLBACK_MODEL",
    "TokenAccountingService",
    "TokenizerContract",
]
