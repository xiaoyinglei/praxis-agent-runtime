from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from rag.schema.core import OcrResult


T = TypeVar("T", bound=BaseModel)

(
    "模型与多模态能力的抽象协议：定义文本生成、结构化生成、向量编码、重排序、OCR "
    "识别和视觉描述等外部能力接口，供不同模型服务或实现类统一接入。"
)


class Generator(Protocol):
    def generate_text(self, *, prompt: str, **kwargs: Any) -> str: ...
    def generate_structured(self, *, prompt: str, schema: type[T], **kwargs: Any) -> T: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]: ...


class Reranker(Protocol):
    def rerank(self, query: str, documents: Sequence[str], **kwargs: Any) -> list[float]: ...


class OcrVisionRepo(Protocol):
    def extract(self, image_path: Path, **kwargs: Any) -> OcrResult: ...


class VisualDescriptionRepo(Protocol):
    def describe_visual(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/png",
        prompt: str | None = None,
        **kwargs: Any,
    ) -> str: ...
