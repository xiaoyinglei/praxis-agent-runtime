from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rag.schema.core import ParsedDocument, SourceType


class DocumentParser(Protocol):
    def parse(
        self,
        file_path: Path,
        *,
        location: str,
        source_type: SourceType,
        title: str | None = None,
        owner: str = "user",
    ) -> ParsedDocument: ...


class ExtractionDispatcher:
    """
    数据加工厂的“前端交警”。
    职责：基于文件后缀识别身份 -> 查路由表 -> 分发给对应的 Parser。
    """

    # 全局格式收口：以后支持新格式，只在这里加一行后缀即可
    _SUFFIX_MAP = {
        ".pdf": SourceType.PDF,
        ".md": SourceType.MARKDOWN,
        ".markdown": SourceType.MARKDOWN,
        ".docx": SourceType.DOCX,
        ".xlsx": SourceType.XLSX,
        ".xls": SourceType.XLSX,
        ".pptx": SourceType.PPTX,
        ".png": SourceType.IMAGE,
        ".jpg": SourceType.IMAGE,
        ".jpeg": SourceType.IMAGE,
    }

    def __init__(
        self,
        docling_parser: DocumentParser,
        excel_parser: DocumentParser,
        pptx_parser: DocumentParser,
        image_parser: DocumentParser,
    ) -> None:
        # 2. 路由表映射：
        self._routes: dict[SourceType, DocumentParser] = {
            SourceType.PDF: docling_parser,
            SourceType.MARKDOWN: docling_parser,
            SourceType.DOCX: docling_parser,
            SourceType.XLSX: excel_parser,
            SourceType.PPTX: pptx_parser,
            SourceType.IMAGE: image_parser,
        }

    def infer_source_type(self, file_path: Path) -> SourceType:
        """集中处理格式识别，Fail Fast 拦截垃圾文件"""
        suffix = file_path.suffix.lower()
        if suffix not in self._SUFFIX_MAP:
            raise ValueError(f"Dispatcher unsupported file extension: {suffix}")
        return self._SUFFIX_MAP[suffix]

    def route_and_parse(
        self,
        file_path: Path,
        *,
        location: str,
        source_type: SourceType | None = None,
        title: str | None = None,
        owner: str = "user",
    ) -> ParsedDocument:
        # 第一步：交警认车牌
        inferred_source_type = self.infer_source_type(file_path)
        if source_type is not None and source_type is not inferred_source_type:
            raise ValueError(
                f"Dispatcher source_type mismatch: declared={source_type.value} inferred={inferred_source_type.value}"
            )
        resolved_source_type = source_type or inferred_source_type

        # 第二步：查表
        parser = self._routes.get(resolved_source_type)
        if not parser:
            raise NotImplementedError(f"No parser registered for {resolved_source_type}")
        # 第三步：分发到对应的 Parser
        return parser.parse(
            file_path, 
            location=location, 
            source_type=resolved_source_type,
            title=title, 
            owner=owner
        )
