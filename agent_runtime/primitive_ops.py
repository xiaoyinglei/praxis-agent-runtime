"""Checkpoint-stable models for structured file previews."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FileKind = Literal["directory", "text", "binary", "unknown"]
CellValue = str | int | float | bool | None


class CandidateHeaderRow(BaseModel):
    row_index: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class StructuredTableProbe(BaseModel):
    table_index: int = Field(ge=0)
    name: str | None = None
    used_range: str | None = None
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    sample_rows: list[list[CellValue]] = Field(default_factory=list)
    candidate_header_rows: list[CandidateHeaderRow] = Field(default_factory=list)
    data_start_row: int | None = Field(default=None, ge=1)
    merged_cells: bool = False
    formula_columns: list[str] = Field(default_factory=list)


class StructuredProbeOutput(BaseModel):
    path: str
    file_kind: FileKind = "unknown"
    mime_type: str | None = None
    tables: list[StructuredTableProbe] = Field(default_factory=list)
    truncated: bool = False
    errors: list[str] = Field(default_factory=list)


__all__ = [
    "CandidateHeaderRow",
    "CellValue",
    "FileKind",
    "StructuredProbeOutput",
    "StructuredTableProbe",
]
