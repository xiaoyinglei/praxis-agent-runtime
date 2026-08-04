from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import cast
from uuid import uuid4

from pydantic import BaseModel

from agent_runtime.memory.models import MemoryPolicy, MemoryRecord, MemoryRef, MemoryRefStatus
from agent_runtime.workspace import WorkspaceRuntime

MEMORY_DIR_NAME = ".agent_memory"
MEMORY_RECORDS_DIR = "records"
_REF_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class MemoryRefError(ValueError):
    """Invalid or unsafe memory reference."""


class WorkspaceMemoryStore:
    """Workspace-backed run-local memory store.

    Raw payloads are stored under `.agent_memory/records/` and are only resolved
    through this object, so normal workspace tools can hide the implementation
    directory.
    """

    def __init__(
        self,
        *,
        workspace: WorkspaceRuntime,
        policy: MemoryPolicy | None = None,
    ) -> None:
        self._workspace = workspace
        self._policy = policy or MemoryPolicy()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def _root(self) -> Path:
        return self._workspace.root / MEMORY_DIR_NAME / MEMORY_RECORDS_DIR

    def write_tool_output(
        self,
        payload: BaseModel,
        *,
        summary: str,
        source_tool_call_id: str | None = None,
        source_tool_name: str | None = None,
        warnings: list[str] | None = None,
    ) -> MemoryRef:
        ref_id = f"mem_{uuid4().hex}"
        path = Path(MEMORY_DIR_NAME) / MEMORY_RECORDS_DIR / f"{ref_id}.json"
        ref = MemoryRef(
            ref_id=ref_id,
            path=path.as_posix(),
            summary=summary,
            source_tool_call_id=source_tool_call_id,
            source_tool_name=source_tool_name,
            status="available",
            warnings=list(warnings or []),
        )
        original_output_model = _model_path(payload)
        record = MemoryRecord(
            original_output_model=original_output_model,
            summary=summary,
            ref=ref,
            status="available",
            warnings=list(warnings or []),
            payload=payload,
        )
        self._write_record(record)
        ref = ref.model_copy(update={"size_bytes": self._record_path(ref).stat().st_size})
        self._write_record(record.model_copy(update={"ref": ref}))
        self._enforce_retention()
        return ref

    def resolve(self, ref: MemoryRef) -> MemoryRecord:
        path = self._record_path(ref)
        if not path.is_file():
            raise FileNotFoundError(f"Memory record not found: {ref.ref_id}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return _record_from_json(raw)

    def tombstone(self, ref: MemoryRef, *, reason: str) -> MemoryRecord:
        record = self.resolve(ref)
        deleted_ref = record.ref.model_copy(update={"status": "deleted"})
        tombstone = MemoryRecord(
            original_output_model=record.original_output_model,
            summary=record.summary,
            ref=deleted_ref,
            status="deleted",
            warnings=record.warnings,
            payload=None,
            ref_deleted=True,
            reason=reason,
        )
        self._write_record(tombstone)
        return tombstone

    def _record_path(self, ref: MemoryRef) -> Path:
        if not _REF_ID_RE.fullmatch(ref.ref_id):
            raise MemoryRefError(f"Invalid memory ref_id: {ref.ref_id!r}")
        raw_path = Path(ref.path)
        expected = Path(MEMORY_DIR_NAME) / MEMORY_RECORDS_DIR / f"{ref.ref_id}.json"
        if raw_path.is_absolute() or ".." in raw_path.parts or raw_path != expected:
            raise MemoryRefError(f"Invalid memory ref path: {ref.path!r}")
        path = (self._workspace.root / raw_path).resolve()
        workspace_root = self._workspace.root.resolve()
        memory_root = (workspace_root / MEMORY_DIR_NAME / MEMORY_RECORDS_DIR).resolve()
        if not str(path).startswith(str(memory_root) + "/"):
            raise MemoryRefError(f"Memory ref escapes memory store: {ref.path!r}")
        return path

    def _write_record(self, record: MemoryRecord) -> None:
        path = self._record_path(record.ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_record_to_json(record), ensure_ascii=False), encoding="utf-8")

    def _enforce_retention(self) -> None:
        available_records: list[tuple[float, MemoryRef]] = []
        for path in self._root.glob("*.json"):
            try:
                record = _record_from_json(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
            if record.status == "available":
                available_records.append((path.stat().st_mtime, record.ref))
        excess = len(available_records) - self._policy.max_memory_records
        if excess <= 0:
            return
        for _, ref in sorted(available_records)[:excess]:
            self.tombstone(ref, reason="retention_limit")


def _record_to_json(record: MemoryRecord) -> dict[str, object]:
    payload_data: object | None = None
    if record.payload is not None:
        payload_data = record.payload.model_dump(mode="json")
    return {
        "schema_version": record.schema_version,
        "original_output_model": record.original_output_model,
        "summary": record.summary,
        "ref": record.ref.model_dump(mode="json"),
        "status": record.status,
        "warnings": record.warnings,
        "payload": payload_data,
        "ref_deleted": record.ref_deleted,
        "reason": record.reason,
    }


def _record_from_json(raw: dict[str, object]) -> MemoryRecord:
    model_path = str(raw["original_output_model"])
    payload_raw = raw.get("payload")
    payload = None if payload_raw is None else _restore_model(model_path, payload_raw)
    return MemoryRecord(
        schema_version=_int_value(raw.get("schema_version"), default=1),
        original_output_model=model_path,
        summary=str(raw.get("summary", "")),
        ref=MemoryRef.model_validate(raw["ref"]),
        status=_status_value(raw.get("status")),
        warnings=_string_list(raw.get("warnings")),
        payload=payload,
        ref_deleted=bool(raw.get("ref_deleted", False)),
        reason=None if raw.get("reason") is None else str(raw.get("reason")),
    )


def _restore_model(model_path: str, payload: object) -> BaseModel:
    if not isinstance(payload, dict):
        raise ValueError("memory payload must be a JSON object")
    module_name, class_name = model_path.rsplit(".", 1)
    model_cls = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
        raise ValueError(f"memory payload model is not a Pydantic model: {model_path}")
    return model_cls.model_validate(payload)


def _int_value(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"expected integer value, got {type(value).__name__}")


def _status_value(value: object) -> MemoryRefStatus:
    if value in {"available", "deleted", "unavailable", "compacted"}:
        return cast(MemoryRefStatus, value)
    return "available"


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _model_path(model: BaseModel) -> str:
    return f"{model.__class__.__module__}.{model.__class__.__name__}"


__all__ = [
    "MEMORY_DIR_NAME",
    "MEMORY_RECORDS_DIR",
    "MemoryRefError",
    "WorkspaceMemoryStore",
]
