from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from threading import Lock
from typing import Any, cast

from rag.schema.core import (
    AssetRecord,
    Document,
    LayoutMetaCacheRecord,
    ProcessingStateRecord,
    SectionRecord,
    Source,
    StorageTier,
)
from rag.schema.runtime import AccessPolicy


class SnowflakeIdGenerator:
    _EPOCH_MS = 1704067200000

    def __init__(self, worker_id: int | None = None) -> None:
        self._worker_id = (os.getpid() if worker_id is None else worker_id) & 0x3FF
        self._sequence = 0
        self._last_timestamp_ms = -1
        self._lock = Lock()

    def next_id(self) -> int:
        with self._lock:
            now_ms = int(time.time() * 1000)
            if now_ms < self._last_timestamp_ms:
                now_ms = self._last_timestamp_ms
            if now_ms == self._last_timestamp_ms:
                self._sequence = (self._sequence + 1) & 0xFFF
                if self._sequence == 0:
                    while now_ms <= self._last_timestamp_ms:
                        time.sleep(0.001)
                        now_ms = int(time.time() * 1000)
            else:
                self._sequence = 0
            self._last_timestamp_ms = now_ms
            return ((now_ms - self._EPOCH_MS) << 22) | (self._worker_id << 12) | self._sequence


class PostgresMetadataRepo:
    def __init__(self, dsn: str, *, schema: str = "public") -> None:
        self._dsn = dsn
        self._schema = schema
        self._id_generator = SnowflakeIdGenerator()
        self._conn: Any = self._connect()
        self._ensure_schema()

    def next_id(self) -> int:
        return self._id_generator.next_id()

    def save_source(self, source: Source) -> Source:
        now = datetime.now(UTC)
        saved_source = source.model_copy(
            update={
                "source_id": source.source_id if source.source_id > 0 else self.next_id(),
                "updated_at": now,
            }
        )
        row = self._fetchone(
            f"""
            INSERT INTO {self._schema}.sources (
                source_id,
                source_type,
                location,
                original_file_name,
                bucket,
                object_key,
                content_hash,
                file_size_bytes,
                mime_type,
                owner_id,
                ingest_version,
                created_at,
                updated_at,
                metadata_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (location, content_hash, ingest_version) DO UPDATE SET
                source_type = EXCLUDED.source_type,
                original_file_name = EXCLUDED.original_file_name,
                bucket = EXCLUDED.bucket,
                object_key = EXCLUDED.object_key,
                file_size_bytes = EXCLUDED.file_size_bytes,
                mime_type = EXCLUDED.mime_type,
                owner_id = EXCLUDED.owner_id,
                updated_at = EXCLUDED.updated_at,
                metadata_json = EXCLUDED.metadata_json
            RETURNING *
            """,
            (
                saved_source.source_id,
                saved_source.source_type.value,
                saved_source.location,
                saved_source.original_file_name,
                saved_source.bucket,
                saved_source.object_key,
                saved_source.content_hash,
                saved_source.file_size_bytes,
                saved_source.mime_type,
                saved_source.owner_id,
                saved_source.ingest_version,
                saved_source.created_at,
                saved_source.updated_at,
                self._json_dumps(saved_source.metadata_json),
            ),
        )
        self._conn.commit()
        if row is None:
            raise RuntimeError("saving source returned no row")
        return self._source_from_row(row)

    def get_source(self, source_id: int) -> Source | None:
        row = self._fetchone(f"SELECT * FROM {self._schema}.sources WHERE source_id = %s", (source_id,))
        return None if row is None else self._source_from_row(row)

    def get_source_by_location_and_hash(self, location: str, content_hash: str) -> Source | None:
        row = self._fetchone(
            f"""
            SELECT *
            FROM {self._schema}.sources
            WHERE location = %s AND content_hash = %s
            ORDER BY ingest_version DESC, updated_at DESC
            LIMIT 1
            """,
            (location, content_hash),
        )
        return None if row is None else self._source_from_row(row)

    def find_source_by_content_hash(self, content_hash: str) -> Source | None:
        row = self._fetchone(
            f"""
            SELECT *
            FROM {self._schema}.sources
            WHERE content_hash = %s
            ORDER BY updated_at DESC, source_id DESC
            LIMIT 1
            """,
            (content_hash,),
        )
        return None if row is None else self._source_from_row(row)

    def get_latest_source_for_location(self, location: str) -> Source | None:
        row = self._fetchone(
            f"""
            SELECT *
            FROM {self._schema}.sources
            WHERE location = %s
            ORDER BY ingest_version DESC, updated_at DESC
            LIMIT 1
            """,
            (location,),
        )
        return None if row is None else self._source_from_row(row)

    def list_sources(self, *, location: str | None = None) -> list[Source]:
        clauses: list[str] = []
        params: list[object] = []
        if location is not None:
            clauses.append("location = %s")
            params.append(location)
        where_sql = "" if not clauses else f" WHERE {' AND '.join(clauses)}"
        rows = self._fetchall(
            f"SELECT * FROM {self._schema}.sources{where_sql} ORDER BY updated_at DESC, source_id DESC",
            tuple(params),
        )
        return [self._source_from_row(row) for row in rows]

    def delete_source(self, source_id: int) -> int:
        cursor = self._conn.execute(f"DELETE FROM {self._schema}.sources WHERE source_id = %s", (source_id,))
        self._conn.commit()
        return int(cursor.rowcount)

    def save_document(self, document: Document) -> Document:
        now = datetime.now(UTC)
        doc_id = document.doc_id if document.doc_id > 0 else self.next_id()
        version_group_id = document.version_group_id if document.version_group_id > 0 else doc_id
        saved_document = document.model_copy(
            update={
                "doc_id": doc_id,
                "version_group_id": version_group_id,
                "updated_at": now,
            }
        )
        row = self._fetchone(
            f"""
            INSERT INTO {self._schema}.documents (
                doc_id,
                source_id,
                title,
                language,
                authors,
                file_hash,
                version_group_id,
                version_no,
                doc_status,
                effective_date,
                is_active,
                is_indexed,
                index_ready,
                index_priority,
                storage_tier,
                reference_count,
                page_count,
                tenant_id,
                department_id,
                auth_tag,
                embedding_model_id,
                indexed_at,
                last_index_error,
                created_at,
                updated_at,
                metadata_json
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (source_id, file_hash, version_no) DO UPDATE SET
                title = EXCLUDED.title,
                language = EXCLUDED.language,
                authors = EXCLUDED.authors,
                doc_status = EXCLUDED.doc_status,
                effective_date = EXCLUDED.effective_date,
                is_active = EXCLUDED.is_active,
                is_indexed = EXCLUDED.is_indexed,
                index_ready = EXCLUDED.index_ready,
                index_priority = EXCLUDED.index_priority,
                storage_tier = EXCLUDED.storage_tier,
                reference_count = EXCLUDED.reference_count,
                page_count = EXCLUDED.page_count,
                tenant_id = EXCLUDED.tenant_id,
                department_id = EXCLUDED.department_id,
                auth_tag = EXCLUDED.auth_tag,
                embedding_model_id = EXCLUDED.embedding_model_id,
                indexed_at = EXCLUDED.indexed_at,
                last_index_error = EXCLUDED.last_index_error,
                updated_at = EXCLUDED.updated_at,
                metadata_json = EXCLUDED.metadata_json
            RETURNING *
            """,
            (
                saved_document.doc_id,
                saved_document.source_id,
                saved_document.title,
                saved_document.language,
                self._json_dumps(saved_document.authors),
                saved_document.file_hash,
                saved_document.version_group_id,
                saved_document.version_no,
                self._enum_value(saved_document.doc_status),
                saved_document.effective_date,
                saved_document.is_active,
                saved_document.is_indexed,
                saved_document.index_ready,
                saved_document.index_priority,
                saved_document.storage_tier.value,
                saved_document.reference_count,
                saved_document.page_count,
                saved_document.tenant_id,
                saved_document.department_id,
                saved_document.auth_tag,
                saved_document.embedding_model_id,
                saved_document.indexed_at,
                saved_document.last_index_error,
                saved_document.created_at,
                saved_document.updated_at,
                self._json_dumps(saved_document.metadata_json),
            ),
        )
        self._conn.commit()
        if row is None:
            raise RuntimeError("saving document returned no row")
        return self._document_from_row(row)

    def get_document(self, doc_id: int) -> Document | None:
        row = self._fetchone(f"SELECT * FROM {self._schema}.documents WHERE doc_id = %s", (doc_id,))
        return None if row is None else self._document_from_row(row)

    def list_documents(
        self,
        *,
        source_id: int | None = None,
        active_only: bool = False,
        version_group_id: int | None = None,
        storage_tier: StorageTier | None = None,
    ) -> list[Document]:
        clauses: list[str] = []
        params: list[object] = []
        if source_id is not None:
            clauses.append("source_id = %s")
            params.append(source_id)
        if active_only:
            clauses.append("is_active = TRUE")
        if version_group_id is not None:
            clauses.append("version_group_id = %s")
            params.append(version_group_id)
        if storage_tier is not None:
            clauses.append("storage_tier = %s")
            params.append(storage_tier.value)
        where_sql = "" if not clauses else f" WHERE {' AND '.join(clauses)}"
        rows = self._fetchall(
            f"""
            SELECT *
            FROM {self._schema}.documents{where_sql}
            ORDER BY version_group_id, version_no DESC, updated_at DESC
            """,
            tuple(params),
        )
        return [self._document_from_row(row) for row in rows]

    def find_document_by_hash(
        self,
        file_hash: str,
        *,
        source_id: int | None = None,
        active_only: bool = False,
    ) -> Document | None:
        clauses = ["file_hash = %s"]
        params: list[object] = [file_hash]
        if source_id is not None:
            clauses.append("source_id = %s")
            params.append(source_id)
        if active_only:
            clauses.append("is_active = TRUE")
        row = self._fetchone(
            f"""
            SELECT *
            FROM {self._schema}.documents
            WHERE {" AND ".join(clauses)}
            ORDER BY updated_at DESC, doc_id DESC
            LIMIT 1
            """,
            tuple(params),
        )
        return None if row is None else self._document_from_row(row)

    def get_document_versions(self, version_group_id: int) -> list[Document]:
        return self.list_documents(version_group_id=version_group_id)

    def activate_document_version(self, doc_id: int) -> Document:
        group_row = self._fetchone(
            f"SELECT version_group_id FROM {self._schema}.documents WHERE doc_id = %s",
            (doc_id,),
        )
        if group_row is None:
            raise KeyError(f"document {doc_id} not found")
        now = datetime.now(UTC)
        self._conn.execute(
            f"""
            UPDATE {self._schema}.documents
            SET is_active = (doc_id = %s),
                updated_at = %s
            WHERE version_group_id = %s
            """,
            (doc_id, now, int(group_row["version_group_id"])),
        )
        self._conn.commit()
        updated = self.get_document(doc_id)
        if updated is None:
            raise RuntimeError("activating document version returned no row")
        return updated

    def set_document_index_state(
        self,
        doc_id: int,
        *,
        is_indexed: bool | None = None,
        index_ready: bool | None = None,
        embedding_model_id: str | None = None,
        indexed_at: datetime | None = None,
        last_index_error: str | None = None,
    ) -> Document:
        now = datetime.now(UTC)
        clauses = ["updated_at = %s"]
        params: list[object] = [now]
        if is_indexed is not None:
            clauses.append("is_indexed = %s")
            params.append(is_indexed)
        if index_ready is not None:
            clauses.append("index_ready = %s")
            params.append(index_ready)
        if embedding_model_id is not None:
            clauses.append("embedding_model_id = %s")
            params.append(embedding_model_id)
        if indexed_at is not None:
            clauses.append("indexed_at = %s")
            params.append(indexed_at)
        elif is_indexed:
            clauses.append("indexed_at = %s")
            params.append(now)
        clauses.append("last_index_error = %s")
        params.append(last_index_error)
        params.append(doc_id)
        row = self._fetchone(
            f"""
            UPDATE {self._schema}.documents
            SET {", ".join(clauses)}
            WHERE doc_id = %s
            RETURNING *
            """,
            tuple(params),
        )
        self._conn.commit()
        if row is None:
            raise KeyError(f"document {doc_id} not found")
        return self._document_from_row(row)

    def increment_document_reference_count(self, doc_id: int, *, amount: int = 1) -> Document:
        row = self._fetchone(
            f"""
            UPDATE {self._schema}.documents
            SET reference_count = reference_count + %s,
                updated_at = %s
            WHERE doc_id = %s
            RETURNING *
            """,
            (amount, datetime.now(UTC), doc_id),
        )
        self._conn.commit()
        if row is None:
            raise KeyError(f"document {doc_id} not found")
        return self._document_from_row(row)

    def set_document_storage_tier(self, doc_id: int, *, storage_tier: StorageTier) -> Document:
        row = self._fetchone(
            f"""
            UPDATE {self._schema}.documents
            SET storage_tier = %s,
                updated_at = %s
            WHERE doc_id = %s
            RETURNING *
            """,
            (storage_tier.value, datetime.now(UTC), doc_id),
        )
        self._conn.commit()
        if row is None:
            raise KeyError(f"document {doc_id} not found")
        return self._document_from_row(row)

    def deactivate_document(self, doc_id: int) -> Document:
        row = self._fetchone(
            f"""
            UPDATE {self._schema}.documents
            SET is_active = FALSE,
                updated_at = %s
            WHERE doc_id = %s
            RETURNING *
            """,
            (datetime.now(UTC), doc_id),
        )
        self._conn.commit()
        if row is None:
            raise KeyError(f"document {doc_id} not found")
        return self._document_from_row(row)

    def delete_document(self, doc_id: int) -> int:
        cursor = self._conn.execute(f"DELETE FROM {self._schema}.documents WHERE doc_id = %s", (doc_id,))
        self._conn.commit()
        return int(cursor.rowcount)

    def save_section(self, section: SectionRecord) -> SectionRecord:
        row = self._save_section(section)
        self._conn.commit()
        return row

    def save_sections(self, sections: list[SectionRecord]) -> list[SectionRecord]:
        saved = [self._save_section(section) for section in sections]
        self._conn.commit()
        return saved

    def get_section(self, section_id: int) -> SectionRecord | None:
        row = self._fetchone(f"SELECT * FROM {self._schema}.sections WHERE section_id = %s", (section_id,))
        return None if row is None else self._section_from_row(row)

    def list_sections(
        self,
        *,
        doc_id: int | None = None,
        source_id: int | None = None,
    ) -> list[SectionRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if doc_id is not None:
            clauses.append("doc_id = %s")
            params.append(doc_id)
        if source_id is not None:
            clauses.append("source_id = %s")
            params.append(source_id)
        where_sql = "" if not clauses else f" WHERE {' AND '.join(clauses)}"
        rows = self._fetchall(
            f"""
            SELECT *
            FROM {self._schema}.sections{where_sql}
            ORDER BY doc_id, order_index, section_id
            """,
            tuple(params),
        )
        return [self._section_from_row(row) for row in rows]

    def delete_sections_for_document(self, *, doc_id: int) -> int:
        cursor = self._conn.execute(f"DELETE FROM {self._schema}.sections WHERE doc_id = %s", (doc_id,))
        self._conn.commit()
        return int(cursor.rowcount)

    def save_asset(self, asset: AssetRecord) -> AssetRecord:
        row = self._save_asset(asset)
        self._conn.commit()
        return row

    def save_assets(self, assets: list[AssetRecord]) -> list[AssetRecord]:
        saved = [self._save_asset(asset) for asset in assets]
        self._conn.commit()
        return saved

    def get_asset(self, asset_id: int) -> AssetRecord | None:
        row = self._fetchone(f"SELECT * FROM {self._schema}.assets WHERE asset_id = %s", (asset_id,))
        return None if row is None else self._asset_from_row(row)

    def list_assets(
        self,
        *,
        doc_id: int | None = None,
        source_id: int | None = None,
        section_id: int | None = None,
    ) -> list[AssetRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if doc_id is not None:
            clauses.append("doc_id = %s")
            params.append(doc_id)
        if source_id is not None:
            clauses.append("source_id = %s")
            params.append(source_id)
        if section_id is not None:
            clauses.append("section_id = %s")
            params.append(section_id)
        where_sql = "" if not clauses else f" WHERE {' AND '.join(clauses)}"
        rows = self._fetchall(
            f"""
            SELECT *
            FROM {self._schema}.assets{where_sql}
            ORDER BY doc_id, page_no, asset_id
            """,
            tuple(params),
        )
        return [self._asset_from_row(row) for row in rows]

    def delete_assets_for_document(self, *, doc_id: int) -> int:
        cursor = self._conn.execute(f"DELETE FROM {self._schema}.assets WHERE doc_id = %s", (doc_id,))
        self._conn.commit()
        return int(cursor.rowcount)

    def save_layout_meta_cache(self, record: LayoutMetaCacheRecord) -> LayoutMetaCacheRecord:
        now = datetime.now(UTC)
        saved_cache = record.model_copy(
            update={
                "cache_id": record.cache_id if record.cache_id > 0 else self.next_id(),
                "updated_at": now,
            }
        )
        row = self._fetchone(
            f"""
            INSERT INTO {self._schema}.layout_meta_cache (
                cache_id,
                source_id,
                doc_id,
                content_hash,
                object_key,
                layout_json,
                layout_version,
                page_count,
                created_at,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_id, content_hash) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                object_key = EXCLUDED.object_key,
                layout_json = EXCLUDED.layout_json,
                layout_version = EXCLUDED.layout_version,
                page_count = EXCLUDED.page_count,
                updated_at = EXCLUDED.updated_at
            RETURNING *
            """,
            (
                saved_cache.cache_id,
                saved_cache.source_id,
                saved_cache.doc_id,
                saved_cache.content_hash,
                saved_cache.object_key,
                self._json_dumps(saved_cache.layout_json),
                saved_cache.layout_version,
                saved_cache.page_count,
                saved_cache.created_at,
                saved_cache.updated_at,
            ),
        )
        self._conn.commit()
        if row is None:
            raise RuntimeError("saving layout meta cache returned no row")
        return self._layout_cache_from_row(row)

    def get_layout_meta_cache(
        self,
        *,
        source_id: int | None = None,
        doc_id: int | None = None,
        content_hash: str | None = None,
    ) -> LayoutMetaCacheRecord | None:
        clauses: list[str] = []
        params: list[object] = []
        if source_id is not None:
            clauses.append("source_id = %s")
            params.append(source_id)
        if doc_id is not None:
            clauses.append("doc_id = %s")
            params.append(doc_id)
        if content_hash is not None:
            clauses.append("content_hash = %s")
            params.append(content_hash)
        if not clauses:
            raise ValueError("at least one layout cache filter is required")
        row = self._fetchone(
            f"""
            SELECT *
            FROM {self._schema}.layout_meta_cache
            WHERE {" AND ".join(clauses)}
            ORDER BY updated_at DESC, cache_id DESC
            LIMIT 1
            """,
            tuple(params),
        )
        return None if row is None else self._layout_cache_from_row(row)

    def list_layout_meta_cache(
        self,
        *,
        source_id: int | None = None,
        doc_id: int | None = None,
    ) -> list[LayoutMetaCacheRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if source_id is not None:
            clauses.append("source_id = %s")
            params.append(source_id)
        if doc_id is not None:
            clauses.append("doc_id = %s")
            params.append(doc_id)
        where_sql = "" if not clauses else f" WHERE {' AND '.join(clauses)}"
        rows = self._fetchall(
            f"""
            SELECT *
            FROM {self._schema}.layout_meta_cache{where_sql}
            ORDER BY updated_at DESC, cache_id DESC
            """,
            tuple(params),
        )
        return [self._layout_cache_from_row(row) for row in rows]

    def save_processing_state(self, record: ProcessingStateRecord) -> ProcessingStateRecord:
        now = datetime.now(UTC)
        saved_state = record.model_copy(update={"updated_at": now})
        row = self._fetchone(
            f"""
            INSERT INTO {self._schema}.processing_state (
                doc_id,
                source_id,
                stage,
                status,
                attempts,
                priority,
                worker_id,
                lease_expires_at,
                error_message,
                created_at,
                updated_at,
                metadata_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                stage = EXCLUDED.stage,
                status = EXCLUDED.status,
                attempts = EXCLUDED.attempts,
                priority = EXCLUDED.priority,
                worker_id = EXCLUDED.worker_id,
                lease_expires_at = EXCLUDED.lease_expires_at,
                error_message = EXCLUDED.error_message,
                updated_at = EXCLUDED.updated_at,
                metadata_json = EXCLUDED.metadata_json
            RETURNING *
            """,
            (
                saved_state.doc_id,
                saved_state.source_id,
                saved_state.stage,
                saved_state.status,
                saved_state.attempts,
                saved_state.priority,
                saved_state.worker_id,
                saved_state.lease_expires_at,
                saved_state.error_message,
                saved_state.created_at,
                saved_state.updated_at,
                self._json_dumps(saved_state.metadata_json),
            ),
        )
        self._conn.commit()
        if row is None:
            raise RuntimeError("saving processing state returned no row")
        return ProcessingStateRecord.model_validate(self._json_columns(dict(row), "metadata_json"))

    def get_processing_state(self, doc_id: int) -> ProcessingStateRecord | None:
        row = self._fetchone(f"SELECT * FROM {self._schema}.processing_state WHERE doc_id = %s", (doc_id,))
        if row is None:
            return None
        return ProcessingStateRecord.model_validate(self._json_columns(dict(row), "metadata_json"))

    def list_processing_states(
        self,
        *,
        source_id: int | None = None,
        status: str | None = None,
        stage: str | None = None,
    ) -> list[ProcessingStateRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if source_id is not None:
            clauses.append("source_id = %s")
            params.append(source_id)
        if status is not None:
            clauses.append("status = %s")
            params.append(status)
        if stage is not None:
            clauses.append("stage = %s")
            params.append(stage)
        where_sql = "" if not clauses else f" WHERE {' AND '.join(clauses)}"
        rows = self._fetchall(
            f"""
            SELECT *
            FROM {self._schema}.processing_state{where_sql}
            ORDER BY updated_at DESC, doc_id DESC
            """,
            tuple(params),
        )
        return [ProcessingStateRecord.model_validate(self._json_columns(dict(row), "metadata_json")) for row in rows]

    def delete_processing_state(self, doc_id: int) -> None:
        self._conn.execute(f"DELETE FROM {self._schema}.processing_state WHERE doc_id = %s", (doc_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _save_section(self, section: SectionRecord) -> SectionRecord:
        now = datetime.now(UTC)
        saved_section = section.model_copy(
            update={
                "section_id": section.section_id if section.section_id > 0 else self.next_id(),
                "updated_at": now,
            }
        )
        row = self._fetchone(
            f"""
            INSERT INTO {self._schema}.sections (
                section_id,
                doc_id,
                source_id,
                parent_section_id,
                toc_path,
                heading_level,
                order_index,
                anchor,
                page_start,
                page_end,
                content_storage_key,
                raw_locator,
                char_range_start,
                char_range_end,
                byte_range_start,
                byte_range_end,
                visible_text_key,
                section_kind,
                content_hash,
                has_table,
                has_figure,
                neighbor_asset_count,
                created_at,
                updated_at,
                metadata_json
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (doc_id, order_index) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                parent_section_id = EXCLUDED.parent_section_id,
                toc_path = EXCLUDED.toc_path,
                heading_level = EXCLUDED.heading_level,
                anchor = EXCLUDED.anchor,
                page_start = EXCLUDED.page_start,
                page_end = EXCLUDED.page_end,
                content_storage_key = EXCLUDED.content_storage_key,
                raw_locator = EXCLUDED.raw_locator,
                char_range_start = EXCLUDED.char_range_start,
                char_range_end = EXCLUDED.char_range_end,
                byte_range_start = EXCLUDED.byte_range_start,
                byte_range_end = EXCLUDED.byte_range_end,
                visible_text_key = EXCLUDED.visible_text_key,
                section_kind = EXCLUDED.section_kind,
                content_hash = EXCLUDED.content_hash,
                has_table = EXCLUDED.has_table,
                has_figure = EXCLUDED.has_figure,
                neighbor_asset_count = EXCLUDED.neighbor_asset_count,
                updated_at = EXCLUDED.updated_at,
                metadata_json = EXCLUDED.metadata_json
            RETURNING *
            """,
            (
                saved_section.section_id,
                saved_section.doc_id,
                saved_section.source_id,
                saved_section.parent_section_id,
                saved_section.toc_path,
                saved_section.heading_level,
                saved_section.order_index,
                saved_section.anchor,
                saved_section.page_start,
                saved_section.page_end,
                saved_section.content_storage_key,
                self._json_dumps(saved_section.raw_locator),
                saved_section.char_range_start,
                saved_section.char_range_end,
                saved_section.byte_range_start,
                saved_section.byte_range_end,
                saved_section.visible_text_key,
                saved_section.section_kind,
                saved_section.content_hash,
                saved_section.has_table,
                saved_section.has_figure,
                saved_section.neighbor_asset_count,
                saved_section.created_at,
                saved_section.updated_at,
                self._json_dumps(saved_section.metadata_json),
            ),
        )
        if row is None:
            raise RuntimeError("saving section returned no row")
        return self._section_from_row(row)

    def _save_asset(self, asset: AssetRecord) -> AssetRecord:
        now = datetime.now(UTC)
        saved_asset = asset.model_copy(
            update={
                "asset_id": asset.asset_id if asset.asset_id > 0 else self.next_id(),
                "updated_at": now,
            }
        )
        row = self._fetchone(
            f"""
            INSERT INTO {self._schema}.assets (
                asset_id,
                doc_id,
                source_id,
                section_id,
                asset_type,
                element_ref,
                page_no,
                bbox,
                caption,
                raw_locator,
                neighbor_section_id,
                content_hash,
                storage_key,
                created_at,
                updated_at,
                metadata_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id, content_hash) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                section_id = EXCLUDED.section_id,
                asset_type = EXCLUDED.asset_type,
                element_ref = EXCLUDED.element_ref,
                page_no = EXCLUDED.page_no,
                bbox = EXCLUDED.bbox,
                caption = EXCLUDED.caption,
                raw_locator = EXCLUDED.raw_locator,
                neighbor_section_id = EXCLUDED.neighbor_section_id,
                storage_key = EXCLUDED.storage_key,
                updated_at = EXCLUDED.updated_at,
                metadata_json = EXCLUDED.metadata_json
            RETURNING *
            """,
            (
                saved_asset.asset_id,
                saved_asset.doc_id,
                saved_asset.source_id,
                saved_asset.section_id,
                saved_asset.asset_type,
                saved_asset.element_ref,
                saved_asset.page_no,
                self._json_dumps(saved_asset.bbox),
                saved_asset.caption,
                self._json_dumps(saved_asset.raw_locator),
                saved_asset.neighbor_section_id,
                saved_asset.content_hash,
                saved_asset.storage_key,
                saved_asset.created_at,
                saved_asset.updated_at,
                self._json_dumps(saved_asset.metadata_json),
            ),
        )
        if row is None:
            raise RuntimeError("saving asset returned no row")
        return self._asset_from_row(row)

    def _ensure_schema(self) -> None:
        self._conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")
        self._create_sources_table()
        self._create_documents_table()
        self._create_sections_table()
        self._create_assets_table()
        self._create_layout_cache_table()
        self._create_processing_state_table()
        self._conn.commit()

    def _create_sources_table(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._schema}.sources (
                source_id BIGINT PRIMARY KEY,
                source_type VARCHAR(32) NOT NULL,
                location TEXT NOT NULL,
                original_file_name TEXT,
                bucket VARCHAR(128),
                object_key TEXT,
                content_hash VARCHAR(64) NOT NULL,
                file_size_bytes BIGINT,
                mime_type VARCHAR(128),
                owner_id VARCHAR(128),
                ingest_version INT NOT NULL DEFAULT 1,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                UNIQUE(location, content_hash, ingest_version)
            )
            """
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_sources_content_hash "
            f"ON {self._schema}.sources(content_hash)"
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_sources_location ON {self._schema}.sources(location)"
        )

    def _create_documents_table(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._schema}.documents (
                doc_id BIGINT PRIMARY KEY,
                source_id BIGINT NOT NULL REFERENCES {self._schema}.sources(source_id) ON DELETE CASCADE,
                title TEXT,
                language VARCHAR(16),
                authors JSONB NOT NULL DEFAULT '[]',
                file_hash VARCHAR(64) NOT NULL,
                version_group_id BIGINT NOT NULL,
                version_no INT NOT NULL DEFAULT 1,
                doc_status VARCHAR(32) NOT NULL,
                effective_date TIMESTAMPTZ,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                is_indexed BOOLEAN NOT NULL DEFAULT FALSE,
                index_ready BOOLEAN NOT NULL DEFAULT FALSE,
                index_priority VARCHAR(16) NOT NULL DEFAULT 'high',
                storage_tier VARCHAR(16) NOT NULL DEFAULT 'hot',
                reference_count INT NOT NULL DEFAULT 1,
                page_count INT,
                tenant_id VARCHAR(64),
                department_id VARCHAR(64),
                auth_tag VARCHAR(128),
                embedding_model_id VARCHAR(64) NOT NULL DEFAULT 'default',
                indexed_at TIMESTAMPTZ,
                last_index_error TEXT,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                UNIQUE(source_id, file_hash, version_no)
            )
            """
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_documents_version "
            f"ON {self._schema}.documents(version_group_id, version_no DESC)"
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_documents_active "
            f"ON {self._schema}.documents(is_active, storage_tier, updated_at DESC)"
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_documents_hash ON {self._schema}.documents(file_hash)"
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_documents_scope "
            f"ON {self._schema}.documents(tenant_id, department_id, auth_tag)"
        )

    def _create_sections_table(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._schema}.sections (
                section_id BIGINT PRIMARY KEY,
                doc_id BIGINT NOT NULL REFERENCES {self._schema}.documents(doc_id) ON DELETE CASCADE,
                source_id BIGINT NOT NULL REFERENCES {self._schema}.sources(source_id) ON DELETE CASCADE,
                parent_section_id BIGINT REFERENCES {self._schema}.sections(section_id) ON DELETE CASCADE,
                toc_path TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                heading_level INT,
                order_index INT NOT NULL,
                anchor TEXT,
                page_start INT,
                page_end INT,
                content_storage_key TEXT,
                raw_locator JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                char_range_start BIGINT,
                char_range_end BIGINT,
                byte_range_start BIGINT,
                byte_range_end BIGINT,
                visible_text_key TEXT,
                section_kind VARCHAR(32) NOT NULL,
                content_hash VARCHAR(64) NOT NULL,
                has_table BOOLEAN NOT NULL DEFAULT FALSE,
                has_figure BOOLEAN NOT NULL DEFAULT FALSE,
                neighbor_asset_count INT NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                UNIQUE(doc_id, order_index)
            )
            """
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_sections_doc_order "
            f"ON {self._schema}.sections(doc_id, order_index)"
        )
        self._conn.execute(f"ALTER TABLE {self._schema}.sections ADD COLUMN IF NOT EXISTS content_storage_key TEXT")
        self._conn.execute(f"ALTER TABLE {self._schema}.sections ADD COLUMN IF NOT EXISTS char_range_start BIGINT")
        self._conn.execute(f"ALTER TABLE {self._schema}.sections ADD COLUMN IF NOT EXISTS char_range_end BIGINT")
        self._conn.execute(f"ALTER TABLE {self._schema}.sections ADD COLUMN IF NOT EXISTS byte_range_start BIGINT")
        self._conn.execute(f"ALTER TABLE {self._schema}.sections ADD COLUMN IF NOT EXISTS byte_range_end BIGINT")
        self._conn.execute(f"ALTER TABLE {self._schema}.sections ADD COLUMN IF NOT EXISTS visible_text_key TEXT")
        self._conn.execute(
            f"ALTER TABLE {self._schema}.sections ADD COLUMN IF NOT EXISTS raw_locator "
            "JSONB NOT NULL DEFAULT '{}'::jsonb"
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_sections_source_doc "
            f"ON {self._schema}.sections(source_id, doc_id)"
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_sections_page_range "
            f"ON {self._schema}.sections(doc_id, page_start, page_end)"
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_sections_toc_path "
            f"ON {self._schema}.sections USING GIN(toc_path)"
        )

    def _create_assets_table(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._schema}.assets (
                asset_id BIGINT PRIMARY KEY,
                doc_id BIGINT NOT NULL REFERENCES {self._schema}.documents(doc_id) ON DELETE CASCADE,
                source_id BIGINT NOT NULL REFERENCES {self._schema}.sources(source_id) ON DELETE CASCADE,
                section_id BIGINT REFERENCES {self._schema}.sections(section_id) ON DELETE CASCADE,
                asset_type VARCHAR(32) NOT NULL,
                element_ref TEXT,
                page_no INT NOT NULL,
                bbox JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                caption TEXT,
                raw_locator JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                neighbor_section_id BIGINT,
                content_hash VARCHAR(64) NOT NULL,
                storage_key TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                UNIQUE(doc_id, content_hash)
            )
            """
        )
        self._conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_assets_doc ON {self._schema}.assets(doc_id)")
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_assets_source_doc "
            f"ON {self._schema}.assets(source_id, doc_id)"
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_assets_section ON {self._schema}.assets(section_id)"
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_assets_page ON {self._schema}.assets(doc_id, page_no)"
        )

    def _create_layout_cache_table(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._schema}.layout_meta_cache (
                cache_id BIGINT PRIMARY KEY,
                source_id BIGINT NOT NULL REFERENCES {self._schema}.sources(source_id) ON DELETE CASCADE,
                doc_id BIGINT REFERENCES {self._schema}.documents(doc_id) ON DELETE CASCADE,
                content_hash VARCHAR(64) NOT NULL,
                object_key TEXT,
                layout_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                layout_version VARCHAR(64) NOT NULL DEFAULT 'v1',
                page_count INT,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                UNIQUE(source_id, content_hash)
            )
            """
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_layout_cache_doc "
            f"ON {self._schema}.layout_meta_cache(doc_id)"
        )

    def _create_processing_state_table(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._schema}.processing_state (
                doc_id BIGINT PRIMARY KEY REFERENCES {self._schema}.documents(doc_id) ON DELETE CASCADE,
                source_id BIGINT NOT NULL REFERENCES {self._schema}.sources(source_id) ON DELETE CASCADE,
                stage VARCHAR(32) NOT NULL,
                status VARCHAR(32) NOT NULL,
                attempts INT NOT NULL DEFAULT 0,
                priority VARCHAR(16) NOT NULL DEFAULT 'normal',
                worker_id VARCHAR(128),
                lease_expires_at TIMESTAMPTZ,
                error_message TEXT,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb
            )
            """
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_processing_state_status "
            f"ON {self._schema}.processing_state(status, stage, priority)"
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._schema}_processing_state_source "
            f"ON {self._schema}.processing_state(source_id, updated_at DESC)"
        )

    def _source_from_row(self, row: dict[str, Any]) -> Source:
        data = self._with_access_policy(row)
        data["metadata_json"] = self._json_mapping(data.get("metadata_json"))
        return Source.model_validate(data)

    def _document_from_row(self, row: dict[str, Any]) -> Document:
        data = self._with_access_policy(row)
        data["authors"] = [str(item) for item in self._json_list(data.get("authors"))]
        data["metadata_json"] = self._json_mapping(data.get("metadata_json"))
        return Document.model_validate(data)

    def _section_from_row(self, row: dict[str, Any]) -> SectionRecord:
        data = self._json_columns(dict(row), "raw_locator", "metadata_json")
        return SectionRecord.model_validate(data)

    def _asset_from_row(self, row: dict[str, Any]) -> AssetRecord:
        data = self._json_columns(dict(row), "bbox", "raw_locator", "metadata_json")
        metadata = data.get("metadata_json", {})
        if isinstance(metadata, dict):
            for key in ("sheet_name", "row_count", "column_count", "sample_rows", "schema"):
                if key not in data and key in metadata:
                    data[key] = metadata[key]
        return AssetRecord.model_validate(data)

    def _layout_cache_from_row(self, row: dict[str, Any]) -> LayoutMetaCacheRecord:
        data = self._json_columns(dict(row), "layout_json")
        return LayoutMetaCacheRecord.model_validate(data)

    def _with_access_policy(self, row: dict[str, Any]) -> dict[str, Any]:
        data = dict(row)
        data["effective_access_policy"] = AccessPolicy()
        return data

    def _json_columns(self, row: dict[str, Any], *columns: str) -> dict[str, Any]:
        data = dict(row)
        for column in columns:
            data[column] = self._json_mapping(data.get(column))
        return data

    @staticmethod
    def _enum_value(value: object) -> object:
        return getattr(value, "value", value)

    @staticmethod
    def _json_dumps(value: object) -> str:
        if hasattr(value, "model_dump") and callable(value.model_dump):
            value = value.model_dump(mode="python")
        return json.dumps(value, ensure_ascii=True, sort_keys=True)

    @staticmethod
    def _json_mapping(value: object) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, str):
            loaded = json.loads(value)
            return cast(dict[str, Any], loaded if isinstance(loaded, dict) else {})
        return cast(dict[str, Any], value if isinstance(value, dict) else {})

    @staticmethod
    def _json_list(value: object) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, str):
            loaded = json.loads(value)
            return loaded if isinstance(loaded, list) else []
        return value if isinstance(value, list) else []

    def _connect(self) -> Any:
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self._dsn, row_factory=dict_row)

    def _fetchone(self, sql: str, params: tuple[object, ...] = ()) -> dict[str, Any] | None:
        row = self._conn.execute(sql, params).fetchone()
        return cast(dict[str, Any] | None, row)

    def _fetchall(self, sql: str, params: tuple[object, ...] = ()) -> list[dict[str, Any]]:
        rows = self._conn.execute(sql, params).fetchall()
        return cast(list[dict[str, Any]], rows)
