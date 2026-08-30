from __future__ import annotations

from dataclasses import dataclass

from rag.schema.core import Document, DocumentStatus, ProcessingStateRecord, StorageTier
from rag.storage.index_sync_service import IndexSyncService


@dataclass(frozen=True, slots=True)
class StorageLifecyclePolicy:
    def desired_tier(self, document: Document) -> StorageTier:
        if not document.is_active:
            return StorageTier.COLD
        if str(document.doc_status).lower() == DocumentStatus.RETIRED.value:
            return StorageTier.COLD
        if not document.index_ready:
            return StorageTier.COLD
        return StorageTier.HOT


@dataclass(slots=True)
class StorageLifecycleService:
    metadata_repo: object
    data_contract_service: object
    policy: StorageLifecyclePolicy = StorageLifecyclePolicy()
    lifecycle_queue: IndexSyncService | None = None

    def __post_init__(self) -> None:
        if self.lifecycle_queue is None:
            self.lifecycle_queue = (
                IndexSyncService(self.metadata_repo, stage="storage_lifecycle")
                if callable(getattr(self.metadata_repo, "save_processing_state", None))
                and callable(getattr(self.metadata_repo, "list_processing_states", None))
                else None
            )

    def enqueue_due_documents(self, *, limit: int = 8) -> list[ProcessingStateRecord]:
        list_documents = getattr(self.metadata_repo, "list_documents", None)
        get_processing_state = getattr(self.metadata_repo, "get_processing_state", None)
        if not callable(list_documents) or self.lifecycle_queue is None:
            return []
        queued: list[ProcessingStateRecord] = []
        for document in list_documents():
            target_tier = self.policy.desired_tier(document)
            if document.storage_tier is target_tier:
                continue
            existing = get_processing_state(document.doc_id) if callable(get_processing_state) else None
            if (
                existing is not None
                and existing.stage != "storage_lifecycle"
                and existing.status in {"pending", "processing"}
            ):
                continue
            state = self.lifecycle_queue.enqueue(
                doc_id=document.doc_id,
                source_id=document.source_id,
                operation="migrate_storage_tier",
                priority=document.index_priority,
                metadata_json={"target_storage_tier": target_tier.value},
            )
            if state is not None:
                queued.append(state)
            if len(queued) >= max(limit, 0):
                break
        return queued

    def process_state(self, state: ProcessingStateRecord) -> int:
        get_document = getattr(self.metadata_repo, "get_document", None)
        set_document_storage_tier = getattr(self.metadata_repo, "set_document_storage_tier", None)
        sync_document_summaries = getattr(self.data_contract_service, "sync_document_summaries", None)
        if (
            not callable(get_document)
            or not callable(set_document_storage_tier)
            or not callable(sync_document_summaries)
        ):
            raise RuntimeError("storage lifecycle requires document lookup, storage tier update, and summary sync")
        document = get_document(state.doc_id)
        if document is None:
            return 0
        target_tier = self.policy.desired_tier(document)
        if document.storage_tier is not target_tier:
            set_document_storage_tier(document.doc_id, storage_tier=target_tier)
        return int(sync_document_summaries(document.doc_id))


__all__ = ["StorageLifecyclePolicy", "StorageLifecycleService"]
