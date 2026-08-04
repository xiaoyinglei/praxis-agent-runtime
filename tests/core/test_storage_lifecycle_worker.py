from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from rag.schema.core import Document, DocumentStatus, ProcessingStateRecord, StorageTier
from rag.storage.storage_lifecycle_service import StorageLifecyclePolicy, StorageLifecycleService
from rag.storage.storage_lifecycle_worker import StorageLifecycleWorker


@dataclass
class _MetadataRepo:
    documents: dict[int, Document] = field(default_factory=dict)
    states: dict[int, ProcessingStateRecord] = field(default_factory=dict)

    def get_document(self, doc_id: int) -> Document | None:
        return self.documents.get(doc_id)

    def save_document(self, document: Document) -> Document:
        self.documents[document.doc_id] = document
        return document

    def list_documents(
        self,
        *,
        source_id: int | None = None,
        active_only: bool = False,
        version_group_id: int | None = None,
        storage_tier: StorageTier | None = None,
    ) -> list[Document]:
        del source_id, version_group_id
        documents = list(self.documents.values())
        if active_only:
            documents = [document for document in documents if document.is_active]
        if storage_tier is not None:
            documents = [document for document in documents if document.storage_tier is storage_tier]
        return documents

    def set_document_storage_tier(self, doc_id: int, *, storage_tier: StorageTier) -> Document:
        document = self.documents[doc_id]
        updated = document.model_copy(update={"storage_tier": storage_tier})
        self.documents[doc_id] = updated
        return updated

    def save_processing_state(self, record: ProcessingStateRecord) -> ProcessingStateRecord:
        self.states[record.doc_id] = record
        return record

    def get_processing_state(self, doc_id: int) -> ProcessingStateRecord | None:
        return self.states.get(doc_id)

    def list_processing_states(
        self, *, source_id: int | None = None, status: str | None = None, stage: str | None = None
    ):
        states = list(self.states.values())
        if source_id is not None:
            states = [state for state in states if state.source_id == source_id]
        if status is not None:
            states = [state for state in states if state.status == status]
        if stage is not None:
            states = [state for state in states if state.stage == stage]
        return sorted(states, key=lambda state: (state.updated_at, state.doc_id))


@dataclass
class _DataContractService:
    synced_doc_ids: list[int] = field(default_factory=list)

    def sync_document_summaries(self, doc_id: int, *, embedding_space: str | None = None) -> int:
        del embedding_space
        self.synced_doc_ids.append(doc_id)
        return 1


def _document(*, doc_id: int, storage_tier: StorageTier, is_active: bool, doc_status: DocumentStatus) -> Document:
    now = datetime(2026, 4, 20, tzinfo=UTC)
    return Document(
        doc_id=doc_id,
        source_id=doc_id + 100,
        title=f"Doc {doc_id}",
        file_hash=f"hash-{doc_id}",
        storage_tier=storage_tier,
        is_active=is_active,
        index_ready=True,
        is_indexed=True,
        doc_status=doc_status,
        updated_at=now,
        created_at=now,
    )


def test_storage_lifecycle_service_enqueues_documents_that_need_tier_migration() -> None:
    repo = _MetadataRepo(
        documents={
            1: _document(doc_id=1, storage_tier=StorageTier.HOT, is_active=False, doc_status=DocumentStatus.RETIRED),
            2: _document(doc_id=2, storage_tier=StorageTier.HOT, is_active=True, doc_status=DocumentStatus.PUBLISHED),
        }
    )
    service = StorageLifecycleService(
        metadata_repo=repo,
        data_contract_service=_DataContractService(),  # type: ignore[arg-type]
        policy=StorageLifecyclePolicy(),
    )

    enqueued = service.enqueue_due_documents()

    assert [state.doc_id for state in enqueued] == [1]
    assert repo.get_processing_state(1) is not None
    assert repo.get_processing_state(1).stage == "storage_lifecycle"  # type: ignore[union-attr]


def test_storage_lifecycle_worker_migrates_document_to_cold_and_reindexes() -> None:
    repo = _MetadataRepo(
        documents={
            1: _document(doc_id=1, storage_tier=StorageTier.HOT, is_active=False, doc_status=DocumentStatus.RETIRED),
        }
    )
    data_contract = _DataContractService()
    service = StorageLifecycleService(
        metadata_repo=repo,
        data_contract_service=data_contract,  # type: ignore[arg-type]
        policy=StorageLifecyclePolicy(),
    )
    worker = StorageLifecycleWorker(service=service, worker_id="storage-worker")
    service.enqueue_due_documents()

    completed = worker.run_once(lease_seconds=30)

    assert completed is not None
    assert completed.status == "completed"
    assert repo.get_document(1).storage_tier is StorageTier.COLD  # type: ignore[union-attr]
    assert data_contract.synced_doc_ids == [1]
