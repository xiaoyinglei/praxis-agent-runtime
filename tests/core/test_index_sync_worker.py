from __future__ import annotations

from dataclasses import dataclass, field

from rag.schema.core import ProcessingStateRecord
from rag.storage.index_sync_service import IndexSyncService
from rag.storage.index_sync_worker import IndexSyncWorker


@dataclass
class _MetadataRepo:
    states: dict[int, ProcessingStateRecord] = field(default_factory=dict)

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
    processed: list[int] = field(default_factory=list)

    def sync_processing_state(self, state: ProcessingStateRecord) -> int:
        self.processed.append(state.doc_id)
        return 1


def test_index_sync_worker_processes_pending_state_and_marks_completion() -> None:
    repo = _MetadataRepo()
    service = IndexSyncService(repo)
    data_contract_service = _DataContractService()
    worker = IndexSyncWorker(
        index_sync_service=service,
        data_contract_service=data_contract_service,  # type: ignore[arg-type]
        worker_id="worker-a",
    )
    service.enqueue(doc_id=20, source_id=10, operation="upsert_summary", priority="high")

    completed = worker.run_once(lease_seconds=30)

    assert completed is not None
    assert completed.status == "completed"
    assert data_contract_service.processed == [20]
    assert repo.get_processing_state(20) is not None
    assert repo.get_processing_state(20).status == "completed"  # type: ignore[union-attr]


def test_index_sync_worker_run_until_idle_processes_multiple_tasks() -> None:
    repo = _MetadataRepo()
    service = IndexSyncService(repo)
    data_contract_service = _DataContractService()
    worker = IndexSyncWorker(
        index_sync_service=service,
        data_contract_service=data_contract_service,  # type: ignore[arg-type]
        worker_id="worker-a",
    )
    service.enqueue(doc_id=20, source_id=10, operation="upsert_summary", priority="high")
    service.enqueue(doc_id=21, source_id=10, operation="upsert_summary", priority="normal")

    processed = worker.run_until_idle(max_tasks=4, lease_seconds=30)

    assert [state.doc_id for state in processed] == [20, 21]
    assert data_contract_service.processed == [20, 21]
