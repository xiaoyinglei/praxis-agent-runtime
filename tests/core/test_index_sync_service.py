from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest

from rag.schema.core import ProcessingStateRecord
from rag.storage.index_sync_service import IndexSyncService, StaleProcessingStateError


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


def test_index_sync_service_claims_high_priority_pending_task() -> None:
    now = datetime(2026, 4, 19, tzinfo=UTC)
    repo = _MetadataRepo(
        states={
            1: ProcessingStateRecord(
                doc_id=1, source_id=7, stage="index_sync", status="pending", priority="normal", updated_at=now
            ),
            2: ProcessingStateRecord(
                doc_id=2, source_id=7, stage="index_sync", status="pending", priority="high", updated_at=now
            ),
        }
    )
    service = IndexSyncService(repo)

    claimed = service.claim_next(worker_id="worker-a", lease_seconds=30, now=now)

    assert claimed is not None
    assert claimed.doc_id == 2
    assert claimed.status == "processing"
    assert claimed.worker_id == "worker-a"
    assert claimed.attempts == 1


def test_index_sync_service_requeues_expired_lease_and_marks_completion() -> None:
    now = datetime(2026, 4, 19, tzinfo=UTC)
    repo = _MetadataRepo(
        states={
            1: ProcessingStateRecord(
                doc_id=1,
                source_id=7,
                stage="index_sync",
                status="processing",
                attempts=1,
                priority="normal",
                worker_id="worker-a",
                lease_expires_at=now - timedelta(seconds=1),
                updated_at=now - timedelta(seconds=1),
            )
        }
    )
    service = IndexSyncService(repo)

    completed = service.process_next(
        worker_id="worker-b",
        lease_seconds=30,
        now=now,
        sync_handler=lambda state: None,
    )

    assert completed is not None
    assert completed.doc_id == 1
    assert completed.status == "completed"
    assert completed.worker_id is None


def test_index_sync_service_marks_retryable_failures_back_to_pending() -> None:
    now = datetime(2026, 4, 19, tzinfo=UTC)
    repo = _MetadataRepo()
    service = IndexSyncService(repo)
    service.enqueue(doc_id=1, source_id=7, operation="upsert_summary", priority="high")

    with pytest.raises(RuntimeError, match="boom"):
        service.process_next(
            worker_id="worker-a",
            lease_seconds=30,
            now=now,
            sync_handler=lambda state: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    state = repo.get_processing_state(1)
    assert state is not None
    assert state.status == "pending"
    assert state.error_message == "boom"


def test_index_sync_service_applies_retry_backoff_before_reclaiming_failed_pending_task() -> None:
    now = datetime(2026, 4, 20, tzinfo=UTC)
    repo = _MetadataRepo(
        states={
            1: ProcessingStateRecord(
                doc_id=1,
                source_id=7,
                stage="index_sync",
                status="pending",
                attempts=2,
                error_message="boom",
                updated_at=now,
            )
        }
    )
    service = IndexSyncService(repo)

    claimed = service.claim_next(worker_id="worker-a", now=now)

    assert claimed is None


def test_index_sync_service_keeps_newer_pending_state_when_claim_is_stale() -> None:
    now = datetime(2026, 4, 20, tzinfo=UTC)
    repo = _MetadataRepo()
    service = IndexSyncService(repo)
    created = service.enqueue(
        doc_id=1,
        source_id=7,
        operation="upsert_summary",
        metadata_json={"commit_anchor": "anchor-a"},
    )
    assert created is not None

    result = service.process_next(
        worker_id="worker-a",
        lease_seconds=30,
        now=now,
        sync_handler=lambda state: (
            repo.save_processing_state(
                state.model_copy(update={"status": "pending", "metadata_json": {"commit_anchor": "anchor-b"}})
            ),
            (_ for _ in ()).throw(StaleProcessingStateError(state.doc_id)),
        )[1],
    )

    assert result is not None
    assert result.status == "pending"
    assert repo.get_processing_state(1) is not None
    assert repo.get_processing_state(1).metadata_json["commit_anchor"] == "anchor-b"  # type: ignore[union-attr]


def test_index_sync_service_reports_monitor_snapshot() -> None:
    now = datetime(2026, 4, 20, tzinfo=UTC)
    repo = _MetadataRepo(
        states={
            1: ProcessingStateRecord(
                doc_id=1, source_id=7, stage="index_sync", status="pending", updated_at=now - timedelta(seconds=400)
            ),
            2: ProcessingStateRecord(
                doc_id=2, source_id=7, stage="index_sync", status="failed", updated_at=now - timedelta(seconds=10)
            ),
            3: ProcessingStateRecord(doc_id=3, source_id=7, stage="index_sync", status="completed", updated_at=now),
        }
    )
    service = IndexSyncService(repo)

    snapshot = service.monitor_snapshot(now=now)

    assert snapshot.pending == 1
    assert snapshot.failed == 1
    assert snapshot.completed == 1
    assert snapshot.alert_level == "critical"
    assert snapshot.oldest_pending_age_seconds == 400.0
