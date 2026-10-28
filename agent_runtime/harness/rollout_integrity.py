"""Snapshot audits and staged reconstruction of rebuildable rollout projections."""

from __future__ import annotations

import hashlib
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import closing
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_runtime.harness import rollout as storage
from agent_runtime.harness.records import ArtifactSnapshot, VerificationReport
from agent_runtime.harness.reducer import ProjectionState, apply_record

if TYPE_CHECKING:
    from agent_runtime.harness.rollout import RolloutStore

_ROW_FORMATS = (
    ("threads", "thread_id", storage._thread_row_payload),
    ("turns", "turn_id", storage._turn_row_payload),
    ("items", "item_id", storage._item_row_payload),
    ("model_operations", "operation_id", storage._model_operation_row_payload),
    ("model_attempts", "attempt_id", storage._model_attempt_row_payload),
    ("budget_reservations", "reservation_id", storage._budget_reservation_row_payload),
    ("child_budget_allocations", "allocation_id", storage._child_budget_allocation_row_payload),
    ("tool_operations", "operation_id", storage._tool_operation_row_payload),
    ("interactions", "request_id", storage._interaction_row_payload),
    ("approvals", "request_id", storage._approval_row_payload),
    ("artifacts", "artifact_id", storage._artifact_row_payload),
)
_TABLES = tuple(name for name, _, _ in _ROW_FORMATS) + ("projection_meta",)
_INVALID_DATA = (ValueError, TypeError, KeyError, IndexError, AttributeError, RuntimeError, OverflowError)


def verify_store(store: RolloutStore, *, thread_ids: frozenset[str] | None = None) -> VerificationReport:
    """Replay and compare one thread at a time inside one coherent read snapshot."""

    def verify() -> VerificationReport:
        errors: list[str] = []
        with store.read_snapshot():
            try:
                for thread_id in _threads(store._connection, thread_ids, include_projections=True):
                    state, digest, position, replay_errors = _replay_thread(store, thread_id)
                    errors.extend(replay_errors)
                    errors.extend(_compare_projections(store._connection, thread_id, state))
                    errors.extend(_artifact_errors(store, state))
                    errors.extend(
                        _attachment_errors(
                            state.items.values(),
                            store._connection,
                            thread_ids,
                            state.artifacts,
                        )
                    )
                    if not _metadata_matches(store._connection, thread_id, state, digest, position):
                        errors.append("projection metadata mismatch")
                if (
                    thread_ids is None
                    and store._connection.execute(
                        "SELECT 1 FROM model_attempts AS a LEFT JOIN model_operations AS o "
                        "ON o.operation_id = a.operation_id WHERE o.operation_id IS NULL LIMIT 1"
                    ).fetchone()
                    is not None
                ):
                    errors.append("model_attempts projection mismatch")
            except (sqlite3.Error, *_INVALID_DATA) as error:
                errors.append(f"store integrity data is invalid: {error}")
        return VerificationReport(valid=not errors, errors=tuple(dict.fromkeys(errors)))

    return store.run_sync(verify)


def _threads(
    connection: sqlite3.Connection,
    selected: frozenset[str] | None,
    *,
    include_projections: bool,
) -> Iterator[str]:
    if selected is not None:
        yield from sorted(selected)
        return
    queries = ["SELECT DISTINCT thread_id FROM rollout_records"]
    if include_projections:
        queries.extend(f"SELECT thread_id FROM {table}" for table in _TABLES if table != "model_attempts")
    for row in connection.execute(" UNION ".join(queries) + " ORDER BY thread_id"):
        yield row[0]


def _replay_thread(
    store: RolloutStore, thread_id: str, *, created: Callable[[str, str, int], None] | None = None,
) -> tuple[ProjectionState, str | None, dict[str, int] | None, list[str]]:
    state = ProjectionState()
    errors: list[str] = []
    digest: str | None = None
    position: dict[str, int] | None = None
    previous_sequence = 0
    previous_record_id = 0
    for row in store._connection.execute(
        "SELECT * FROM rollout_records WHERE thread_id = ? ORDER BY record_id",
        (thread_id,),
    ):
        record_id = row["record_id"]
        try:
            sequence = row["thread_sequence"]
            if type(record_id) is not int or record_id <= previous_record_id:
                raise ValueError("record IDs must increase")
            if type(sequence) is not int or sequence != previous_sequence + 1:
                raise ValueError("thread sequence is not contiguous")
            previous_record_id, previous_sequence = record_id, sequence
            position = {"record_id": record_id, "thread_sequence": sequence}
            for key in ("record_uuid", "thread_id", "record_type", "producer"):
                if not isinstance(row[key], str) or not row[key]:
                    raise ValueError(f"invalid {key}")
            if row["turn_id"] is not None and (not isinstance(row["turn_id"], str) or not row["turn_id"]):
                raise ValueError("invalid turn_id")
            if type(row["payload_schema_version"]) is not int:
                raise ValueError("invalid payload schema version")
            if hashlib.sha256(row["payload_json"].encode()).hexdigest() != row["payload_hash"]:
                raise ValueError("payload hash mismatch")
            record = storage._record_snapshot(row)
            digest = storage._record_prefix_hash(digest, record)
            counts = {name: len(getattr(state, name)) for name, _, _ in _ROW_FORMATS} if created else {}
            apply_record(state, record, reducer_version=storage._REDUCER_VERSION)
            if created is not None:
                for name, _, _ in _ROW_FORMATS:
                    entities = getattr(state, name)
                    added = len(entities) - counts[name]
                    # Dict insertion order is canonical first appearance. Persist
                    # only newly created identities, without scanning old state.
                    for identity in reversed(tuple(islice(reversed(entities), added))):
                        created(name, identity, record_id)
        except _INVALID_DATA as error:
            errors.append(f"record {record_id} is corrupt: {error}")
    return state, digest, position, errors


def _compare_projections(connection: sqlite3.Connection, thread_id: str, state: ProjectionState) -> list[str]:
    errors = []
    for table, key, row_payload in _ROW_FORMATS:
        expected = getattr(state, table)
        count, mismatch = 0, False
        if table == "model_attempts":
            rows = connection.execute(
                "SELECT a.* FROM model_attempts AS a JOIN model_operations AS o "
                "ON o.operation_id = a.operation_id WHERE o.thread_id = ?",
                (thread_id,),
            )
        else:
            rows = connection.execute(f"SELECT * FROM {table} WHERE thread_id = ?", (thread_id,))
        for row in rows:
            count += 1
            try:
                if storage._canonical_json(expected.get(row[key])) != storage._canonical_json(row_payload(row)):
                    mismatch = True
            except _INVALID_DATA:
                mismatch = True
        if mismatch or count != len(expected):
            errors.append(f"{table} projection mismatch")
    return errors


def _metadata_matches(
    connection: sqlite3.Connection, thread_id: str, state: ProjectionState, digest: str | None,
    position: Mapping[str, int] | None,
) -> bool:
    row = connection.execute("SELECT * FROM projection_meta WHERE thread_id = ?", (thread_id,)).fetchone()
    if thread_id not in state.threads:
        return row is None and position is None
    if row is None or position is None:
        return False
    try:
        version = storage._metadata_hash_version(row)
        expected = {
            "applied_record_id": position["record_id"],
            "applied_thread_sequence": position["thread_sequence"],
            "reducer_version": storage._REDUCER_VERSION,
            "canonical_hash": storage._projection_hash(state, thread_id) if version == 1 else digest,
            "hash_version": 1 if version == 1 else 2,
        }
        actual = {key: row[key] for key in expected if key != "hash_version"}
        actual["hash_version"] = version
        return storage._canonical_json(actual) == storage._canonical_json(expected)
    except _INVALID_DATA:
        return False


def _artifact_errors(store: RolloutStore, state: ProjectionState) -> list[str]:
    errors = []
    for artifact_id, payload in state.artifacts.items():
        try:
            digest, size = payload["blob_sha256"], payload["size_bytes"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or type(size) is not int
                or size < 0
            ):
                raise ValueError("invalid blob identity or size")
            artifact = ArtifactSnapshot(
                artifact_id=artifact_id,
                thread_id=payload["thread_id"],
                turn_id=payload["turn_id"],
                blob_sha256=digest,
                size_bytes=size,
                media_type=payload["media_type"],
                name=payload["name"],
                applied_thread_sequence=payload["applied_thread_sequence"],
            )
            error = storage._artifact_blob_error(artifact, store._artifact_root / "blobs" / digest)
            if error is not None:
                errors.append(f"artifact {artifact_id}: {error}")
        except (OSError, *_INVALID_DATA) as error:
            errors.append(f"artifact {artifact_id}: {error}")
    return errors


def _attachment_errors(
    items: Iterable[Mapping[str, Any]], connection: sqlite3.Connection, selected: frozenset[str] | None,
    local_artifacts: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    errors = []
    for item in items:
        try:
            attachments = item["payload"].get("attachments")
            if attachments is None:
                continue
            if not isinstance(attachments, (list, tuple)):
                raise ValueError("attachments must be a list")
            for attachment in attachments:
                if not isinstance(attachment, Mapping) or not isinstance(attachment.get("artifact_id"), str):
                    raise ValueError("artifact attachment is malformed")
                artifact_id = attachment["artifact_id"]
                durable = local_artifacts.get(artifact_id)
                if durable is None:
                    sql = "SELECT * FROM artifacts WHERE artifact_id = ?"
                    parameters: tuple[Any, ...] = (artifact_id,)
                    if selected is not None:
                        sql += " AND thread_id IN (" + ",".join("?" for _ in selected) + ")"
                        parameters += tuple(sorted(selected))
                    row = connection.execute(sql, parameters).fetchone()
                    durable = storage._artifact_row_payload(row) if row is not None else None
                if durable is None:
                    raise ValueError("references unknown artifact")
                if attachment.get("media_type") != durable["media_type"] or attachment.get("name") != durable["name"]:
                    raise ValueError("artifact attachment metadata mismatch")
        except _INVALID_DATA as error:
            errors.append(f"artifact attachment in Item {item.get('item_id')}: {error}")
    return errors


def rebuild_store(store: RolloutStore) -> None:
    """Stage a streamed replay without a write lock; publish only if its source is unchanged."""

    def rebuild() -> None:
        store.assert_writable()
        source = store._connection
        if source.in_transaction:
            raise RuntimeError("projection rebuild requires its own transaction boundary")
        baseline = source.execute("PRAGMA main.data_version").fetchone()[0]
        with tempfile.TemporaryDirectory(prefix="praxis-rollout-rebuild-") as directory:
            path = Path(directory) / "projections.sqlite3"
            with closing(sqlite3.connect(path, isolation_level=None)) as staging:
                staging.row_factory = sqlite3.Row
                with store.read_snapshot():
                    # BEGIN is deferred. Force the first read before checking that no
                    # writer slipped between the pre-transaction baseline and snapshot.
                    source.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
                    if source.execute("PRAGMA main.data_version").fetchone()[0] != baseline:
                        raise RuntimeError("store changed while preparing projection rebuild")
                    for table in _TABLES:
                        row = source.execute(
                            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                            (table,),
                        ).fetchone()
                        if row is None:
                            raise RuntimeError(f"projection table is missing: {table}")
                        staging.execute(row[0])
                    staging.execute("BEGIN")
                    staging.execute(
                        'CREATE TABLE rebuild_order (table_name TEXT, entity_id TEXT, record_id INTEGER, '
                        'PRIMARY KEY(table_name, entity_id)) WITHOUT ROWID'
                    )

                    def created(table: str, identity: str, record_id: int) -> None:
                        staging.execute('INSERT INTO rebuild_order VALUES (?, ?, ?)', (table, identity, record_id))

                    for thread_id in _threads(source, None, include_projections=False):
                        state, digest, position, errors = _replay_thread(store, thread_id, created=created)
                        errors.extend(_artifact_errors(store, state))
                        if errors or position is None or digest is None or thread_id not in state.threads:
                            raise RuntimeError(f"cannot rebuild from corrupt canonical records: {errors}")
                        try:
                            store._persist_projection_state(
                                state,
                                chain_hashes={thread_id: digest},
                                connection=staging,
                                positions={thread_id: position},
                            )
                            errors = _compare_projections(staging, thread_id, state)
                            if errors:
                                raise ValueError(f"canonical projection types are invalid: {errors}")
                        except (sqlite3.Error, *_INVALID_DATA) as error:
                            raise RuntimeError(f"cannot rebuild corrupt canonical records: {error}") from error
                    # All dependency threads now exist in staging; attachment checks do
                    # not trust the old projection rows that this operation will replace.
                    items = (
                        storage._item_row_payload(row) for row in staging.execute("SELECT * FROM items ORDER BY rowid")
                    )
                    errors = _attachment_errors(items, staging, None, {})
                    if errors:
                        raise RuntimeError(f"cannot rebuild corrupt canonical artifact references: {errors}")
                    staging.execute("COMMIT")
            source.execute("ATTACH DATABASE ? AS rollout_rebuild_stage", (str(path),))
            try:
                with store.transaction():
                    if source.execute("PRAGMA main.data_version").fetchone()[0] != baseline:
                        raise RuntimeError("store changed during projection rebuild; staged projections discarded")
                    for table in reversed(_TABLES):
                        source.execute(f"DELETE FROM main.{table}")
                    for table in _TABLES:
                        key = 'thread_id' if table == 'projection_meta' else storage._PROJECTIONS[table][0]
                        ordered_table = 'threads' if table == 'projection_meta' else table
                        source.execute(
                            f'INSERT INTO main.{table} SELECT p.* FROM rollout_rebuild_stage.{table} AS p '
                            'JOIN rollout_rebuild_stage.rebuild_order AS o '
                            f'ON o.table_name = ? AND o.entity_id = p.{key} ORDER BY o.record_id, p.rowid',
                            (ordered_table,),
                        )
            finally:
                source.execute("DETACH DATABASE rollout_rebuild_stage")

    store.run_sync(rebuild)
