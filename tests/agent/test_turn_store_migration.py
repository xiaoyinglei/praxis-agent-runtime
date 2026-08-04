from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from uuid import UUID

import pytest

from agent_runtime.core.messages import (
    ModelMessage,
    canonical_json_text,
    model_message_payload,
)
from agent_runtime.turns import RuntimeBinding, TurnStore


def _create_session_schema(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE agent_sessions (
            session_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            runtime_json TEXT NOT NULL,
            history_revision INTEGER NOT NULL DEFAULT 0,
            active_turn_id TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            archived_at REAL
        );
        CREATE TABLE agent_turns (
            turn_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            status TEXT NOT NULL,
            user_message TEXT NOT NULL,
            runtime_json TEXT NOT NULL,
            checkpoint_id TEXT NOT NULL,
            lease_owner TEXT,
            lease_expires_at REAL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(session_id, ordinal),
            FOREIGN KEY(session_id) REFERENCES agent_sessions(session_id)
        );
        CREATE TABLE agent_session_events (
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            event_index INTEGER NOT NULL,
            session_sequence INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY(turn_id, event_index),
            UNIQUE(session_id, session_sequence),
            FOREIGN KEY(session_id) REFERENCES agent_sessions(session_id),
            FOREIGN KEY(turn_id) REFERENCES agent_turns(turn_id)
        );
        """
    )
    return connection


def _insert_session(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    kind: str,
    runtime_json: str,
) -> None:
    connection.execute(
        """
        INSERT INTO agent_sessions (
            session_id, kind, runtime_json, history_revision,
            active_turn_id, created_at, updated_at, archived_at
        ) VALUES (?, ?, ?, 0, NULL, 1, 1, NULL)
        """,
        (session_id, kind, runtime_json),
    )


def _insert_turn(
    connection: sqlite3.Connection,
    *,
    turn_id: str,
    session_id: str,
    ordinal: int,
    message: str,
    runtime_json: str,
) -> None:
    connection.execute(
        """
        INSERT INTO agent_turns (
            turn_id, session_id, ordinal, status, user_message,
            runtime_json, checkpoint_id, lease_owner, lease_expires_at,
            created_at, updated_at
        ) VALUES (?, ?, ?, 'completed', ?, ?, ?, NULL, NULL, ?, ?)
        """,
        (turn_id, session_id, ordinal, message, runtime_json, turn_id, ordinal, ordinal),
    )
    connection.execute(
        """
        INSERT INTO agent_session_events (
            session_id, turn_id, event_index, session_sequence, payload_json
        ) VALUES (?, ?, 0, ?, ?)
        """,
        (
            session_id,
            turn_id,
            ordinal,
            canonical_json_text(
                model_message_payload(ModelMessage(role="user", content=message))
            ),
        ),
    )


def test_session_rows_migrate_to_turn_lineage_and_drop_session_tables(
    tmp_path: Path,
) -> None:
    database = tmp_path / "agent.sqlite"
    connection = _create_session_schema(database)
    runtime = RuntimeBinding(
        model_alias="model-a",
        workspace_path=str(tmp_path),
    ).model_dump_json()
    conversation_id = str(UUID(int=1))
    one_shot_id = str(UUID(int=2))
    first_id = str(UUID(int=11))
    second_id = str(UUID(int=12))
    single_id = str(UUID(int=13))
    _insert_session(
        connection,
        session_id=conversation_id,
        kind="conversation",
        runtime_json=runtime,
    )
    _insert_session(
        connection,
        session_id=one_shot_id,
        kind="one_shot",
        runtime_json=runtime,
    )
    _insert_turn(
        connection,
        turn_id=first_id,
        session_id=conversation_id,
        ordinal=1,
        message="first",
        runtime_json=runtime,
    )
    _insert_turn(
        connection,
        turn_id=second_id,
        session_id=conversation_id,
        ordinal=2,
        message="second",
        runtime_json=runtime,
    )
    _insert_turn(
        connection,
        turn_id=single_id,
        session_id=one_shot_id,
        ordinal=1,
        message="single",
        runtime_json=runtime,
    )
    connection.commit()
    connection.close()

    store = TurnStore(database)
    assert store.get_turn(first_id).previous_turn_id is None
    assert store.get_turn(second_id).previous_turn_id == first_id
    assert store.get_turn(single_id).previous_turn_id is None
    assert store.history_before_turn(second_id) == (
        ModelMessage(role="user", content="first"),
    )
    store.close()

    connection = sqlite3.connect(database)
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    connection.close()
    assert "agent_turns" in tables
    assert "agent_turn_messages" in tables
    assert "agent_sessions" not in tables
    assert "agent_session_events" not in tables


def test_migration_rejects_invalid_multi_turn_one_shot_without_data_loss(
    tmp_path: Path,
) -> None:
    database = tmp_path / "agent.sqlite"
    connection = _create_session_schema(database)
    runtime = RuntimeBinding(workspace_path=str(tmp_path)).model_dump_json()
    session_id = str(UUID(int=20))
    _insert_session(
        connection,
        session_id=session_id,
        kind="one_shot",
        runtime_json=runtime,
    )
    for ordinal in (1, 2):
        _insert_turn(
            connection,
            turn_id=str(UUID(int=20 + ordinal)),
            session_id=session_id,
            ordinal=ordinal,
            message=f"message-{ordinal}",
            runtime_json=runtime,
        )
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="expected one Turn, found 2"):
        TurnStore(database)

    connection = sqlite3.connect(database)
    assert connection.execute("SELECT COUNT(*) FROM agent_sessions").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM agent_turns").fetchone()[0] == 2
    connection.close()


def test_legacy_runtime_binding_is_canonicalized_during_migration(
    tmp_path: Path,
) -> None:
    database = tmp_path / "agent.sqlite"
    connection = _create_session_schema(database)
    session_id = str(UUID(int=30))
    turn_id = str(UUID(int=31))
    legacy_runtime = json.dumps(
        {
            "agent_type": "generic",
            "model_alias": "legacy-model",
            "workspace_path": str(tmp_path),
            "knowledge": [],
            "vector_backend": "in_memory",
        }
    )
    _insert_session(
        connection,
        session_id=session_id,
        kind="conversation",
        runtime_json=legacy_runtime,
    )
    _insert_turn(
        connection,
        turn_id=turn_id,
        session_id=session_id,
        ordinal=1,
        message="legacy",
        runtime_json=legacy_runtime,
    )
    connection.commit()
    connection.close()

    store = TurnStore(database)
    assert store.get_turn(turn_id).runtime == RuntimeBinding(
        model_alias="legacy-model",
        workspace_path=str(tmp_path),
    )
    store.close()
