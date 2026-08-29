from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from agent_runtime.harness import RolloutEventReader, RolloutStore


def _start_turn(store: RolloutStore, workspace: Path) -> tuple[str, str]:
    thread = store.create_thread(workspace=workspace)
    turn = store.start_turn(
        thread_id=thread.thread_id,
        user_message="project this turn",
        binding_manifest={"model_alias": "projection-test"},
    )
    return thread.thread_id, turn.turn_id


def test_unknown_internal_item_kind_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread_id, _turn_id = _start_turn(store, workspace)
        record = next(
            record
            for record in store.list_records(thread_id)
            if record.record_type == "item_started"
        )
        payload = {"item_id": record.payload["item_id"], "kind": "future_internal_kind"}
        payload_json = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        with sqlite3.connect(database) as connection:
            connection.execute(
                """
                UPDATE rollout_records
                SET payload_json = ?, payload_hash = ?
                WHERE record_id = ?
                """,
                (
                    payload_json,
                    hashlib.sha256(payload_json.encode()).hexdigest(),
                    record.record_id,
                ),
            )

        with pytest.raises(RuntimeError, match="unknown internal Item kind"):
            RolloutEventReader(store).read(thread_id)
