"""praxis.store.verify: offline verification smoke-tests + regression guard.

Praxis verify 离线校验——只读扫描、问题累积、永不修复。
"""

from __future__ import annotations

import pytest

from praxis.protocol.items import ItemKind
from praxis.store.log import ItemEntry, Store
from praxis.store.verify import verify_store


# ---------------------------------------------------------------------------
# Step 2 brief test 1: seq gap detection
# ---------------------------------------------------------------------------


def test_verify_detects_seq_gap(store: Store) -> None:
    store.append_items("t", None, [ItemEntry(ItemKind.USER_MESSAGE, {"text": "a"})])
    # Directly insert with seq=3 to simulate a gap (after delete, max+1 would fill back)
    store.transact(
        lambda c: c.execute(
            "INSERT INTO items VALUES ("
            "'manual', 't', NULL, 3, 'tool_call', "
            "'{\"call_id\":\"c1\",\"name\":\"x\",\"input\":{}}', 0.0)",
        )
    )
    report = verify_store(store)
    assert not report.ok
    assert any("seq" in p for p in report.problems)


# ---------------------------------------------------------------------------
# Step 2 brief test 2: B21 regression — write path never calls verify
# ---------------------------------------------------------------------------


def test_write_path_never_calls_verify(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    """B21 回归:把 verify 变成炸弹,append 必须照常成功。"""
    import praxis.store.verify as verify_mod

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("verify called on write path")

    monkeypatch.setattr(verify_mod, "verify_store", boom)
    assert store.append_items("t", None, [ItemEntry("user_message", {"text": "ok"})])


# ---------------------------------------------------------------------------
# Clean-store: no problems expected
# ---------------------------------------------------------------------------


def test_verify_clean_store(store: Store) -> None:
    """Fresh store with valid items → ok=True, empty problems."""
    store.create_thread()
    store.append_items(
        "t",
        "turn-1",
        [
            ItemEntry(ItemKind.USER_MESSAGE, {"text": "hello"}),
            ItemEntry(ItemKind.TURN_STATUS, {"from": "running", "to": "completed", "reason": ""}),
        ],
    )
    report = verify_store(store)
    assert report.ok
    assert report.problems == ()
    assert report.checked_items == 2


# ---------------------------------------------------------------------------
# Bad JSON payload detection
# ---------------------------------------------------------------------------


def test_verify_detects_corrupt_payload(store: Store) -> None:
    """A stored item with non-JSON payload is flagged as corrupt."""
    # Two valid items, then corrupt one via direct SQL (bypasses Store's JSON encoding).
    store.append_items(
        "t",
        "turn-1",
        [
            ItemEntry(ItemKind.USER_MESSAGE, {"text": "valid"}),
            ItemEntry(ItemKind.USER_MESSAGE, {"text": "also_valid"}),
        ],
    )
    store.transact(
        lambda c: c.execute(
            "UPDATE items SET payload = ? WHERE seq = 1",
            ("{not json}",),
        ),
    )
    report = verify_store(store)
    assert not report.ok
    assert any("corrupt" in p or "payload" in p for p in report.problems)
    assert report.checked_items >= 2


# ---------------------------------------------------------------------------
# Unknown turn status detection
# ---------------------------------------------------------------------------


def test_verify_detects_unknown_turn_status(store: Store) -> None:
    """Turns.status outside known set is listed as a problem."""
    from praxis.store.turns import create_turn

    store.create_thread()
    store.ensure_thread("t")
    create_turn(store, turn_id="bad-turn", thread_id="t", binding={})
    store.transact(
        lambda c: c.execute("UPDATE turns SET status = ? WHERE turn_id = ?", ("evil_status", "bad-turn")),
    )
    report = verify_store(store)
    assert not report.ok
    assert any("status" in p.lower() or "evil" in p for p in report.problems)


# ---------------------------------------------------------------------------
# Orphan tool_call detection (no matching tool_result by call_id in same turn)
# ---------------------------------------------------------------------------


def test_verify_orphan_tool_call(store: Store) -> None:
    """tool_call without a matching tool_result is informational."""
    call_id = "call-001"
    store.append_items(
        "t",
        "turn-1",
        [
            ItemEntry(ItemKind.TOOL_CALL, {"call_id": call_id, "name": "echo", "input": {"msg": "hi"}}),
            ItemEntry(ItemKind.USER_MESSAGE, {"text": "other"}),
        ],
    )
    report = verify_store(store)
    assert not report.ok
    assert any(call_id in p for p in report.problems)


def test_verify_tool_call_with_matching_result(store: Store) -> None:
    """tool_call paired with tool_result (same call_id, same turn) → clean."""
    call_id = "call-002"
    store.append_items(
        "t",
        "turn-1",
        [
            ItemEntry(ItemKind.TOOL_CALL, {"call_id": call_id, "name": "greet", "input": {}}),
            ItemEntry(ItemKind.TOOL_RESULT, {"call_id": call_id, "output": {"value": "hello"}}),
        ],
    )
    report = verify_store(store)
    assert report.ok
    assert report.problems == ()


# ---------------------------------------------------------------------------
# Fix Round 1: Finding 1 — non-dict JSON payload must not raise AttributeError
# ---------------------------------------------------------------------------


def test_verify_non_dict_payload_no_crash(store: Store) -> None:
    """A tool_call with payload=[1,2] (valid JSON but not object) must NOT crash.

    Previously: json.loads succeeds, then payload.get('call_id') → AttributeError escapes.
    Expected: problem entry for corrupt payload type, ok=False.
    """
    # Insert a valid item so checked_items ≥ 1, plus the bad tool_call at seq=3
    store.append_items("t", "turn-1", [ItemEntry(ItemKind.USER_MESSAGE, {"text": "ok"})])
    store.transact(
        lambda c: c.execute(
            "INSERT INTO items VALUES ("
            "'bad', 't', 'turn-1', 3, 'tool_call', "
            "'[1, 2]', 0.0)",
        ),
    )
    report = verify_store(store)
    assert not report.ok
    assert any("bad" in p or "corrupt" in p or "payload" in p for p in report.problems)


# ---------------------------------------------------------------------------
# Fix Round 1: Finding 2 — NULL-turn orphan tool_call must be reported
# ---------------------------------------------------------------------------


def test_verify_null_turn_orphan_tool_call(store: Store) -> None:
    """A tool_call with turn_id=NULL (no result can match) must be flagged as orphan."""
    # Build an empty thread first, then inject a naked tool_call via raw SQL
    store.create_thread()
    store.transact(
        lambda c: c.execute(
            "INSERT INTO items VALUES ("
            "'orphan-n', 't', NULL, 1, 'tool_call', "
            "'{\"call_id\":\"naked\",\"name\":\"x\",\"input\":{}}', 0.0)",
        ),
    )
    report = verify_store(store)
    assert not report.ok
    assert any("naked" in p for p in report.problems)
