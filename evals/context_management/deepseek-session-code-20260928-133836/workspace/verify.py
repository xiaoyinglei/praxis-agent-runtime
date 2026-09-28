"""Verify reconcile() against identity.md, deletion.md and ordering.md."""

import copy

from ledger import reconcile, totals


def test_empty():
    assert reconcile([]) == []


def test_identity_is_tenant_and_id():
    # Same id, different tenants -> independent identities.
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1},
        {"tenant": "b", "id": "x", "seq": 2, "amount": 2},
    ]
    result = reconcile(events)
    assert [(e["tenant"], e["id"]) for e in result] == [("a", "x"), ("b", "x")]


def test_greatest_seq_wins():
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1},
        {"tenant": "a", "id": "x", "seq": 5, "amount": 5},
        {"tenant": "a", "id": "x", "seq": 3, "amount": 3},
    ]
    result = reconcile(events)
    assert len(result) == 1
    assert result[0]["seq"] == 5 and result[0]["amount"] == 5


def test_equal_seq_later_position_wins():
    events = [
        {"tenant": "a", "id": "x", "seq": 7, "amount": 1},
        {"tenant": "a", "id": "x", "seq": 7, "amount": 2},
    ]
    result = reconcile(events)
    assert len(result) == 1
    assert result[0]["amount"] == 2


def test_deletion_removes_identity():
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1},
        {"tenant": "a", "id": "x", "seq": 2, "amount": 2, "deleted": True},
    ]
    assert reconcile(events) == []


def test_resurrection_by_later_nondeleted():
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1, "deleted": True},
        {"tenant": "a", "id": "x", "seq": 2, "amount": 2, "deleted": False},
    ]
    result = reconcile(events)
    assert len(result) == 1 and result[0]["amount"] == 2


def test_deleted_defaults_false():
    events = [{"tenant": "a", "id": "x", "seq": 1, "amount": 1}]
    assert len(reconcile(events)) == 1


def test_winner_selection_before_deletion():
    # Highest seq is deleted -> identity removed even though a live lower seq exists.
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1},
        {"tenant": "a", "id": "x", "seq": 9, "amount": 9, "deleted": True},
    ]
    assert reconcile(events) == []


def test_sorted_lexicographically_by_tenant_then_id():
    events = [
        {"tenant": "b", "id": "a", "seq": 1},
        {"tenant": "a", "id": "z", "seq": 1},
        {"tenant": "a", "id": "a", "seq": 1},
    ]
    result = reconcile(events)
    assert [(e["tenant"], e["id"]) for e in result] == [
        ("a", "a"),
        ("a", "z"),
        ("b", "a"),
    ]


def test_preserves_all_fields():
    event = {
        "tenant": "a",
        "id": "x",
        "seq": 1,
        "amount": -225179981368524825,
        "deleted": False,
        "note": "audit sample 0; preserve this field",
    }
    result = reconcile([event])
    assert result[0] == event
    assert result[0] is event


def test_does_not_modify_input():
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1},
        {"tenant": "a", "id": "x", "seq": 2, "amount": 2, "deleted": True},
        {"tenant": "b", "id": "y", "seq": 1, "amount": 3},
    ]
    snapshot = copy.deepcopy(events)
    order = list(events)
    reconcile(events)
    assert events == snapshot
    assert events == order


def test_returns_new_list():
    events = [{"tenant": "a", "id": "x", "seq": 1}]
    result = reconcile(events)
    assert result is not events


def test_totals_empty_input():
    assert totals([]) == {}


def test_totals_sums_per_tenant():
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 10},
        {"tenant": "a", "id": "y", "seq": 1, "amount": 5},
        {"tenant": "b", "id": "z", "seq": 1, "amount": 7},
    ]
    assert totals(events) == {"a": 15, "b": 7}


def test_totals_uses_reconciled_live_events_only():
    # Superseded and deleted events must not contribute.
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 100},
        {"tenant": "a", "id": "x", "seq": 2, "amount": 3},
        {"tenant": "a", "id": "y", "seq": 1, "amount": 50, "deleted": True},
        {"tenant": "b", "id": "z", "seq": 1, "amount": 4},
    ]
    assert totals(events) == {"a": 3, "b": 4}


def test_totals_missing_amount_is_zero():
    events = [
        {"tenant": "a", "id": "x", "seq": 1},
        {"tenant": "a", "id": "y", "seq": 1, "amount": 6},
    ]
    assert totals(events) == {"a": 6}


def test_totals_tenant_with_no_live_events_absent():
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1, "deleted": True},
        {"tenant": "b", "id": "y", "seq": 1, "amount": 2},
    ]
    assert totals(events) == {"b": 2}


def test_totals_negative_amounts():
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 10},
        {"tenant": "a", "id": "y", "seq": 1, "amount": -4},
    ]
    assert totals(events) == {"a": 6}


def test_totals_does_not_modify_input():
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1},
        {"tenant": "a", "id": "x", "seq": 2, "amount": 2, "deleted": True},
        {"tenant": "b", "id": "y", "seq": 1, "amount": 3},
    ]
    snapshot = copy.deepcopy(events)
    order = list(events)
    totals(events)
    assert events == snapshot
    assert events == order


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("all", len(tests), "tests passed")


if __name__ == "__main__":
    main()
