"""Verification for ledger.reconcile against the specification documents."""

import copy

from ledger import reconcile


def test_empty_input():
    assert reconcile([]) == []


def test_identity_is_tenant_and_id():
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1},
        {"tenant": "b", "id": "x", "seq": 2, "amount": 2},
    ]
    result = reconcile(events)
    assert result == [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1},
        {"tenant": "b", "id": "x", "seq": 2, "amount": 2},
    ]


def test_greatest_seq_wins():
    events = [
        {"tenant": "a", "id": "x", "seq": 5, "amount": 5},
        {"tenant": "a", "id": "x", "seq": 9, "amount": 9},
        {"tenant": "a", "id": "x", "seq": 2, "amount": 2},
    ]
    assert reconcile(events) == [{"tenant": "a", "id": "x", "seq": 9, "amount": 9}]


def test_equal_seq_later_position_wins():
    events = [
        {"tenant": "a", "id": "x", "seq": 3, "amount": 1},
        {"tenant": "a", "id": "x", "seq": 3, "amount": 2},
    ]
    assert reconcile(events) == [{"tenant": "a", "id": "x", "seq": 3, "amount": 2}]


def test_deleted_winner_removes_identity():
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1},
        {"tenant": "a", "id": "x", "seq": 2, "amount": 2, "deleted": True},
    ]
    assert reconcile(events) == []


def test_later_nondeleted_resurrects():
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1, "deleted": True},
        {"tenant": "a", "id": "x", "seq": 2, "amount": 2},
    ]
    assert reconcile(events) == [{"tenant": "a", "id": "x", "seq": 2, "amount": 2}]


def test_deleted_defaults_to_false():
    events = [{"tenant": "a", "id": "x", "seq": 1, "amount": 1}]
    assert reconcile(events) == [{"tenant": "a", "id": "x", "seq": 1, "amount": 1}]


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


def test_preserves_every_field():
    event = {
        "tenant": "a",
        "id": "x",
        "seq": 1,
        "amount": -225179981368524825,
        "deleted": False,
        "note": "audit sample; preserve this field",
    }
    result = reconcile([event])
    assert result == [event]
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


def test_combined_scenario():
    events = [
        {"tenant": "t", "id": "a", "seq": 1, "amount": 1},
        {"tenant": "t", "id": "b", "seq": 1, "amount": 2, "deleted": True},
        {"tenant": "t", "id": "a", "seq": 1, "amount": 3},
        {"tenant": "t", "id": "c", "seq": 4, "amount": 4},
        {"tenant": "t", "id": "c", "seq": 4, "amount": 5, "deleted": True},
    ]
    assert reconcile(events) == [
        {"tenant": "t", "id": "a", "seq": 1, "amount": 3},
    ]


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("all", len(tests), "checks passed")


if __name__ == "__main__":
    main()
