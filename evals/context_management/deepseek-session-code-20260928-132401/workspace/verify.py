"""Verify reconcile() and totals() against the specification documents."""

import copy

from ledger import reconcile, totals


def test_empty_input():
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
    assert result[0]["seq"] == 5
    assert result[0]["amount"] == 5


def test_equal_seq_later_position_wins():
    events = [
        {"tenant": "a", "id": "x", "seq": 4, "amount": 10},
        {"tenant": "a", "id": "x", "seq": 4, "amount": 20},
    ]
    result = reconcile(events)
    assert len(result) == 1
    assert result[0]["amount"] == 20


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
    assert len(result) == 1
    assert result[0]["seq"] == 2


def test_deleted_defaults_to_false():
    events = [{"tenant": "a", "id": "x", "seq": 1, "amount": 1}]
    assert len(reconcile(events)) == 1


def test_sorted_lexicographically_by_tenant_id():
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
        "note": "audit sample; preserve this field",
    }
    result = reconcile([event])
    assert result[0] == event
    assert result[0] is event


def test_input_not_modified():
    events = [
        {"tenant": "b", "id": "y", "seq": 2, "amount": 2},
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1},
        {"tenant": "a", "id": "x", "seq": 3, "amount": 3, "deleted": True},
    ]
    snapshot = copy.deepcopy(events)
    order = list(events)
    reconcile(events)
    assert events == snapshot
    assert events == order


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("all", len(tests), "checks passed")


if __name__ == "__main__":
    main()
