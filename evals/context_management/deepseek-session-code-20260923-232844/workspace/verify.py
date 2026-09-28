"""Specification checks for ledger.reconcile."""

import copy

from ledger import reconcile


def test_empty_input():
    assert reconcile([]) == []


def test_identity_is_tenant_and_id_pair():
    events = [
        {"tenant": "a", "id": "1", "seq": 1, "amount": 10},
        {"tenant": "b", "id": "1", "seq": 2, "amount": 20},
    ]
    result = reconcile(events)
    assert result == [
        {"tenant": "a", "id": "1", "seq": 1, "amount": 10},
        {"tenant": "b", "id": "1", "seq": 2, "amount": 20},
    ]


def test_greatest_seq_wins():
    events = [
        {"tenant": "a", "id": "1", "seq": 1, "amount": 10},
        {"tenant": "a", "id": "1", "seq": 5, "amount": 50},
        {"tenant": "a", "id": "1", "seq": 3, "amount": 30},
    ]
    assert reconcile(events) == [
        {"tenant": "a", "id": "1", "seq": 5, "amount": 50}
    ]


def test_equal_seq_later_input_position_wins():
    events = [
        {"tenant": "a", "id": "1", "seq": 2, "amount": 10},
        {"tenant": "a", "id": "1", "seq": 2, "amount": 99},
    ]
    assert reconcile(events) == [
        {"tenant": "a", "id": "1", "seq": 2, "amount": 99}
    ]


def test_deleted_winner_removes_identity():
    events = [
        {"tenant": "a", "id": "1", "seq": 1, "amount": 10},
        {"tenant": "a", "id": "1", "seq": 2, "amount": 20, "deleted": True},
    ]
    assert reconcile(events) == []


def test_later_nondeleted_resurrects():
    events = [
        {"tenant": "a", "id": "1", "seq": 1, "amount": 10},
        {"tenant": "a", "id": "1", "seq": 2, "amount": 20, "deleted": True},
        {"tenant": "a", "id": "1", "seq": 3, "amount": 30},
    ]
    assert reconcile(events) == [
        {"tenant": "a", "id": "1", "seq": 3, "amount": 30}
    ]


def test_deleted_defaults_to_false():
    events = [{"tenant": "a", "id": "1", "seq": 1, "amount": 10}]
    assert reconcile(events) == [
        {"tenant": "a", "id": "1", "seq": 1, "amount": 10}
    ]


def test_sorted_lexicographically_by_tenant_then_id():
    events = [
        {"tenant": "b", "id": "1", "seq": 1, "amount": 1},
        {"tenant": "a", "id": "2", "seq": 1, "amount": 2},
        {"tenant": "a", "id": "10", "seq": 1, "amount": 3},
        {"tenant": "a", "id": "1", "seq": 1, "amount": 4},
    ]
    result = reconcile(events)
    assert [(e["tenant"], e["id"]) for e in result] == [
        ("a", "1"),
        ("a", "10"),
        ("a", "2"),
        ("b", "1"),
    ]


def test_all_fields_preserved():
    event = {
        "tenant": "a",
        "id": "1",
        "seq": 1,
        "amount": 10,
        "deleted": False,
        "note": "keep me",
    }
    assert reconcile([event]) == [event]


def test_input_list_and_dicts_not_modified():
    events = [
        {"tenant": "b", "id": "1", "seq": 1, "amount": 1},
        {"tenant": "a", "id": "1", "seq": 2, "amount": 2},
        {"tenant": "a", "id": "1", "seq": 1, "amount": 3},
    ]
    snapshot = copy.deepcopy(events)
    result = reconcile(events)
    assert events == snapshot
    assert result is not events
    assert all(r is not e for r in result for e in events)


def main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("all", len(tests), "checks passed")


if __name__ == "__main__":
    main()
