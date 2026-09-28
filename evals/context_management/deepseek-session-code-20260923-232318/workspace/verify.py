"""Verify reconcile(events) against identity.md, deletion.md, ordering.md."""

import copy

from ledger import reconcile


def test_empty_input():
    assert reconcile([]) == []


def test_identity_is_tenant_and_id_pair():
    # Same id in different tenants are distinct identities.
    events = [
        {"tenant": "a", "id": "1", "seq": 1, "amount": 10},
        {"tenant": "b", "id": "1", "seq": 1, "amount": 20},
    ]
    result = reconcile(events)
    assert len(result) == 2
    assert [(e["tenant"], e["id"]) for e in result] == [("a", "1"), ("b", "1")]


def test_greatest_seq_wins():
    events = [
        {"tenant": "a", "id": "1", "seq": 1, "amount": 10},
        {"tenant": "a", "id": "1", "seq": 5, "amount": 50},
        {"tenant": "a", "id": "1", "seq": 3, "amount": 30},
    ]
    result = reconcile(events)
    assert len(result) == 1
    assert result[0]["seq"] == 5
    assert result[0]["amount"] == 50


def test_equal_seq_later_input_position_wins():
    events = [
        {"tenant": "a", "id": "1", "seq": 2, "amount": 10},
        {"tenant": "a", "id": "1", "seq": 2, "amount": 99},
    ]
    result = reconcile(events)
    assert len(result) == 1
    assert result[0]["amount"] == 99


def test_deleted_winner_removes_identity():
    events = [
        {"tenant": "a", "id": "1", "seq": 1, "amount": 10},
        {"tenant": "a", "id": "1", "seq": 2, "deleted": True},
    ]
    assert reconcile(events) == []


def test_later_nondeleted_event_resurrects():
    events = [
        {"tenant": "a", "id": "1", "seq": 1, "amount": 10},
        {"tenant": "a", "id": "1", "seq": 2, "deleted": True},
        {"tenant": "a", "id": "1", "seq": 3, "amount": 30},
    ]
    result = reconcile(events)
    assert len(result) == 1
    assert result[0]["seq"] == 3
    assert result[0]["amount"] == 30


def test_deleted_defaults_to_false():
    events = [{"tenant": "a", "id": "1", "seq": 1, "amount": 10}]
    assert len(reconcile(events)) == 1


def test_deletion_applied_after_winner_selection():
    # A deleted event with a lower seq must not remove a higher-seq winner.
    events = [
        {"tenant": "a", "id": "1", "seq": 1, "deleted": True},
        {"tenant": "a", "id": "1", "seq": 2, "amount": 20},
    ]
    result = reconcile(events)
    assert len(result) == 1
    assert result[0]["seq"] == 2


def test_ordering_lexicographic_by_tenant_then_id():
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
    event = {"tenant": "a", "id": "1", "seq": 1, "amount": 7, "note": "x"}
    result = reconcile([event])
    assert result[0] == event
    assert result[0] is event


def test_input_list_and_dicts_not_modified():
    events = [
        {"tenant": "b", "id": "1", "seq": 1, "amount": 1},
        {"tenant": "a", "id": "1", "seq": 2, "amount": 2},
        {"tenant": "a", "id": "1", "seq": 1, "amount": 3},
    ]
    snapshot = copy.deepcopy(events)
    order = list(events)
    reconcile(events)
    assert events == snapshot
    assert events == order


def test_returns_new_list():
    events = [{"tenant": "a", "id": "1", "seq": 1, "amount": 1}]
    result = reconcile(events)
    assert result is not events


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print("all", len(tests), "checks passed")


if __name__ == "__main__":
    main()
