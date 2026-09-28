"""Verify reconcile() against identity.md, deletion.md and ordering.md."""

import copy

from ledger import reconcile


def test_empty_input():
    assert reconcile([]) == []


def test_identity_is_tenant_and_id_not_id_alone():
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1},
        {"tenant": "b", "id": "x", "seq": 2, "amount": 2},
    ]
    result = reconcile(events)
    assert len(result) == 2
    assert {(e["tenant"], e["id"]) for e in result} == {("a", "x"), ("b", "x")}


def test_greatest_seq_wins():
    events = [
        {"tenant": "t", "id": "e", "seq": 1, "amount": 10},
        {"tenant": "t", "id": "e", "seq": 5, "amount": 50},
        {"tenant": "t", "id": "e", "seq": 3, "amount": 30},
    ]
    result = reconcile(events)
    assert len(result) == 1
    assert result[0]["amount"] == 50


def test_equal_seq_later_input_position_wins():
    events = [
        {"tenant": "t", "id": "e", "seq": 7, "amount": 1},
        {"tenant": "t", "id": "e", "seq": 7, "amount": 2},
    ]
    result = reconcile(events)
    assert len(result) == 1
    assert result[0]["amount"] == 2


def test_deleted_winner_removes_identity():
    events = [
        {"tenant": "t", "id": "e", "seq": 1, "deleted": True},
        {"tenant": "t", "id": "e", "seq": 2, "deleted": True},
    ]
    assert reconcile(events) == []


def test_later_nondeleted_resurrects():
    events = [
        {"tenant": "t", "id": "e", "seq": 1, "deleted": True},
        {"tenant": "t", "id": "e", "seq": 2, "deleted": False, "amount": 9},
    ]
    result = reconcile(events)
    assert len(result) == 1
    assert result[0]["amount"] == 9


def test_deleted_defaults_to_false():
    events = [{"tenant": "t", "id": "e", "seq": 1, "amount": 3}]
    result = reconcile(events)
    assert len(result) == 1
    assert result[0]["amount"] == 3


def test_winner_selection_before_deletion():
    # Highest seq is deleted, so the identity is removed even though a
    # lower-seq non-deleted event exists.
    events = [
        {"tenant": "t", "id": "e", "seq": 1, "deleted": False},
        {"tenant": "t", "id": "e", "seq": 2, "deleted": True},
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


def test_preserves_every_field():
    event = {
        "tenant": "t",
        "id": "e",
        "seq": 1,
        "amount": -225179981368524825,
        "deleted": False,
        "note": "audit sample 0; preserve this field",
    }
    result = reconcile([event])
    assert result[0] == event
    assert result[0] is event


def test_input_list_and_dicts_not_modified():
    events = [
        {"tenant": "t", "id": "e", "seq": 1, "amount": 1},
        {"tenant": "t", "id": "e", "seq": 2, "amount": 2},
        {"tenant": "u", "id": "f", "seq": 1, "deleted": True},
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
