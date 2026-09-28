"""Verification for ledger.reconcile against the specification documents."""

import copy

from ledger import reconcile, totals


def check_identity_is_tenant_and_id():
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1},
        {"tenant": "b", "id": "x", "seq": 2, "amount": 2},
    ]
    result = reconcile(events)
    assert [(e["tenant"], e["id"]) for e in result] == [("a", "x"), ("b", "x")]


def check_greatest_seq_wins():
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1},
        {"tenant": "a", "id": "x", "seq": 5, "amount": 5},
        {"tenant": "a", "id": "x", "seq": 3, "amount": 3},
    ]
    result = reconcile(events)
    assert len(result) == 1
    assert result[0]["seq"] == 5
    assert result[0]["amount"] == 5


def check_equal_seq_later_position_wins():
    events = [
        {"tenant": "a", "id": "x", "seq": 7, "amount": 1},
        {"tenant": "a", "id": "x", "seq": 7, "amount": 2},
    ]
    result = reconcile(events)
    assert len(result) == 1
    assert result[0]["amount"] == 2


def check_deletion_removes_identity():
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1},
        {"tenant": "a", "id": "x", "seq": 2, "amount": 2, "deleted": True},
    ]
    assert reconcile(events) == []


def check_resurrection_by_later_nondeleted():
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1, "deleted": True},
        {"tenant": "a", "id": "x", "seq": 2, "amount": 2, "deleted": False},
    ]
    result = reconcile(events)
    assert len(result) == 1
    assert result[0]["seq"] == 2


def check_deleted_defaults_to_false():
    events = [{"tenant": "a", "id": "x", "seq": 1, "amount": 1}]
    assert len(reconcile(events)) == 1


def check_winner_selection_before_deletion():
    # The deleted event has the greatest seq, so it wins and removes the
    # identity even though an earlier non-deleted event exists.
    events = [
        {"tenant": "a", "id": "x", "seq": 1, "amount": 1, "deleted": False},
        {"tenant": "a", "id": "x", "seq": 9, "amount": 9, "deleted": True},
    ]
    assert reconcile(events) == []


def check_sorted_by_tenant_then_id():
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


def check_empty_input():
    assert reconcile([]) == []


def check_preserves_fields_and_input():
    events = [
        {
            "tenant": "tenant-0",
            "id": "event-0",
            "seq": 0,
            "amount": -225179981368524825,
            "deleted": True,
            "note": "audit sample 0; preserve this field",
        },
        {
            "tenant": "tenant-1",
            "id": "event-1",
            "seq": 1,
            "amount": -216172782113783832,
            "deleted": False,
            "note": "audit sample 1; preserve this field",
        },
    ]
    snapshot = copy.deepcopy(events)
    result = reconcile(events)
    # Input list and dictionaries are untouched.
    assert events == snapshot
    # The winning dictionary is the very same object, with every field intact.
    assert result == [events[1]]
    assert result[0] is events[1]
    assert result[0]["note"] == "audit sample 1; preserve this field"


def main():
    checks = [
        check_identity_is_tenant_and_id,
        check_greatest_seq_wins,
        check_equal_seq_later_position_wins,
        check_deletion_removes_identity,
        check_resurrection_by_later_nondeleted,
        check_deleted_defaults_to_false,
        check_winner_selection_before_deletion,
        check_sorted_by_tenant_then_id,
        check_empty_input,
        check_preserves_fields_and_input,
    ]
    for check in checks:
        check()
        print("ok:", check.__name__)
    print("all checks passed")


if __name__ == "__main__":
    main()
