"""Tenant event ledger.

reconcile(events) selects the winning event for each identity, applies
deletions, and returns the surviving events ordered by (tenant, id).

Specifications: identity.md, deletion.md, ordering.md.
"""


def reconcile(events):
    """Return the reconciled list of winning event dictionaries.

    Identity is the pair (tenant, id). For each identity the event with the
    greatest seq wins; on equal seq the later input position wins. A winning
    event with deleted=True removes that identity, and a later non-deleted
    event can resurrect it. Surviving events are sorted lexicographically by
    (tenant, id). The input list and its dictionaries are never modified.
    """
    winners = {}
    for position, event in enumerate(events):
        key = (event["tenant"], event["id"])
        current = winners.get(key)
        if current is None:
            winners[key] = (event["seq"], position, event)
            continue
        current_seq, current_position, _ = current
        if event["seq"] > current_seq or (
            event["seq"] == current_seq and position > current_position
        ):
            winners[key] = (event["seq"], position, event)

    live = [
        event
        for _, _, event in winners.values()
        if not event.get("deleted", False)
    ]
    live.sort(key=lambda event: (event["tenant"], event["id"]))
    return live
