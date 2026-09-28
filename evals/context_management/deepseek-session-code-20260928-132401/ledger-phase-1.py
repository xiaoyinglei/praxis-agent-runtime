"""Tenant event ledger."""

def reconcile(events):
    """Return the winning live events for the given input events.

    Identity is (tenant, id). The greatest seq wins; on equal seq the later
    input position wins. Winner selection happens before deletion: a winning
    event with deleted=true removes that identity, and a later nondeleted
    event can resurrect it. Live winners are sorted lexicographically by
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
