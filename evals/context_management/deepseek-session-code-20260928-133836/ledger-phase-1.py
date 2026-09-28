"""Tenant event ledger."""


def reconcile(events):
    """Return the winning, live event dictionaries.

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
        if current is None or (event["seq"], position) > (current[0], current[1]):
            winners[key] = (event["seq"], position, event)

    live = [
        event
        for _, _, event in winners.values()
        if not event.get("deleted", False)
    ]
    live.sort(key=lambda event: (event["tenant"], event["id"]))
    return live
