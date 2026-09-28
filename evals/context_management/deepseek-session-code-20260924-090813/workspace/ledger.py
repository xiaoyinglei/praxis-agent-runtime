"""Tenant event ledger."""

def reconcile(events):
    """Return the winning live events for each identity.

    Identity is (tenant, id). The winner is the event with the greatest
    seq; ties are broken by the later input position. Winner selection
    happens before deletion: a winning event with deleted=true removes
    that identity, while a later non-deleted event can resurrect it.
    Live winners are sorted lexicographically by (tenant, id). The input
    list and its dictionaries are never modified.
    """
    winners = {}
    for position, event in enumerate(events):
        key = (event["tenant"], event["id"])
        current = winners.get(key)
        if current is None or event["seq"] > current[0][0] or (
            event["seq"] == current[0][0] and position > current[0][1]
        ):
            winners[key] = ((event["seq"], position), event)

    live = [
        event
        for (seq, position), event in winners.values()
        if not event.get("deleted", False)
    ]
    live.sort(key=lambda event: (event["tenant"], event["id"]))
    return live
