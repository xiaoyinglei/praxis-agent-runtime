"""Tenant event ledger."""

def reconcile(events):
    """Return the winning, live events sorted by (tenant, id).

    Identity is the pair (tenant, id). The winner for an identity is the
    event with the greatest seq; ties are broken by later input position.
    A winning event with deleted=True removes the identity, and a later
    non-deleted winner can resurrect it. Input events and the input list
    are never modified.
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
