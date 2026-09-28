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


def totals(events):
    """Return per-tenant summed amounts over the reconciled live events.

    Aggregation runs on reconcile(events), so deleted and superseded events
    are excluded. Each tenant maps to the sum of its live events' amounts,
    with a missing amount treated as 0. Tenants with no live events are
    absent, so an empty input yields an empty dict. The input list and its
    dictionaries are never modified.
    """
    result = {}
    for event in reconcile(events):
        tenant = event["tenant"]
        result[tenant] = result.get(tenant, 0) + event.get("amount", 0)
    return result
