"""Tenant event ledger."""


def reconcile(events):
    """Return the winning live events for the given input events.

    Identity is the pair (tenant, id). For each identity the event with the
    greatest seq wins; ties on seq are broken by the later input position.
    Winner selection happens before deletion: a winning event with
    deleted=True removes that identity, while a later non-deleted event can
    resurrect it. The surviving winners are sorted lexicographically by
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


def totals(events):
    """Return per-tenant totals of the live reconciled events.

    The input is first reduced with reconcile, so only winning, non-deleted
    events contribute. Each live event's amount is added to its tenant's
    running total; amounts are summed exactly as given (no rounding or
    coercion). Tenants with no live events are absent from the result. The
    returned mapping is keyed by tenant and the input is never modified.
    """
    result = {}
    for event in reconcile(events):
        tenant = event["tenant"]
        result[tenant] = result.get(tenant, 0) + event["amount"]
    return result
