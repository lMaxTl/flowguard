from flowguard.defenses.query.base import QueryDefense


class NoOpQueryDefense(QueryDefense):
    name = "noop"
