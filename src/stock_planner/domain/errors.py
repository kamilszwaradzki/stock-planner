class InsufficientHistoryError(ValueError):
    """Not enough data points to produce a forecast."""


class UnknownProductError(LookupError):
    """No sales data for the requested SKU in the requested warehouse."""


class UnknownWarehouseError(LookupError):
    """No stock records for the requested warehouse."""
