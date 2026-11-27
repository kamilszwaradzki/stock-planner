import math
import statistics
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from stock_planner.domain.errors import InsufficientHistoryError
from stock_planner.domain.forecasting import History


@dataclass(frozen=True)
class ReplenishmentPlan:
    average_daily_demand: float
    safety_stock: float
    reorder_point: float
    available_stock: int
    order_quantity: int

    @property
    def needs_order(self) -> bool:
        return self.order_quantity > 0


def safety_stock(
    daily_demand: History,
    lead_time_days: int,
    service_level: float = 0.95,
) -> float:
    """Stock buffer covering demand variability during the lead time."""
    if not 0 < service_level < 1:
        raise ValueError("service_level must be in the range (0, 1)")
    if lead_time_days < 1:
        raise ValueError("lead_time_days must be at least 1")

    values = np.asarray(daily_demand, dtype=np.float64)

    if len(values) < 2:
        raise InsufficientHistoryError(
            f"Safety stock needs at least 2 days, got {len(values)}"
        )

    z = statistics.NormalDist().inv_cdf(service_level)
    sigma = float(np.std(values, ddof=1))

    return float(z * sigma * np.sqrt(lead_time_days))


class _ReorderPointComponents(NamedTuple):
    average_daily_demand: float
    safety_stock: float
    reorder_point: float


def _reorder_point_components(
    daily_demand: History, lead_time_days: int, service_level: float
) -> _ReorderPointComponents:
    """Compute the reorder point and its parts from one pass over the history."""
    buffer = safety_stock(daily_demand, lead_time_days, service_level)
    average_daily_demand = float(np.mean(np.asarray(daily_demand, dtype=np.float64)))

    return _ReorderPointComponents(
        average_daily_demand=average_daily_demand,
        safety_stock=buffer,
        reorder_point=average_daily_demand * lead_time_days + buffer,
    )


def reorder_point(
    daily_demand: History,
    lead_time_days: int,
    service_level: float = 0.95,
) -> float:
    """Stock level at which a new order should be placed."""
    return _reorder_point_components(
        daily_demand, lead_time_days, service_level
    ).reorder_point


def recommended_order_quantity(
    on_hand: int,
    on_order: int,
    reorder_level: float,
    review_period_demand: float,
) -> int:
    """Units to order so that stock covers the reorder point and the review period."""
    if on_hand < 0:
        raise ValueError("on_hand must not be negative")
    if on_order < 0:
        raise ValueError("on_order must not be negative")
    if reorder_level < 0:
        raise ValueError("reorder_level must not be negative")
    if review_period_demand < 0:
        raise ValueError("review_period_demand must not be negative")

    available_stock = on_hand + on_order

    if available_stock > reorder_level:
        return 0

    return math.ceil(round(reorder_level + review_period_demand - available_stock, 6))


def plan_replenishment(
    daily_demand: History,
    on_hand: int,
    on_order: int,
    lead_time_days: int,
    service_level: float = 0.95,
    review_period_days: int = 7,
) -> ReplenishmentPlan:
    """Combine demand statistics and current stock into an ordering decision."""
    if review_period_days < 1:
        raise ValueError("review_period_days must be at least 1")

    components = _reorder_point_components(daily_demand, lead_time_days, service_level)
    available_stock = on_hand + on_order
    review_period_demand = components.average_daily_demand * review_period_days

    order_quantity = recommended_order_quantity(
        on_hand=on_hand,
        on_order=on_order,
        reorder_level=components.reorder_point,
        review_period_demand=review_period_demand,
    )

    return ReplenishmentPlan(
        average_daily_demand=components.average_daily_demand,
        safety_stock=components.safety_stock,
        reorder_point=components.reorder_point,
        available_stock=available_stock,
        order_quantity=order_quantity,
    )
