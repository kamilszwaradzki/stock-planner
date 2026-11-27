"""Use cases: fetch data from the repository and run domain logic on it.

Nothing here knows about HTTP, so the same functions can be called from the
API, a CLI or a batch job.
"""

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from stock_planner.domain.errors import UnknownProductError, UnknownWarehouseError
from stock_planner.domain.forecasting import (
    FloatArray,
    ForecastMethod,
    backtest,
    best_method,
    exponential_smoothing_forecast,
    moving_average_forecast,
)
from stock_planner.domain.replenishment import plan_replenishment
from stock_planner.infrastructure.sales_repository import SalesRepository


class MethodChoice(StrEnum):
    """A forecasting method, or AUTO to let backtesting pick one."""

    AUTO = "auto"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"


@dataclass(frozen=True)
class ForecastResult:
    sku: str
    warehouse: str
    method: ForecastMethod
    forecast: list[float]


@dataclass(frozen=True)
class BacktestResult:
    sku: str
    warehouse: str
    errors: dict[ForecastMethod, float]
    best_method: ForecastMethod


@dataclass(frozen=True)
class ReplenishmentItem:
    sku: str
    average_daily_demand: float
    safety_stock: float
    reorder_point: float
    available_stock: int
    order_quantity: int
    days_of_cover: float

    @property
    def needs_order(self) -> bool:
        return self.order_quantity > 0


def _weekly_history(
    repository: SalesRepository, sku: str, warehouse: str
) -> FloatArray:
    weekly = repository.weekly_sales(sku, warehouse)

    if weekly.empty:
        raise UnknownProductError(f"No sales for {sku} in {warehouse}")

    return weekly["quantity"].to_numpy(dtype=np.float64)


def forecast_product(
    repository: SalesRepository,
    sku: str,
    warehouse: str,
    method: MethodChoice,
    horizon: int,
) -> ForecastResult:
    history = _weekly_history(repository, sku, warehouse)

    used_method = (
        best_method(history)
        if method is MethodChoice.AUTO
        else ForecastMethod(method.value)
    )

    if used_method is ForecastMethod.MOVING_AVERAGE:
        forecast = moving_average_forecast(history, horizon=horizon)
    else:
        forecast = exponential_smoothing_forecast(history, horizon=horizon)

    return ForecastResult(
        sku=sku,
        warehouse=warehouse,
        method=used_method,
        forecast=forecast.tolist(),
    )


def backtest_product(
    repository: SalesRepository,
    sku: str,
    warehouse: str,
) -> BacktestResult:
    history = _weekly_history(repository, sku, warehouse)

    return BacktestResult(
        sku=sku,
        warehouse=warehouse,
        errors=backtest(history),
        best_method=best_method(history),
    )


def _days_of_cover(available_stock: int, average_daily_demand: float) -> float:
    if average_daily_demand == 0:
        return math.inf
    return available_stock / average_daily_demand


def plan_warehouse_replenishment(
    repository: SalesRepository,
    warehouse: str,
    service_level: float = 0.95,
    only_needing_order: bool = True,
) -> list[ReplenishmentItem]:
    """Plan orders for every product in a warehouse, most urgent first."""
    stock = repository.stock_levels(warehouse)

    if stock.empty:
        raise UnknownWarehouseError(f"No stock records for warehouse {warehouse}")

    items: list[ReplenishmentItem] = []

    rows = zip(
        stock["sku"].astype(str).tolist(),
        stock["on_hand"].astype(int).tolist(),
        stock["on_order"].astype(int).tolist(),
        stock["lead_time_days"].astype(int).tolist(),
        strict=True,
    )

    for sku, on_hand, on_order, lead_time_days in rows:
        daily = repository.daily_demand(sku, warehouse)

        plan = plan_replenishment(
            daily.to_numpy(dtype=np.float64),
            on_hand=on_hand,
            on_order=on_order,
            lead_time_days=lead_time_days,
            service_level=service_level,
        )

        items.append(
            ReplenishmentItem(
                sku=sku,
                average_daily_demand=plan.average_daily_demand,
                safety_stock=plan.safety_stock,
                reorder_point=plan.reorder_point,
                available_stock=plan.available_stock,
                order_quantity=plan.order_quantity,
                days_of_cover=_days_of_cover(
                    plan.available_stock, plan.average_daily_demand
                ),
            )
        )

    if only_needing_order:
        items = [item for item in items if item.needs_order]

    return sorted(items, key=lambda item: item.days_of_cover)
