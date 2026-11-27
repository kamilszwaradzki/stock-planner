import math

import pytest
from conftest import WAREHOUSE

from stock_planner.application.services import (
    MethodChoice,
    backtest_product,
    forecast_product,
    plan_warehouse_replenishment,
)
from stock_planner.domain.errors import (
    InsufficientHistoryError,
    UnknownProductError,
    UnknownWarehouseError,
)
from stock_planner.domain.forecasting import ForecastMethod
from stock_planner.infrastructure.sales_repository import SalesRepository


def test_forecast_product_constant_sales_returns_weekly_level(
    planning_repository: SalesRepository,
) -> None:
    result = forecast_product(
        planning_repository,
        "SKU-001",
        WAREHOUSE,
        MethodChoice.MOVING_AVERAGE,
        horizon=3,
    )

    assert result.method is ForecastMethod.MOVING_AVERAGE
    assert result.forecast == pytest.approx([70.0, 70.0, 70.0])
    assert all(type(value) is float for value in result.forecast)


def test_forecast_product_uses_requested_method(
    planning_repository: SalesRepository,
) -> None:
    result = forecast_product(
        planning_repository,
        "SKU-001",
        WAREHOUSE,
        MethodChoice.EXPONENTIAL_SMOOTHING,
        horizon=2,
    )

    assert result.method is ForecastMethod.EXPONENTIAL_SMOOTHING
    assert result.forecast == pytest.approx([70.0, 70.0])


def test_forecast_product_auto_picks_a_method(
    planning_repository: SalesRepository,
) -> None:
    result = forecast_product(
        planning_repository, "SKU-001", WAREHOUSE, MethodChoice.AUTO, horizon=4
    )

    assert result.method is ForecastMethod.MOVING_AVERAGE
    assert len(result.forecast) == 4


def test_forecast_product_unknown_sku_raises(
    planning_repository: SalesRepository,
) -> None:
    with pytest.raises(UnknownProductError):
        forecast_product(
            planning_repository, "UNKNOWN", WAREHOUSE, MethodChoice.AUTO, 4
        )


def test_forecast_product_unknown_warehouse_raises(
    planning_repository: SalesRepository,
) -> None:
    with pytest.raises(UnknownProductError):
        forecast_product(planning_repository, "SKU-001", "WH-99", MethodChoice.AUTO, 4)


@pytest.mark.parametrize(
    "method",
    [MethodChoice.AUTO, MethodChoice.MOVING_AVERAGE],
)
def test_forecast_product_short_history_raises_insufficient_history(
    planning_repository: SalesRepository, method: MethodChoice
) -> None:
    with pytest.raises(InsufficientHistoryError):
        forecast_product(planning_repository, "SKU-002", WAREHOUSE, method, 4)


def test_forecast_product_exponential_smoothing_works_with_one_week(
    planning_repository: SalesRepository,
) -> None:
    result = forecast_product(
        planning_repository, "SKU-002", WAREHOUSE, MethodChoice.EXPONENTIAL_SMOOTHING, 2
    )

    assert result.forecast == pytest.approx([35.0, 35.0])


def test_backtest_product_constant_sales(planning_repository: SalesRepository) -> None:
    result = backtest_product(planning_repository, "SKU-001", WAREHOUSE)

    assert result.errors == {
        ForecastMethod.MOVING_AVERAGE: pytest.approx(0.0),
        ForecastMethod.EXPONENTIAL_SMOOTHING: pytest.approx(0.0),
    }
    assert result.best_method is ForecastMethod.MOVING_AVERAGE


def test_backtest_product_unknown_sku_raises(
    planning_repository: SalesRepository,
) -> None:
    with pytest.raises(UnknownProductError):
        backtest_product(planning_repository, "UNKNOWN", WAREHOUSE)


def test_plan_warehouse_replenishment_sorts_by_days_of_cover(
    planning_repository: SalesRepository,
) -> None:
    items = plan_warehouse_replenishment(
        planning_repository, WAREHOUSE, only_needing_order=False
    )

    assert [item.sku for item in items] == ["SKU-001", "SKU-002", "SKU-003", "SKU-004"]
    assert [item.days_of_cover for item in items] == pytest.approx(
        [0.0, 4.0, 250.0, math.inf]
    )


def test_plan_warehouse_replenishment_only_needing_order(
    planning_repository: SalesRepository,
) -> None:
    items = plan_warehouse_replenishment(planning_repository, WAREHOUSE)

    assert [item.sku for item in items] == ["SKU-001", "SKU-002"]
    assert all(item.needs_order for item in items)


def test_plan_warehouse_replenishment_item_values(
    planning_repository: SalesRepository,
) -> None:
    first = plan_warehouse_replenishment(planning_repository, WAREHOUSE)[0]

    assert first.sku == "SKU-001"
    assert first.average_daily_demand == pytest.approx(10.0)
    assert first.safety_stock == pytest.approx(0.0)
    assert first.reorder_point == pytest.approx(70.0)
    assert first.available_stock == 0
    assert first.order_quantity == 140


def test_plan_warehouse_replenishment_unknown_warehouse_raises(
    planning_repository: SalesRepository,
) -> None:
    with pytest.raises(UnknownWarehouseError):
        plan_warehouse_replenishment(planning_repository, "WH-99")
