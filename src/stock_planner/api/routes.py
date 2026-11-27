from typing import Annotated

from fastapi import APIRouter, Depends, Query

from stock_planner.api.dependencies import get_repository
from stock_planner.api.schemas import (
    BacktestResponse,
    ForecastResponse,
    HealthResponse,
    ReplenishmentResponse,
    WeeklySalesPoint,
)
from stock_planner.application import services
from stock_planner.application.services import MethodChoice
from stock_planner.infrastructure.sales_repository import SalesRepository

router = APIRouter()

Repository = Annotated[SalesRepository, Depends(get_repository)]
Warehouse = Annotated[str, Query(min_length=1, examples=["WH-01"])]


@router.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/products")
def list_products(repository: Repository) -> list[str]:
    return services.list_products(repository)


@router.get("/products/{sku}/sales")
def product_sales(
    repository: Repository, sku: str, warehouse: Warehouse
) -> list[WeeklySalesPoint]:
    sales = services.product_weekly_sales(repository, sku, warehouse)
    return [WeeklySalesPoint.model_validate(point) for point in sales]


@router.get("/products/{sku}/forecast")
def product_forecast(
    repository: Repository,
    sku: str,
    warehouse: Warehouse,
    method: MethodChoice = MethodChoice.AUTO,
    horizon: Annotated[int, Query(ge=1, le=26)] = 4,
) -> ForecastResponse:
    result = services.forecast_product(repository, sku, warehouse, method, horizon)
    return ForecastResponse.model_validate(result)


@router.get("/products/{sku}/backtest")
def product_backtest(
    repository: Repository, sku: str, warehouse: Warehouse
) -> BacktestResponse:
    result = services.backtest_product(repository, sku, warehouse)
    return BacktestResponse.model_validate(result)


@router.get("/replenishment")
def replenishment(
    repository: Repository,
    warehouse: Warehouse,
    service_level: Annotated[float, Query(gt=0, lt=1)] = 0.95,
    only_needing_order: bool = True,
) -> ReplenishmentResponse:
    plan = services.plan_warehouse_replenishment(
        repository, warehouse, service_level, only_needing_order
    )
    return ReplenishmentResponse.model_validate(plan)
