import math
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from stock_planner.domain.forecasting import ForecastMethod


class _FromAttributes(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class HealthResponse(BaseModel):
    status: str


class WeeklySalesPoint(_FromAttributes):
    week: date
    quantity: int


class ForecastResponse(_FromAttributes):
    sku: str
    warehouse: str
    method: ForecastMethod
    forecast: list[float]


class BacktestResponse(_FromAttributes):
    sku: str
    warehouse: str
    errors: dict[ForecastMethod, float]
    best_method: ForecastMethod


class ReplenishmentItemResponse(_FromAttributes):
    sku: str
    average_daily_demand: float
    safety_stock: float
    reorder_point: float
    available_stock: int
    order_quantity: int
    needs_order: bool
    days_of_cover: float | None = Field(
        description="Days the available stock lasts; null when demand is zero.",
    )

    @field_serializer("days_of_cover")
    def _infinite_cover_as_null(self, value: float | None) -> float | None:
        if value is None or math.isinf(value):
            return None
        return value
