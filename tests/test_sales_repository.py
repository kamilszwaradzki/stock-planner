import pandas as pd
import pytest
from conftest import RepositoryFactory

from stock_planner.infrastructure.sales_repository import SalesRepository


@pytest.fixture
def repository(make_repository: RepositoryFactory) -> SalesRepository:
    sales = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-03",
                "2024-01-04",
                "2024-01-08",
                "2024-01-09",
            ],
            "sku": [
                "SKU-001",
                "SKU-001",
                "SKU-001",
                "SKU-001",
                "SKU-002",
            ],
            "warehouse": [
                "WH-01",
                "WH-01",
                "WH-01",
                "WH-01",
                "WH-01",
            ],
            "quantity": [10, 2, 3, 3, 7],
        }
    )

    stock = pd.DataFrame(
        {
            "sku": ["SKU-001", "SKU-002"],
            "warehouse": ["WH-01", "WH-01"],
            "on_hand": [100, 50],
            "on_order": [20, 10],
            "lead_time_days": [7, 5],
        }
    )

    return make_repository(sales, stock)


def test_weekly_sales_sums_days_into_weeks(
    repository: SalesRepository,
) -> None:
    result = repository.weekly_sales("SKU-001", "WH-01")

    assert len(result) == 2
    assert result["quantity"].tolist() == [15, 3]

    assert result["week"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-08"),
    ]


def test_daily_demand_fills_missing_days_with_zero(
    repository: SalesRepository,
) -> None:
    result = repository.daily_demand("SKU-001", "WH-01")

    assert len(result) == 9

    assert result.loc[pd.Timestamp("2024-01-01")] == 10
    assert result.loc[pd.Timestamp("2024-01-09")] == 0
    assert result.loc[pd.Timestamp("2024-01-03")] == 2
    assert result.loc[pd.Timestamp("2024-01-04")] == 3

    assert result.index.is_monotonic_increasing


def test_unknown_sku_returns_empty(
    repository: SalesRepository,
) -> None:
    weekly = repository.weekly_sales("UNKNOWN", "WH-01")
    daily = repository.daily_demand("UNKNOWN", "WH-01")

    assert weekly.empty
    assert daily.empty


def test_list_skus_returns_unique_sorted(
    repository: SalesRepository,
) -> None:
    assert repository.list_skus() == [
        "SKU-001",
        "SKU-002",
    ]


def test_weekly_sales_drops_incomplete_weeks(
    repository: SalesRepository,
) -> None:
    result = repository.weekly_sales("SKU-001", "WH-01")

    assert pd.Timestamp("2024-01-15") not in result["week"].tolist()
