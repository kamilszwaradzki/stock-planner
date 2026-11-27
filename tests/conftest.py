from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd
import pytest

from stock_planner.infrastructure.sales_repository import SalesRepository

RepositoryFactory = Callable[[pd.DataFrame, pd.DataFrame], SalesRepository]


@pytest.fixture
def make_repository(tmp_path: Path) -> Iterator[RepositoryFactory]:
    """Build a SalesRepository from in-memory sales and stock frames."""
    repositories: list[SalesRepository] = []

    def factory(sales: pd.DataFrame, stock: pd.DataFrame) -> SalesRepository:
        sales_path = tmp_path / f"sales_{len(repositories)}.csv"
        stock_path = tmp_path / f"stock_{len(repositories)}.csv"
        sales.to_csv(sales_path, index=False)
        stock.to_csv(stock_path, index=False)

        repository = SalesRepository(sales_path=sales_path, stock_path=stock_path)
        repositories.append(repository)
        return repository

    yield factory

    for repository in repositories:
        repository.close()


DATES = pd.date_range("2024-01-01", periods=12 * 7, freq="D")
WAREHOUSE = "WH-01"


def _daily_sales(sku: str, quantity: int, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "sku": sku,
            "warehouse": WAREHOUSE,
            "quantity": quantity,
        }
    )


@pytest.fixture
def planning_repository(make_repository: RepositoryFactory) -> SalesRepository:
    sales = pd.concat(
        [
            _daily_sales("SKU-001", 10, DATES),
            _daily_sales("SKU-002", 5, DATES[-7:]),
            _daily_sales("SKU-003", 2, DATES),
            _daily_sales("SKU-004", 0, DATES),
        ]
    )
    stock = pd.DataFrame(
        {
            "sku": ["SKU-001", "SKU-002", "SKU-003", "SKU-004"],
            "warehouse": WAREHOUSE,
            "on_hand": [0, 20, 500, 10],
            "on_order": [0, 0, 0, 0],
            "lead_time_days": [7, 5, 7, 7],
        }
    )
    return make_repository(sales, stock)
