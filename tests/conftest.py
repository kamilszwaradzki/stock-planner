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
