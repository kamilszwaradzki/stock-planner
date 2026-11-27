from fastapi import Request

from stock_planner.infrastructure.sales_repository import SalesRepository


def get_repository(request: Request) -> SalesRepository:
    repository: SalesRepository = request.app.state.repository
    return repository
