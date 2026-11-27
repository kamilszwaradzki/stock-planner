from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from stock_planner.api.routes import router
from stock_planner.domain.errors import (
    InsufficientHistoryError,
    UnknownProductError,
    UnknownWarehouseError,
)
from stock_planner.infrastructure.sales_repository import SalesRepository
from stock_planner.settings import Settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    repository = SalesRepository(
        sales_path=settings.sales_path,
        stock_path=settings.stock_path,
        memory_limit=settings.duckdb_memory_limit,
    )
    app.state.repository = repository
    try:
        yield
    finally:
        repository.close()


app = FastAPI(title="Stock Planner", lifespan=lifespan)
app.include_router(router)


@app.exception_handler(UnknownProductError)
@app.exception_handler(UnknownWarehouseError)
async def not_found_handler(request: Request, exc: LookupError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(InsufficientHistoryError)
async def insufficient_history_handler(
    request: Request, exc: InsufficientHistoryError
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})
