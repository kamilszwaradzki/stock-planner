# Stock Planner

Demand forecasting and replenishment planning API built with FastAPI, pandas, NumPy and DuckDB.

## Running

```bash
uv sync
uv run python scripts/generate_data.py
uv run uvicorn stock_planner.api.main:app --reload
```

Interactive docs: http://localhost:8000/docs

## Configuration

Settings are read from environment variables with the `STOCK_PLANNER_` prefix.

| Variable | Default | Meaning |
|---|---|---|
| `STOCK_PLANNER_DATA_DIR` | `data` | Directory with `sales.csv` and `stock.csv` |
| `STOCK_PLANNER_DUCKDB_MEMORY_LIMIT` | `512MB` | DuckDB memory limit |

## Architecture

```
api/             HTTP only: routes, schemas, error mapping
application/     use cases: fetch data, run domain logic
domain/          pure logic: forecasting, backtesting, replenishment
infrastructure/  DuckDB sales repository
```

Dependencies point one way: `api → application → domain + infrastructure`. The domain layer does no I/O and knows nothing about HTTP, so the same use cases can run from a CLI or a batch job.

Endpoints are plain `def`, not `async def`: DuckDB and pandas are synchronous, and FastAPI runs `def` endpoints in a thread pool so they do not block the event loop.

## Endpoints

| Method and URL | Returns |
|---|---|
| `GET /health` | `{"status": "ok"}` |
| `GET /products` | list of SKUs |
| `GET /products/{sku}/sales?warehouse=WH-01` | weekly sales |
| `GET /products/{sku}/forecast?warehouse=WH-01&method=auto&horizon=4` | forecast and the method used |
| `GET /products/{sku}/backtest?warehouse=WH-01` | MAE per method and the best one |
| `GET /replenishment?warehouse=WH-01&service_level=0.95&only_needing_order=true` | order plan, most urgent first |

`method=auto` backtests both methods on the last 8 weeks and uses the one with the lower error; on a tie the simpler moving average wins.

`/replenishment` is sorted by days of cover (available stock divided by average daily demand). With zero demand the stock never runs out, and `days_of_cover` is `null`.

## Errors

| Status | When |
|---|---|
| 404 | `UnknownProductError`: no sales for the SKU in that warehouse |
| 404 | `UnknownWarehouseError`: no stock records for the warehouse |
| 422 | `InsufficientHistoryError`: the product exists, but its history is too short for the requested calculation |
| 422 | invalid query parameters (unknown `method`, `horizon` outside 1–26, `service_level` outside (0, 1)) |

An unknown warehouse has its own error rather than reusing `UnknownProductError`, so the message says what is actually missing.

## Development

```bash
uv run pytest
uv run ruff check .
uv run mypy src tests
```
