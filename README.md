# Stock Planner

[![CI](https://github.com/kamilszwaradzki/stock-planner/actions/workflows/ci.yml/badge.svg)](https://github.com/kamilszwaradzki/stock-planner/actions/workflows/ci.yml)

A warehouse has to keep enough stock that customers do not hit "out of stock", but every unit sitting on a shelf is cash that cannot be spent elsewhere. Stock Planner forecasts demand per product and warehouse from sales history, and turns it into a concrete answer: which products to reorder now, and how many units.

It is a FastAPI service over a DuckDB repository, with the forecasting and replenishment logic in a pure Python domain layer built on NumPy.

## Example

```bash
curl "http://localhost:8000/replenishment?warehouse=WH-01"
```

```json
{
  "items": [
    {
      "sku": "SKU-006",
      "average_daily_demand": 10.087551299589604,
      "safety_stock": 30.65417828793804,
      "reorder_point": 303.01806337685736,
      "available_stock": 57,
      "order_quantity": 317,
      "needs_order": true,
      "days_of_cover": 5.650528885272579
    },
    {
      "sku": "SKU-004",
      "average_daily_demand": 0.5704514363885089,
      "safety_stock": 8.99217996468482,
      "reorder_point": 23.82391731078605,
      "available_stock": 7,
      "order_quantity": 21,
      "needs_order": true,
      "days_of_cover": 12.270983213429256
    }
  ],
  "skipped": []
}
```

The response above is shortened to the first two of six items. SKU-006 sells about 10 units a day and takes 27 days to arrive, so it should be reordered at 303 units. With only 57 available, the stock lasts under 6 days, so it comes first.

## How it works

### Forecasting

Daily sales are summed into Monday-to-Sunday weeks, and each product is forecast from its weekly history. Two methods are available:

- **Moving average:** the forecast is the mean of the last 4 weeks. It is simple and stable, but slow to react when demand shifts.
- **Simple exponential smoothing (SES):** a running level updated every week as `level = α × sales + (1 − α) × level`, with α = 0.3. Recent weeks weigh more, and older weeks fade out gradually instead of dropping off a window.

Both produce a flat forecast: the same value for every future week.

**Backtesting** decides which method to trust. Both methods are fitted on the history except the last 8 weeks, forecast those 8 weeks, and are scored by mean absolute error (MAE) against what actually sold. With `method=auto` the API uses the method with the lower error. On a tie the simpler moving average wins.

### Replenishment

Replenishment works on daily demand. Days without sales count as zero.

- **Safety stock** covers demand that turns out higher than average while an order is on its way:

  `safety stock = z × σ × √L`

  where σ is the standard deviation of daily demand, L is the lead time in days, and z comes from the service level. At 95% (z ≈ 1.645) the stock should cover the lead time in 95% of cases. Higher service levels need disproportionately more stock.

- **Reorder point** is the stock level at which to place an order: expected demand during the lead time plus the safety stock.

  `reorder point = average daily demand × L + safety stock`

- **Order quantity:** available stock is what is on hand plus what is already on order. If it is above the reorder point, nothing is ordered. Otherwise the order brings stock up to the reorder point plus the demand of the next review period (7 days), rounded up to whole units.

- **Days of cover** is available stock divided by average daily demand. `/replenishment` is sorted by it, so the products closest to running out come first.

## Running

```bash
uv sync
uv run python scripts/generate_data.py
uv run uvicorn stock_planner.api.main:app --reload
```

Interactive docs: http://localhost:8000/docs

### With Docker

```bash
docker compose build
docker compose run --rm generate-data
docker compose up -d
```

The image is a two-stage build on `python:3.12-slim`. The first stage installs the locked dependencies with uv. The second copies only the resulting virtual environment, so neither uv nor any build tooling ships in the final image. The app runs as a non-root user, and a health check calls `/health`.

`data/` is mounted as a volume: read-only for the API, writable for the one-off `generate-data` service. The container user has uid 1000, so generated files belong to a typical host user. On a host where your uid differs, build with `docker compose build --build-arg UID=$(id -u) --build-arg GID=$(id -g)`.

`scripts/generate_data.py` writes synthetic data to `data/`: two years of daily sales (2024–2025) for 20 products in 2 warehouses, with weekly seasonality, per-product trends and a few slow movers, plus a stock snapshot.

### Data

| File | Columns |
|---|---|
| `sales.csv` | `date`, `sku`, `warehouse`, `quantity` |
| `stock.csv` | `sku`, `warehouse`, `on_hand`, `on_order`, `lead_time_days` |

### Configuration

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

Endpoints are plain `def`, not `async def`. DuckDB and pandas are synchronous, and FastAPI runs `def` endpoints in a thread pool so they do not block the event loop.

## Endpoints

| Method and URL | Returns |
|---|---|
| `GET /health` | `{"status": "ok"}` |
| `GET /products` | list of SKUs |
| `GET /products/{sku}/sales?warehouse=WH-01` | weekly sales |
| `GET /products/{sku}/forecast?warehouse=WH-01&method=auto&horizon=4` | forecast and the method used |
| `GET /products/{sku}/backtest?warehouse=WH-01` | MAE per method and the best one |
| `GET /replenishment?warehouse=WH-01&service_level=0.95&only_needing_order=true` | `{"items": [...], "skipped": [...]}`, most urgent first |

With fewer than 12 complete weeks a backtest is impossible, so `method=auto` falls back to the moving average. The `method` field in the response says which method was used.

A product whose history is too short to plan, such as a new product with no sales yet, does not fail the whole report. It is listed in `skipped` with the reason. When demand is zero the stock never runs out, and `days_of_cover` is `null`.

## Errors

| Status | When |
|---|---|
| 404 | `UnknownProductError`: no sales for the SKU in that warehouse |
| 404 | `UnknownWarehouseError`: no stock records for the warehouse |
| 422 | `InsufficientHistoryError`: the product exists, but its history is too short for the requested calculation |
| 422 | invalid query parameters (unknown `method`, `horizon` outside 1–26, `service_level` outside (0, 1)) |

An unknown warehouse has its own error rather than reusing `UnknownProductError`, so the message says what is actually missing.

## Design decisions

- **DuckDB for aggregation, not pandas.** Filtering, grouping by day and joining against a calendar run as one SQL query in DuckDB. It is columnar, multi-threaded and bounded by a memory limit, so the same code scales from this dataset to sales files that would not fit comfortably in a pandas DataFrame. pandas is used only for the final weekly resample on one product's series.
- **In-memory table, not a view.** The CSV files are loaded into DuckDB temp tables once, at startup. A view over `read_csv` would re-parse the files on every request. The trade-off: new data needs a restart.
- **Missing days are filled with zeros.** A day with no sales row means nothing sold, not that data is missing. Without the zeros, average demand and its standard deviation would only describe days that had sales, overstating demand and understating its variability. Each product's calendar runs from its first sale to the last date in the dataset, so a product that stopped selling shows trailing zeros.
- **Only complete weeks.** If the data ends on a Wednesday, the last "week" holds 3 days of sales and looks like a sudden drop in demand. That week weighs most in both forecasting methods, so partial first and last weeks are dropped.
- **Order quantity rounds up, after removing float noise.** You cannot order 74.65 units, and rounding down would leave stock below target, so the quantity is rounded up. Floating-point sums can land just above a whole number (`(0.1 + 0.2) * 10` is `3.0000000000000004`), and a plain `ceil` would order one extra unit. The value is rounded to 6 decimal places before `ceil`.
- **Results are frozen dataclasses, separate from the Pydantic models.** The application layer returns immutable dataclasses, the result of a calculation. The API layer has its own Pydantic models, the contract with clients. Either can change without forcing a change in the other. For example, infinite days of cover is `inf` internally and `null` in JSON.
- **DuckDB is pinned to 1.4.1.** Newer DuckDB wheels require a CPU with AVX2, which the development machine does not have. The pin is deliberate, and can be lifted when deploying to hardware with AVX2.

## Limitations and next steps

- **No yearly seasonality or trend.** Both methods produce flat forecasts, so they lag behind trends and miss seasonal peaks. Holt–Winters or a seasonal model would be the next step, once there are a few years of history.
- **Replenishment uses historical average demand, not the forecast.** The reorder point is based on the average of all daily history. Feeding the chosen forecast into it would make it react to recent changes.
- **Normal demand assumption.** The safety stock formula assumes roughly normal daily demand. Slow movers that sell nothing on most days break that assumption, and a method for intermittent demand (for example Croston) would fit them better.
- **Constant lead time.** Lead-time variability is not part of the safety stock, though late deliveries are often a larger risk than demand swings.
- **No ordering constraints.** There are no minimum order quantities, pack sizes, supplier calendars or budget limits.
- **Synthetic data.** The numbers come from a generator, not from a real business.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

GitHub Actions runs the same checks on every push to `main` and on every pull request.
