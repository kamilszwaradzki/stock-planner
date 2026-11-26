from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

SKUS = [f"SKU-{i:03d}" for i in range(1, 21)]
WAREHOUSES = ["WH-01", "WH-02"]

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

NUM_SKUS = len(SKUS)
NUM_WAREHOUSES = len(WAREHOUSES)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def generate_sales(rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    num_days = len(dates)

    base_demand = rng.integers(3, 50, size=NUM_SKUS)

    slow_moving = rng.choice(NUM_SKUS, size=3, replace=False)
    base_demand[slow_moving] = rng.integers(1, 4, size=3)

    weekly_seasonality = np.array([0.9, 0.95, 1.0, 1.0, 1.05, 1.3, 1.3])

    trends = rng.uniform(-0.10, 0.15, size=NUM_SKUS)

    weekdays = dates.day_of_week

    weekly_factor = weekly_seasonality[weekdays][:, None]

    product_trend = 1 + np.linspace(0, 1, num_days)[:, None] * trends[None, :]

    expected_sales = base_demand[None, :] * weekly_factor * product_trend

    total_sales = rng.poisson(expected_sales)

    slow_mask = rng.random((num_days, len(slow_moving))) < 0.65

    total_sales[:, slow_moving] = np.where(
        slow_mask,
        0,
        total_sales[:, slow_moving],
    )
    base_shares = rng.dirichlet(np.full(NUM_WAREHOUSES, 5.0), size=NUM_SKUS)

    daily_noise = rng.normal(
        0,
        0.025,
        size=(num_days, NUM_SKUS, NUM_WAREHOUSES),
    )

    shares = np.clip(
        base_shares[None, :, :] + daily_noise,
        0.01,
        None,
    )

    shares /= shares.sum(axis=-1, keepdims=True)

    warehouse_sales = rng.multinomial(total_sales, shares)

    return pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), NUM_SKUS * NUM_WAREHOUSES),
            "sku": np.tile(np.repeat(SKUS, NUM_WAREHOUSES), num_days),
            "warehouse": np.tile(WAREHOUSES, num_days * NUM_SKUS),
            "quantity": warehouse_sales.reshape(-1),
        }
    )


def generate_stock(
    rng: np.random.Generator,
    sales: pd.DataFrame,
) -> pd.DataFrame:
    size = NUM_SKUS * NUM_WAREHOUSES

    index = pd.MultiIndex.from_product(
        [SKUS, WAREHOUSES],
        names=["sku", "warehouse"],
    )

    avg_daily_demand = (
        sales.groupby(["sku", "warehouse"])["quantity"].mean().reindex(index).to_numpy()
    )

    stock_coverage_days = rng.uniform(0, 45, size=size)

    on_hand = np.rint(avg_daily_demand * stock_coverage_days).astype(int)

    stockout_indices = rng.choice(
        size,
        size=min(2, size),
        replace=False,
    )

    on_hand[stockout_indices] = 0

    on_order_coverage_days = rng.uniform(0, 15, size=size)

    on_order = np.rint(avg_daily_demand * on_order_coverage_days).astype(int)

    lead_time_days = rng.integers(2, 31, size=size)

    return pd.DataFrame(
        {
            "sku": np.repeat(SKUS, NUM_WAREHOUSES),
            "warehouse": np.tile(WAREHOUSES, NUM_SKUS),
            "on_hand": on_hand,
            "on_order": on_order,
            "lead_time_days": lead_time_days,
        }
    )


def main() -> None:
    rng = np.random.default_rng(seed=SEED)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    sales = generate_sales(rng)
    stock = generate_stock(rng, sales)

    sales.to_csv(DATA_DIR / "sales.csv", index=False)
    stock.to_csv(DATA_DIR / "stock.csv", index=False)

    print(f"Generated {len(sales)} sales records.")
    print(f"Generated {len(stock)} stock records.")
    print(f"Saved files to: {DATA_DIR}")


if __name__ == "__main__":
    main()
