from pathlib import Path

import duckdb
import pandas as pd

DAYS_PER_WEEK = 7


class SalesRepository:
    def __init__(
        self,
        sales_path: Path,
        stock_path: Path,
        memory_limit: str = "512MB",
    ) -> None:
        self._con: duckdb.DuckDBPyConnection = duckdb.connect(
            config={"memory_limit": memory_limit}
        )

        self._con.execute(
            """
            CREATE TEMP TABLE sales AS
            SELECT *
            FROM read_csv(
                ?,
                columns = {
                    'date': 'DATE',
                    'sku': 'VARCHAR',
                    'warehouse': 'VARCHAR',
                    'quantity': 'INTEGER'
                }
            )
            """,
            [str(sales_path)],
        )

        self._con.execute(
            """
            CREATE TEMP TABLE stock AS
            SELECT *
            FROM read_csv(
                ?,
                columns = {
                    'sku': 'VARCHAR',
                    'warehouse': 'VARCHAR',
                    'on_hand': 'INTEGER',
                    'on_order': 'INTEGER',
                    'lead_time_days': 'INTEGER'
                }
            )
            """,
            [str(stock_path)],
        )

    def list_skus(self) -> list[str]:
        result = self._con.execute(
            """
            SELECT DISTINCT sku
            FROM sales
            ORDER BY sku
            """
        ).df()

        return result["sku"].tolist()

    def daily_demand(
        self,
        sku: str,
        warehouse: str,
    ) -> pd.Series:
        result = self._con.execute(
            """
            WITH filtered_sales AS (
                SELECT
                    date,
                    SUM(quantity) AS quantity
                FROM sales
                WHERE sku = ?
                  AND warehouse = ?
                GROUP BY date
            ),
            date_range AS (
                SELECT
                    (SELECT MIN(date) FROM filtered_sales) AS start_date,
                    (SELECT MAX(date) FROM sales) AS end_date
            ),
            calendar AS (
                SELECT generated_date::DATE AS date
                FROM date_range,
                generate_series(
                    start_date,
                    end_date,
                    INTERVAL 1 DAY
                ) AS dates(generated_date)
            )
            SELECT
                calendar.date,
                COALESCE(filtered_sales.quantity, 0)::BIGINT
                    AS quantity
            FROM calendar
            LEFT JOIN filtered_sales
                ON calendar.date = filtered_sales.date
            ORDER BY calendar.date
            """,
            [sku, warehouse],
        ).df()

        if result.empty:
            return pd.Series(
                dtype="int64",
                name="quantity",
                index=pd.DatetimeIndex([], name="date"),
            )

        return pd.Series(
            result["quantity"].to_numpy(dtype="int64"),
            index=pd.DatetimeIndex(result["date"], name="date"),
            name="quantity",
        )

    def weekly_sales(
        self,
        sku: str,
        warehouse: str,
    ) -> pd.DataFrame:
        """Sum daily demand into Monday-to-Sunday weeks, keeping only full weeks."""
        daily = self.daily_demand(sku, warehouse)

        weeks = daily.resample(
            "W-MON",
            label="left",
            closed="left",
        )
        weekly = weeks.sum()[weeks.count() == DAYS_PER_WEEK]

        return weekly.rename_axis("week").reset_index(name="quantity")

    def stock_levels(self, warehouse: str) -> pd.DataFrame:
        return self._con.execute(
            """
            SELECT
                sku,
                warehouse,
                on_hand,
                on_order,
                lead_time_days
            FROM stock
            WHERE warehouse = ?
            ORDER BY sku
            """,
            [warehouse],
        ).df()

    def close(self) -> None:
        self._con.close()
