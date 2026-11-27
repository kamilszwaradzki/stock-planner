from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STOCK_PLANNER_")

    data_dir: Path = Path("data")
    duckdb_memory_limit: str = "512MB"

    @property
    def sales_path(self) -> Path:
        return self.data_dir / "sales.csv"

    @property
    def stock_path(self) -> Path:
        return self.data_dir / "stock.csv"
