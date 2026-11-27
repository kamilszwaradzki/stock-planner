from collections.abc import Iterator

import pytest
from conftest import WAREHOUSE
from fastapi.testclient import TestClient

from stock_planner.api.dependencies import get_repository
from stock_planner.api.main import app
from stock_planner.infrastructure.sales_repository import SalesRepository


@pytest.fixture
def client(planning_repository: SalesRepository) -> Iterator[TestClient]:
    app.dependency_overrides[get_repository] = lambda: planning_repository
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_products(client: TestClient) -> None:
    response = client.get("/products")

    assert response.status_code == 200
    assert response.json() == ["SKU-001", "SKU-002", "SKU-003", "SKU-004"]


def test_sales_returns_weekly_points(client: TestClient) -> None:
    response = client.get("/products/SKU-001/sales", params={"warehouse": WAREHOUSE})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 12
    assert body[0] == {"week": "2024-01-01", "quantity": 70}


def test_sales_unknown_sku_returns_404(client: TestClient) -> None:
    response = client.get("/products/UNKNOWN/sales", params={"warehouse": WAREHOUSE})

    assert response.status_code == 404


def test_forecast_returns_horizon_values(client: TestClient) -> None:
    response = client.get(
        "/products/SKU-001/forecast",
        params={"warehouse": WAREHOUSE, "method": "moving_average", "horizon": 6},
    )

    assert response.status_code == 200
    assert response.json() == {
        "sku": "SKU-001",
        "warehouse": WAREHOUSE,
        "method": "moving_average",
        "forecast": [70.0] * 6,
    }


def test_forecast_defaults_to_auto_with_four_weeks(client: TestClient) -> None:
    response = client.get("/products/SKU-001/forecast", params={"warehouse": WAREHOUSE})

    assert response.status_code == 200
    assert response.json()["method"] == "moving_average"
    assert len(response.json()["forecast"]) == 4


def test_forecast_unknown_sku_returns_404(client: TestClient) -> None:
    response = client.get("/products/UNKNOWN/forecast", params={"warehouse": WAREHOUSE})

    assert response.status_code == 404
    assert "UNKNOWN" in response.json()["detail"]


def test_forecast_short_history_returns_422(client: TestClient) -> None:
    response = client.get("/products/SKU-002/forecast", params={"warehouse": WAREHOUSE})

    assert response.status_code == 422
    assert "12 weeks" in response.json()["detail"]


def test_forecast_missing_warehouse_returns_422(client: TestClient) -> None:
    response = client.get("/products/SKU-001/forecast")

    assert response.status_code == 422


def test_forecast_invalid_method_returns_422(client: TestClient) -> None:
    response = client.get(
        "/products/SKU-001/forecast",
        params={"warehouse": WAREHOUSE, "method": "magic"},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "method"]


@pytest.mark.parametrize("horizon", [0, 27])
def test_forecast_horizon_out_of_range_returns_422(
    client: TestClient, horizon: int
) -> None:
    response = client.get(
        "/products/SKU-001/forecast",
        params={"warehouse": WAREHOUSE, "horizon": horizon},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "horizon"]


def test_backtest_returns_errors_and_best_method(client: TestClient) -> None:
    response = client.get("/products/SKU-001/backtest", params={"warehouse": WAREHOUSE})

    assert response.status_code == 200
    assert response.json() == {
        "sku": "SKU-001",
        "warehouse": WAREHOUSE,
        "errors": {"moving_average": 0.0, "exponential_smoothing": 0.0},
        "best_method": "moving_average",
    }


def test_replenishment_lists_most_urgent_first(client: TestClient) -> None:
    response = client.get("/replenishment", params={"warehouse": WAREHOUSE})

    assert response.status_code == 200
    body = response.json()
    assert [item["sku"] for item in body] == ["SKU-001", "SKU-002"]
    assert body[0] == {
        "sku": "SKU-001",
        "average_daily_demand": 10.0,
        "safety_stock": 0.0,
        "reorder_point": 70.0,
        "available_stock": 0,
        "order_quantity": 140,
        "needs_order": True,
        "days_of_cover": 0.0,
    }


def test_replenishment_full_overview_serializes_infinite_cover_as_null(
    client: TestClient,
) -> None:
    response = client.get(
        "/replenishment",
        params={"warehouse": WAREHOUSE, "only_needing_order": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert [item["sku"] for item in body] == [
        "SKU-001",
        "SKU-002",
        "SKU-003",
        "SKU-004",
    ]
    assert body[-1]["days_of_cover"] is None
    assert body[-1]["needs_order"] is False


def test_replenishment_unknown_warehouse_returns_404(client: TestClient) -> None:
    response = client.get("/replenishment", params={"warehouse": "WH-99"})

    assert response.status_code == 404


@pytest.mark.parametrize("service_level", [0, 1, 1.5, -0.1])
def test_replenishment_invalid_service_level_returns_422(
    client: TestClient, service_level: float
) -> None:
    response = client.get(
        "/replenishment",
        params={"warehouse": WAREHOUSE, "service_level": service_level},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "service_level"]


def test_sales_returns_only_full_weeks(client: TestClient) -> None:
    response = client.get("/products/SKU-002/sales", params={"warehouse": WAREHOUSE})

    assert response.status_code == 200
    assert response.json() == [{"week": "2024-03-18", "quantity": 35}]
