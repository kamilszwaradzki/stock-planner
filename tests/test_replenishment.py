from dataclasses import FrozenInstanceError

import pytest

from stock_planner.domain.errors import InsufficientHistoryError
from stock_planner.domain.replenishment import (
    ReplenishmentPlan,
    plan_replenishment,
    recommended_order_quantity,
    reorder_point,
    safety_stock,
)

DEMAND = [8, 10, 10, 10, 12]


def test_safety_stock_is_zero_for_constant_demand() -> None:
    assert safety_stock([10.0] * 14, lead_time_days=5) == pytest.approx(0.0)


def test_safety_stock_known_value() -> None:
    assert safety_stock(DEMAND, lead_time_days=4) == pytest.approx(4.6523, rel=1e-4)


def test_higher_service_level_gives_higher_safety_stock() -> None:
    low = safety_stock(DEMAND, lead_time_days=4, service_level=0.90)
    medium = safety_stock(DEMAND, lead_time_days=4, service_level=0.95)
    high = safety_stock(DEMAND, lead_time_days=4, service_level=0.99)

    assert high > medium > low


@pytest.mark.parametrize("service_level", [0, 1, 1.5, -0.1])
def test_safety_stock_rejects_invalid_service_level(service_level: float) -> None:
    with pytest.raises(ValueError, match="service_level"):
        safety_stock(DEMAND, lead_time_days=4, service_level=service_level)


@pytest.mark.parametrize("lead_time_days", [0, -1])
def test_safety_stock_rejects_invalid_lead_time(lead_time_days: int) -> None:
    with pytest.raises(ValueError, match="lead_time_days"):
        safety_stock(DEMAND, lead_time_days=lead_time_days)


@pytest.mark.parametrize("history", [[], [10.0]])
def test_safety_stock_needs_at_least_two_days(history: list[float]) -> None:
    with pytest.raises(InsufficientHistoryError):
        safety_stock(history, lead_time_days=4)


def test_reorder_point_known_value() -> None:
    assert reorder_point(DEMAND, lead_time_days=4) == pytest.approx(44.6523, rel=1e-4)


def test_reorder_point_delegates_validation_to_safety_stock() -> None:
    with pytest.raises(InsufficientHistoryError):
        reorder_point([10.0], lead_time_days=4)
    with pytest.raises(ValueError, match="service_level"):
        reorder_point(DEMAND, lead_time_days=4, service_level=1)


def test_no_order_when_stock_above_reorder_point() -> None:
    quantity = recommended_order_quantity(
        on_hand=60, on_order=0, reorder_level=50.0, review_period_demand=70.0
    )

    assert quantity == 0


def test_order_quantity_known_value() -> None:
    quantity = recommended_order_quantity(
        on_hand=40, on_order=0, reorder_level=50.0, review_period_demand=70.0
    )

    assert quantity == 80


def test_on_order_counts_towards_available_stock() -> None:
    without_on_order = recommended_order_quantity(
        on_hand=30, on_order=0, reorder_level=50.0, review_period_demand=70.0
    )
    with_on_order = recommended_order_quantity(
        on_hand=30, on_order=10, reorder_level=50.0, review_period_demand=70.0
    )
    covered_by_on_order = recommended_order_quantity(
        on_hand=30, on_order=30, reorder_level=50.0, review_period_demand=70.0
    )

    assert without_on_order == 90
    assert with_on_order == 80
    assert covered_by_on_order == 0


def test_order_quantity_rounds_up() -> None:
    quantity = recommended_order_quantity(
        on_hand=40, on_order=0, reorder_level=44.6523, review_period_demand=70.0
    )

    assert quantity == 75


def test_order_quantity_triggers_order_from_zero_stock() -> None:
    quantity = recommended_order_quantity(
        on_hand=0, on_order=0, reorder_level=0.1 + 0.2, review_period_demand=0.0
    )

    assert quantity == 1


def test_order_quantity_ignores_float_noise_before_rounding_up() -> None:
    quantity = recommended_order_quantity(
        on_hand=0, on_order=0, reorder_level=(0.1 + 0.2) * 10, review_period_demand=0.0
    )

    assert quantity == 3


@pytest.mark.parametrize(
    ("on_hand", "on_order", "reorder_level", "review_period_demand", "field"),
    [
        (-1, 0, 50.0, 70.0, "on_hand"),
        (0, -1, 50.0, 70.0, "on_order"),
        (0, 0, -0.1, 70.0, "reorder_level"),
        (0, 0, 50.0, -0.1, "review_period_demand"),
    ],
)
def test_order_quantity_rejects_negative_values(
    on_hand: int,
    on_order: int,
    reorder_level: float,
    review_period_demand: float,
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        recommended_order_quantity(
            on_hand=on_hand,
            on_order=on_order,
            reorder_level=reorder_level,
            review_period_demand=review_period_demand,
        )


def test_plan_replenishment_full_plan() -> None:
    plan = plan_replenishment(
        DEMAND,
        on_hand=30,
        on_order=10,
        lead_time_days=4,
        service_level=0.95,
        review_period_days=7,
    )

    assert isinstance(plan, ReplenishmentPlan)
    assert plan.average_daily_demand == pytest.approx(10.0)
    assert plan.safety_stock == pytest.approx(4.6523, rel=1e-4)
    assert plan.reorder_point == pytest.approx(44.6523, rel=1e-4)
    assert plan.available_stock == 40

    assert plan.order_quantity == 75
    assert plan.needs_order is True


def test_plan_replenishment_without_order() -> None:
    plan = plan_replenishment(DEMAND, on_hand=50, on_order=0, lead_time_days=4)

    assert plan.order_quantity == 0
    assert plan.needs_order is False


def test_plan_is_immutable() -> None:
    plan = plan_replenishment(DEMAND, on_hand=30, on_order=10, lead_time_days=4)

    with pytest.raises(FrozenInstanceError):
        plan.order_quantity = 1


def test_plan_replenishment_rejects_zero_review_period() -> None:
    with pytest.raises(ValueError, match="review_period_days"):
        plan_replenishment(
            DEMAND, on_hand=30, on_order=10, lead_time_days=4, review_period_days=0
        )


@pytest.mark.parametrize(
    ("on_hand", "on_order", "field"),
    [(-1, 0, "on_hand"), (0, -1, "on_order")],
)
def test_plan_replenishment_rejects_negative_stock(
    on_hand: int, on_order: int, field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        plan_replenishment(DEMAND, on_hand=on_hand, on_order=on_order, lead_time_days=4)


def test_plan_replenishment_rejects_short_history() -> None:
    with pytest.raises(InsufficientHistoryError):
        plan_replenishment([10.0], on_hand=30, on_order=10, lead_time_days=4)
