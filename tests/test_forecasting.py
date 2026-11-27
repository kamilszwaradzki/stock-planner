import pytest

from stock_planner.domain.errors import InsufficientHistoryError
from stock_planner.domain.forecasting import (
    ForecastMethod,
    backtest,
    best_method,
    exponential_smoothing_forecast,
    mean_absolute_error,
    moving_average_forecast,
)


def test_best_method_picks_lower_error() -> None:
    history = [10, 20, 30, 40, 50, 60, 70, 80]

    assert (
        best_method(history, holdout=2, window=4, alpha=0.9)
        == ForecastMethod.EXPONENTIAL_SMOOTHING
    )
    assert (
        best_method(history, holdout=2, window=4, alpha=0.3)
        == ForecastMethod.MOVING_AVERAGE
    )


def test_best_method_prefers_moving_average_on_tie() -> None:
    history = [10.0] * 12

    assert best_method(history, holdout=4, window=4) == ForecastMethod.MOVING_AVERAGE


def test_backtest_known_values() -> None:
    history = [10, 20, 30, 40, 50, 60, 70, 80]

    errors = backtest(history, holdout=2, window=4, alpha=0.9)

    assert errors[ForecastMethod.MOVING_AVERAGE] == pytest.approx(30.0)
    assert errors[ForecastMethod.EXPONENTIAL_SMOOTHING] == pytest.approx(
        16.1111, rel=1e-4
    )


def test_backtest_rejects_too_short_history() -> None:
    with pytest.raises(InsufficientHistoryError):
        backtest([1, 2, 3, 4, 5], holdout=2, window=4)


def test_moving_average_forecast_known_values() -> None:
    result = moving_average_forecast(
        [10, 20, 30, 40, 50],
        window=3,
        horizon=2,
    )

    assert result.tolist() == [40.0, 40.0]


def test_moving_average_forecast_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="window and horizon"):
        moving_average_forecast([10, 20, 30], window=0)


def test_moving_average_forecast_rejects_invalid_horizon() -> None:
    with pytest.raises(ValueError, match="window and horizon"):
        moving_average_forecast([10, 20, 30], horizon=0)


def test_moving_average_forecast_rejects_insufficient_history() -> None:
    with pytest.raises(InsufficientHistoryError):
        moving_average_forecast([10, 20], window=3)


def test_exponential_smoothing_forecast_known_values() -> None:
    result = exponential_smoothing_forecast(
        [10, 20, 30],
        alpha=0.5,
        horizon=2,
    )

    assert result.tolist() == [22.5, 22.5]


def test_exponential_smoothing_forecast_alpha_one_returns_last_value() -> None:
    result = exponential_smoothing_forecast(
        [10, 20, 30],
        alpha=1.0,
    )

    assert result.tolist() == [30.0] * 4


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1])
def test_exponential_smoothing_forecast_rejects_invalid_alpha(
    alpha: float,
) -> None:
    with pytest.raises(ValueError, match="alpha"):
        exponential_smoothing_forecast([10, 20, 30], alpha=alpha)


def test_exponential_smoothing_forecast_rejects_empty_history() -> None:
    with pytest.raises(InsufficientHistoryError):
        exponential_smoothing_forecast([])


def test_mean_absolute_error_known_values() -> None:
    result = mean_absolute_error(
        actual=[10, 20, 30],
        predicted=[12, 18, 33],
    )

    assert result == pytest.approx(2.333333)


def test_mean_absolute_error_rejects_different_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        mean_absolute_error(
            actual=[10, 20, 30],
            predicted=[10, 20],
        )


def test_mean_absolute_error_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        mean_absolute_error([], [])
