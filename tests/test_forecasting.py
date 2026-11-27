import pytest

from stock_planner.domain.errors import InsufficientHistoryError
from stock_planner.domain.forecasting import ForecastMethod, backtest, best_method


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
    history = [10.0] * 12  # stała sprzedaż: obie metody trafiają idealnie, MAE = 0

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
        backtest([1, 2, 3, 4, 5], holdout=2, window=4)  # potrzeba 6, jest 5
