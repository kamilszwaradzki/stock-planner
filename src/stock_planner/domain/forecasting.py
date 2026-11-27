from collections.abc import Sequence
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from stock_planner.domain.errors import InsufficientHistoryError

FloatArray = npt.NDArray[np.float64]
History = Sequence[float] | FloatArray


class ForecastMethod(StrEnum):
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"


def moving_average_forecast(
    history: History,
    window: int = 4,
    horizon: int = 4,
) -> FloatArray:
    if window < 1 or horizon < 1:
        raise ValueError("window and horizon must be at least 1")

    values = np.asarray(history, dtype=np.float64)

    if len(values) < window:
        raise InsufficientHistoryError(
            f"Moving average needs at least {window} weeks, got {len(values)}"
        )

    level = values[-window:].mean()

    return np.full(horizon, level)


def exponential_smoothing_forecast(
    history: History,
    alpha: float = 0.3,
    horizon: int = 4,
) -> FloatArray:
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in the range (0, 1]")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    values = np.asarray(history, dtype=np.float64)

    if len(values) == 0:
        raise InsufficientHistoryError("Exponential smoothing needs at least 1 week")

    level = values[0]
    for value in values[1:]:
        level = alpha * value + (1 - alpha) * level

    return np.full(horizon, level)


def mean_absolute_error(actual: History, predicted: History) -> float:
    actual_values = np.asarray(actual, dtype=np.float64)
    predicted_values = np.asarray(predicted, dtype=np.float64)

    if actual_values.size == 0:
        raise ValueError("actual and predicted must not be empty")

    if actual_values.shape != predicted_values.shape:
        raise ValueError("actual and predicted must have the same length")

    return float(np.mean(np.abs(actual_values - predicted_values)))


def backtest(
    history: History,
    holdout: int = 8,
    window: int = 4,
    alpha: float = 0.3,
) -> dict[ForecastMethod, float]:
    """Compare forecasting methods on the last `holdout` weeks.

    The methods are fitted only on data before the holdout period,
    so they are evaluated on weeks they have never seen.
    """
    values = np.asarray(history, dtype=np.float64)

    if holdout < 1:
        raise ValueError("holdout must be at least 1")
    if len(values) < holdout + window:
        raise InsufficientHistoryError(
            f"Backtest needs at least {holdout + window} weeks, got {len(values)}"
        )

    train = values[:-holdout]
    actual = values[-holdout:]

    moving_average = moving_average_forecast(train, window=window, horizon=holdout)
    smoothing = exponential_smoothing_forecast(train, alpha=alpha, horizon=holdout)

    return {
        ForecastMethod.MOVING_AVERAGE: mean_absolute_error(actual, moving_average),
        ForecastMethod.EXPONENTIAL_SMOOTHING: mean_absolute_error(actual, smoothing),
    }


def best_method(
    history: History,
    holdout: int = 8,
    window: int = 4,
    alpha: float = 0.3,
) -> ForecastMethod:
    """Return the method with the lower backtest error.

    On a tie the moving average wins, because it is the simpler method.
    """
    errors = backtest(history, holdout=holdout, window=window, alpha=alpha)

    if (
        errors[ForecastMethod.EXPONENTIAL_SMOOTHING]
        < errors[ForecastMethod.MOVING_AVERAGE]
    ):
        return ForecastMethod.EXPONENTIAL_SMOOTHING
    return ForecastMethod.MOVING_AVERAGE
