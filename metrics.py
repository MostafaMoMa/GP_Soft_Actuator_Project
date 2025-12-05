# metrics.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def negative_log_likelihood_gaussian(y_true, y_pred, y_std):
    """
    Average negative log likelihood under N(y_pred, y_std^2).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_var = np.asarray(y_std) ** 2
    return 0.5 * np.mean(
        np.log(2 * np.pi * y_var) + ((y_true - y_pred) ** 2) / y_var
    )


def absolute_errors_list(y_true, y_pred):
    """
    Requirement Part 2: comprehension for at least one data type.
    Returns list[float] of absolute errors.
    """
    return [abs(float(t) - float(p)) for t, p in zip(y_true, y_pred)]
