# data_handler.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_pwm_force_data(
    csv_path: str,
    sample_start: int = 1,
    sample_step: int = 2,
    test_ratio: float = 0.2,
    random_state: int = 42,
):
    """
    Load PWM & force data from CSV, build (force_t, PWM_t) -> force_{t+1} dataset,
    and return train/test splits with scalers.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, PWM, force
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {csv_path}")
        raise e
    except pd.errors.EmptyDataError as e:
        print(f"[ERROR] File is empty: {csv_path}")
        raise e

    # Mutable objects: DataFrame, ndarray, list (Requirement Part 2)
    df_sampled = df.iloc[sample_start:600:sample_step]

    PWM = df_sampled.iloc[:, 5].to_numpy()
    force = df_sampled.iloc[:, 6].to_numpy() / 100.0

    assert len(PWM) == len(force), "PWM and Force must have the same length"
    np.random.seed(random_state)

    # Build supervised dataset: (force_t, PWM_t) -> force_{t+1}
    X_raw = np.column_stack((force[:-1], PWM[:-1]))
    y_raw = force[1:]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_full = scaler_X.fit_transform(X_raw)
    y_full = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

    N = len(X_full)
    split_idx = int((1.0 - test_ratio) * N)

    X_train, X_test = X_full[:split_idx], X_full[split_idx:]
    y_train, y_test = y_full[:split_idx], y_full[split_idx:]

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, PWM, force
