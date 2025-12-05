# visualization.py
import numpy as np
import matplotlib.pyplot as plt


def plot_pwm_force(PWM, force):
    plt.figure(figsize=(12, 5))

    plt.subplot(2, 1, 1)
    plt.plot(PWM, label="PWM Input")
    plt.title("PWM Signal Over Time")
    plt.ylabel("PWM")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(force, label="Force")
    plt.title("Force Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Force")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_gp_prediction_vs_truth(y_true, y_pred, y_std, title_suffix="Δforce"):
    timesteps = np.arange(len(y_pred))

    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label=f"True {title_suffix}", linewidth=2)
    plt.plot(y_pred, label=f"Predicted {title_suffix}", linewidth=2)
    plt.fill_between(
        timesteps,
        y_pred - 2 * y_std,
        y_pred + 2 * y_std,
        alpha=0.3,
        label="±2σ",
    )
    plt.legend()
    plt.title("GP Prediction vs Ground Truth")
    plt.xlabel("Test Time Step")
    plt.ylabel(title_suffix)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_gp_confidence(y_true, y_pred, y_std, title_suffix="Δforce"):
    idx = np.arange(len(y_pred))

    plt.figure(figsize=(10, 5))
    plt.plot(y_pred, linewidth=2, label="Predicted Mean")
    plt.fill_between(idx, y_pred - 2 * y_std, y_pred + 2 * y_std,
                     alpha=0.4, label="95% CI")
    plt.scatter(idx, y_true, s=30, label="Test Data")

    plt.title("GP Prediction with Confidence")
    plt.xlabel("Time Step")
    plt.ylabel(title_suffix)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
