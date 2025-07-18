import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

def calculate_regression_stats(x, y, regression_type):
    stats = {
        "x": x,
        "y": y,
        "regression_type": regression_type
    }
    if regression_type == "Linear":
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        y_pred = p(x)
        n = len(x)
        m, b = z[0], z[1]
        residuals = y - y_pred
        se = np.sqrt(np.sum(residuals**2) / (n - 2))
        x_mean = np.mean(x)
        stderr_slope = se / np.sqrt(np.sum((x - x_mean)**2))
        r2 = r2_score(y, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = np.mean(np.abs(residuals))
        stats.update({
            "coefficients": (m, b),
            "residuals": residuals,
            "se": se,
            "stderr_slope": stderr_slope,
            "r2": r2,
            "adj_r2": adj_r2,
            "rmse": rmse,
            "mae": mae,
            "formula": f"y = {m:.6f}x + {b:.6f}"
        })
    elif regression_type == "Exponential":
        mask = y > 0
        x_clean, y_clean = x[mask], y[mask]
        if len(x_clean) < 2:
            return None
        z = np.polyfit(x_clean, np.log(y_clean), 1)
        p = np.poly1d(z)
        y_pred = np.exp(p(x_clean))
        r2 = r2_score(y_clean, y_pred)
        rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
        mae = np.mean(np.abs(y_clean - y_pred))
        stats.update({
            "coefficients": (z[1], z[0]),
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "formula": f"y = {np.exp(z[1]):.6f} * e^({z[0]:.6f}x)"
        })
    elif regression_type == "Logarithmic":
        mask = x > 0
        x_clean, y_clean = x[mask], y[mask]
        if len(x_clean) < 2:
            return None
        z = np.polyfit(np.log(x_clean), y_clean, 1)
        p = np.poly1d(z)
        y_pred = p(np.log(x_clean))
        r2 = r2_score(y_clean, y_pred)
        rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
        mae = np.mean(np.abs(y_clean - y_pred))
        stats.update({
            "coefficients": (z[0], z[1]),
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "formula": f"y = {z[0]:.6f} * ln(x) + {z[1]:.6f}"
        })
    return stats 