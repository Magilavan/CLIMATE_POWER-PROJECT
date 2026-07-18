"""
baselines.py — floor models to compare the LSTM against.

Models:
1. Naive (lag-1)
2. Seasonal naive (weekly, k=7)
3. Seasonal naive (annual, k=365)
4. Linear regression: lag-1, lag-7, rolling-7 mean, temperature, humidity
5. Fourier + exogenous regression ("SARIMAX-style" seasonal baseline):
   uses annual + weekly Fourier terms (sin/cos pairs) plus temperature and
   humidity as regressors, fit with linear regression on the residual
   structure. This replaces a literal SARIMAX(m=365) fit, which is
   computationally impractical on ~6000 daily points and — as observed
   in an earlier run of this project with m=7 — fails to capture annual
   seasonality at all if the seasonal period is set too short. Fourier
   terms capture long seasonal cycles far more efficiently for daily data.
   (statsmodels SARIMAX is provided separately in src/sarimax_optional.py
   for users who have statsmodels installed and want the literal model.)

Target: peak_demand_mw
Horizon: next 7 days (i.e., predict t+7 using information available at t)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET = "peak_demand_mw"
HORIZON = 7

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"


def load():
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    val = pd.read_csv(DATA_DIR / "val.csv", parse_dates=["date"])
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    return train, val, test


def metrics(actual, pred):
    actual, pred = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    mask = ~np.isnan(pred) & ~np.isnan(actual)
    actual, pred = actual[mask], pred[mask]
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mae = np.mean(np.abs(actual - pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def naive_lag1(full_df, test_index):
    # forecast for t+HORIZON = value at t (persistence)
    return full_df[TARGET].shift(HORIZON).loc[test_index]


def seasonal_naive(full_df, test_index, k):
    return full_df[TARGET].shift(k).loc[test_index]


def linreg_baseline(full_df, train_idx, test_idx):
    feat = pd.DataFrame(index=full_df.index)
    feat["lag1"] = full_df[TARGET].shift(1)
    feat["lag7"] = full_df[TARGET].shift(7)
    feat["roll7"] = full_df[TARGET].shift(1).rolling(7).mean()
    feat["temperature"] = full_df["temperature_2m_c"]
    feat["humidity"] = full_df["humidity_2m_pct"]
    feat["target"] = full_df[TARGET].shift(-HORIZON)  # predict t+HORIZON
    feat = feat.dropna()

    train_mask = feat.index.isin(train_idx)
    test_mask = feat.index.isin(test_idx)

    X_train = feat.loc[train_mask, ["lag1", "lag7", "roll7", "temperature", "humidity"]]
    y_train = feat.loc[train_mask, "target"]
    X_test = feat.loc[test_mask, ["lag1", "lag7", "roll7", "temperature", "humidity"]]

    model = LinearRegression().fit(X_train, y_train)
    preds = pd.Series(model.predict(X_test), index=feat.loc[test_mask].index)
    return preds, model


def fourier_regression_baseline(full_df, train_idx, test_idx):
    doy = full_df["date"].dt.dayofyear.values
    dow = full_df["date"].dt.dayofweek.values
    feat = pd.DataFrame(index=full_df.index)
    feat["sin365"] = np.sin(2 * np.pi * doy / 365.25)
    feat["cos365"] = np.cos(2 * np.pi * doy / 365.25)
    feat["sin182"] = np.sin(4 * np.pi * doy / 365.25)
    feat["cos182"] = np.cos(4 * np.pi * doy / 365.25)
    feat["sin7"] = np.sin(2 * np.pi * dow / 7)
    feat["cos7"] = np.cos(2 * np.pi * dow / 7)
    feat["trend"] = np.arange(len(full_df))  # linear trend to capture multi-year demand growth
    feat["temperature"] = full_df["temperature_2m_c"]
    feat["humidity"] = full_df["humidity_2m_pct"]
    feat["lag1"] = full_df[TARGET].shift(1)
    feat["target"] = full_df[TARGET].shift(-HORIZON)
    feat = feat.dropna()

    train_mask = feat.index.isin(train_idx)
    test_mask = feat.index.isin(test_idx)
    cols = ["sin365", "cos365", "sin182", "cos182", "sin7", "cos7", "trend", "temperature", "humidity", "lag1"]

    X_train, y_train = feat.loc[train_mask, cols], feat.loc[train_mask, "target"]
    X_test = feat.loc[test_mask, cols]

    model = LinearRegression().fit(X_train, y_train)
    preds = pd.Series(model.predict(X_test), index=feat.loc[test_mask].index)
    return preds, model


def main():
    train, val, test = load()
    full = pd.concat([train, val, test]).reset_index(drop=True)
    full = full.sort_values("date").reset_index(drop=True)
    test_idx = full.index[full["date"].isin(test["date"])]
    train_idx = full.index[full["date"].isin(train["date"])]

    results = {}
    preds_dict = {}

    preds = naive_lag1(full, test_idx)
    preds_dict["Naive (lag-1)"] = preds
    results["Naive (lag-1)"] = metrics(full.loc[test_idx, TARGET], preds)

    preds = seasonal_naive(full, test_idx, 7)
    preds_dict["Seasonal Naive (weekly, k=7)"] = preds
    results["Seasonal Naive (weekly, k=7)"] = metrics(full.loc[test_idx, TARGET], preds)

    preds = seasonal_naive(full, test_idx, 365)
    preds_dict["Seasonal Naive (annual, k=365)"] = preds
    results["Seasonal Naive (annual, k=365)"] = metrics(full.loc[test_idx, TARGET], preds)

    preds, lr_model = linreg_baseline(full, train_idx, test_idx)
    preds_dict["Linear Regression (lag1,lag7,roll7,temp,humid)"] = preds
    results["Linear Regression (lag1,lag7,roll7,temp,humid)"] = metrics(
        full.loc[preds.index, TARGET], preds
    )

    preds, fourier_model = fourier_regression_baseline(full, train_idx, test_idx)
    preds_dict["Fourier+Exog Regression (annual+weekly seasonality)"] = preds
    results["Fourier+Exog Regression (annual+weekly seasonality)"] = metrics(
        full.loc[preds.index, TARGET], preds
    )

    # ---- comparison table ----
    table = pd.DataFrame(results).T
    table = table[["RMSE", "MAE", "MAPE", "R2"]].round(3)
    print(table.to_string())
    table.to_csv(REPORT_DIR / "baseline_comparison.csv")

    # ---- chart ----
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(full.loc[test_idx, "date"], full.loc[test_idx, TARGET], label="Actual", color="black", linewidth=1.5)
    for name, preds in preds_dict.items():
        ax.plot(full.loc[preds.index, "date"], preds, label=name, alpha=0.75, linewidth=1)
    ax.set_title("Tamil Nadu Peak Demand — Baseline Forecasts vs Actual (Test Period)")
    ax.set_ylabel("Peak Demand (MW)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "baseline_comparison.png", dpi=130)
    print(f"\nSaved chart to {OUT_DIR / 'baseline_comparison.png'}")
    print(f"Saved table to {REPORT_DIR / 'baseline_comparison.csv'}")

    return table


if __name__ == "__main__":
    main()
