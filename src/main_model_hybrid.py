"""
main_model_hybrid.py — Fourier+trend (captures smooth seasonal/growth structure)
+ Gradient Boosting on residuals (captures nonlinear climate effects the linear
stage misses, e.g. heat-wave spikes, humidity interactions).

This two-stage design exists because two things were empirically observed on
this dataset while building this project:
1. A pure Fourier+trend linear regression outperformed a pure Gradient
   Boosting model on peak_demand_mw — tree models can't extrapolate a
   raw trend feature, and this dataset's dominant signal IS a rising trend
   plus a smooth annual cycle, which linear regression fits well natively.
2. However a pure linear model can't capture nonlinear effects like sharp
   heat-wave demand spikes. Modeling the RESIDUAL of the linear fit with
   Gradient Boosting lets the nonlinear model focus only on what's left
   over, without re-learning (and mis-extrapolating) the trend itself.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

TARGET = "peak_demand_mw"
HORIZON = 7

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"


def metrics(actual, pred):
    actual, pred = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mae = np.mean(np.abs(actual - pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def main():
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    val = pd.read_csv(DATA_DIR / "val.csv", parse_dates=["date"])
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    full = pd.concat([train, val, test]).sort_values("date").reset_index(drop=True)

    doy = full["date"].dt.dayofyear.values
    dow = full["date"].dt.dayofweek.values

    feat = pd.DataFrame(index=full.index)
    feat["date"] = full["date"]
    feat["sin365"] = np.sin(2 * np.pi * doy / 365.25)
    feat["cos365"] = np.cos(2 * np.pi * doy / 365.25)
    feat["sin182"] = np.sin(4 * np.pi * doy / 365.25)
    feat["cos182"] = np.cos(4 * np.pi * doy / 365.25)
    feat["sin7"] = np.sin(2 * np.pi * dow / 7)
    feat["cos7"] = np.cos(2 * np.pi * dow / 7)
    feat["trend"] = np.arange(len(full))
    feat["lag1"] = full[TARGET].shift(1)
    for lag in [2, 3, 7, 14, 30]:
        feat[f"lag{lag}"] = full[TARGET].shift(lag)
    feat["roll7_mean"] = full[TARGET].shift(1).rolling(7).mean()
    feat["roll7_std"] = full[TARGET].shift(1).rolling(7).std()
    feat["roll30_mean"] = full[TARGET].shift(1).rolling(30).mean()
    feat["dayofweek"] = dow
    feat["is_weekend"] = (dow >= 5).astype(int)
    feat["temperature"] = full["temperature_2m_c"]
    feat["humidity"] = full["humidity_2m_pct"]
    feat["wind_speed"] = full["wind_speed_10m_ms"]
    feat["solar_irr"] = full["solar_irradiance_kwh_m2"]
    feat["precip"] = full["precipitation_mm"]
    doy_climatology = full.groupby(full["date"].dt.dayofyear)["temperature_2m_c"].transform("mean")
    feat["temp_anomaly"] = full["temperature_2m_c"] - doy_climatology
    feat["heat_spike"] = np.clip(full["temperature_2m_c"] - 33, 0, None)

    feat["target"] = full[TARGET].shift(-HORIZON)
    feat_clean = feat.dropna().reset_index(drop=True)

    train_end_date, val_end_date = train["date"].max(), val["date"].max()
    train_mask = feat_clean["date"] <= train_end_date
    val_mask = (feat_clean["date"] > train_end_date) & (feat_clean["date"] <= val_end_date)
    test_mask = feat_clean["date"] > val_end_date

    # ---- Stage 1: linear (Fourier + trend + lag1) ----
    stage1_cols = ["sin365", "cos365", "sin182", "cos182", "sin7", "cos7", "trend", "lag1"]
    X1_train, y_train = feat_clean.loc[train_mask, stage1_cols], feat_clean.loc[train_mask, "target"]
    stage1 = LinearRegression().fit(X1_train, y_train)

    stage1_pred_train = stage1.predict(X1_train)
    residual_train = y_train - stage1_pred_train

    # ---- Stage 2: GBM on residuals, using climate + short lags (no trend/Fourier —
    # those are already handled by stage 1) ----
    stage2_cols = ["lag1", "lag2", "lag3", "lag7", "lag14", "lag30", "roll7_mean",
                   "roll7_std", "roll30_mean", "dayofweek", "is_weekend",
                   "temperature", "humidity", "wind_speed", "solar_irr", "precip",
                   "temp_anomaly", "heat_spike"]
    X2_train = feat_clean.loc[train_mask, stage2_cols]
    stage2 = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=42
    ).fit(X2_train, residual_train)

    def predict(mask):
        X1 = feat_clean.loc[mask, stage1_cols]
        X2 = feat_clean.loc[mask, stage2_cols]
        return stage1.predict(X1) + stage2.predict(X2)

    val_pred = predict(val_mask)
    test_pred = predict(test_mask)
    y_val = feat_clean.loc[val_mask, "target"]
    y_test = feat_clean.loc[test_mask, "target"]

    val_metrics = metrics(y_val, val_pred)
    test_metrics = metrics(y_test, test_pred)
    print("Hybrid model — Validation metrics:", {k: round(v, 3) for k, v in val_metrics.items()})
    print("Hybrid model — Test metrics:      ", {k: round(v, 3) for k, v in test_metrics.items()})

    try:
        combined = pd.read_csv(f"{REPORT_DIR}/model_comparison_final.csv", index_col=0)
    except FileNotFoundError:
        combined = pd.DataFrame()
    combined = pd.concat([combined, pd.DataFrame([test_metrics], index=["Hybrid (Fourier+Trend + GBM residual) — MAIN MODEL"])])
    combined.round(3).to_csv(REPORT_DIR / "model_comparison_final.csv")
    print("\nFull model comparison so far:")
    print(combined.round(3).to_string())

    # feature importance on the residual stage (what the ML part is actually learning)
    stage1_pred_test = stage1.predict(feat_clean.loc[test_mask, stage1_cols])
    residual_test = y_test.values - stage1_pred_test
    perm = permutation_importance(
        stage2, feat_clean.loc[test_mask, stage2_cols], residual_test,
        n_repeats=10, random_state=42, n_jobs=-1
    )
    importance_df = pd.DataFrame({
        "feature": stage2_cols,
        "importance_mean": perm.importances_mean,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(REPORT_DIR / "hybrid_residual_feature_importance.csv", index=False)
    print("\nResidual-stage feature importance (what climate variables add beyond trend+seasonality):")
    print(importance_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(14, 6))
    test_dates = feat_clean.loc[test_mask, "date"]
    ax.plot(test_dates, y_test, label="Actual", color="black", linewidth=1.5)
    ax.plot(test_dates, test_pred, label="Hybrid Model (main model)", color="#2a9d8f", linewidth=1.2)
    ax.set_title("Tamil Nadu Peak Demand — Main Hybrid Model vs Actual (Test Period)")
    ax.set_ylabel("Peak Demand (MW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hybrid_model_vs_actual.png", dpi=130)

    joblib.dump({"stage1": stage1, "stage2": stage2, "stage1_cols": stage1_cols, "stage2_cols": stage2_cols},
                MODEL_DIR / "hybrid_peak_demand_model.joblib")
    print(f"\nSaved hybrid model to {MODEL_DIR / 'hybrid_peak_demand_model.joblib'}")
    print(f"Saved chart to {OUT_DIR / 'hybrid_model_vs_actual.png'}")


if __name__ == "__main__":
    main()
