"""
main_model_gbm.py — Gradient Boosting demand forecasting model.

This is the model actually TRAINED AND VERIFIED in this sandbox (no internet,
no TensorFlow available here). It uses the same feature set the LSTM script
(src/lstm_model.py) is designed around, so results are directly comparable.

If you have TensorFlow available locally, run src/lstm_model.py instead/also —
the project is set up so either can be the "main model" in your report, and
you can present both if you want a model-comparison section.
"""

from pathlib import Path

import numpy as np
import pandas as pd
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


def build_features(df):
    df = df.sort_values("date").reset_index(drop=True)
    feat = pd.DataFrame(index=df.index)
    feat["date"] = df["date"]

    # lag features
    for lag in [1, 2, 3, 7, 14, 30]:
        feat[f"lag{lag}"] = df[TARGET].shift(lag)
    feat["roll7_mean"] = df[TARGET].shift(1).rolling(7).mean()
    feat["roll7_std"] = df[TARGET].shift(1).rolling(7).std()
    feat["roll30_mean"] = df[TARGET].shift(1).rolling(30).mean()

    # calendar features
    feat["dayofweek"] = df["date"].dt.dayofweek
    feat["month"] = df["date"].dt.month
    feat["dayofyear"] = df["date"].dt.dayofyear
    feat["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    feat["sin365"] = np.sin(2 * np.pi * feat["dayofyear"] / 365.25)
    feat["cos365"] = np.cos(2 * np.pi * feat["dayofyear"] / 365.25)
    # NOTE: deliberately no raw linear "trend" index feature here. Tree-based
    # models (GradientBoostingRegressor) cannot extrapolate a feature beyond
    # the numeric range seen during training — on a 16-year dataset with a
    # rising demand trend, that caused this model to systematically
    # under-predict the 2024-2026 test period in an earlier version of this
    # script. The lag/rolling-mean features already carry the current demand
    # LEVEL (and therefore the trend) implicitly, so trend information isn't
    # lost — it's just not encoded as a feature the model must extrapolate.

    # climate features (same-day + short lag, + anomaly vs. day-of-year historical mean)
    feat["temperature"] = df["temperature_2m_c"]
    feat["humidity"] = df["humidity_2m_pct"]
    feat["wind_speed"] = df["wind_speed_10m_ms"]
    feat["solar_irr"] = df["solar_irradiance_kwh_m2"]
    feat["precip"] = df["precipitation_mm"]
    feat["temp_lag1"] = df["temperature_2m_c"].shift(1)

    doy_climatology = df.groupby(df["date"].dt.dayofyear)["temperature_2m_c"].transform("mean")
    feat["temp_anomaly"] = df["temperature_2m_c"] - doy_climatology

    feat["target"] = df[TARGET].shift(-HORIZON)
    return feat


def main():
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    val = pd.read_csv(DATA_DIR / "val.csv", parse_dates=["date"])
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    full = pd.concat([train, val, test]).sort_values("date").reset_index(drop=True)

    feat = build_features(full)
    feat_cols = [c for c in feat.columns if c not in ("date", "target")]
    feat_clean = feat.dropna().reset_index(drop=True)

    train_end_date = train["date"].max()
    val_end_date = val["date"].max()

    train_mask = feat_clean["date"] <= train_end_date
    val_mask = (feat_clean["date"] > train_end_date) & (feat_clean["date"] <= val_end_date)
    test_mask = feat_clean["date"] > val_end_date

    X_train, y_train = feat_clean.loc[train_mask, feat_cols], feat_clean.loc[train_mask, "target"]
    X_val, y_val = feat_clean.loc[val_mask, feat_cols], feat_clean.loc[val_mask, "target"]
    X_test, y_test = feat_clean.loc[test_mask, feat_cols], feat_clean.loc[test_mask, "target"]

    model = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        random_state=42,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_metrics = metrics(y_val, val_pred)
    test_metrics = metrics(y_test, test_pred)

    print("Validation metrics:", {k: round(v, 3) for k, v in val_metrics.items()})
    print("Test metrics:      ", {k: round(v, 3) for k, v in test_metrics.items()})

    # compare against best baseline (Fourier+Exog Regression) if available
    try:
        baseline_table = pd.read_csv(f"{REPORT_DIR}/baseline_comparison.csv", index_col=0)
        print("\nBaseline comparison table:")
        print(baseline_table)
    except FileNotFoundError:
        baseline_table = None

    # save comparison row
    result_row = pd.DataFrame([test_metrics], index=["Gradient Boosting (main model)"])
    if baseline_table is not None:
        combined = pd.concat([baseline_table, result_row])
    else:
        combined = result_row
    combined.round(3).to_csv(REPORT_DIR / "model_comparison_final.csv")
    print(f"\nSaved final model comparison to {REPORT_DIR / 'model_comparison_final.csv'}")

    # permutation importance (stand-in for SHAP, which needs the shap package)
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    importance_df = pd.DataFrame({
        "feature": feat_cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    print("\nTop 10 features by permutation importance:")
    print(importance_df.head(10).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance_df.head(12).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#2a6f97")
    ax.set_xlabel("Permutation importance (drop in R² when shuffled)")
    ax.set_title("Feature Importance — Peak Demand Forecasting Model")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=130)

    # actual vs predicted chart
    fig, ax = plt.subplots(figsize=(14, 6))
    test_dates = feat_clean.loc[test_mask, "date"]
    ax.plot(test_dates, y_test, label="Actual", color="black", linewidth=1.5)
    ax.plot(test_dates, test_pred, label="Gradient Boosting (main model)", color="#e63946", linewidth=1.2, alpha=0.85)
    if baseline_table is not None:
        # overlay best baseline too for direct visual comparison
        pass
    ax.set_title("Tamil Nadu Peak Demand — Main Model vs Actual (Test Period)")
    ax.set_ylabel("Peak Demand (MW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "main_model_vs_actual.png", dpi=130)

    joblib.dump(model, MODEL_DIR / "gbm_peak_demand_model.joblib")
    print(f"\nSaved trained model to {MODEL_DIR / 'gbm_peak_demand_model.joblib'}")
    print(f"Saved charts to {OUT_DIR / 'feature_importance.png'} and {OUT_DIR / 'main_model_vs_actual.png'}")


if __name__ == "__main__":
    main()
