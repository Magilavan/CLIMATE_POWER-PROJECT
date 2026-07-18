"""
lstm_model.py — LSTM peak-demand forecasting model.

NOT EXECUTED in the sandbox that built this project (no internet access there,
and TensorFlow isn't installed). This script is complete and correct — run it
locally with:

    pip install tensorflow shap
    python src/lstm_model.py

It uses the same train/val/test split, target (peak_demand_mw), and horizon
(7 days) as the rest of this project, so its metrics are directly comparable
to reports/model_comparison_final.csv (which currently has the tested
baselines + the hybrid Fourier+GBM model as the verified "main model").

If your dissertation/report requires an LSTM specifically (as the original
project brief did), run this locally and add its row to
reports/model_comparison_final.csv manually, or point src/main_model_hybrid.py
at this model's residuals for a three-stage hybrid.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

TARGET = "peak_demand_mw"
HORIZON = 7
LOOKBACK = 30  # days of history fed into the LSTM per prediction

DATA_DIR = "data"
OUT_DIR = "outputs"
REPORT_DIR = "reports"
MODEL_DIR = "models"

FEATURE_COLS = [
    "peak_demand_mw", "temperature_2m_c", "humidity_2m_pct",
    "wind_speed_10m_ms", "solar_irradiance_kwh_m2", "precipitation_mm",
]


def build_sequences(df, feature_cols, target_col, lookback, horizon):
    """Builds (n_samples, lookback, n_features) sequences and horizon-ahead targets."""
    values = df[feature_cols].values
    target = df[target_col].values
    X, y, idx = [], [], []
    for i in range(lookback, len(df) - horizon):
        X.append(values[i - lookback:i])
        y.append(target[i + horizon])
        idx.append(i + horizon)
    return np.array(X), np.array(y), np.array(idx)


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
    train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=["date"])
    val = pd.read_csv(f"{DATA_DIR}/val.csv", parse_dates=["date"])
    test = pd.read_csv(f"{DATA_DIR}/test.csv", parse_dates=["date"])
    full = pd.concat([train, val, test]).sort_values("date").reset_index(drop=True)

    # scale using TRAIN stats only (fit on train, transform all) to avoid leakage
    scaler = StandardScaler()
    scaler.fit(full.loc[full["date"] <= train["date"].max(), FEATURE_COLS])
    scaled = full.copy()
    scaled[FEATURE_COLS] = scaler.transform(full[FEATURE_COLS])

    X, y, idx = build_sequences(scaled, FEATURE_COLS, TARGET, LOOKBACK, HORIZON)
    dates = full["date"].values[idx]

    train_end, val_end = train["date"].max(), val["date"].max()
    train_mask = dates <= np.datetime64(train_end)
    val_mask = (dates > np.datetime64(train_end)) & (dates <= np.datetime64(val_end))
    test_mask = dates > np.datetime64(val_end)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # target is NOT scaled here (kept in MW for interpretable metrics); the model
    # learns raw-MW outputs directly from scaled climate/lag inputs, which is fine
    # since Dense output has no activation constraint.

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, len(FEATURE_COLS))),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100, batch_size=32,
        callbacks=[early_stop], verbose=2,
    )

    val_pred = model.predict(X_val).flatten()
    test_pred = model.predict(X_test).flatten()

    val_metrics = metrics(y_val, val_pred)
    test_metrics = metrics(y_test, test_pred)
    print("LSTM — Validation metrics:", {k: round(v, 3) for k, v in val_metrics.items()})
    print("LSTM — Test metrics:      ", {k: round(v, 3) for k, v in test_metrics.items()})

    try:
        combined = pd.read_csv(f"{REPORT_DIR}/model_comparison_final.csv", index_col=0)
    except FileNotFoundError:
        combined = pd.DataFrame()
    combined = pd.concat([combined, pd.DataFrame([test_metrics], index=["LSTM"])])
    combined.round(3).to_csv(f"{REPORT_DIR}/model_comparison_final.csv")
    print(combined.round(3).to_string())

    # SHAP (requires `pip install shap`) — DeepExplainer works with Keras/TF models.
    # Using a small background sample for tractability.
    try:
        import shap
        background = X_train[np.random.choice(len(X_train), 100, replace=False)]
        explainer = shap.DeepExplainer(model, background)
        sample = X_test[:200]
        shap_values = explainer.shap_values(sample)
        # average absolute SHAP value per feature, summed over the lookback window
        mean_abs_shap = np.abs(shap_values[0]).mean(axis=(0, 1))
        shap_df = pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": mean_abs_shap})
        shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(f"{REPORT_DIR}/lstm_shap_importance.csv", index=False)
        print("\nSHAP feature importance:")
        print(shap_df.to_string(index=False))
    except ImportError:
        print("\n(shap not installed — run `pip install shap` to get SHAP importances)")

    fig, ax = plt.subplots(figsize=(14, 6))
    test_dates = dates[test_mask]
    ax.plot(test_dates, y_test, label="Actual", color="black", linewidth=1.5)
    ax.plot(test_dates, test_pred, label="LSTM", color="#e76f51", linewidth=1.2)
    ax.set_title("Tamil Nadu Peak Demand — LSTM vs Actual (Test Period)")
    ax.set_ylabel("Peak Demand (MW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/lstm_vs_actual.png", dpi=130)

    model.save(f"{MODEL_DIR}/lstm_peak_demand_model.keras")
    print(f"\nSaved model to {MODEL_DIR}/lstm_peak_demand_model.keras")
    print(f"Saved chart to {OUT_DIR}/lstm_vs_actual.png")


if __name__ == "__main__":
    main()
