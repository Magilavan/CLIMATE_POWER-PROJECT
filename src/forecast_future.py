"""
forecast_future.py — Rolls the saved hybrid model forward 3 years
(2026-07-18 to 2029-07-17) using:
  - Stage 1 (Fourier+trend linear model): extrapolates trend + seasonality
    purely from the date — no lag dependency, safe to project indefinitely.
  - Stage 2 (GBM residual model): corrects for climate/weekday effects
    recursively — predicted demand feeds back as lag1/lag2/... for the
    next day's prediction.
  - Future climate: generated from day-of-year historical means + noise,
    matching the same seasonal patterns in the training data.
    Replace with real climate forecasts (NASA POWER, Open-Meteo, IMD,
    CMIP6) for a production-grade version.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR   = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"

FORECAST_START = "2026-07-18"
FORECAST_END   = "2029-07-17"   # ~3 years


def build_climate_climatology(hist_df):
    """Compute day-of-year mean + std for each climate variable from history."""
    hist_df = hist_df.copy()
    hist_df["doy"] = pd.to_datetime(hist_df["date"]).dt.dayofyear
    clim_cols = ["temperature_2m_c", "humidity_2m_pct", "wind_speed_10m_ms",
                 "solar_irradiance_kwh_m2", "precipitation_mm"]
    clim = hist_df.groupby("doy")[clim_cols].agg(["mean", "std"]).reset_index()
    clim.columns = ["doy"] + [f"{c}_{s}" for c in clim_cols for s in ["mean", "std"]]
    return clim


def generate_future_climate(dates, clim):
    """Sample future climate from historical day-of-year distributions."""
    np.random.seed(0)
    doys = dates.dayofyear
    clim_indexed = clim.set_index("doy")

    def sample(col):
        means = clim_indexed.loc[doys, f"{col}_mean"].values
        stds  = clim_indexed.loc[doys, f"{col}_std"].values
        return means + np.random.normal(0, stds)

    return pd.DataFrame({
        "date": dates,
        "temperature_2m_c":        np.clip(sample("temperature_2m_c"), 18, 40),
        "humidity_2m_pct":         np.clip(sample("humidity_2m_pct"), 30, 98),
        "wind_speed_10m_ms":       np.clip(sample("wind_speed_10m_ms"), 0.5, 10),
        "solar_irradiance_kwh_m2": np.clip(sample("solar_irradiance_kwh_m2"), 0.4, 8.0),
        "precipitation_mm":        np.clip(sample("precipitation_mm"), 0, None),
    })


def main():
    # load history and model
    hist = pd.read_csv(DATA_DIR / "merged_dataset.csv")
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").reset_index(drop=True)

    bundle = joblib.load(MODEL_DIR / "hybrid_peak_demand_model.joblib")
    stage1       = bundle["stage1"]
    stage2       = bundle["stage2"]
    stage1_cols  = bundle["stage1_cols"]
    stage2_cols  = bundle["stage2_cols"]

    # total rows in history (trend index must continue from here)
    n_hist = len(hist)

    # build climatology from history for future climate generation
    clim = build_climate_climatology(hist)

    # future date range
    future_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
    future_climate = generate_future_climate(future_dates, clim)

    # seed lag buffer with last 30 days of actual demand
    lag_buffer = list(hist["peak_demand_mw"].values[-30:])  # index -30..-1

    predictions = []

    for i, row in future_climate.iterrows():
        date = row["date"]
        doy  = date.dayofyear
        dow  = date.dayofweek
        trend_idx = n_hist + i  # continues from end of historical index

        # ---- Stage 1 features ----
        s1 = {
            "sin365": np.sin(2 * np.pi * doy / 365.25),
            "cos365": np.cos(2 * np.pi * doy / 365.25),
            "sin182": np.sin(4 * np.pi * doy / 365.25),
            "cos182": np.cos(4 * np.pi * doy / 365.25),
            "sin7":   np.sin(2 * np.pi * dow / 7),
            "cos7":   np.cos(2 * np.pi * dow / 7),
            "trend":  trend_idx,
            "lag1":   lag_buffer[-1],
        }
        X1 = pd.DataFrame([s1])[stage1_cols]
        stage1_pred = stage1.predict(X1)[0]

        # ---- Stage 2 features ----
        buf = lag_buffer  # shorthand
        roll7_vals  = buf[-7:]
        roll30_vals = buf[-30:]
        doy_clim_temp = clim.set_index("doy").loc[doy, "temperature_2m_c_mean"]

        s2 = {
            "lag1":        buf[-1],
            "lag2":        buf[-2],
            "lag3":        buf[-3],
            "lag7":        buf[-7],
            "lag14":       buf[-14],
            "lag30":       buf[-30],
            "roll7_mean":  np.mean(roll7_vals),
            "roll7_std":   np.std(roll7_vals),
            "roll30_mean": np.mean(roll30_vals),
            "dayofweek":   dow,
            "is_weekend":  int(dow >= 5),
            "temperature": row["temperature_2m_c"],
            "humidity":    row["humidity_2m_pct"],
            "wind_speed":  row["wind_speed_10m_ms"],
            "solar_irr":   row["solar_irradiance_kwh_m2"],
            "precip":      row["precipitation_mm"],
            "temp_anomaly": row["temperature_2m_c"] - doy_clim_temp,
            "heat_spike":  max(row["temperature_2m_c"] - 33, 0),
        }
        X2 = pd.DataFrame([s2])[stage2_cols]
        residual_pred = stage2.predict(X2)[0]

        pred = stage1_pred + residual_pred
        pred = float(np.clip(pred, 5000, 15000))

        predictions.append({"date": date, "predicted_peak_demand_mw": round(pred, 1),
                             **row[["temperature_2m_c","humidity_2m_pct",
                                    "wind_speed_10m_ms","solar_irradiance_kwh_m2"]].to_dict()})
        lag_buffer.append(pred)
        lag_buffer.pop(0)  # keep buffer at 30 entries

    result = pd.DataFrame(predictions)
    out_csv = REPORT_DIR / "future_forecast_2026_2029.csv"
    result.to_csv(out_csv, index=False)

    # ---- summary stats ----
    print(f"Forecast period: {result['date'].min().date()} to {result['date'].max().date()} ({len(result)} days)")
    print(f"\nYearly average predicted peak demand:")
    result["year"] = result["date"].dt.year
    print(result.groupby("year")["predicted_peak_demand_mw"].mean().round(1).to_string())
    print(f"\nOverall: min={result['predicted_peak_demand_mw'].min():.0f} MW  "
          f"max={result['predicted_peak_demand_mw'].max():.0f} MW  "
          f"mean={result['predicted_peak_demand_mw'].mean():.0f} MW")

    # ---- chart ----
    # include last 1 year of history for context
    hist_tail = hist[hist["date"] >= hist["date"].max() - pd.Timedelta(days=365)]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(hist_tail["date"], hist_tail["peak_demand_mw"],
            color="black", linewidth=1.2, label="Historical (actual)")
    ax.plot(result["date"], result["predicted_peak_demand_mw"],
            color="#2a9d8f", linewidth=1.1, label="Forecast (2026-2029)")
    ax.axvline(pd.Timestamp(FORECAST_START), color="red", linestyle="--", linewidth=1, label="Forecast start")

    # yearly mean bands
    for yr, grp in result.groupby("year"):
        ax.axhline(grp["predicted_peak_demand_mw"].mean(), color="#e76f51",
                   linestyle=":", linewidth=0.9, alpha=0.7)
        ax.text(grp["date"].iloc[len(grp)//2], grp["predicted_peak_demand_mw"].mean() + 80,
                f"{yr} avg\n{grp['predicted_peak_demand_mw'].mean():.0f} MW",
                fontsize=7, color="#e76f51", ha="center")

    ax.set_title("Tamil Nadu Peak Demand — 3-Year Future Forecast (2026–2029)")
    ax.set_ylabel("Peak Demand (MW)")
    ax.legend()
    fig.tight_layout()
    chart_path = OUT_DIR / "future_forecast_2026_2029.png"
    fig.savefig(chart_path, dpi=130)

    print(f"\nSaved forecast to {out_csv}")
    print(f"Saved chart to {chart_path}")


if __name__ == "__main__":
    main()
