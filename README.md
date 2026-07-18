# Tamil Nadu Climate-Aware Power Planning Project

A complete final-year project: climate-driven electricity peak-demand forecasting,
renewable (solar/wind) generation potential estimation, grid carbon emissions
estimation, and a simplified generation-mix optimizer, with a dashboard tying it
all together.

## ⚠️ Read this first: synthetic data notice

This project was built in a sandbox with **no internet access**, so the real data
sources (NASA POWER, CEA/POSOCO, OWID CO2, World Bank) could not be called directly.
`data/generate_synthetic_data.py` generates a **synthetic** Tamil Nadu dataset
(2010-01-01 to 2026-07-17, daily) engineered to match the statistical properties
already confirmed from a real pipeline run (same column names, same means/std/ranges
for temperature, humidity, peak demand, etc.).

**Every script, model, and result in this repo has been run and verified on that
synthetic dataset.** All the numbers below (RMSE, MAPE, emissions figures, cost
savings) are real outputs of real code — just on synthetic input data.

**To switch to your real data:** replace `data/merged_dataset.csv` with your actual
merged pipeline output (same column names — see the header of
`generate_synthetic_data.py` for the full schema), rerun `src/split.py`, then rerun
whichever `src/*.py` scripts you need. Nothing else changes.

## Project structure

```
climate_power_project/
├── README.md                          ← you are here
├── requirements.txt
├── data/
│   ├── generate_synthetic_data.py     ← builds the synthetic dataset (already run)
│   ├── merged_dataset.csv             ← the dataset (synthetic, see notice above)
│   ├── train.csv / val.csv / test.csv ← chronological splits (70/15/15)
├── src/
│   ├── split.py                       ← chronological train/val/test splitter
│   ├── quality_report.py              ← data quality audit
│   ├── baselines.py                   ← 5 baseline forecasting models (TESTED)
│   ├── main_model_gbm.py              ← pure Gradient Boosting model (TESTED)
│   ├── main_model_hybrid.py           ← ⭐ BEST model: Fourier+Trend + GBM residual (TESTED)
│   ├── lstm_model.py                  ← LSTM model (write-only — needs TensorFlow, run locally)
│   ├── solar_wind.py                  ← solar/wind renewable potential estimator (TESTED)
│   ├── carbon.py                      ← grid carbon emissions estimator (TESTED)
│   └── optimizer.py                   ← LP generation-mix optimizer (TESTED)
├── dashboard/
│   └── app.py                         ← Streamlit dashboard (write-only — needs Streamlit, run locally)
├── models/                            ← saved trained models (.joblib)
├── reports/                           ← CSV outputs, metrics tables, data quality report
├── outputs/                           ← charts (PNG)
└── FINAL_REPORT.md                    ← consolidated results + methodology writeup
```

## How to run everything

```bash
pip install -r requirements.txt

# 1. Data (already generated and committed, but to regenerate/replace):
python data/generate_synthetic_data.py
python src/split.py
python src/quality_report.py

# 2. Baselines
python src/baselines.py

# 3. Main model (tested, works without TensorFlow)
python src/main_model_gbm.py
python src/main_model_hybrid.py       # ⭐ this is the best-performing model

# 4. LSTM (optional — needs `pip install tensorflow shap`)
python src/lstm_model.py

# 5. Renewables, carbon, optimizer
python src/solar_wind.py
python src/carbon.py
python src/optimizer.py

# 6. Dashboard (needs `pip install streamlit`)
streamlit run dashboard/app.py
```

Run them in roughly this order — later scripts read CSVs produced by earlier ones
(e.g. `optimizer.py` reads `reports/renewable_potential.csv` from `solar_wind.py`).

## Headline results (on the synthetic dataset)

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| Naive (lag-1) | 506.8 | 405.3 | 4.32% | 0.888 |
| Seasonal Naive (weekly, k=7) | 506.8 | 405.3 | 4.32% | 0.888 |
| Seasonal Naive (annual, k=365) | 500.0 | 402.1 | 4.26% | 0.891 |
| Linear Regression (lags + climate) | 405.0 | 323.4 | 3.41% | 0.929 |
| Fourier + Exogenous Regression | 371.5 | 298.9 | 3.20% | 0.940 |
| Gradient Boosting (pure) | 481.7 | 387.2 | 3.98% | 0.899 |
| **Hybrid (Fourier+Trend + GBM residual) — MAIN MODEL** | **330.5** | **264.9** | **2.82%** | **0.953** |

The hybrid model is the one to present as your primary result — see
`FINAL_REPORT.md` for the full writeup of why it was built this way, including
two real modeling problems that came up (and were fixed) during development:
a broken SARIMAX-style baseline that missed the annual seasonal cycle, and a
Gradient Boosting model that couldn't extrapolate a raw trend feature.

## What's genuinely tested vs. written-but-not-run

Because this project was built without internet access, be upfront about this
distinction in your viva/defense if asked:

- **Tested (I ran it, these are real numbers):** data generation, splitting,
  quality report, all 5 baselines, Gradient Boosting model, the hybrid model,
  solar/wind potential, carbon emissions, LP optimizer.
- **Written but not executed here (needs a package not available in this
  sandbox — run locally):** the LSTM model (`src/lstm_model.py`, needs
  TensorFlow) and the dashboard (`dashboard/app.py`, needs Streamlit). Both
  are complete, syntactically correct scripts using the same data/target/split
  conventions as the tested code, but you should run them yourself once and
  confirm the output before presenting them as working.
