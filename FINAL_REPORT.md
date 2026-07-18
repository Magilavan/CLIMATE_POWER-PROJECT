# Climate-Aware Power Planning for Tamil Nadu — Final Report

## 1. Problem Statement

Given climate variables (temperature, humidity, wind speed, solar irradiance,
precipitation) and historical grid data (peak demand, generation mix), this
project builds:

1. A peak electricity demand forecasting model (7-day horizon)
2. A solar + wind generation potential estimator
3. A grid carbon emissions estimator, with renewable-penetration scenarios
4. A simplified generation-mix optimizer (cost + emissions weighted)
5. A dashboard consolidating all of the above

**Target variable:** `peak_demand_mw` (not total daily energy `demand_mwh`) —
chosen because peak load, not total energy, is what determines grid capacity
planning decisions, and it is more directly climate-sensitive (afternoon/evening
cooling-load driven) than a 24-hour energy total, which is what makes the
climate-demand link this project investigates worth demonstrating.

**Data:** daily, Tamil Nadu, 2010-01-01 to 2026-07-17 (6,042 days, 100% coverage,
near-zero missingness after linear interpolation). See the data notice in
`README.md` regarding the synthetic dataset used in this build.

**Split:** chronological 70/15/15 (train: 2010–2021, val: 2021–2024, test:
2024–2026) — never randomly shuffled, since random splitting on time-series data
leaks future information into training.

## 2. Baseline Models

Five baselines were built to establish a floor before attempting anything more
complex:

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| Naive (lag-1) | 506.8 | 405.3 | 4.32% | 0.888 |
| Seasonal Naive (weekly, k=7) | 506.8 | 405.3 | 4.32% | 0.888 |
| Seasonal Naive (annual, k=365) | 500.0 | 402.1 | 4.26% | 0.891 |
| Linear Regression (lag1, lag7, roll7, temp, humidity) | 405.0 | 323.4 | 3.41% | 0.929 |
| Fourier + Exogenous Regression (annual+weekly cycles + trend) | 371.5 | 298.9 | 3.20% | 0.940 |

**Note on Naive (lag-1) == Seasonal Naive (weekly):** these two produce identical
numbers here because the forecast horizon (7 days) exactly equals the weekly
seasonal period (k=7) — persistence-at-lag-7 and weekly-seasonal-naive are
mathematically the same forecast in this specific configuration. This isn't a
bug; it's a direct consequence of the horizon choice, worth noting if an examiner
asks why two "different" baselines produced identical numbers.

**A real problem that came up and was fixed:** an earlier SARIMAX-style baseline
using a literal seasonal period of `m=7` completely failed to capture the annual
demand cycle — it produced a smooth upward-drifting forecast that ignored the
summer/winter swing entirely, because m=7 only models weekly seasonality. A
literal `m=365` SARIMAX fit is computationally impractical on ~6,000 daily
points. The fix implemented here uses **Fourier terms** (sin/cos pairs at annual
and semi-annual frequencies) plus weekly Fourier terms and a linear trend term,
fit via ordinary linear regression — this captures long seasonal cycles far more
efficiently than a literal high-period SARIMAX model, and is the standard
alternative used in practice (e.g., Facebook Prophet uses the same idea).

## 3. Main Model: Hybrid (Fourier+Trend + Gradient Boosting Residual)

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| Gradient Boosting (pure, all features) | 481.7 | 387.2 | 3.98% | 0.899 |
| **Hybrid (Fourier+Trend + GBM residual) — FINAL MODEL** | **330.5** | **264.9** | **2.82%** | **0.953** |

**Why a hybrid model, and not a pure Gradient Boosting or pure LSTM model:**

Two genuine findings drove this design, both worth stating plainly in a defense:

1. A pure Gradient Boosting model, given the same features as everything else,
   scored *worse* than the simple Fourier+trend linear regression baseline
   (RMSE 481.7 vs. 371.5). The reason: **tree-based models cannot extrapolate a
   feature beyond the numeric range seen during training.** A raw linear "trend"
   feature (day index) grows monotonically, and by the 2024–2026 test period it
   exceeds every value the trees ever split on during training — so the model
   systematically misjudged the level of demand growth. Removing the raw trend
   feature and relying on lag features (which carry the current demand level
   implicitly) improved this somewhat (RMSE 481.7) but still didn't match the
   linear model, which extrapolates a trend term natively without this problem.
2. However, a *pure* linear/Fourier model can't capture nonlinear effects — like
   the sharper-than-linear jump in cooling demand on very hot days, or how
   humidity and demand interact. Modeling the **residual** of the linear fit with
   Gradient Boosting lets the nonlinear model focus only on what the linear stage
   missed, without ever needing to extrapolate the trend itself (that's handled
   natively by the linear stage).

This two-stage design (Stage 1: linear Fourier+trend+lag1 → Stage 2: GBM on
residuals using climate variables + short lags) beat every baseline and both
pure-model attempts, and gives an honest, defensible story for why the final
architecture looks the way it does — this is a better viva answer than simply
presenting the best-scoring model with no explanation of what didn't work.

**What the residual stage is actually learning (permutation importance on
Stage 2, i.e. what adds value *beyond* trend+seasonality):**

| Feature | Importance |
|---|---|
| day of week | 0.046 |
| 30-day rolling mean | 0.023 |
| humidity | 0.016 |
| temperature anomaly (vs. day-of-year average) | 0.015 |
| lag-1 | 0.012 |
| is_weekend | 0.012 |

This confirms the climate-driven story the project set out to demonstrate:
**temperature anomaly** — not raw temperature, which the Fourier stage already
absorbs as part of the seasonal cycle — is what the nonlinear stage picks up as
additional signal, alongside humidity and short-term demand momentum.

**LSTM:** a complete LSTM script (`src/lstm_model.py`) was written using the same
target, horizon, and train/val/test split as everything above, with SHAP
importance and a 30-day lookback window. It was **not executed** in the sandbox
that built this project (no internet access, TensorFlow not installed) — run it
locally and report its metrics alongside the table above if your project brief
specifically requires an LSTM as the headline model.

## 4. Solar + Wind Renewable Generation Potential

**Method:** solar output scaled from irradiance relative to a 5.5 kWh/m²/day
reference, at 19% panel efficiency and 0.80 performance ratio, against an
assumed 6,000 MW installed solar capacity. Wind output uses a standard cubic
power-curve approximation (cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s) against
an assumed 10,000 MW installed wind capacity.

**A real modeling problem that came up and was fixed:** climate wind speed data
(from NASA POWER-style sources) is measured at 10m height, but turbine hubs sit
around 80m, where wind is substantially stronger. Applying the power curve
directly to 10m wind speed made turbines look almost useless (average output
~0.5% of installed capacity). The fix applies the standard wind shear power-law
extrapolation (`v_hub = v_10m × (80/10)^0.14`) before the power curve, which
raised average wind output to a more realistic ~3.3% capacity factor for this
dataset's modest wind resource.

**Result:** solar+wind potential covers **~60% of average peak demand**, given
the assumed installed capacities above — leaving a substantial gap that
dispatchable sources (coal, gas, hydro, nuclear) or storage must cover,
especially since renewable output and peak demand don't always coincide in
time (solar peaks at midday, demand peaks in the evening).

**Disclosed limitations:** installed capacities are assumed, not sourced from
real TN grid figures; the solar model is a simplified flat-plate calculation, not
a full PVWatts-style model; the wind power curve is a generic textbook
approximation, not a specific turbine's certified curve.

## 5. Grid Carbon Emissions

**Method:** daily generation mix (coal/gas/hydro/nuclear/solar/wind shares) ×
published emission factors (gCO2/kWh): coal 950, gas 450, hydro 24, nuclear 12,
solar 41, wind 11 (IPCC lifecycle estimates / CEA baseline methodology).

**Result:** grid emissions intensity declines from ~721 gCO2/kWh (2010) to
~508 gCO2/kWh (2026) as coal's share of the mix falls — consistent with a
renewables-growth narrative. Scenario analysis shows a +20 percentage-point
shift from coal to solar/wind would cut annual emissions by roughly a third
relative to the current-mix trajectory.

**Disclosed limitation:** these are fixed national/international average
emission factors, not measured for Tamil Nadu's specific plants, and don't vary
by time of day or plant age/efficiency.

## 6. Generation-Mix Optimizer (Simplified LP, not full NSGA-II)

**Method:** for each day in a 90-day test window, `scipy.optimize.linprog`
allocates generation across 6 sources to meet forecasted demand, minimizing a
weighted sum of cost (60%) and emissions (40%), subject to illustrative
min/max capacity constraints per source and renewable output capped at that
day's estimated potential.

**Result (vs. a naive fixed-priority-order dispatch):** the optimizer achieves
**~21% lower cost** and **~71% lower emissions** over the 90-day test window.
The large emissions gap arises because naive priority dispatch defaults to
maximum coal use before considering gas, even though gas costs only moderately
more per MWh while emitting roughly half as much CO2 per kWh — the optimizer
correctly trades a modest cost increase for a large emissions reduction where
the weighting favors it.

**Explicitly NOT a full multi-objective optimizer:** this collapses cost and
emissions into a single weighted-sum objective rather than producing a full
Pareto front of trade-off solutions (which NSGA-II would do); it optimizes each
day independently with no ramp-rate, storage, or start-up-cost constraints; and
the cost figures are illustrative, not real-time market prices. State this
plainly in your report's methodology/limitations section — examiners respond
better to an honestly-scoped simplification than an overclaimed "AI-optimized"
system.

## 7. Dashboard

A Streamlit dashboard (`dashboard/app.py`) consolidates all of the above into
four tabs (demand forecast, renewable potential, emissions, optimized mix) with
a date-range filter and top-line summary metrics. It loads pre-computed CSVs
rather than retraining on load. Not executed in this sandbox (Streamlit not
installed) — run locally to confirm.

## 8. Summary of Genuine Problems Found and Fixed During Development

Presenting these in a report or viva is a strength, not a weakness — it shows
the modeling process was actually scrutinized rather than accepted at face value:

1. **SARIMAX with m=7 missed the annual demand cycle entirely** → fixed with
   Fourier-term regression capturing both annual and weekly seasonality plus a
   trend term.
2. **Gradient Boosting underperformed simple linear regression** due to an
   inability to extrapolate a raw trend feature beyond the training range →
   fixed by building a hybrid model where the linear stage handles trend and
   seasonality natively, and GBM only models the residual.
3. **Wind turbine output was drastically understated** using raw 10m climate
   wind speed data → fixed by applying standard wind-shear extrapolation to
   hub height before applying the turbine power curve.

## 9. Suggested Report Sections to Write Next

- Abstract / Introduction — use Section 1 above
- Literature review — standard for your institution's format
- Methodology — Sections 2–6 above, condensed
- Results — the tables above, plus the charts in `outputs/`
- Discussion / Limitations — Section 8 above, plus each module's disclosed
  limitations
- Conclusion & Future Work — real data ingestion (replacing the synthetic
  dataset), LSTM comparison once run locally, full NSGA-II multi-objective
  optimization as a stretch extension
