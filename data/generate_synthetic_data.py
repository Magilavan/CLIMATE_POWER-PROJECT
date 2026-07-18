"""
generate_synthetic_data.py

Generates a SYNTHETIC Tamil Nadu climate + electricity + emissions dataset that
mimics the statistical properties of the real dataset already produced by the
user's local pipeline (NASA POWER + CEA/POSOCO + OWID CO2 + World Bank population).

WHY THIS EXISTS:
This project was built in a sandbox with no internet access, so the real
ingestion APIs (NASA POWER, CEA/POSOCO, OWID, World Bank) could not be called.
This script produces a realistic stand-in dataset with the correct seasonality,
trend, and noise structure so every downstream model (baselines, LSTM, solar/wind,
carbon, optimizer, dashboard) can be built and verified end-to-end right now.

TO USE YOUR REAL DATA INSTEAD:
Replace data/merged_dataset.csv with your actual pipeline's output (same column
names as below) and rerun any script in src/ — nothing else needs to change.
"""

from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

START_DATE = "2010-01-01"
END_DATE = "2026-07-17"

dates = pd.date_range(START_DATE, END_DATE, freq="D")
n = len(dates)
doy = dates.dayofyear.values
year_frac = (dates.year - dates.year.min()) + (doy / 365.25)
t = np.arange(n)

# ---------------------------------------------------------------------------
# Climate variables — annual seasonal cycle + noise, matching reported stats:
# temp mean 27.67C std 2.80 | humidity mean 69.6% std 11.5
# wind mean 3.73 m/s std 1.34 | solar mean 5.34 kWh/m2 std 1.20
# ---------------------------------------------------------------------------
annual = np.sin(2 * np.pi * (doy - 60) / 365.25)  # peaks ~ day 60+90=~May/June

temperature_2m_c = 27.7 + 4.0 * annual + np.random.normal(0, 1.3, n)
temperature_2m_c = np.clip(temperature_2m_c, 18, 40)

humidity_2m_pct = 69.6 - 12.0 * annual + np.random.normal(0, 6.0, n)
humidity_2m_pct = np.clip(humidity_2m_pct, 30, 98)

# monsoon bump (Tamil Nadu NE monsoon ~ Oct-Dec) for precipitation
monsoon = np.exp(-((doy - 305) ** 2) / (2 * 35 ** 2)) * 3.5
precipitation_mm = np.clip(
    np.random.exponential(1.0, n) * (0.3 + monsoon) + monsoon, 0, None
)
# force ~35% dry days like real rainfall data
precipitation_mm = precipitation_mm * (np.random.rand(n) > 0.55)

wind_speed_10m_ms = 3.7 + 1.6 * np.sin(2 * np.pi * (doy - 150) / 365.25) + np.random.normal(0, 0.6, n)
wind_speed_10m_ms = np.clip(wind_speed_10m_ms, 0.5, 10)

solar_irradiance_kwh_m2 = 5.3 + 1.1 * np.sin(2 * np.pi * (doy - 80) / 365.25) + np.random.normal(0, 0.35, n)
solar_irradiance_kwh_m2 = np.clip(solar_irradiance_kwh_m2, 0.4, 8.0)

# ---------------------------------------------------------------------------
# Electricity demand — driven by temperature (cooling load), weekly pattern,
# multi-year growth trend, matching reported stats:
# peak_demand_mw mean 8330 std 1786 range 5308-11565
# demand_mwh mean 148099 std 31754
# ---------------------------------------------------------------------------
dow = dates.dayofweek.values
weekday_effect = np.where(dow < 5, 90, -180)  # weekday bump / weekend dip (MW)

trend_component = -1250 + 2500 * (year_frac / year_frac.max())  # long-run growth, centered
seasonal_component = 2100 * annual  # summer cooling peak / winter trough
cooling_kick = np.clip(temperature_2m_c - 33, 0, None) * 140  # extra heat-wave spikes

peak_demand_mw = (
    8330 + trend_component + seasonal_component + weekday_effect + cooling_kick
    + np.random.normal(0, 330, n)
)
peak_demand_mw = np.clip(peak_demand_mw, 5100, 11900)

# daily energy scales with peak but with a load-factor around ~0.74
load_factor = 0.74 + np.random.normal(0, 0.02, n)
demand_mwh = peak_demand_mw * 24 * load_factor
generation_mwh = demand_mwh * (1 + np.random.normal(0.0005, 0.01, n))  # small T&D loss/gain noise

# ---------------------------------------------------------------------------
# Generation mix (for carbon module) — coal declining share, solar/wind rising
# ---------------------------------------------------------------------------
coal_share = np.clip(0.60 - 0.012 * year_frac, 0.28, 0.62)
gas_share = np.full(n, 0.06)
hydro_share = np.full(n, 0.06)
nuclear_share = np.full(n, 0.05)
solar_share = np.clip(0.03 + 0.011 * year_frac, 0.03, 0.20) * (0.6 + 0.4 * (solar_irradiance_kwh_m2 / 6.0))
wind_share = np.clip(0.06 + 0.006 * year_frac, 0.06, 0.16) * (0.6 + 0.4 * (wind_speed_10m_ms / 4.0))
remaining = 1.0 - (gas_share + hydro_share + nuclear_share)
solar_share = np.clip(solar_share, 0, remaining * 0.5)
wind_share = np.clip(wind_share, 0, remaining * 0.5)
coal_share = remaining - solar_share - wind_share
coal_share = np.clip(coal_share, 0.20, 0.70)

gen_mix = pd.DataFrame({
    "coal_share": coal_share,
    "gas_share": gas_share,
    "hydro_share": hydro_share,
    "nuclear_share": nuclear_share,
    "solar_share": solar_share,
    "wind_share": wind_share,
})
gen_mix = gen_mix.div(gen_mix.sum(axis=1), axis=0)  # normalize to sum to 1

# ---------------------------------------------------------------------------
# OWID-style national annual emissions data, forward-filled to daily
# mean owid_co2 2469 std 474 (Mt) | co2_per_capita mean 1.80 std 0.26
# ---------------------------------------------------------------------------
years = np.arange(dates.year.min(), dates.year.max() + 1)
owid_annual = pd.DataFrame({
    "year": years,
    "owid_co2": np.clip(1650 + 95 * (years - years.min()) + np.random.normal(0, 40, len(years)), 1600, 3250),
})
owid_annual["owid_coal_co2"] = owid_annual["owid_co2"] * 0.645
owid_annual["owid_oil_co2"] = owid_annual["owid_co2"] * 0.247
owid_annual["owid_gas_co2"] = owid_annual["owid_co2"] * 0.051
owid_annual["population"] = np.clip(1.243e9 + 1.35e7 * (years - years.min()), None, 1.47e9)
owid_annual["owid_co2_per_capita"] = owid_annual["owid_co2"] * 1e6 / owid_annual["population"]

df = pd.DataFrame({
    "date": dates,
    "temperature_2m_c": temperature_2m_c.round(2),
    "humidity_2m_pct": humidity_2m_pct.round(2),
    "precipitation_mm": precipitation_mm.round(2),
    "wind_speed_10m_ms": wind_speed_10m_ms.round(2),
    "solar_irradiance_kwh_m2": solar_irradiance_kwh_m2.round(3),
    "demand_mwh": demand_mwh.round(1),
    "generation_mwh": generation_mwh.round(1),
    "peak_demand_mw": peak_demand_mw.round(1),
})
df = pd.concat([df, gen_mix.round(4)], axis=1)
df["year"] = df["date"].dt.year
df = df.merge(owid_annual, on="year", how="left").drop(columns=["year"])
df["owid_co2"] = df["owid_co2"].round(1)
df["owid_coal_co2"] = df["owid_coal_co2"].round(1)
df["owid_oil_co2"] = df["owid_oil_co2"].round(1)
df["owid_gas_co2"] = df["owid_gas_co2"].round(1)
df["population"] = df["population"].round(0)
df["owid_co2_per_capita"] = df["owid_co2_per_capita"].round(3)

out_path = DATA_DIR / "merged_dataset.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows, {df.shape[1]} columns to {out_path}")
print(df.describe().T[["mean", "std", "min", "max"]])
