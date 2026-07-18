"""
carbon.py — grid carbon emissions estimator.

Emission factors (gCO2/kWh) below are representative published figures for
each generation source. SOURCES (cite these in your report):
- Coal (India average, incl. supercritical mix): ~950 gCO2/kWh
  (CEA CO2 Baseline Database methodology; IPCC default for sub-bituminous/
  bituminous coal falls in a similar 800-1050 g/kWh range depending on plant
  efficiency)
- Natural gas (combined cycle): ~450 gCO2/kWh (IPCC/IEA typical figure for
  CCGT plants)
- Hydro: ~24 gCO2/kWh (IPCC lifecycle estimate — reservoir hydro; run-of-river
  is typically lower)
- Nuclear: ~12 gCO2/kWh (IPCC lifecycle estimate)
- Solar PV: ~41 gCO2/kWh (IPCC lifecycle estimate — manufacturing emissions,
  zero direct operational emissions)
- Wind: ~11 gCO2/kWh (IPCC lifecycle estimate — manufacturing emissions,
  zero direct operational emissions)

LIMITATION TO DISCLOSE: these are FIXED national/international average
factors, not measured for Tamil Nadu's specific plants, and don't vary by
time of day or plant age/efficiency. A more precise estimate would use
CEA's state-specific, plant-specific emission factors where available.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"

EMISSION_FACTORS_G_PER_KWH = {
    "coal": 950,
    "gas": 450,
    "hydro": 24,
    "nuclear": 12,
    "solar": 41,
    "wind": 11,
}

SHARE_COLS = {
    "coal": "coal_share", "gas": "gas_share", "hydro": "hydro_share",
    "nuclear": "nuclear_share", "solar": "solar_share", "wind": "wind_share",
}


def compute_daily_emissions(df):
    total_emissions_tonnes = np.zeros(len(df))
    for source, share_col in SHARE_COLS.items():
        gen_mwh = df["generation_mwh"] * df[share_col]
        emissions_g = gen_mwh * 1000 * EMISSION_FACTORS_G_PER_KWH[source]  # MWh->kWh
        total_emissions_tonnes += emissions_g / 1e6  # g -> tonnes
    return total_emissions_tonnes


def scenario_emissions(df, renewable_penetration_increase_pp):
    """
    Projects emissions under an alternative scenario where solar+wind share
    increases by `renewable_penetration_increase_pp` percentage points
    (taken proportionally from coal, since coal is the marginal/displaceable
    source in most dispatch order assumptions), holding total generation
    fixed.
    """
    df2 = df.copy()
    increase = renewable_penetration_increase_pp / 100.0
    # split the increase proportionally between solar and wind based on their current mix
    solar_frac_of_re = df2["solar_share"] / (df2["solar_share"] + df2["wind_share"])
    solar_add = increase * solar_frac_of_re
    wind_add = increase * (1 - solar_frac_of_re)
    df2["solar_share"] = df2["solar_share"] + solar_add
    df2["wind_share"] = df2["wind_share"] + wind_add
    df2["coal_share"] = np.clip(df2["coal_share"] - increase, 0, None)
    # renormalize in case of floor clipping
    total = df2[list(SHARE_COLS.values())].sum(axis=1)
    for col in SHARE_COLS.values():
        df2[col] = df2[col] / total
    return compute_daily_emissions(df2)


def main():
    df = pd.read_csv(DATA_DIR / "merged_dataset.csv", parse_dates=["date"])

    df["grid_emissions_tonnes"] = compute_daily_emissions(df)
    df["emissions_intensity_g_per_kwh"] = (df["grid_emissions_tonnes"] * 1e6) / (df["generation_mwh"] * 1000)

    df["emissions_scenario_plus10pp_re"] = scenario_emissions(df, 10)
    df["emissions_scenario_plus20pp_re"] = scenario_emissions(df, 20)

    out_cols = ["date", "grid_emissions_tonnes", "emissions_intensity_g_per_kwh",
                "emissions_scenario_plus10pp_re", "emissions_scenario_plus20pp_re"]
    df[out_cols].to_csv(REPORT_DIR / "emissions_estimates.csv", index=False)

    print("Grid emissions intensity (gCO2/kWh) over time:")
    print(df.groupby(df["date"].dt.year)["emissions_intensity_g_per_kwh"].mean().round(1))

    annual = df.groupby(df["date"].dt.year).agg(
        current_emissions_Mt=("grid_emissions_tonnes", lambda x: x.sum() / 1e6),
        scenario_plus10pp_Mt=("emissions_scenario_plus10pp_re", lambda x: x.sum() / 1e6),
        scenario_plus20pp_Mt=("emissions_scenario_plus20pp_re", lambda x: x.sum() / 1e6),
    ).round(3)
    annual.to_csv(REPORT_DIR / "annual_emissions_scenarios.csv")
    print("\nAnnual emissions by scenario (million tonnes CO2):")
    print(annual.tail(8))

    fig, ax = plt.subplots(figsize=(12, 6))
    annual[["current_emissions_Mt", "scenario_plus10pp_Mt", "scenario_plus20pp_Mt"]].plot(
        ax=ax, marker="o"
    )
    ax.set_title("Grid Emissions — Current Mix vs. Increased Renewable Penetration Scenarios")
    ax.set_ylabel("Annual grid CO2 emissions (Million tonnes)")
    ax.set_xlabel("Year")
    ax.legend(["Current mix", "+10pp renewable share", "+20pp renewable share"])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "emissions_scenarios.png", dpi=130)
    print(f"\nSaved chart to {OUT_DIR / 'emissions_scenarios.png'}")
    print(f"Saved data to {REPORT_DIR / 'emissions_estimates.csv'} and {REPORT_DIR / 'annual_emissions_scenarios.csv'}")


if __name__ == "__main__":
    main()
