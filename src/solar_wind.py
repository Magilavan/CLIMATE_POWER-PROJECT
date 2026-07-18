"""
solar_wind.py — estimates daily solar PV and wind turbine generation potential
from climate data.

SIMPLIFYING ASSUMPTIONS (disclose these in your report's limitations section):
- Solar: output = irradiance x panel_area x panel_efficiency x performance_ratio.
  This is a simplified flat-plate PV model. It does NOT account for panel
  temperature derating, tilt/orientation optimization, or inverter losses in
  detail (folded into a single performance_ratio constant). A professional
  estimate would use NREL's PVWatts model or panel-specific datasheets.
- Wind: output follows a standard cubic power-curve approximation between
  cut-in and rated wind speed, flat at rated power up to cut-out speed. Real
  turbines have manufacturer-specific (non-cubic) power curves; this is a
  standard textbook approximation, not a specific turbine's certified curve.
- Both are scaled to an ASSUMED installed capacity (see CONFIG below) — for
  a real project, replace with your target region's actual/planned installed
  solar and wind capacity (MW).
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

# ---- CONFIG: assumed installed capacity (MW) — replace with real regional figures ----
SOLAR_INSTALLED_MW = 6000     # Tamil Nadu had ~ multi-GW solar capacity as of recent years;
                               # replace with your project's actual/assumed figure
WIND_INSTALLED_MW = 10000     # Tamil Nadu is one of India's largest wind states;
                               # replace with your project's actual/assumed figure

# ---- Solar model constants ----
PANEL_EFFICIENCY = 0.19       # typical modern crystalline-silicon panel, ~19%
PERFORMANCE_RATIO = 0.80      # accounts for inverter/wiring/soiling/temperature losses (typical 0.75-0.85)

# ---- Wind model constants (generic 2-3 MW class turbine power curve) ----
CUT_IN_MS = 3.0
RATED_MS = 12.0
CUT_OUT_MS = 25.0

# NASA POWER wind speed is measured at 10m height, but turbine hubs sit far
# higher (~80m for a 2-3MW class turbine). Wind speed increases substantially
# with height, so applying a power curve directly to 10m data badly
# understates real turbine output. Extrapolate using the standard wind
# power-law profile: v_hub = v_ref * (h_hub / h_ref) ^ alpha.
# alpha = 0.14 is a common default for open/flat terrain (coastal Tamil Nadu
# wind corridors are reasonably open) — a real project should use a
# site-specific alpha or, better, pull NASA POWER's 50m wind speed field
# directly if available and extrapolate the smaller remaining gap to 80m.
WIND_MEASUREMENT_HEIGHT_M = 10
TURBINE_HUB_HEIGHT_M = 80
WIND_SHEAR_ALPHA = 0.14


def extrapolate_to_hub_height(v_ref_ms):
    return v_ref_ms * (TURBINE_HUB_HEIGHT_M / WIND_MEASUREMENT_HEIGHT_M) ** WIND_SHEAR_ALPHA


def solar_output_mw(irradiance_kwh_m2, installed_mw=SOLAR_INSTALLED_MW):
    """
    irradiance_kwh_m2: daily total irradiance (kWh/m2/day)
    Returns: estimated daily average solar output (MW), scaled to installed capacity.
    Reference case: STC irradiance of 1 kW/m2 x panel_efficiency x performance_ratio
    defines the "nameplate" output fraction; we scale daily irradiance relative to
    a reference full-sun-equivalent-hours benchmark of 5.5 kWh/m2/day (rough
    India annual average) to get a capacity utilization factor.
    """
    reference_irradiance = 5.5  # kWh/m2/day, rough long-run average reference
    capacity_factor = np.clip(irradiance_kwh_m2 / reference_irradiance, 0, 1.3) * PERFORMANCE_RATIO
    return installed_mw * capacity_factor


def wind_output_mw(wind_speed_ms, installed_mw=WIND_INSTALLED_MW):
    """Standard cubic power-curve approximation, vectorized.
    Input wind_speed_ms is assumed to be at WIND_MEASUREMENT_HEIGHT_M (10m,
    matching NASA POWER) and is extrapolated to hub height before applying
    the power curve.
    """
    v = extrapolate_to_hub_height(np.asarray(wind_speed_ms, dtype=float))
    cf = np.zeros_like(v)
    ramp = (v >= CUT_IN_MS) & (v < RATED_MS)
    cf[ramp] = ((v[ramp] - CUT_IN_MS) / (RATED_MS - CUT_IN_MS)) ** 3
    rated = (v >= RATED_MS) & (v <= CUT_OUT_MS)
    cf[rated] = 1.0
    # above cut-out: turbines shut down for safety -> 0 output
    return installed_mw * cf


def main():
    df = pd.read_csv(DATA_DIR / "merged_dataset.csv", parse_dates=["date"])

    df["solar_potential_mw"] = solar_output_mw(df["solar_irradiance_kwh_m2"])
    df["wind_potential_mw"] = wind_output_mw(df["wind_speed_10m_ms"])
    df["renewable_potential_mw"] = df["solar_potential_mw"] + df["wind_potential_mw"]

    df["demand_gap_mw"] = df["peak_demand_mw"] - df["renewable_potential_mw"]
    # positive gap = renewables cannot cover peak demand alone; negative = surplus

    out_cols = ["date", "solar_potential_mw", "wind_potential_mw",
                "renewable_potential_mw", "peak_demand_mw", "demand_gap_mw"]
    df[out_cols].to_csv(REPORT_DIR / "renewable_potential.csv", index=False)

    print("Renewable potential summary (whole dataset):")
    print(df[["solar_potential_mw", "wind_potential_mw", "renewable_potential_mw", "demand_gap_mw"]].describe().T[["mean", "std", "min", "max"]])

    avg_coverage_pct = (df["renewable_potential_mw"] / df["peak_demand_mw"]).mean() * 100
    print(f"\nOn average, solar+wind potential could cover {avg_coverage_pct:.1f}% of peak demand"
          f" (given assumed {SOLAR_INSTALLED_MW} MW solar + {WIND_INSTALLED_MW} MW wind installed capacity).")

    # chart: last 2 years for readability
    recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=730)]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(recent["date"], recent["peak_demand_mw"], label="Peak Demand", color="black", linewidth=1.3)
    ax.plot(recent["date"], recent["renewable_potential_mw"], label="Solar+Wind Potential", color="#2a9d8f", linewidth=1.1)
    ax.fill_between(recent["date"], recent["peak_demand_mw"], recent["renewable_potential_mw"],
                     where=(recent["peak_demand_mw"] > recent["renewable_potential_mw"]),
                     color="#e63946", alpha=0.15, label="Supply gap (thermal/other needed)")
    ax.set_title(f"Peak Demand vs Renewable Potential (assumed {SOLAR_INSTALLED_MW}MW solar + {WIND_INSTALLED_MW}MW wind)")
    ax.set_ylabel("MW")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "renewable_potential_vs_demand.png", dpi=130)
    print(f"\nSaved chart to {OUT_DIR / 'renewable_potential_vs_demand.png'}")
    print(f"Saved data to {REPORT_DIR / 'renewable_potential.csv'}")


if __name__ == "__main__":
    main()
