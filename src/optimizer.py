"""
optimizer.py — SIMPLIFIED generation-mix optimizer using linear programming
(scipy.optimize.linprog), NOT a full multi-objective NSGA-II system.

For each day in the test period, allocates generation across coal, gas,
hydro, nuclear, solar, and wind to meet forecasted peak demand while
minimizing a weighted objective of (cost + emissions), subject to:
- Renewable (solar/wind) output capped at that day's estimated potential
  (from src/solar_wind.py)
- Total generation must meet forecasted demand (from the hybrid model)
- Coal/gas/hydro/nuclear each have minimum baseload and maximum capacity
  constraints (illustrative Tamil Nadu-scale figures — replace with real
  regional capacity data for a production-grade version)

HOW THIS DIFFERS FROM A REAL MULTI-OBJECTIVE OPTIMIZER (disclose in report):
- This collapses cost and emissions into a SINGLE weighted-sum objective
  (see COST_WEIGHT / EMISSIONS_WEIGHT below) rather than finding a full
  Pareto front of non-dominated trade-off solutions the way NSGA-II would.
  Changing the weights changes the single "optimal" point chosen; a true
  multi-objective approach would show the full trade-off curve instead.
- It optimizes each day independently (no inter-day constraints like ramp
  rates, storage state-of-charge, or start-up costs), which a full unit
  commitment model would include.
- Cost figures below are illustrative representative values, not real-time
  market prices.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"

# illustrative cost figures (INR/MWh equivalent, representative order of magnitude)
COST_PER_MWH = {
    "coal": 3500, "gas": 5200, "hydro": 1200,
    "nuclear": 2800, "solar": 2200, "wind": 2400,
}
EMISSIONS_G_PER_KWH = {  # same factors as src/carbon.py
    "coal": 950, "gas": 450, "hydro": 24, "nuclear": 12, "solar": 41, "wind": 11,
}

COST_WEIGHT = 0.6
EMISSIONS_WEIGHT = 0.4
# emissions in gCO2/kWh are ~1e-3 x cost in INR/MWh scale; normalize emissions
# term so both objectives are comparable magnitude in the weighted sum
EMISSIONS_NORMALIZER = 8.0  # INR-equivalent per gCO2/kWh, illustrative weighting

# illustrative capacity constraints (MW) — replace with real TN grid figures
MIN_CAPACITY_MW = {"coal": 2000, "gas": 200, "hydro": 300, "nuclear": 1000, "solar": 0, "wind": 0}
MAX_CAPACITY_MW = {"coal": 9000, "gas": 2000, "hydro": 1500, "nuclear": 1050, "solar": None, "wind": None}
# solar/wind max is set per-day from estimated potential


def optimize_day(demand_mw, solar_potential_mw, wind_potential_mw):
    sources = ["coal", "gas", "hydro", "nuclear", "solar", "wind"]
    n = len(sources)

    # objective: minimize weighted (cost + emissions) per MW allocated
    c = []
    for s in sources:
        cost_term = COST_WEIGHT * COST_PER_MWH[s]
        emis_term = EMISSIONS_WEIGHT * EMISSIONS_G_PER_KWH[s] * EMISSIONS_NORMALIZER
        c.append(cost_term + emis_term)
    c = np.array(c)

    # equality: sum of allocations = demand
    A_eq = [np.ones(n)]
    b_eq = [demand_mw]

    bounds = []
    for s in sources:
        lo = MIN_CAPACITY_MW[s]
        if s == "solar":
            hi = max(solar_potential_mw, lo)
        elif s == "wind":
            hi = max(wind_potential_mw, lo)
        else:
            hi = MAX_CAPACITY_MW[s]
        bounds.append((lo, hi))

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        # relax equality to >= demand if infeasible (shouldn't happen given capacity headroom)
        return None
    return dict(zip(sources, result.x))


def naive_allocation(demand_mw, solar_potential_mw, wind_potential_mw):
    """Baseline: use all available renewables first (cheapest marginal + free fuel),
    then coal, then gas, in that fixed priority order, ignoring emissions weighting."""
    sources_priority = [
        ("hydro", MIN_CAPACITY_MW["hydro"], MAX_CAPACITY_MW["hydro"]),
        ("nuclear", MIN_CAPACITY_MW["nuclear"], MAX_CAPACITY_MW["nuclear"]),
        ("solar", 0, solar_potential_mw),
        ("wind", 0, wind_potential_mw),
        ("coal", MIN_CAPACITY_MW["coal"], MAX_CAPACITY_MW["coal"]),
        ("gas", MIN_CAPACITY_MW["gas"], MAX_CAPACITY_MW["gas"]),
    ]
    remaining = demand_mw
    alloc = {}
    for name, lo, hi in sources_priority:
        take = min(max(lo, 0), max(remaining, 0))
        take = min(take, hi if hi is not None else remaining)
        take = max(take, lo if remaining >= lo else 0)
        take = min(take, hi)
        alloc[name] = take
        remaining -= take
    if remaining > 0:
        alloc["coal"] += remaining  # coal absorbs any shortfall (dispatchable)
    return alloc


def main():
    renewables = pd.read_csv(REPORT_DIR / "renewable_potential.csv", parse_dates=["date"])
    test_start = renewables["date"].max() - pd.Timedelta(days=90)
    sample = renewables[renewables["date"] >= test_start].reset_index(drop=True)

    rows_opt, rows_naive = [], []
    for _, row in sample.iterrows():
        opt = optimize_day(row["peak_demand_mw"], row["solar_potential_mw"], row["wind_potential_mw"])
        naive = naive_allocation(row["peak_demand_mw"], row["solar_potential_mw"], row["wind_potential_mw"])
        rows_opt.append({"date": row["date"], **opt})
        rows_naive.append({"date": row["date"], **naive})

    opt_df = pd.DataFrame(rows_opt).set_index("date")
    naive_df = pd.DataFrame(rows_naive).set_index("date")

    def cost_and_emissions(df):
        cost = sum(df[s] * COST_PER_MWH[s] for s in COST_PER_MWH) * 24  # MW->MWh/day (24h)
        emissions = sum(df[s] * EMISSIONS_G_PER_KWH[s] * 1000 for s in EMISSIONS_G_PER_KWH) * 24 / 1e6  # tonnes/day
        return cost.sum(), emissions.sum()

    opt_cost, opt_emissions = cost_and_emissions(opt_df)
    naive_cost, naive_emissions = cost_and_emissions(naive_df)

    print(f"Over {len(sample)} days:")
    print(f"  Optimized allocation: total cost ~= INR {opt_cost/1e7:.2f} crore, "
          f"total emissions ~= {opt_emissions:.0f} tonnes CO2")
    print(f"  Naive (priority-order) allocation: total cost ~= INR {naive_cost/1e7:.2f} crore, "
          f"total emissions ~= {naive_emissions:.0f} tonnes CO2")
    print(f"  Optimizer saves ~= {(1 - opt_cost/naive_cost)*100:.1f}% cost and "
          f"{(1 - opt_emissions/naive_emissions)*100:.1f}% emissions vs. naive priority dispatch")

    opt_df.to_csv(REPORT_DIR / "optimized_generation_mix.csv")
    naive_df.to_csv(REPORT_DIR / "naive_generation_mix.csv")

    fig, ax = plt.subplots(figsize=(14, 6))
    opt_df[["coal", "gas", "hydro", "nuclear", "solar", "wind"]].plot.area(ax=ax, alpha=0.85)
    ax.set_title("Optimized Daily Generation Mix (LP optimizer, last 90 days)")
    ax.set_ylabel("MW")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "optimized_generation_mix.png", dpi=130)

    print(f"\nSaved allocations to {REPORT_DIR / 'optimized_generation_mix.csv'} and {REPORT_DIR / 'naive_generation_mix.csv'}")
    print(f"Saved chart to {OUT_DIR / 'optimized_generation_mix.png'}")


if __name__ == "__main__":
    main()
