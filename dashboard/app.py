"""
app.py — Combined dashboard for Tamil Nadu Climate-Aware Power Planning.
Run from project root:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
DATA_DIR   = ROOT / "data"

st.set_page_config(page_title="TN Power Planning", layout="wide", page_icon="⚡")

@st.cache_data
def load_csv(name, dates=None):
    p = REPORT_DIR / name
    if not p.exists():
        p = ROOT / name
    return pd.read_csv(p, parse_dates=dates) if p.exists() else None

# ── Sidebar toggle ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👁️ View Mode")
    view = st.radio(
        label="view",
        options=["🔬 Expert Mode", "✨ Power Story"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    if view == "✨ Power Story":
        st.info("You're in **Power Story** mode — plain English, big picture, no jargon.")
    else:
        st.info("You're in **Expert Mode** — full metrics, model comparisons, and technical charts.")

# ── Load data (shared) ──────────────────────────────────────────────────────
renewables = load_csv("renewable_potential.csv",      dates=["date"])
emissions  = load_csv("emissions_estimates.csv",       dates=["date"])
opt_mix    = load_csv("optimized_generation_mix.csv",  dates=["date"])
model_comp = load_csv("model_comparison_final.csv")
forecast   = load_csv("future_forecast_2026_2029.csv", dates=["date"])
hist       = load_csv("data/merged_dataset.csv",       dates=["date"])

COLORS = {"coal": "#555555", "gas": "#f4a261", "hydro": "#457b9d",
          "nuclear": "#9b2226", "solar": "#e9c46a", "wind": "#2a9d8f"}

# ════════════════════════════════════════════════════════════════════════════
# EXPERT MODE
# ════════════════════════════════════════════════════════════════════════════
if view == "🔬 Expert Mode":

    st.title("🌦️ Tamil Nadu Climate-Aware Power Planning Dashboard")
    st.caption("Peak demand forecasting, renewable generation potential, carbon emissions, "
               "and generation-mix optimization for Tamil Nadu, driven by climate data.")

    col1, col2, col3, col4 = st.columns(4)

    if model_comp is not None:
        model_comp = model_comp.set_index(model_comp.columns[0])
        for c in ["RMSE", "MAE", "MAPE", "R2"]:
            model_comp[c] = pd.to_numeric(model_comp[c], errors="coerce")
        model_comp = model_comp.dropna(subset=["MAPE"])
        if not model_comp.empty:
            best_idx  = int(model_comp["MAPE"].astype(float).argmin())
            best_mape = model_comp.iloc[best_idx]["MAPE"]
            best_row  = model_comp.index[best_idx]
            col1.metric("Best model MAPE", f"{float(best_mape):.2f}%", help=f"Model: {best_row}")
        else:
            col1.metric("Best model MAPE", "—")
    else:
        col1.metric("Best model MAPE", "—")

    if renewables is not None:
        latest = renewables.iloc[-1]
        col2.metric("Latest forecasted peak demand", f"{latest['peak_demand_mw']:,.0f} MW")
        col3.metric("Latest renewable potential",    f"{latest['renewable_potential_mw']:,.0f} MW")

    if emissions is not None:
        col4.metric("Latest grid emissions intensity",
                    f"{emissions.iloc[-1]['emissions_intensity_g_per_kwh']:.0f} gCO2/kWh")

    st.divider()

    if renewables is not None:
        min_date, max_date = renewables["date"].min(), renewables["date"].max()
        date_range = st.slider(
            "Date range",
            min_value=min_date.to_pydatetime(), max_value=max_date.to_pydatetime(),
            value=(max(min_date, max_date - pd.Timedelta(days=730)).to_pydatetime(),
                   max_date.to_pydatetime()),
            format="YYYY-MM-DD",
        )
    else:
        date_range = None

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Demand Forecast", "☀️ Renewable Potential", "🌍 Emissions", "⚡ Optimized Generation Mix"]
    )

    with tab1:
        st.subheader("Peak Demand — Model Comparison")
        if model_comp is not None:
            st.dataframe(model_comp.style.format("{:.3f}"), use_container_width=True)
        img_path = ROOT / "outputs" / "hybrid_model_vs_actual.png"
        if img_path.exists():
            st.image(str(img_path), caption="Main model (Fourier+Trend + GBM residual) vs. actual",
                     use_container_width=True)

    with tab2:
        st.subheader("Solar + Wind Generation Potential vs. Demand")
        if renewables is not None and date_range:
            filtered = renewables[(renewables["date"] >= date_range[0]) & (renewables["date"] <= date_range[1])]
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(filtered["date"], filtered["peak_demand_mw"], label="Peak Demand", color="black")
            ax.plot(filtered["date"], filtered["renewable_potential_mw"], label="Renewable Potential", color="#2a9d8f")
            ax.fill_between(filtered["date"], filtered["peak_demand_mw"], filtered["renewable_potential_mw"],
                            where=(filtered["peak_demand_mw"] > filtered["renewable_potential_mw"]),
                            color="#e63946", alpha=0.15, label="Supply gap")
            ax.legend(); ax.set_ylabel("MW")
            st.pyplot(fig)
            coverage = (filtered["renewable_potential_mw"] / filtered["peak_demand_mw"]).mean() * 100
            st.metric("Avg. renewable coverage of peak demand (selected range)", f"{coverage:.1f}%")

    with tab3:
        st.subheader("Grid Carbon Emissions")
        if emissions is not None and date_range:
            filtered = emissions[(emissions["date"] >= date_range[0]) & (emissions["date"] <= date_range[1])]
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(filtered["date"], filtered["grid_emissions_tonnes"], label="Current mix", color="black")
            ax.plot(filtered["date"], filtered["emissions_scenario_plus10pp_re"], label="+10pp renewable", color="#f4a261")
            ax.plot(filtered["date"], filtered["emissions_scenario_plus20pp_re"], label="+20pp renewable", color="#2a9d8f")
            ax.legend(); ax.set_ylabel("Tonnes CO2 / day")
            st.pyplot(fig)

    with tab4:
        st.subheader("Optimized Generation Mix (last 90 days of data)")
        if opt_mix is not None:
            om = opt_mix.set_index("date")
            fig, ax = plt.subplots(figsize=(12, 5))
            om[["coal", "gas", "hydro", "nuclear", "solar", "wind"]].plot.area(ax=ax, alpha=0.85)
            ax.set_ylabel("MW")
            st.pyplot(fig)
            st.caption("Simplified LP optimizer (cost + emissions weighted objective).")

    st.divider()

# ════════════════════════════════════════════════════════════════════════════
# POWER STORY MODE
# ════════════════════════════════════════════════════════════════════════════
else:

    st.title("⚡ Tamil Nadu Power at a Glance")
    st.markdown("### How much electricity do we need — and how clean can we make it?")
    st.markdown("---")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    if renewables is not None:
        latest_demand    = renewables["peak_demand_mw"].iloc[-1]
        latest_renewable = renewables["renewable_potential_mw"].iloc[-1]
        k1.metric("🏙️ Today's Peak Power Need", f"{latest_demand:,.0f} MW",
                  help="How much electricity Tamil Nadu needs at its busiest moment today")
        k2.metric("☀️ Clean Energy Available",  f"{latest_renewable:,.0f} MW",
                  help="How much solar + wind power is available right now")
        k3.metric("🌿 Renewable Coverage",       f"{latest_renewable/latest_demand*100:.0f}%",
                  help="How much of today's need can be met by solar and wind alone")
    if emissions is not None:
        k4.metric("💨 Grid Carbon Intensity",
                  f"{emissions['emissions_intensity_g_per_kwh'].iloc[-1]:.0f} g CO₂/kWh",
                  help="How much CO₂ is released per unit of electricity — lower is better.")
    st.markdown("---")

    # Section 1 — Future demand
    st.subheader("📅 How much power will Tamil Nadu need in the coming years?")
    st.markdown(
        "Our AI model looked at 16 years of electricity usage and weather patterns "
        "to predict how demand will grow. Think of it like a weather forecast — "
        "but for electricity, years into the future."
    )
    if forecast is not None and hist is not None:
        hist_tail = hist[hist["date"] >= "2024-01-01"][["date", "peak_demand_mw"]]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(hist_tail["date"], hist_tail["peak_demand_mw"],
                color="#333333", linewidth=1.5, label="Past (actual usage)")
        ax.plot(forecast["date"], forecast["predicted_peak_demand_mw"],
                color="#2a9d8f", linewidth=2, label="Future (AI forecast)")
        ax.axvline(pd.Timestamp("2026-07-18"), color="red", linestyle="--",
                   linewidth=1.2, label="Forecast starts here")
        ax.fill_between(forecast["date"], forecast["predicted_peak_demand_mw"],
                        alpha=0.12, color="#2a9d8f")
        ax.set_ylabel("Peak Power Demand (MW)", fontsize=11)
        ax.set_title("Electricity Demand — Past & Future (2024–2029)", fontsize=13)
        ax.legend(fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        fig.tight_layout()
        st.pyplot(fig)

        forecast["year"] = forecast["date"].dt.year
        yearly = forecast.groupby("year")["predicted_peak_demand_mw"].mean().round(0)
        cols = st.columns(len(yearly))
        for col, (yr, val) in zip(cols, yearly.items()):
            col.metric(f"📆 {yr} average", f"{val:,.0f} MW")
        st.info(
            f"💡 **What this means:** Demand is expected to grow by roughly "
            f"{((yearly.iloc[-1]/yearly.iloc[0])-1)*100:.0f}% over the next 3 years. "
            "That's like adding the electricity needs of several new cities on top of today's usage."
        )
    st.markdown("---")

    # Section 2 — Renewables
    st.subheader("☀️ Can solar and wind keep up with our growing needs?")
    st.markdown(
        "Tamil Nadu is one of India's biggest solar and wind states. "
        "The chart below shows how much clean energy is available each day "
        "compared to what we actually need. The **red area** is the gap that "
        "still needs to be filled by coal, gas, or other sources."
    )
    if renewables is not None:
        recent = renewables[renewables["date"] >= renewables["date"].max() - pd.Timedelta(days=365)]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(recent["date"], recent["peak_demand_mw"],
                color="#333333", linewidth=1.5, label="Power needed")
        ax.plot(recent["date"], recent["renewable_potential_mw"],
                color="#2a9d8f", linewidth=1.5, label="Solar + Wind available")
        ax.fill_between(recent["date"], recent["peak_demand_mw"], recent["renewable_potential_mw"],
                        where=(recent["peak_demand_mw"] > recent["renewable_potential_mw"]),
                        color="#e63946", alpha=0.25, label="Gap (needs other sources)")
        ax.fill_between(recent["date"], recent["peak_demand_mw"], recent["renewable_potential_mw"],
                        where=(recent["renewable_potential_mw"] >= recent["peak_demand_mw"]),
                        color="#2a9d8f", alpha=0.2, label="Surplus clean energy")
        ax.set_ylabel("Power (MW)", fontsize=11)
        ax.set_title("Daily Power Need vs Clean Energy Available (Last 12 Months)", fontsize=13)
        ax.legend(fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        fig.tight_layout()
        st.pyplot(fig)

        avg_cov      = (recent["renewable_potential_mw"] / recent["peak_demand_mw"]).mean() * 100
        surplus_days = (recent["renewable_potential_mw"] >= recent["peak_demand_mw"]).sum()
        c1, c2 = st.columns(2)
        c1.metric("Average clean energy coverage",      f"{avg_cov:.0f}% of daily need")
        c2.metric("Days with 100% renewable surplus",   f"{surplus_days} days last year")
    st.markdown("---")

    # Section 3 — Emissions
    st.subheader("🌍 Is our electricity getting cleaner?")
    st.markdown(
        "Every unit of electricity from coal releases CO₂ into the atmosphere. "
        "The good news: as Tamil Nadu adds more solar and wind, the grid gets cleaner every year. "
        "The chart below shows what happens if we push even harder on renewables."
    )
    if emissions is not None:
        annual = load_csv("annual_emissions_scenarios.csv")
        if annual is not None:
            annual = annual.rename(columns={annual.columns[0]: "year"})
            annual["year"] = pd.to_numeric(annual["year"], errors="coerce")
            annual = annual.dropna(subset=["year"])
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(annual["year"].astype(int) - 0.25, annual["current_emissions_Mt"],
                   width=0.25, color="#555555", label="Today's energy mix")
            ax.bar(annual["year"].astype(int),        annual["scenario_plus10pp_Mt"],
                   width=0.25, color="#f4a261",       label="If we add 10% more renewables")
            ax.bar(annual["year"].astype(int) + 0.25, annual["scenario_plus20pp_Mt"],
                   width=0.25, color="#2a9d8f",       label="If we add 20% more renewables")
            ax.set_ylabel("CO₂ Emissions (Million Tonnes)", fontsize=11)
            ax.set_title("Annual CO₂ Emissions — What If We Used More Renewables?", fontsize=13)
            ax.legend(fontsize=10)
            fig.tight_layout()
            st.pyplot(fig)

            if len(annual) >= 2:
                saved_10 = annual["current_emissions_Mt"].iloc[-1] - annual["scenario_plus10pp_Mt"].iloc[-1]
                saved_20 = annual["current_emissions_Mt"].iloc[-1] - annual["scenario_plus20pp_Mt"].iloc[-1]
                c1, c2 = st.columns(2)
                c1.metric("CO₂ saved with +10% renewables", f"{saved_10:.1f} Mt/year")
                c2.metric("CO₂ saved with +20% renewables", f"{saved_20:.1f} Mt/year")
            st.info("💡 **In simple terms:** Adding 20% more renewable energy is like taking millions of cars off the road every year.")
    st.markdown("---")

    # Section 4 — Generation mix
    st.subheader("⚡ Where does Tamil Nadu's electricity actually come from?")
    st.markdown(
        "Electricity comes from many sources — coal, gas, hydro, nuclear, solar, and wind. "
        "Our smart optimizer figures out the best mix each day to keep costs low "
        "and pollution minimal. Here's what the last 90 days looked like."
    )
    if opt_mix is not None:
        om      = opt_mix.set_index("date")
        sources = ["coal", "gas", "hydro", "nuclear", "solar", "wind"]
        labels  = {"coal": "Coal 🏭", "gas": "Gas 🔥", "hydro": "Hydro 💧",
                   "nuclear": "Nuclear ⚛️", "solar": "Solar ☀️", "wind": "Wind 🌬️"}

        fig, ax = plt.subplots(figsize=(14, 5))
        bottom = np.zeros(len(om))
        for s in sources:
            if s in om.columns:
                ax.fill_between(om.index, bottom, bottom + om[s].values,
                                label=labels[s], color=COLORS[s], alpha=0.85)
                bottom += om[s].values
        ax.set_ylabel("Power Generated (MW)", fontsize=11)
        ax.set_title("Daily Power Generation Mix — Last 90 Days (Optimized)", fontsize=13)
        ax.legend(loc="upper left", fontsize=9, ncol=3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        fig.tight_layout()
        st.pyplot(fig)

        avg = om[sources].mean()
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.pie(avg.values, labels=[labels[s] for s in sources],
                colors=[COLORS[s] for s in sources],
                autopct="%1.0f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax2.set_title("Average Energy Mix", fontsize=12)
        c1, c2, c3 = st.columns([1, 1.2, 1])
        with c2:
            st.pyplot(fig2)

        re_pct = (avg["solar"] + avg["wind"]) / avg.sum() * 100
        st.success(f"🌿 On average, **{re_pct:.0f}%** of optimized generation comes from clean solar and wind energy.")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:grey; font-size:13px;'>"
        "Tamil Nadu Climate-Aware Power Planning Project"
        "</div>",
        unsafe_allow_html=True
    )
