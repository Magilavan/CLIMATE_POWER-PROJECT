"""
public_dashboard.py — Plain-English dashboard for non-technical users.
Run from project root:
    streamlit run dashboard/public_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
OUT_DIR    = ROOT / "outputs"

st.set_page_config(page_title="Tamil Nadu Power Planner", layout="wide", page_icon="⚡")

# ── helpers ────────────────────────────────────────────────────────────────
@st.cache_data
def load(name, dates=None):
    p = REPORT_DIR / name
    return pd.read_csv(p, parse_dates=dates) if p.exists() else None

renewables = load("renewable_potential.csv",      dates=["date"])
emissions  = load("emissions_estimates.csv",       dates=["date"])
forecast   = load("future_forecast_2026_2029.csv", dates=["date"])
opt_mix    = load("optimized_generation_mix.csv",  dates=["date"])
hist       = load("../data/merged_dataset.csv",    dates=["date"])

COLORS = {"coal":"#555555","gas":"#f4a261","hydro":"#457b9d",
          "nuclear":"#9b2226","solar":"#e9c46a","wind":"#2a9d8f"}

# ── header ─────────────────────────────────────────────────────────────────
st.title("⚡ Tamil Nadu Power at a Glance")
st.markdown("### How much electricity do we need — and how clean can we make it?")
st.markdown("---")

# ── 4 big plain-English KPIs ───────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

if renewables is not None:
    latest_demand    = renewables["peak_demand_mw"].iloc[-1]
    latest_renewable = renewables["renewable_potential_mw"].iloc[-1]
    coverage         = latest_renewable / latest_demand * 100
    k1.metric("🏙️ Today's Peak Power Need",  f"{latest_demand:,.0f} MW",
              help="How much electricity Tamil Nadu needs at its busiest moment today")
    k2.metric("☀️ Clean Energy Available",   f"{latest_renewable:,.0f} MW",
              help="How much solar + wind power is available right now")
    k3.metric("🌿 Renewable Coverage",        f"{coverage:.0f}%",
              help="How much of today's need can be met by solar and wind alone")

if emissions is not None:
    intensity = emissions["emissions_intensity_g_per_kwh"].iloc[-1]
    k4.metric("💨 Grid Carbon Intensity",     f"{intensity:.0f} g CO₂/kWh",
              help="How much CO₂ is released per unit of electricity — lower is better. "
                   "A phone charger running for an hour emits about 10g CO₂.")

st.markdown("---")

# ── SECTION 1 — Future demand ──────────────────────────────────────────────
st.subheader("📅 How much power will Tamil Nadu need in the coming years?")
st.markdown(
    "Our AI model looked at 16 years of electricity usage and weather patterns "
    "to predict how demand will grow. Think of it like a weather forecast — "
    "but for electricity, years into the future."
)

if forecast is not None and hist is not None:
    hist_tail = hist[hist["date"] >= "2024-01-01"][["date","peak_demand_mw"]]

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

    # plain yearly summary
    forecast["year"] = forecast["date"].dt.year
    yearly = forecast.groupby("year")["predicted_peak_demand_mw"].mean().round(0)
    cols = st.columns(len(yearly))
    for col, (yr, val) in zip(cols, yearly.items()):
        col.metric(f"📆 {yr} average", f"{val:,.0f} MW")

    st.info(
        "💡 **What this means:** Demand is expected to grow by roughly "
        f"{((yearly.iloc[-1]/yearly.iloc[0])-1)*100:.0f}% over the next 3 years. "
        "That's like adding the electricity needs of several new cities on top of today's usage."
    )

st.markdown("---")

# ── SECTION 2 — Renewables vs demand ──────────────────────────────────────
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
    ax.fill_between(recent["date"],
                    recent["peak_demand_mw"], recent["renewable_potential_mw"],
                    where=(recent["peak_demand_mw"] > recent["renewable_potential_mw"]),
                    color="#e63946", alpha=0.25, label="Gap (needs other sources)")
    ax.fill_between(recent["date"],
                    recent["peak_demand_mw"], recent["renewable_potential_mw"],
                    where=(recent["renewable_potential_mw"] >= recent["peak_demand_mw"]),
                    color="#2a9d8f", alpha=0.2, label="Surplus clean energy")
    ax.set_ylabel("Power (MW)", fontsize=11)
    ax.set_title("Daily Power Need vs Clean Energy Available (Last 12 Months)", fontsize=13)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    st.pyplot(fig)

    avg_cov = (recent["renewable_potential_mw"] / recent["peak_demand_mw"]).mean() * 100
    surplus_days = (recent["renewable_potential_mw"] >= recent["peak_demand_mw"]).sum()
    c1, c2 = st.columns(2)
    c1.metric("Average clean energy coverage", f"{avg_cov:.0f}% of daily need")
    c2.metric("Days with 100% renewable surplus", f"{surplus_days} days last year")

st.markdown("---")

# ── SECTION 3 — Emissions ──────────────────────────────────────────────────
st.subheader("🌍 Is our electricity getting cleaner?")
st.markdown(
    "Every unit of electricity from coal releases CO₂ into the atmosphere. "
    "The good news: as Tamil Nadu adds more solar and wind, the grid gets cleaner every year. "
    "The chart below shows what happens if we push even harder on renewables."
)

if emissions is not None:
    annual = load("annual_emissions_scenarios.csv")
    if annual is not None:
        annual.columns = annual.columns.str.strip()
        annual["date"] = pd.to_numeric(annual["date"] if "date" in annual.columns
                                       else annual.iloc[:, 0], errors="coerce")
        annual = annual.dropna(subset=[annual.columns[0]])
        annual = annual.rename(columns={annual.columns[0]: "year"})

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
            c1.metric("CO₂ saved with +10% renewables", f"{saved_10:.1f} Mt/year",
                      help="Million tonnes of CO₂ avoided per year")
            c2.metric("CO₂ saved with +20% renewables", f"{saved_20:.1f} Mt/year",
                      help="Million tonnes of CO₂ avoided per year")

        st.info(
            "💡 **In simple terms:** Adding 20% more renewable energy is like taking "
            f"millions of cars off the road every year."
        )

st.markdown("---")

# ── SECTION 4 — Generation mix ─────────────────────────────────────────────
st.subheader("⚡ Where does Tamil Nadu's electricity actually come from?")
st.markdown(
    "Electricity comes from many sources — coal, gas, hydro, nuclear, solar, and wind. "
    "Our smart optimizer figures out the best mix each day to keep costs low "
    "and pollution minimal. Here's what the last 90 days looked like."
)

if opt_mix is not None:
    opt_mix = opt_mix.set_index("date")
    sources = ["coal","gas","hydro","nuclear","solar","wind"]
    labels  = {"coal":"Coal 🏭","gas":"Gas 🔥","hydro":"Hydro 💧",
               "nuclear":"Nuclear ⚛️","solar":"Solar ☀️","wind":"Wind 🌬️"}

    fig, ax = plt.subplots(figsize=(14, 5))
    bottom = np.zeros(len(opt_mix))
    for s in sources:
        if s in opt_mix.columns:
            ax.fill_between(opt_mix.index, bottom, bottom + opt_mix[s].values,
                            label=labels[s], color=COLORS[s], alpha=0.85)
            bottom += opt_mix[s].values
    ax.set_ylabel("Power Generated (MW)", fontsize=11)
    ax.set_title("Daily Power Generation Mix — Last 90 Days (Optimized)", fontsize=13)
    ax.legend(loc="upper left", fontsize=9, ncol=3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    st.pyplot(fig)

    # donut chart of average mix
    avg = opt_mix[sources].mean()
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax2.pie(
        avg.values, labels=[labels[s] for s in sources],
        colors=[COLORS[s] for s in sources],
        autopct="%1.0f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax2.set_title("Average Energy Mix", fontsize=12)
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        st.pyplot(fig2)

    re_pct = (avg["solar"] + avg["wind"]) / avg.sum() * 100
    st.success(f"🌿 On average, **{re_pct:.0f}%** of optimized generation comes from clean solar and wind energy.")

st.markdown("---")

# ── footer ─────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center; color:grey; font-size:13px;'>"
    "Tamil Nadu Climate-Aware Power Planning Project"
    "</div>",
    unsafe_allow_html=True
)
