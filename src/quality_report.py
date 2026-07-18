"""quality_report.py — audits date coverage, gaps, and missing values."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

df = pd.read_csv(DATA_DIR / "merged_dataset.csv", parse_dates=["date"])

full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
gaps = full_range.difference(df["date"])

lines = []
lines.append("=" * 70)
lines.append("  DATA QUALITY REPORT — Tamil Nadu Climate-Electricity Pipeline")
lines.append("=" * 70)
lines.append("")
lines.append("NOTE: this run uses a SYNTHETIC dataset (see data/generate_synthetic_data.py)")
lines.append("built to match the statistical properties of the real pipeline output.")
lines.append("Swap in your real merged_dataset.csv to get a report on real data.")
lines.append("")
lines.append(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
lines.append(f"Total rows: {len(df)}   Expected calendar days: {len(full_range)}")
lines.append(f"Coverage: {len(df) / len(full_range) * 100:.1f}%")
lines.append(f"Date gaps: {len(gaps)}")
lines.append("")
lines.append("Missing values per column:")
miss = df.isna().sum()
for col, cnt in miss.items():
    if cnt > 0:
        lines.append(f"  {col:30s} {cnt:6d}  ({cnt/len(df)*100:.2f}%)")
if miss.sum() == 0:
    lines.append("  none")
lines.append("")
lines.append("Descriptive statistics:")
lines.append(df.describe().T[["mean", "std", "min", "max"]].to_string())
lines.append("=" * 70)

report = "\n".join(lines)
print(report)
with open(REPORT_DIR / "data_quality_report.txt", "w") as f:
    f.write(report)
