"""
split.py — chronological (non-shuffled) train/val/test split.

Time-series data must be split by DATE, never randomly: a random split lets
future information leak into training (the model would effectively be
"tested on the past using knowledge of the future"), which silently inflates
every evaluation metric and gives a false sense of model quality.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def chronological_split(df: pd.DataFrame, date_col: str = "date",
                         train_frac: float = 0.70, val_frac: float = 0.15):
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "merged_dataset.csv", parse_dates=["date"])
    train, val, test = chronological_split(df)
    train.to_csv(DATA_DIR / "train.csv", index=False)
    val.to_csv(DATA_DIR / "val.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)
    print(f"train: {train['date'].min().date()} to {train['date'].max().date()}  ({len(train)} rows)")
    print(f"val:   {val['date'].min().date()} to {val['date'].max().date()}  ({len(val)} rows)")
    print(f"test:  {test['date'].min().date()} to {test['date'].max().date()}  ({len(test)} rows)")
