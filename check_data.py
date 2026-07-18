import pandas as pd

def load(path):
    df = pd.read_csv(path)
    # handle date saved as unnamed index column
    if 'date' not in df.columns and 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    return df

files = {
    'merged_dataset': 'data/merged_dataset.csv',
    'train': 'data/train.csv',
    'val': 'data/val.csv',
    'test': 'data/test.csv',
}

for name, path in files.items():
    df = load(path)
    nulls = df.isnull().sum().sum()
    print(f"{name}: {len(df)} rows | {df.shape[1]} cols | nulls={nulls} | {df['date'].min().date()} to {df['date'].max().date()}")

train = load('data/train.csv')
val   = load('data/val.csv')
test  = load('data/test.csv')

print("\nSplit continuity:")
print(f"  train ends {train['date'].max().date()} | val starts {val['date'].min().date()} | gap={(val['date'].min()-train['date'].max()).days-1}d")
print(f"  val ends   {val['date'].max().date()} | test starts {test['date'].min().date()} | gap={(test['date'].min()-val['date'].max()).days-1}d")
print(f"  train+val+test={len(train)+len(val)+len(test)} | merged={len(pd.read_csv('data/merged_dataset.csv'))}")

required = ['date','temperature_2m_c','humidity_2m_pct','precipitation_mm',
            'wind_speed_10m_ms','solar_irradiance_kwh_m2','peak_demand_mw',
            'demand_mwh','generation_mwh','coal_share','solar_share','wind_share']
missing = [c for c in required if c not in train.columns]
print(f"  missing required columns: {missing if missing else 'None'}")

print("\nValue ranges (merged_dataset):")
full = pd.read_csv('data/merged_dataset.csv')
for col in ['temperature_2m_c','humidity_2m_pct','wind_speed_10m_ms','solar_irradiance_kwh_m2','peak_demand_mw']:
    print(f"  {col}: min={full[col].min():.2f} max={full[col].max():.2f} mean={full[col].mean():.2f} nulls={full[col].isnull().sum()}")

print("\nReports check:")
import os
for f in os.listdir('reports'):
    size = os.path.getsize(f'reports/{f}')
    print(f"  reports/{f}: {size} bytes {'⚠ EMPTY' if size==0 else 'OK'}")

print("\nModels check:")
for f in os.listdir('models'):
    size = os.path.getsize(f'models/{f}')
    print(f"  models/{f}: {size/1024:.1f} KB {'⚠ EMPTY' if size==0 else 'OK'}")
