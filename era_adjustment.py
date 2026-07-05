import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# Load the basic box scores
try:
    df = pd.read_csv(DATA_DIR / "raw_historical_nba.csv")
except FileNotFoundError:
    print("Error: 'raw_historical_nba.csv' not found. Run fetch_history.py first.")
    exit()

# We need to exclude identifying columns from the math so we don't accidentally turn a GAME_ID into a Z-Score!
exclude_cols = ['SEASON_ID', 'TEAM_ID', 'GAME_ID', 'MIN', 'VIDEO_AVAILABLE']
stats_to_normalize = [col for col in df.select_dtypes(include='number').columns if col not in exclude_cols]

for stat in stats_to_normalize:
    if stat in df.columns:
        z_col_name = f"Z_{stat}"
        df[z_col_name] = df.groupby('SEASON_ID')[stat].transform(lambda x: (x - x.mean()) / x.std())

# If a stat is missing, we fill it with 0. 
# Because these are Z-scores, '0' mathematically represents the exact league average.
z_cols = [col for col in df.columns if col.startswith('Z_')]
df[z_cols] = df[z_cols].fillna(0)

output_file = DATA_DIR / "era_adjusted_nba.csv"
df.to_csv(output_file, index=False)
print(f"Success! Era-adjusted data saved to '{output_file.name}'.")