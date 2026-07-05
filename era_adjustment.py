import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# Load the basic box scores
try:
    df = pd.read_csv(DATA_DIR / "raw_historical_nba.csv")
except FileNotFoundError:
    print("Error: 'raw_historical_nba.csv' not found in the data folder.")
    exit()

# Isolate only the actual stats. We ignore ID columns like GAME_ID so we don't accidentally normalize them
exclude_cols = ['SEASON_ID', 'TEAM_ID', 'GAME_ID', 'MIN', 'VIDEO_AVAILABLE']
stats_to_normalize = [col for col in df.select_dtypes(include='number').columns if col not in exclude_cols]

# Convert every raw stat into a Z-Score based strictly on its specific season
for stat in stats_to_normalize:
    if stat in df.columns:
        z_col_name = f"Z_{stat}"
        df[z_col_name] = df.groupby('SEASON_ID')[stat].transform(lambda x: (x - x.mean()) / x.std())

# Fill any missing data with 0 (which perfectly represents the league average in a Z-Score)
z_cols = [col for col in df.columns if col.startswith('Z_')]
df[z_cols] = df[z_cols].fillna(0)

# Save the era-adjusted basic stats
output_file = DATA_DIR / "era_adjusted_nba.csv"
df.to_csv(output_file, index=False)
print(f"Success! Era-adjusted data saved to '{output_file}'.")