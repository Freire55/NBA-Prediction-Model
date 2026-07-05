import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# Load the season-long advanced team stats
df = pd.read_csv(DATA_DIR / "advanced_team_stats.csv")

# Isolate only the advanced stats we want to normalize
exclude_cols = ['TEAM_ID', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'SEASON_ID']
stats_to_normalize = [col for col in df.select_dtypes(include='number').columns if col not in exclude_cols]

# Convert every advanced stat into a Z-Score compared to the rest of the league for that specific year
for stat in stats_to_normalize:
    if stat in df.columns:
        z_col_name = f"Z_{stat}"
        df[z_col_name] = df.groupby('SEASON_ID')[stat].transform(lambda x: (x - x.mean()) / x.std())

# Clean up any missing or blank data by setting it to the league average (0)
z_cols = [col for col in df.columns if col.startswith('Z_')]
df[z_cols] = df[z_cols].fillna(0)

# Save the era-adjusted advanced stats
output_file = DATA_DIR / "era_adjusted_advanced.csv"
df.to_csv(output_file, index=False)
print(f"Success! Era-adjusted advanced data saved to '{output_file}'.")