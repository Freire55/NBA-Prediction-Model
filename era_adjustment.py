import pandas as pd
import numpy as np

print("Loading raw historical data...")
try:
    # 1. Load the master dataset we created in Phase 1
    df = pd.read_csv("raw_historical_nba.csv")
    print(f"Successfully loaded {len(df)} games.")
except FileNotFoundError:
    print("Error: 'raw_historical_nba.csv' not found. Please run fetch_history.py first.")
    exit()

# 2. Define the core stats we want to era-adjust.
# These are the counting stats that fluctuate heavily depending on the era's pace.
stats_to_normalize = [
    'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 
    'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF'
]

print("Calculating era-adjusted Z-scores for each season...")

# 3. Calculate Z-Scores grouped strictly by Season
# The NBA API provides a 'SEASON_ID' column. We group the data by this ID so that 
# the mean and standard deviation are calculated purely within that single year.
for stat in stats_to_normalize:
    if stat in df.columns:
        z_col_name = f"Z_{stat}"
        
        # We use pandas .transform() to apply the Z-score formula: (Value - Mean) / Standard Deviation
        # This keeps the dataframe the exact same shape but appends our new advanced columns.
        df[z_col_name] = df.groupby('SEASON_ID')[stat].transform(
            lambda x: (x - x.mean()) / x.std()
        )

# 4. Fill any potential NaN values (in case a stat had 0 standard deviation, though rare in NBA data)
z_cols = [col for col in df.columns if col.startswith('Z_')]
df[z_cols] = df[z_cols].fillna(0)

# 5. Save the mathematically normalized dataset
output_file = "era_adjusted_nba.csv"
df.to_csv(output_file, index=False)

print(f"Success! Era-adjusted data saved to '{output_file}'.")
print("\nHere is a preview of the raw Points vs the new Era-Adjusted Z-Points:")
# Print a quick preview of the transformation to verify it worked
print(df[['SEASON_ID', 'GAME_DATE', 'TEAM_ABBREVIATION', 'PTS', 'Z_PTS']].head(10))