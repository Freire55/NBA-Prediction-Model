import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# Load both the base matchups and the advanced stats
matchups_df = pd.read_csv(DATA_DIR / "ml_ready_matchups.csv")
adv_df = pd.read_csv(DATA_DIR / "era_adjusted_advanced.csv")

# Keep only the IDs and Z-scores from the advanced dataset
adv_cols = ['TEAM_ID', 'SEASON_ID'] + [col for col in adv_df.columns if col.startswith('Z_')]
adv_clean = adv_df[adv_cols].copy()

# Convert season IDs to strings so they merge cleanly without pandas throwing type errors
matchups_df['HOME_SEASON_ID'] = matchups_df['HOME_SEASON_ID'].astype(str)
matchups_df['AWAY_SEASON_ID'] = matchups_df['AWAY_SEASON_ID'].astype(str)
adv_clean['SEASON_ID'] = adv_clean['SEASON_ID'].astype(str)

# Map the advanced stats to the Home team
adv_home = adv_clean.add_prefix('HOME_ADV_')
matchups_df = pd.merge(matchups_df, adv_home, left_on=['HOME_TEAM_ID', 'HOME_SEASON_ID'], right_on=['HOME_ADV_TEAM_ID', 'HOME_ADV_SEASON_ID'], how='left')

# Map the advanced stats to the Away team
adv_away = adv_clean.add_prefix('AWAY_ADV_')
matchups_df = pd.merge(matchups_df, adv_away, left_on=['AWAY_TEAM_ID', 'AWAY_SEASON_ID'], right_on=['AWAY_ADV_TEAM_ID', 'AWAY_ADV_SEASON_ID'], how='left')

# Calculate the Deltas for the advanced stats (Home Adv - Away Adv)
z_cols = [col for col in adv_clean.columns if col not in ('TEAM_ID', 'SEASON_ID')]
delta_dict = {}
for col in z_cols:
    delta_col = f"DELTA_ADV_{col}"
    delta_dict[delta_col] = matchups_df[f"HOME_ADV_{col}"] - matchups_df[f"AWAY_ADV_{col}"]

# Add all the new advanced Deltas to our main dataframe at once to prevent memory fragmentation
deltas_df = pd.DataFrame(delta_dict)
matchups_df = pd.concat([matchups_df, deltas_df], axis=1)

# Clean up any NaN values in numeric columns (replace with 0)
numeric_cols = matchups_df.select_dtypes(include='number').columns
matchups_df[numeric_cols] = matchups_df[numeric_cols].fillna(0)

output_file = DATA_DIR / "ml_ready_matchups_advanced.csv"
matchups_df.to_csv(output_file, index=False)
print(f"Success! Advanced stats merged into '{output_file}'.")