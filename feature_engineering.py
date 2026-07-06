import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# Load the data and keep it in time order
df = pd.read_csv(DATA_DIR / "era_adjusted_nba.csv")
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df['GAME_ID'] = df['GAME_ID'].astype(str)
df['SEASON_ID'] = df['SEASON_ID'].astype(str)
df = df.sort_values(by=['TEAM_ABBREVIATION', 'GAME_DATE']).reset_index(drop=True)

# Add a simple season-year signal
df['SEASON_YEAR'] = df['SEASON_ID'].astype(str).str[1:].astype(int)

# Build the fatigue features
print("Calculating Deep Schedule Density (Fatigue Flags)...")
df['PREV_GAME_DATE'] = df.groupby('TEAM_ABBREVIATION')['GAME_DATE'].shift(1)
df['REST_DAYS'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days
df['REST_DAYS'] = df['REST_DAYS'].fillna(5.0)  # Start the season with a full rest guess
df['B2B'] = np.where(df['REST_DAYS'] == 1, 1, 0)  # Mark back-to-backs

# Check for short rest stretches
df['DATE_MINUS_2'] = df.groupby('TEAM_ABBREVIATION')['GAME_DATE'].shift(2)
df['DATE_MINUS_3'] = df.groupby('TEAM_ABBREVIATION')['GAME_DATE'].shift(3)
df['3_IN_4'] = np.where((df['GAME_DATE'] - df['DATE_MINUS_2']).dt.days <= 3, 1, 0)
df['4_IN_5'] = np.where((df['GAME_DATE'] - df['DATE_MINUS_3']).dt.days <= 4, 1, 0)

# Track road trips
df['IS_AWAY'] = df['MATCHUP'].str.contains(' @ ').astype(int)
df['AWAY_GROUP'] = (df['IS_AWAY'] != df.groupby('TEAM_ABBREVIATION')['IS_AWAY'].shift(1)).cumsum()
df['ROAD_TRIP_LENGTH'] = np.where(df['IS_AWAY'] == 1, df.groupby(['TEAM_ABBREVIATION', 'AWAY_GROUP']).cumcount() + 1, 0)

# Build rolling team stats from past games only
print("Calculating chronologically pure rolling ratings...")
# Estimate possessions
df['POSSESSIONS'] = df['FGA'] + 0.44 * df['FTA'] - df['OREB'] + df['TOV']

# Use the last five games
df['ROLLING_PTS_5'] = df.groupby('TEAM_ABBREVIATION')['PTS'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
df['ROLLING_POSS_5'] = df.groupby('TEAM_ABBREVIATION')['POSSESSIONS'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
# Turn that into offensive rating
df['ROLLING_OFF_RATING'] = (df['ROLLING_PTS_5'] / df['ROLLING_POSS_5']) * 100
df['ROLLING_OFF_RATING'] = df['ROLLING_OFF_RATING'].fillna(100.0)  # Use a neutral baseline

# Roll the other Z stats too
z_columns = [col for col in df.columns if col.startswith('Z_')]
for col in z_columns:
    df[f"{col}_ROLLING_5"] = df.groupby('TEAM_ABBREVIATION')[col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

# Add strength of schedule
df['OPP_ABBREVIATION'] = df['MATCHUP'].str[-3:]
team_strength = df[['GAME_ID', 'TEAM_ABBREVIATION', 'Z_PLUS_MINUS_ROLLING_5']].copy()
team_strength.rename(columns={'Z_PLUS_MINUS_ROLLING_5': 'OPP_PRE_GAME_STRENGTH'}, inplace=True)
df = pd.merge(df, team_strength, left_on=['GAME_ID', 'OPP_ABBREVIATION'], right_on=['GAME_ID', 'TEAM_ABBREVIATION'], suffixes=('', '_DROP'))
df.drop(columns=['TEAM_ABBREVIATION_DROP'], inplace=True)
df['OPP_PRE_GAME_STRENGTH'] = df['OPP_PRE_GAME_STRENGTH'].fillna(0)
# Average the last five opponents
df['SOS_ROLLING_5'] = df.groupby('TEAM_ABBREVIATION')['OPP_PRE_GAME_STRENGTH'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)

# Simulate Elo game by game
print("Simulating historical Elo Ratings (Chronological Engine)...")
elo_dict = {}
elo_records = []

# Go through games in order
unique_games = df.drop_duplicates(subset=['GAME_ID']).sort_values(by='GAME_DATE')

for index, row in unique_games.iterrows():
    game_id = row['GAME_ID']
    game_rows = df[df['GAME_ID'] == game_id]
    
    home_subset = game_rows[game_rows['MATCHUP'].str.contains(' vs. ', na=False)]
    away_subset = game_rows[game_rows['MATCHUP'].str.contains(' @ ', na=False)]
    if len(home_subset) == 0 or len(away_subset) == 0:
        continue  # Skip bad API rows
        
    home_team = home_subset.iloc[0]['TEAM_ABBREVIATION']
    away_team = away_subset.iloc[0]['TEAM_ABBREVIATION']
    
    # Start new teams at 1500
    if home_team not in elo_dict: elo_dict[home_team] = 1500
    if away_team not in elo_dict: elo_dict[away_team] = 1500
    
    home_elo_pre = elo_dict[home_team]
    away_elo_pre = elo_dict[away_team]
    
    # Save the pre-game Elo
    elo_records.append({'GAME_ID': game_id, 'TEAM_ABBREVIATION': home_team, 'PRE_GAME_ELO': home_elo_pre})
    elo_records.append({'GAME_ID': game_id, 'TEAM_ABBREVIATION': away_team, 'PRE_GAME_ELO': away_elo_pre})
    
    # Update both sides after the result
    home_prob = 1.0 / (1.0 + 10.0 ** ((away_elo_pre - home_elo_pre) / 400.0))
    home_won = 1 if home_subset.iloc[0]['PTS'] > away_subset.iloc[0]['PTS'] else 0
    
    elo_dict[home_team] = home_elo_pre + 20 * (home_won - home_prob)
    elo_dict[away_team] = away_elo_pre + 20 * ((1 - home_won) - (1 - home_prob))

elo_df = pd.DataFrame(elo_records)
df = pd.merge(df, elo_df, on=['GAME_ID', 'TEAM_ABBREVIATION'], how='left')

# Merge home and away rows
home_df = df[df['MATCHUP'].str.contains(' vs. ')].copy().add_prefix('HOME_')
away_df = df[df['MATCHUP'].str.contains(' @ ')].copy().add_prefix('AWAY_')
matchups_df = pd.merge(home_df, away_df, left_on='HOME_GAME_ID', right_on='AWAY_GAME_ID')

# Set the target
matchups_df['HOME_WIN'] = np.where(matchups_df['HOME_PTS'] > matchups_df['AWAY_PTS'], 1, 0)

# Build home-away differences
matchups_df['REST_ADVANTAGE'] = matchups_df['HOME_REST_DAYS'] - matchups_df['AWAY_REST_DAYS']
matchups_df['SEASON_YEAR'] = matchups_df['HOME_SEASON_YEAR'] 
matchups_df['DELTA_ELO'] = matchups_df['HOME_PRE_GAME_ELO'] - matchups_df['AWAY_PRE_GAME_ELO']
matchups_df['DELTA_ROAD_TRIP_LENGTH'] = matchups_df['HOME_ROAD_TRIP_LENGTH'] - matchups_df['AWAY_ROAD_TRIP_LENGTH']
matchups_df['DELTA_3_IN_4'] = matchups_df['HOME_3_IN_4'] - matchups_df['AWAY_3_IN_4']
matchups_df['DELTA_4_IN_5'] = matchups_df['HOME_4_IN_5'] - matchups_df['AWAY_4_IN_5']
matchups_df['DELTA_SOS_ROLLING_5'] = matchups_df['HOME_SOS_ROLLING_5'] - matchups_df['AWAY_SOS_ROLLING_5']
matchups_df['DELTA_ROLLING_OFF_RATING'] = matchups_df['HOME_ROLLING_OFF_RATING'] - matchups_df['AWAY_ROLLING_OFF_RATING']

for col in z_columns:
    delta_col = f"DELTA_{col}_ROLLING_5"
    matchups_df[delta_col] = matchups_df[f"HOME_{col}_ROLLING_5"] - matchups_df[f"AWAY_{col}_ROLLING_5"]

matchups_df.dropna(subset=['DELTA_Z_PTS_ROLLING_5'], inplace=True)
output_file = DATA_DIR / "ml_ready_matchups.csv"
matchups_df.to_csv(output_file, index=False)
print(f"Success! Built matchup rows with pure rolling features.")