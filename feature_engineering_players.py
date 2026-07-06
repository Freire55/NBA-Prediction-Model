import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

print("Loading player logs and matchups...")
logs_df = pd.read_csv(DATA_DIR / "raw_player_game_logs.csv")
matchups_df = pd.read_csv(DATA_DIR / "ml_ready_matchups.csv")

logs_df['PLAYER_ID'] = logs_df['PLAYER_ID'].astype(str)
logs_df['GAME_DATE'] = pd.to_datetime(logs_df['GAME_DATE'])
logs_df = logs_df.sort_values(by=['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

# Parse minutes
def parse_minutes(min_val):
    try:
        if isinstance(min_val, str) and ':' in min_val:
            m, s = min_val.split(':')
            return float(m) + (float(s) / 60)
        return float(min_val)
    except:
        return 0.0

logs_df['MINUTES_NUM'] = logs_df['MIN'].apply(parse_minutes)

# Calculate Game Score
logs_df['GAME_SCORE'] = (
    logs_df['PTS'] + (0.4 * logs_df['FGM']) - (0.7 * logs_df['FGA']) - 
    (0.4 * (logs_df['FTA'] - logs_df['FTM'])) + (0.7 * logs_df['OREB']) + 
    (0.3 * logs_df['DREB']) + logs_df['STL'] + (0.7 * logs_df['AST']) + 
    (0.7 * logs_df['BLK']) - (0.4 * logs_df['PF']) - logs_df['TOV']
)

print("Calculating chronologically pure player impact and minutes...")
# Use past games only
logs_df['PLAYER_FORM_ROLLING_5'] = logs_df.groupby('PLAYER_ID')['GAME_SCORE'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
logs_df['MINUTES_ROLLING_5'] = logs_df.groupby('PLAYER_ID')['MINUTES_NUM'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

logs_df['PLAYER_FORM_ROLLING_5'] = logs_df['PLAYER_FORM_ROLLING_5'].fillna(0.0)
logs_df['MINUTES_ROLLING_5'] = logs_df['MINUTES_ROLLING_5'].fillna(0.0)

# Estimate expected impact
logs_df['EXPECTED_IMPACT'] = logs_df['PLAYER_FORM_ROLLING_5'] * logs_df['MINUTES_ROLLING_5']

# Roll players up to the team level
roster_agg = logs_df.groupby(['GAME_ID', 'TEAM_ID'])['EXPECTED_IMPACT'].agg(['sum', 'std']).reset_index()
roster_agg = roster_agg.rename(columns={'sum': 'ACTIVE_ROSTER_FORM_SUM', 'std': 'ACTIVE_ROSTER_FORM_STD'})
roster_agg['ACTIVE_ROSTER_FORM_STD'] = roster_agg['ACTIVE_ROSTER_FORM_STD'].fillna(0.0)

# Merge into matchups
matchups_df['HOME_GAME_ID'] = matchups_df['HOME_GAME_ID'].astype(str)
matchups_df['AWAY_GAME_ID'] = matchups_df['AWAY_GAME_ID'].astype(str)
matchups_df['HOME_TEAM_ID'] = matchups_df['HOME_TEAM_ID'].astype(str)
matchups_df['AWAY_TEAM_ID'] = matchups_df['AWAY_TEAM_ID'].astype(str)
roster_agg['GAME_ID'] = roster_agg['GAME_ID'].astype(str)
roster_agg['TEAM_ID'] = roster_agg['TEAM_ID'].astype(str)

matchups_df = pd.merge(matchups_df, roster_agg.add_prefix('HOME_'), left_on=['HOME_GAME_ID', 'HOME_TEAM_ID'], right_on=['HOME_GAME_ID', 'HOME_TEAM_ID'], how='left')
matchups_df = pd.merge(matchups_df, roster_agg.add_prefix('AWAY_'), left_on=['AWAY_GAME_ID', 'AWAY_TEAM_ID'], right_on=['AWAY_GAME_ID', 'AWAY_TEAM_ID'], how='left')

# Build roster deltas
matchups_df['DELTA_ACTIVE_ROSTER_FORM_SUM'] = matchups_df['HOME_ACTIVE_ROSTER_FORM_SUM'] - matchups_df['AWAY_ACTIVE_ROSTER_FORM_SUM']
matchups_df['DELTA_ACTIVE_ROSTER_FORM_STD'] = matchups_df['HOME_ACTIVE_ROSTER_FORM_STD'] - matchups_df['AWAY_ACTIVE_ROSTER_FORM_STD']

matchups_df = matchups_df.fillna(0)  # Fill any gaps
output_file = DATA_DIR / "ml_ready_matchups_players.csv"
matchups_df.to_csv(output_file, index=False)
print(f"Success! Active Roster impact mapped to '{output_file.name}'.")