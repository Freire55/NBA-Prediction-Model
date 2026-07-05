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

# 1. Parse Minutes
def parse_minutes(min_val):
    try:
        if isinstance(min_val, str) and ':' in min_val:
            m, s = min_val.split(':')
            return float(m) + (float(s) / 60)
        return float(min_val)
    except:
        return 0.0

logs_df['MINUTES_NUM'] = logs_df['MIN'].apply(parse_minutes)

# 2. Calculate Hollinger's Game Score (A proxy for Player Impact)
# Formula: PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA - FTM) + 0.7*OREB + 0.3*DREB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV
logs_df['GAME_SCORE'] = (
    logs_df['PTS'] + (0.4 * logs_df['FGM']) - (0.7 * logs_df['FGA']) - 
    (0.4 * (logs_df['FTA'] - logs_df['FTM'])) + (0.7 * logs_df['OREB']) + 
    (0.3 * logs_df['DREB']) + logs_df['STL'] + (0.7 * logs_df['AST']) + 
    (0.7 * logs_df['BLK']) - (0.4 * logs_df['PF']) - logs_df['TOV']
)

print("Calculating chronologically pure player impact and minutes...")
# 3. CRITICAL LEAK-PROOFING: The shift(1) ensures we only judge a player based on past games.
# We also calculate "Rolling Minutes" to estimate playing time BEFORE tip-off, removing rotational leaks.
logs_df['PLAYER_FORM_ROLLING_5'] = logs_df.groupby('PLAYER_ID')['GAME_SCORE'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
logs_df['MINUTES_ROLLING_5'] = logs_df.groupby('PLAYER_ID')['MINUTES_NUM'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

logs_df['PLAYER_FORM_ROLLING_5'] = logs_df['PLAYER_FORM_ROLLING_5'].fillna(0.0)
logs_df['MINUTES_ROLLING_5'] = logs_df['MINUTES_ROLLING_5'].fillna(0.0)

# Calculate expected impact based on expected playing time
logs_df['EXPECTED_IMPACT'] = logs_df['PLAYER_FORM_ROLLING_5'] * logs_df['MINUTES_ROLLING_5']

# 4. Aggregate to Team Level (Volume + Dispersion)
# We aggregate the players who actually dressed for THIS specific game.
roster_agg = logs_df.groupby(['GAME_ID', 'TEAM_ID'])['EXPECTED_IMPACT'].agg(['sum', 'std']).reset_index()
roster_agg = roster_agg.rename(columns={'sum': 'ACTIVE_ROSTER_FORM_SUM', 'std': 'ACTIVE_ROSTER_FORM_STD'})
roster_agg['ACTIVE_ROSTER_FORM_STD'] = roster_agg['ACTIVE_ROSTER_FORM_STD'].fillna(0.0)

# 5. Merge into Matchups
matchups_df['HOME_GAME_ID'] = matchups_df['HOME_GAME_ID'].astype(str)
matchups_df['AWAY_GAME_ID'] = matchups_df['AWAY_GAME_ID'].astype(str)
matchups_df['HOME_TEAM_ID'] = matchups_df['HOME_TEAM_ID'].astype(str)
matchups_df['AWAY_TEAM_ID'] = matchups_df['AWAY_TEAM_ID'].astype(str)
roster_agg['GAME_ID'] = roster_agg['GAME_ID'].astype(str)
roster_agg['TEAM_ID'] = roster_agg['TEAM_ID'].astype(str)

matchups_df = pd.merge(matchups_df, roster_agg.add_prefix('HOME_'), left_on=['HOME_GAME_ID', 'HOME_TEAM_ID'], right_on=['HOME_GAME_ID', 'HOME_TEAM_ID'], how='left')
matchups_df = pd.merge(matchups_df, roster_agg.add_prefix('AWAY_'), left_on=['AWAY_GAME_ID', 'AWAY_TEAM_ID'], right_on=['AWAY_GAME_ID', 'AWAY_TEAM_ID'], how='left')

# 6. Calculate the Deltas (Home Roster Strength vs Away Roster Strength)
matchups_df['DELTA_ACTIVE_ROSTER_FORM_SUM'] = matchups_df['HOME_ACTIVE_ROSTER_FORM_SUM'] - matchups_df['AWAY_ACTIVE_ROSTER_FORM_SUM']
matchups_df['DELTA_ACTIVE_ROSTER_FORM_STD'] = matchups_df['HOME_ACTIVE_ROSTER_FORM_STD'] - matchups_df['AWAY_ACTIVE_ROSTER_FORM_STD']

matchups_df = matchups_df.fillna(0) # Catch-all for any edge cases
output_file = DATA_DIR / "ml_ready_matchups_players.csv"
matchups_df.to_csv(output_file, index=False)
print(f"Success! Active Roster impact mapped to '{output_file.name}'.")