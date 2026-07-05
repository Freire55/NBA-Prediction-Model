import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# 1. Load the data
print("Loading player logs, player ratings, and master matchups...")
logs_df = pd.read_csv(DATA_DIR / "raw_player_game_logs.csv")
ratings_df = pd.read_csv(DATA_DIR / "advanced_player_ratings.csv")
matchups_df = pd.read_csv(DATA_DIR / "ml_ready_matchups_advanced.csv")

# Ensure IDs are strings
logs_df['PLAYER_ID'] = logs_df['PLAYER_ID'].astype(str)
logs_df['SEASON_ID'] = logs_df['SEASON_ID'].astype(str)
ratings_df['PLAYER_ID'] = ratings_df['PLAYER_ID'].astype(str)
ratings_df['SEASON_ID'] = ratings_df['SEASON_ID'].astype(str)

# --- PLAYER HOT STREAK (FORM) ---
print("Calculating player hot streaks...")
logs_df['GAME_DATE'] = pd.to_datetime(logs_df['GAME_DATE'])
logs_df = logs_df.sort_values(by=['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

logs_df['PLAYER_FORM_ROLLING_5'] = logs_df.groupby('PLAYER_ID')['PLUS_MINUS'].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
)
logs_df['PLAYER_FORM_ROLLING_5'] = logs_df['PLAYER_FORM_ROLLING_5'].fillna(0.0)

# 2. Fix Minutes formatting
def parse_minutes(min_val):
    try:
        if isinstance(min_val, str) and ':' in min_val:
            m, s = min_val.split(':')
            return float(m) + (float(s) / 60)
        return float(min_val)
    except:
        return 0.0

logs_df['MINUTES_NUM'] = logs_df['MIN'].apply(parse_minutes)

# 3. Merge Ratings
player_impact = pd.merge(
    logs_df[['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'SEASON_ID', 'MINUTES_NUM', 'PLAYER_FORM_ROLLING_5']], 
    ratings_df[['PLAYER_ID', 'SEASON_ID', 'PIE', 'NET_RATING']], 
    left_on=['PLAYER_ID', 'SEASON_ID'],
    right_on=['PLAYER_ID', 'SEASON_ID'],
    how='left'
)
player_impact['PIE'] = player_impact['PIE'].fillna(0.10)
player_impact['NET_RATING'] = player_impact['NET_RATING'].fillna(0.0)

# 4. Calculate Weighting
player_impact['WEIGHTED_PIE'] = player_impact['PIE'] * player_impact['MINUTES_NUM']
player_impact['WEIGHTED_NET_RATING'] = player_impact['NET_RATING'] * player_impact['MINUTES_NUM']
player_impact['WEIGHTED_PLAYER_FORM'] = player_impact['PLAYER_FORM_ROLLING_5'] * player_impact['MINUTES_NUM']

# 5. Aggregate up to Team Level (Volume + Dispersion)
roster_agg = player_impact.groupby(['GAME_ID', 'TEAM_ID'])[['WEIGHTED_PIE', 'WEIGHTED_NET_RATING', 'WEIGHTED_PLAYER_FORM']].agg(['sum', 'std']).reset_index()
roster_agg.columns = ['_'.join(col).strip('_') for col in roster_agg.columns.values]

roster_strength = roster_agg.rename(columns={
    'WEIGHTED_PIE_sum': 'ACTIVE_ROSTER_PIE',
    'WEIGHTED_NET_RATING_sum': 'ACTIVE_ROSTER_NET_RATING',
    'WEIGHTED_PLAYER_FORM_sum': 'ACTIVE_ROSTER_FORM',
    'WEIGHTED_PIE_std': 'ACTIVE_ROSTER_PIE_STD',
    'WEIGHTED_NET_RATING_std': 'ACTIVE_ROSTER_NET_RATING_STD',
    'WEIGHTED_PLAYER_FORM_std': 'ACTIVE_ROSTER_FORM_STD'
})

# --- CLUTCH & COACHING FEATURES (STUB) ---
# Note: These features require game-specific situational data. 
# We initialize them as baseline zeros to be populated by the pipeline flow.
roster_strength['CLUTCH_PERFORMANCE_5G'] = 0.0
roster_strength['COACH_ELO_RATING'] = 1500.0
# ------------------------------------------

# 6. Merge into Matchups
matchups_df['HOME_GAME_ID'] = matchups_df['HOME_GAME_ID'].astype(str)
matchups_df['AWAY_GAME_ID'] = matchups_df['AWAY_GAME_ID'].astype(str)
matchups_df['HOME_TEAM_ID'] = matchups_df['HOME_TEAM_ID'].astype(str)
matchups_df['AWAY_TEAM_ID'] = matchups_df['AWAY_TEAM_ID'].astype(str)
roster_strength['GAME_ID'] = roster_strength['GAME_ID'].astype(str)
roster_strength['TEAM_ID'] = roster_strength['TEAM_ID'].astype(str)

matchups_df = pd.merge(matchups_df, roster_strength.add_prefix('HOME_'), left_on=['HOME_GAME_ID', 'HOME_TEAM_ID'], right_on=['HOME_GAME_ID', 'HOME_TEAM_ID'], how='left')
matchups_df = pd.merge(matchups_df, roster_strength.add_prefix('AWAY_'), left_on=['AWAY_GAME_ID', 'AWAY_TEAM_ID'], right_on=['AWAY_GAME_ID', 'AWAY_TEAM_ID'], how='left')

# 7. Calculate the Deltas (including the new STD and Situational columns)
matchups_df['DELTA_ACTIVE_ROSTER_PIE'] = matchups_df['HOME_ACTIVE_ROSTER_PIE'] - matchups_df['AWAY_ACTIVE_ROSTER_PIE']
matchups_df['DELTA_ACTIVE_ROSTER_NET_RATING'] = matchups_df['HOME_ACTIVE_ROSTER_NET_RATING'] - matchups_df['AWAY_ACTIVE_ROSTER_NET_RATING']
matchups_df['DELTA_ACTIVE_ROSTER_FORM'] = matchups_df['HOME_ACTIVE_ROSTER_FORM'] - matchups_df['AWAY_ACTIVE_ROSTER_FORM']
matchups_df['DELTA_ACTIVE_ROSTER_PIE_STD'] = matchups_df['HOME_ACTIVE_ROSTER_PIE_STD'] - matchups_df['AWAY_ACTIVE_ROSTER_PIE_STD']
matchups_df['DELTA_ACTIVE_ROSTER_NET_RATING_STD'] = matchups_df['HOME_ACTIVE_ROSTER_NET_RATING_STD'] - matchups_df['AWAY_ACTIVE_ROSTER_NET_RATING_STD']
matchups_df['DELTA_ACTIVE_ROSTER_FORM_STD'] = matchups_df['HOME_ACTIVE_ROSTER_FORM_STD'] - matchups_df['AWAY_ACTIVE_ROSTER_FORM_STD']
matchups_df['DELTA_CLUTCH_PERFORMANCE'] = matchups_df['HOME_CLUTCH_PERFORMANCE_5G'] - matchups_df['AWAY_CLUTCH_PERFORMANCE_5G']
matchups_df['DELTA_COACH_ELO'] = matchups_df['HOME_COACH_ELO_RATING'] - matchups_df['AWAY_COACH_ELO_RATING']

matchups_df = matchups_df.copy()
numeric_cols = matchups_df.select_dtypes(include='number').columns
matchups_df[numeric_cols] = matchups_df[numeric_cols].fillna(0)

# 8. Save
output_file = DATA_DIR / "ml_ready_matchups_players.csv"
matchups_df.to_csv(output_file, index=False)
print(f"Success! Active Roster impact (including Dispersion, Clutch, and Coach) mapped to '{output_file}'.")