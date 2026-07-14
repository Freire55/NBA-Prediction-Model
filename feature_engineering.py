"""
Builds chronologically correct matchup-level features from historical NBA
team box scores.

The script engineers fatigue, rolling team performance, strength of schedule,
historical Elo ratings, and home-away delta features using only information
available prior to each game, preventing target leakage.

Input:
    data/era_adjusted_nba.csv

Output:
    data/ml_ready_matchups.csv
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# ======================================================
# Constants & Configuration
# ======================================================
DATA_DIR = Path(__file__).resolve().parent / "data"

# Rolling feature parameters
ROLLING_WINDOW = 8

# Elo parameters
INITIAL_ELO = 1500
ELO_K_FACTOR = 20
ELO_DIVISOR = 400

# Basketball constants
POSSESSION_FT_WEIGHT = 0.44
DEFAULT_REST_DAYS = 5.0
DEFAULT_OFF_RATING = 100.0

# ======================================================
# Logging Setup
# ======================================================
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ======================================================
# Helper Functions
# ======================================================
def load_and_sort_data(filepath: Path) -> pd.DataFrame:
    """Loads era-adjusted data and enforces strict chronological sorting."""
    df = pd.read_csv(filepath)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['GAME_ID'] = df['GAME_ID'].astype(str)
    df['SEASON_ID'] = df['SEASON_ID'].astype(str)
    
    df = df.sort_values(by=['TEAM_ABBREVIATION', 'GAME_DATE']).reset_index(drop=True)
    df['SEASON_YEAR'] = df['SEASON_ID'].astype(str).str[1:].astype(int)
    
    return df

def rolling_mean(series: pd.Series, rolling_window: int = ROLLING_WINDOW) -> pd.Series:
    """Computes a rolling mean using only prior observations."""
    return series.shift(1).rolling(rolling_window, min_periods=1).mean()

def ewma(series: pd.Series, span: int = ROLLING_WINDOW) -> pd.Series:
    """Computes an exponentially weighted moving average using only prior observations."""
    return series.shift(1).ewm(adjust=False, span=span).mean()

def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers fatigue and schedule density flags."""
    team_groups = df.groupby('TEAM_ABBREVIATION')
    
    df['PREV_GAME_DATE'] = team_groups['GAME_DATE'].shift(1)
    df['REST_DAYS'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days
    df['REST_DAYS'] = df['REST_DAYS'].fillna(DEFAULT_REST_DAYS)
    df['B2B'] = np.where(df['REST_DAYS'] == 1, 1, 0)

    # Identify grueling stretches
    df['DATE_MINUS_2'] = team_groups['GAME_DATE'].shift(2)
    df['DATE_MINUS_3'] = team_groups['GAME_DATE'].shift(3)
    df['3_IN_4'] = np.where((df['GAME_DATE'] - df['DATE_MINUS_2']).dt.days <= 3, 1, 0)
    df['4_IN_5'] = np.where((df['GAME_DATE'] - df['DATE_MINUS_3']).dt.days <= 4, 1, 0)

    # Track road trip exhaustion
    df['IS_AWAY'] = df['MATCHUP'].str.contains(' @ ').astype(int)
    df['AWAY_GROUP'] = (df['IS_AWAY'] != team_groups['IS_AWAY'].shift(1)).cumsum()
    df['ROAD_TRIP_LENGTH'] = np.where(df['IS_AWAY'] == 1, df.groupby(['TEAM_ABBREVIATION', 'AWAY_GROUP']).cumcount() + 1, 0)
    
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates chronologically pure rolling averages and strength of schedule."""
    team_groups = df.groupby('TEAM_ABBREVIATION')
    
    df['POSSESSIONS'] = df['FGA'] + POSSESSION_FT_WEIGHT * df['FTA'] - df['OREB'] + df['TOV']

    df[f'ROLLING_PTS_{ROLLING_WINDOW}'] = team_groups['PTS'].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).sum())
    df[f'ROLLING_POSS_{ROLLING_WINDOW}'] = team_groups['POSSESSIONS'].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).sum())
    
    df['ROLLING_OFF_RATING'] = (df[f'ROLLING_PTS_{ROLLING_WINDOW}'] / df[f'ROLLING_POSS_{ROLLING_WINDOW}']) * 100
    df['ROLLING_OFF_RATING'] = df['ROLLING_OFF_RATING'].fillna(DEFAULT_OFF_RATING)

    # Standardize historical memory for Z-stats
    z_columns = [col for col in df.columns if col.startswith('Z_')]
    for col in z_columns:
        df[f"{col}_ROLLING_{ROLLING_WINDOW}"] = team_groups[col].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())

    # Map past opponent strength
    df['OPP_ABBREVIATION'] = df['MATCHUP'].str[-3:]
    team_strength = df[['GAME_ID', 'TEAM_ABBREVIATION', f'Z_PLUS_MINUS_ROLLING_{ROLLING_WINDOW}']].copy()
    team_strength = team_strength.rename(columns={f'Z_PLUS_MINUS_ROLLING_{ROLLING_WINDOW}': 'OPP_PRE_GAME_STRENGTH'})
    
    df = df.merge(team_strength, left_on=['GAME_ID', 'OPP_ABBREVIATION'], right_on=['GAME_ID', 'TEAM_ABBREVIATION'], suffixes=('', '_DROP'))
    df = df.drop(columns=['TEAM_ABBREVIATION_DROP'])
    df['OPP_PRE_GAME_STRENGTH'] = df['OPP_PRE_GAME_STRENGTH'].fillna(0)
    
    df[f'SOS_ROLLING_{ROLLING_WINDOW}'] = df.groupby('TEAM_ABBREVIATION')['OPP_PRE_GAME_STRENGTH'].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()).fillna(0)

    return df

def simulate_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Simulates a continuous Elo rating timeline using pure Python iterations for speed."""
    current_elo = {}
    pre_game_elo_records = []

    games_by_id = {}
    for game_id, group in df.groupby("GAME_ID"):
        if len(group) != 2:
            continue
            
        team_a = group.iloc[0]
        team_b = group.iloc[1]

        if " vs. " in str(team_a["MATCHUP"]):
            home = team_a
            away = team_b
        else:
            home = team_b
            away = team_a

        games_by_id[game_id] = (
            home["TEAM_ABBREVIATION"],
            away["TEAM_ABBREVIATION"],
            home["PTS"],
            away["PTS"]
        )

    unique_games = df.drop_duplicates(subset=['GAME_ID']).sort_values(by='GAME_DATE')

    # Iterate over IDs only to prevent pandas iterrows() overhead
    for game_id in unique_games["GAME_ID"]:
        if game_id not in games_by_id:
            continue
            
        home_team, away_team, home_pts, away_pts = games_by_id[game_id]
        
        if home_team not in current_elo: current_elo[home_team] = INITIAL_ELO
        if away_team not in current_elo: current_elo[away_team] = INITIAL_ELO
        
        home_elo_pre = current_elo[home_team]
        away_elo_pre = current_elo[away_team]
        
        pre_game_elo_records.append({'GAME_ID': game_id, 'TEAM_ABBREVIATION': home_team, 'PRE_GAME_ELO': home_elo_pre})
        pre_game_elo_records.append({'GAME_ID': game_id, 'TEAM_ABBREVIATION': away_team, 'PRE_GAME_ELO': away_elo_pre})
        
        home_prob = 1.0 / (1.0 + 10.0 ** ((away_elo_pre - home_elo_pre) / ELO_DIVISOR))
        home_won = 1 if home_pts > away_pts else 0
        
        current_elo[home_team] = home_elo_pre + ELO_K_FACTOR * (home_won - home_prob)
        current_elo[away_team] = away_elo_pre + ELO_K_FACTOR * ((1 - home_won) - (1 - home_prob))

    elo_df = pd.DataFrame(pre_game_elo_records)
    df = df.merge(elo_df, on=['GAME_ID', 'TEAM_ABBREVIATION'], how='left')
    return df

def build_matchups(df: pd.DataFrame) -> pd.DataFrame:
    """Combines home and away rows into single matchup-level observations."""
    home_df = df[df['MATCHUP'].str.contains(' vs. ')].copy().add_prefix('HOME_')
    away_df = df[df['MATCHUP'].str.contains(' @ ')].copy().add_prefix('AWAY_')
    matchups_df = home_df.merge(away_df, left_on='HOME_GAME_ID', right_on='AWAY_GAME_ID')

    matchups_df['HOME_WIN'] = np.where(matchups_df['HOME_PTS'] > matchups_df['AWAY_PTS'], 1, 0)

    # Establish relative advantages
    matchups_df['REST_ADVANTAGE'] = matchups_df['HOME_REST_DAYS'] - matchups_df['AWAY_REST_DAYS']
    matchups_df['SEASON_YEAR'] = matchups_df['HOME_SEASON_YEAR'] 
    matchups_df['DELTA_ELO'] = matchups_df['HOME_PRE_GAME_ELO'] - matchups_df['AWAY_PRE_GAME_ELO']
    matchups_df['DELTA_ROAD_TRIP_LENGTH'] = matchups_df['HOME_ROAD_TRIP_LENGTH'] - matchups_df['AWAY_ROAD_TRIP_LENGTH']
    matchups_df['DELTA_3_IN_4'] = matchups_df['HOME_3_IN_4'] - matchups_df['AWAY_3_IN_4']
    matchups_df['DELTA_4_IN_5'] = matchups_df['HOME_4_IN_5'] - matchups_df['AWAY_4_IN_5']
    matchups_df[f'DELTA_SOS_ROLLING_{ROLLING_WINDOW}'] = matchups_df[f'HOME_SOS_ROLLING_{ROLLING_WINDOW}'] - matchups_df[f'AWAY_SOS_ROLLING_{ROLLING_WINDOW}']
    matchups_df['DELTA_ROLLING_OFF_RATING'] = matchups_df['HOME_ROLLING_OFF_RATING'] - matchups_df['AWAY_ROLLING_OFF_RATING']

    z_rolling_cols = [col.replace("HOME_", "") for col in home_df.columns if col.startswith("HOME_Z_") and col.endswith(f"_ROLLING_{ROLLING_WINDOW}")]
    for col in z_rolling_cols:
        delta_col = f"DELTA_{col}"
        matchups_df[delta_col] = matchups_df[f"HOME_{col}"] - matchups_df[f"AWAY_{col}"]

    matchups_df = matchups_df.dropna(subset=[f'DELTA_Z_PTS_ROLLING_{ROLLING_WINDOW}'])

    # ---------------------------------------------------------
    # Drop Post-Game Box Score Stats
    # ---------------------------------------------------------
    raw_box_score_stats = [
        'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
        'BLK', 'TOV', 'PF', 'PLUS_MINUS', 'POSSESSIONS'
    ]
    
    cols_to_drop = []
    for stat in raw_box_score_stats:
        cols_to_drop.extend([
            f"HOME_{stat}", f"AWAY_{stat}", 
            f"HOME_Z_{stat}", f"AWAY_Z_{stat}"
        ])
        
    matchups_df = matchups_df.drop(
        columns=[c for c in cols_to_drop if c in matchups_df.columns]
    )

    return matchups_df

# ======================================================
# Main Execution
# ======================================================
if __name__ == "__main__":
    logger.info("Loading and sorting era-adjusted data...")
    df = load_and_sort_data(DATA_DIR / "era_adjusted_nba.csv")

    logger.info("Calculating Deep Schedule Density (Fatigue Flags)...")
    df = add_schedule_features(df)

    logger.info("Calculating chronologically pure rolling ratings...")
    df = add_rolling_features(df)

    logger.info("Simulating historical Elo Ratings (Chronological Engine)...")
    df = simulate_elo(df)

    logger.info("Constructing final matchup-level dataset...")
    matchups_df = build_matchups(df)

    output_file = DATA_DIR / "ml_ready_matchups.csv"
    matchups_df.to_csv(output_file, index=False)
    
    rolling_feature_count = len([col for col in matchups_df.columns if col.startswith('DELTA_Z_')])
    logger.info(
        f"Success! Generated {len(matchups_df):,} matchup observations "
        f"with {rolling_feature_count} rolling statistical features."
    )