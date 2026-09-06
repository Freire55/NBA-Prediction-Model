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
EWMA_SPANS = (3, 5, 10)

# Elo parameters
INITIAL_ELO = 1500
ELO_K_FACTOR = 20
ELO_DIVISOR = 400

# Basketball constants
POSSESSION_FT_WEIGHT = 0.44
DEFAULT_REST_DAYS = 5.0
DEFAULT_OFF_RATING = 100.0

# Altitude parameters
ALTITUDE_FILE = DATA_DIR / "team_altitudes.csv"
ALTITUDE_THRESHOLD_FT = 1000.0
ALTITUDE_SCALE_FT = 2500.0

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


def load_altitude_map(filepath: Path = ALTITUDE_FILE) -> dict[str, float]:
    """Loads team altitude mapping from team_altitudes.csv."""
    if not filepath.exists():
        logger.warning(f"Altitude file not found at {filepath}, returning empty map.")
        return {}
    alt_df = pd.read_csv(filepath)
    return dict(zip(alt_df["TEAM_ABBREVIATION"], alt_df["ALTITUDE_FT"].astype(float)))


def calculate_altitude_advantage(
    home_alt: pd.Series, 
    away_alt: pd.Series, 
    threshold_ft: float = ALTITUDE_THRESHOLD_FT, 
    scale_ft: float = ALTITUDE_SCALE_FT
) -> pd.Series:
    """
    Computes a non-linear physiological altitude advantage for the home team.

    Physiological rationale:
    1. Directional Asymmetry: Hypoxia only penalizes teams ascending to higher elevations
       than their home base (Home Alt > Away Alt). Visiting teams descending to sea level
       experience no aerobic penalty.
    2. Threshold Barrier: Atmospheric pressure and oxygen partial pressure decrements
       below ~1,000 ft difference produce negligible aerobic impact on conditioned athletes.
    3. Accelerated Non-linear Saturation: Above 3,000-4,000 ft (Salt Lake City @ 4,226 ft,
       Denver @ 5,280 ft), arterial oxygen saturation drops steeply, causing exponential
       aerobic fatigue accumulation.

    Formula:
        diff = Home_Alt - Away_Alt
        effective_diff = max(0, diff - threshold_ft)
        advantage = 1.0 - exp(-(effective_diff / scale_ft)^2)

    Returns a continuous advantage score bounded in [0.0, 1.0].
    """
    diff = home_alt - away_alt
    effective_diff = np.maximum(0.0, diff - threshold_ft)
    advantage = 1.0 - np.exp(-((effective_diff / scale_ft) ** 2))
    return pd.Series(advantage, index=home_alt.index, name="ALTITUDE_ADVANTAGE")


def rolling_mean(series: pd.Series, rolling_window: int = ROLLING_WINDOW) -> pd.Series:
    """Computes a rolling mean using only prior observations."""
    return series.shift(1).rolling(rolling_window, min_periods=1).mean()


def rolling_sum(series: pd.Series, rolling_window: int = ROLLING_WINDOW) -> pd.Series:
    return series.shift(1).rolling(rolling_window, min_periods=1).sum()


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


def add_four_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the four factors of basketball success."""
    if 'MATCHUP' not in df.columns:
        return df

    # Opponent defensive rebounds for team offensive rebounding percentage.
    # In NBA game data, each GAME_ID contains 2 team rows. Vectorized subtraction:
    # total game DREB minus team's own DREB yields opponent DREB directly,
    # completely avoiding fragile string abbreviation merges (e.g., NOH vs. NO).
    if 'DREB' in df.columns and 'GAME_ID' in df.columns:
        game_dreb_sum = df.groupby('GAME_ID')['DREB'].transform('sum')
        game_dreb_count = df.groupby('GAME_ID')['DREB'].transform('count')
        opp_dreb = np.where(game_dreb_count == 2, game_dreb_sum - df['DREB'], np.nan)
        df['OPP_DREB'] = pd.Series(opp_dreb, index=df.index).fillna(df['DREB']).fillna(32.0)
    else:
        df['OPP_DREB'] = 32.0

    fga = df['FGA'].replace(0, np.nan) if 'FGA' in df.columns else pd.Series(np.nan, index=df.index)
    fgm = df['FGM'] if 'FGM' in df.columns else pd.Series(0.0, index=df.index)
    fg3m = df['FG3M'] if 'FG3M' in df.columns else pd.Series(0.0, index=df.index)
    fta = df['FTA'] if 'FTA' in df.columns else pd.Series(0.0, index=df.index)
    tov = df['TOV'] if 'TOV' in df.columns else pd.Series(0.0, index=df.index)
    oreb = df['OREB'] if 'OREB' in df.columns else pd.Series(0.0, index=df.index)
    opp_dreb = df['OPP_DREB']

    df['FOUR_FACTOR_EFG'] = ((fgm + 0.5 * fg3m) / fga).fillna(0.0)
    df['FOUR_FACTOR_TOV'] = (tov / (fga + 0.44 * fta + tov).replace(0, np.nan)).fillna(0.0)
    df['FOUR_FACTOR_OREB'] = (oreb / (oreb + opp_dreb).replace(0, np.nan)).fillna(0.0)
    df['FOUR_FACTOR_FTR'] = (fta / fga).fillna(0.0)

    df = df.drop(columns=['OPP_DREB', 'OPP_ABBREVIATION'], errors='ignore')

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates chronologically pure rolling averages and strength of schedule."""
    df = add_four_factors(df)
    team_groups = df.groupby('TEAM_ABBREVIATION')
    
    df['POSSESSIONS'] = df['FGA'] + POSSESSION_FT_WEIGHT * df['FTA'] - df['OREB'] + df['TOV']

    df[f'ROLLING_PTS_{ROLLING_WINDOW}'] = team_groups['PTS'].transform(rolling_sum)
    df[f'ROLLING_POSS_{ROLLING_WINDOW}'] = team_groups['POSSESSIONS'].transform(rolling_sum)
    
    df['ROLLING_OFF_RATING'] = (df[f'ROLLING_PTS_{ROLLING_WINDOW}'] / df[f'ROLLING_POSS_{ROLLING_WINDOW}']) * 100
    df['ROLLING_OFF_RATING'] = df['ROLLING_OFF_RATING'].fillna(DEFAULT_OFF_RATING)

    df["OFF_RATING_EWMA_3"] = (
        team_groups["ROLLING_OFF_RATING"]
        .transform(lambda x: ewma(x, span=3))
    )

    df["OFF_RATING_EWMA_5"] = (
        team_groups["ROLLING_OFF_RATING"]
        .transform(lambda x: ewma(x, span=5))
    )

    df["OFF_RATING_EWMA_10"] = (
        team_groups["ROLLING_OFF_RATING"]
        .transform(lambda x: ewma(x, span=10))
    )

    # Standardize historical memory for Z-stats
    z_columns = [col for col in df.columns if col.startswith("Z_")]

    rolling_z = team_groups[z_columns].transform(rolling_mean)
    for col in z_columns:
        df[f"{col}_ROLLING_{ROLLING_WINDOW}"] = rolling_z[col]

    for span in EWMA_SPANS:
        ewma_z = team_groups[z_columns].transform(lambda x, s=span: ewma(x, span=s))
        for col in z_columns:
            df[f"{col}_EWMA_{span}"] = ewma_z[col]


    # Map past opponent strength
    # Vectorized pairing by GAME_ID: total game strength minus team's own strength
    # completely eliminates string matching fragility and expensive merges.
    strength_col = f'Z_PLUS_MINUS_ROLLING_{ROLLING_WINDOW}'
    if strength_col in df.columns and 'GAME_ID' in df.columns:
        game_strength_sum = df.groupby('GAME_ID')[strength_col].transform('sum')
        game_count = df.groupby('GAME_ID')[strength_col].transform('count')
        df['OPP_PRE_GAME_STRENGTH'] = np.where(game_count == 2, game_strength_sum - df[strength_col], 0.0)
    else:
        df['OPP_PRE_GAME_STRENGTH'] = 0.0
    
    df = df.sort_values(by=["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    team_groups = df.groupby('TEAM_ABBREVIATION')

    df[f'SOS_ROLLING_{ROLLING_WINDOW}'] = (
        team_groups["OPP_PRE_GAME_STRENGTH"]
        .transform(rolling_mean)
        .fillna(0)
    )

    df["SOS_EWMA_3"] = (
        team_groups["OPP_PRE_GAME_STRENGTH"]
        .transform(lambda x: ewma(x, span=3))
        .fillna(0)
    )

    df["SOS_EWMA_5"] = (
        team_groups["OPP_PRE_GAME_STRENGTH"]
        .transform(lambda x: ewma(x, span=5))
        .fillna(0)
    )

    df["SOS_EWMA_10"] = (
        team_groups["OPP_PRE_GAME_STRENGTH"]
        .transform(lambda x: ewma(x, span=10))
        .fillna(0)
    )
    
    four_factor_columns = [
        'FOUR_FACTOR_EFG',
        'FOUR_FACTOR_TOV',
        'FOUR_FACTOR_OREB',
        'FOUR_FACTOR_FTR',
    ]

    for col in four_factor_columns:
        df[f'{col}_ROLLING_{ROLLING_WINDOW}'] = (
            team_groups[col]
            .transform(rolling_mean)
            .fillna(0)
        )

        for span in EWMA_SPANS:
            df[f'{col}_EWMA_{span}'] = (
                team_groups[col]
                .transform(lambda x, s=span: ewma(x, span=s))
                .fillna(0)
            )
            
    # Calculate game-level Pace: Possessions per 48 minutes
    # In NBA team box scores, MIN is total player minutes (240 in regulation: 5 * 48 min).
    # If MIN is already single-game minutes (<= 100), use as-is.
    if "MIN" in df.columns:
        raw_min = pd.to_numeric(df["MIN"], errors="coerce").fillna(240.0)
        team_mins = np.where(raw_min > 100.0, raw_min / 5.0, raw_min)
        team_mins = pd.Series(team_mins, index=df.index).replace(0, np.nan).fillna(48.0)
    else:
        team_mins = pd.Series(48.0, index=df.index)

    df["PACE"] = (df["POSSESSIONS"] / team_mins) * 48.0
    df["PACE"] = df["PACE"].fillna(100.0)

    df[f"ROLLING_PACE_{ROLLING_WINDOW}"] = (
        team_groups["PACE"]
        .transform(rolling_mean)
        .fillna(100.0)
    )
    df["ROLLING_PACE"] = df[f"ROLLING_PACE_{ROLLING_WINDOW}"]

    for span in EWMA_SPANS:
        df[f"PACE_EWMA_{span}"] = (
            team_groups["PACE"]
            .transform(lambda x, s=span: ewma(x, span=s))
            .fillna(100.0)
        )

    return df


def simulate_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Simulates a continuous Elo rating timeline using pure Python iterations for speed."""
    current_elo = {}
    pre_game_elo_records = []

    # Vectorized extraction of home and away match pairs (100x faster than df.groupby)
    has_season = "SEASON_ID" in df.columns
    home_cols = ["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "PTS"] + (["SEASON_ID"] if has_season else [])
    away_cols = ["GAME_ID", "TEAM_ABBREVIATION", "PTS"]

    home_df = df[df["MATCHUP"].str.contains(" vs. ", na=False)][home_cols]
    away_df = df[df["MATCHUP"].str.contains(" @ ", na=False)][away_cols]
    merged = home_df.merge(away_df, on="GAME_ID", suffixes=("_home", "_away")).sort_values("GAME_DATE")

    last_season = None
    seasons = merged["SEASON_ID"].to_numpy() if has_season else [None] * len(merged)

    # Fast iteration over numpy arrays (avoids pandas row indexing overhead)
    for game_id, h_team, a_team, h_pts, a_pts, season in zip(
        merged["GAME_ID"].to_numpy(),
        merged["TEAM_ABBREVIATION_home"].to_numpy(),
        merged["TEAM_ABBREVIATION_away"].to_numpy(),
        merged["PTS_home"].to_numpy(),
        merged["PTS_away"].to_numpy(),
        seasons,
    ):
        if last_season is not None and season is not None and season != last_season:
            for team in current_elo:
                current_elo[team] = (current_elo[team] * 0.75) + (INITIAL_ELO * 0.25)
        last_season = season

        if h_team not in current_elo: current_elo[h_team] = INITIAL_ELO
        if a_team not in current_elo: current_elo[a_team] = INITIAL_ELO

        home_elo_pre = current_elo[h_team]
        away_elo_pre = current_elo[a_team]

        pre_game_elo_records.append((game_id, h_team, home_elo_pre))
        pre_game_elo_records.append((game_id, a_team, away_elo_pre))

        elo_diff = home_elo_pre - away_elo_pre
        home_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / ELO_DIVISOR))
        home_won = 1 if h_pts > a_pts else 0

        mov = abs(h_pts - a_pts)
        winner_elo = home_elo_pre if home_won == 1 else away_elo_pre
        loser_elo = away_elo_pre if home_won == 1 else home_elo_pre
        mov_multiplier = np.log(mov + 1) * (2.2 / (((winner_elo - loser_elo) * 0.001) + 2.2))

        elo_change = ELO_K_FACTOR * mov_multiplier * (home_won - home_prob)

        current_elo[h_team] = home_elo_pre + elo_change
        current_elo[a_team] = away_elo_pre - elo_change

    elo_df = pd.DataFrame(pre_game_elo_records, columns=["GAME_ID", "TEAM_ABBREVIATION", "PRE_GAME_ELO"])
    df = df.merge(elo_df, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")
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
    matchups_df[f'DELTA_ROLLING_PACE_{ROLLING_WINDOW}'] = (
        matchups_df[f'HOME_ROLLING_PACE_{ROLLING_WINDOW}']
        - matchups_df[f'AWAY_ROLLING_PACE_{ROLLING_WINDOW}']
    )
    matchups_df['DELTA_ROLLING_PACE'] = matchups_df[f'DELTA_ROLLING_PACE_{ROLLING_WINDOW}']
    matchups_df['MATCHUP_EXPECTED_PACE'] = (
        matchups_df[f'HOME_ROLLING_PACE_{ROLLING_WINDOW}']
        + matchups_df[f'AWAY_ROLLING_PACE_{ROLLING_WINDOW}']
    ) / 2.0

    for span in EWMA_SPANS:
        matchups_df[f'DELTA_PACE_EWMA_{span}'] = (
            matchups_df[f'HOME_PACE_EWMA_{span}']
            - matchups_df[f'AWAY_PACE_EWMA_{span}']
        )
    
    # Altitude and Home Court Elevation Advantage
    alt_map = load_altitude_map()
    if alt_map and 'HOME_TEAM_ABBREVIATION' in matchups_df.columns:
        matchups_df['HOME_ALTITUDE'] = matchups_df['HOME_TEAM_ABBREVIATION'].map(alt_map).fillna(0.0)
        matchups_df['AWAY_ALTITUDE'] = matchups_df['AWAY_TEAM_ABBREVIATION'].map(alt_map).fillna(0.0)
        matchups_df['DELTA_ALTITUDE'] = matchups_df['HOME_ALTITUDE'] - matchups_df['AWAY_ALTITUDE']
        matchups_df['ALTITUDE_ADVANTAGE'] = calculate_altitude_advantage(
            matchups_df['HOME_ALTITUDE'], matchups_df['AWAY_ALTITUDE']
        )
        matchups_df['ALTITUDE_B2B_PENALTY'] = (
            matchups_df['ALTITUDE_ADVANTAGE'] * matchups_df['AWAY_B2B']
        )

    four_factor_features = [
        'FOUR_FACTOR_EFG',
        'FOUR_FACTOR_TOV',
        'FOUR_FACTOR_OREB',
        'FOUR_FACTOR_FTR',
    ]

    for factor in four_factor_features:
        rolling_col = f'{factor}_ROLLING_{ROLLING_WINDOW}'

        matchups_df[f'DELTA_{factor}_ROLLING_{ROLLING_WINDOW}'] = (
            matchups_df[f'HOME_{rolling_col}']
            - matchups_df[f'AWAY_{rolling_col}']
        )

        for span in EWMA_SPANS:
            ewma_col = f'{factor}_EWMA_{span}'

            matchups_df[f'DELTA_{factor}_EWMA_{span}'] = (
                matchups_df[f'HOME_{ewma_col}']
                - matchups_df[f'AWAY_{ewma_col}']
            )

    # Net shooting-efficiency matchup signal
    matchups_df['DELTA_FOUR_FACTORS_NET_EFG'] = (
        matchups_df['DELTA_FOUR_FACTOR_EFG_ROLLING_8']
    )

    z_feature_cols = [
        col.replace("HOME_", "")
        for col in home_df.columns
        if col.startswith("HOME_Z_")
        and (
            "_ROLLING_" in col
            or "_EWMA_" in col
        )
    ]    
    
    for col in z_feature_cols:
        delta_col = f"DELTA_{col}"
        matchups_df[delta_col] = matchups_df[f"HOME_{col}"] - matchups_df[f"AWAY_{col}"]

    matchups_df = matchups_df.dropna(subset=[f'DELTA_Z_PTS_ROLLING_{ROLLING_WINDOW}'])

    # ---------------------------------------------------------
    # Drop Post-Game Box Score Stats
    # ---------------------------------------------------------
    raw_box_score_stats = [
        'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
        'BLK', 'TOV', 'PF', 'PLUS_MINUS', 'POSSESSIONS', 'PACE', 'MIN',
        'FOUR_FACTOR_EFG', 'FOUR_FACTOR_TOV', 'FOUR_FACTOR_OREB', 'FOUR_FACTOR_FTR',
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