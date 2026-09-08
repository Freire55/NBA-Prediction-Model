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
EWMA_SPANS = (5, 10)

# Elo parameters
INITIAL_ELO = 1500
ELO_K_FACTOR = 20
ELO_DIVISOR = 400
ELO_EARLY_SEASON_BOOST = 0.3
ELO_EARLY_SEASON_HALF_LIFE = 8.0
HOME_COURT_ADVANTAGE_ELO = 70.0

# Basketball constants
POSSESSION_FT_WEIGHT = 0.44
DEFAULT_REST_DAYS = 5.0
DEFAULT_OFF_RATING = 100.0
LEAGUE_AVERAGE_3PT_PCT = 0.360
OPP_3PT_VARIANCE_REGRESSION_WEIGHT = 0.50

# Altitude parameters
ALTITUDE_FILE = DATA_DIR / "team_altitudes.csv"
ALTITUDE_THRESHOLD_FT = 1000.0
ALTITUDE_SCALE_FT = 2500.0

# Team coordinates (lat, lon) and standard UTC timezone offsets for circadian fatigue
TEAM_COORDINATES_AND_TZ: dict[str, tuple[float, float, float]] = {
    "ATL": (33.7573, -84.3963, -5.0),
    "BKN": (40.6826, -73.9754, -5.0),
    "BOS": (42.3662, -71.0621, -5.0),
    "CHA": (35.2251, -80.8392, -5.0),
    "CHH": (35.2251, -80.8392, -5.0),
    "CHI": (41.8807, -87.6742, -6.0),
    "CLE": (41.4965, -81.6882, -5.0),
    "DAL": (32.7905, -96.8103, -6.0),
    "DEN": (39.7487, -105.0076, -7.0),
    "DET": (42.3411, -83.0553, -5.0),
    "GSW": (37.7680, -122.3877, -8.0),
    "HOU": (29.7508, -95.3621, -6.0),
    "IND": (39.7640, -86.1555, -5.0),
    "LAC": (33.9450, -118.3418, -8.0),
    "LAL": (34.0430, -118.2673, -8.0),
    "MEM": (35.1382, -90.0505, -6.0),
    "MIA": (25.7814, -80.1870, -5.0),
    "MIL": (43.0451, -87.9174, -6.0),
    "MIN": (44.9795, -93.2761, -6.0),
    "NJN": (40.8122, -74.0744, -5.0),
    "NOH": (29.9490, -90.0821, -6.0),
    "NOK": (35.4634, -97.5151, -6.0),
    "NOP": (29.9490, -90.0821, -6.0),
    "NYK": (40.7505, -73.9934, -5.0),
    "OKC": (35.4634, -97.5151, -6.0),
    "ORL": (28.5392, -81.3839, -5.0),
    "PHI": (39.9012, -75.1720, -5.0),
    "PHX": (33.4457, -112.0712, -7.0),
    "POR": (45.5316, -122.6668, -8.0),
    "SAC": (38.5802, -121.4997, -8.0),
    "SAS": (29.4270, -98.4375, -6.0),
    "SEA": (47.6221, -122.3540, -8.0),
    "TOR": (43.6435, -79.3791, -5.0),
    "UTA": (40.7683, -111.9011, -7.0),
    "VAN": (49.2778, -123.1089, -8.0),
    "WAS": (38.8982, -77.0209, -5.0),
}


def haversine_np(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Computes vectorized great-circle distance between coordinates in miles."""
    r_miles = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return r_miles * 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))

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
    """Engineers fatigue, schedule density flags, and continuous circadian travel index."""
    df = df.sort_values(by=['TEAM_ABBREVIATION', 'GAME_DATE']).reset_index(drop=True)
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
    df['4_IN_6'] = np.where((df['GAME_DATE'] - df['DATE_MINUS_3']).dt.days <= 5, 1, 0)

    # Track road trip exhaustion and venue host team
    has_matchup = 'MATCHUP' in df.columns
    if has_matchup:
        df['IS_AWAY'] = df['MATCHUP'].str.contains(' @ ', na=False).astype(int)
        away_split = df['MATCHUP'].str.split(' @ ')
        opp_abbrev = np.where(df['IS_AWAY'] == 1, away_split.str[1], df['TEAM_ABBREVIATION'])
        host_team = pd.Series(opp_abbrev, index=df.index).str.strip().fillna(df['TEAM_ABBREVIATION'])
    else:
        df['IS_AWAY'] = 0
        host_team = df['TEAM_ABBREVIATION']

    df['HOST_TEAM'] = host_team
    df['AWAY_GROUP'] = (df['IS_AWAY'] != team_groups['IS_AWAY'].shift(1)).cumsum()
    df['ROAD_TRIP_LENGTH'] = np.where(df['IS_AWAY'] == 1, df.groupby(['TEAM_ABBREVIATION', 'AWAY_GROUP']).cumcount() + 1, 0)

    # Circadian travel distance and timezone tracking (strictly pre-game: venue of prior game to venue of this game)
    prev_host = team_groups['HOST_TEAM'].shift(1).fillna(df['TEAM_ABBREVIATION'])
    
    default_coord_tz = (39.0, -95.0, -6.0)
    lat_curr = df['HOST_TEAM'].map(lambda x: TEAM_COORDINATES_AND_TZ.get(x, default_coord_tz)[0]).to_numpy()
    lon_curr = df['HOST_TEAM'].map(lambda x: TEAM_COORDINATES_AND_TZ.get(x, default_coord_tz)[1]).to_numpy()
    tz_curr = df['HOST_TEAM'].map(lambda x: TEAM_COORDINATES_AND_TZ.get(x, default_coord_tz)[2]).to_numpy()

    lat_prev = prev_host.map(lambda x: TEAM_COORDINATES_AND_TZ.get(x, default_coord_tz)[0]).to_numpy()
    lon_prev = prev_host.map(lambda x: TEAM_COORDINATES_AND_TZ.get(x, default_coord_tz)[1]).to_numpy()
    tz_prev = prev_host.map(lambda x: TEAM_COORDINATES_AND_TZ.get(x, default_coord_tz)[2]).to_numpy()

    same_host = (df['HOST_TEAM'] == prev_host).to_numpy()
    game_dist = np.where(same_host, 0.0, haversine_np(lat_prev, lon_prev, lat_curr, lon_curr))
    df['TRAVEL_DIST_GAME'] = game_dist

    # Rolling 7-day cumulative travel distance
    df_idx = df[['TEAM_ABBREVIATION', 'GAME_DATE', 'TRAVEL_DIST_GAME']].set_index('GAME_DATE')
    prior_7d = (
        df_idx.groupby('TEAM_ABBREVIATION')['TRAVEL_DIST_GAME']
        .rolling('7D', closed='left')
        .sum()
        .fillna(0.0)
        .to_numpy()
    )
    df['TRAVEL_7D'] = prior_7d + df['TRAVEL_DIST_GAME']

    # Directional Circadian Jet Lag: West-to-East (tz_curr > tz_prev) loses recovery hours
    tz_diff = tz_curr - tz_prev
    df['TZ_EASTWARD_LOSS'] = np.maximum(0.0, tz_diff)
    df['TZ_CIRCADIAN_PENALTY'] = df['TZ_EASTWARD_LOSS'] * np.where(
        df['B2B'] == 1, 1.5, np.where(df['REST_DAYS'] <= 2, 1.0, 0.5)
    )

    # Mile-high altitude acute interaction (DEN @ 5280 ft, UTA @ 4226 ft)
    altitude_acute_penalty = np.where(
        (df['HOST_TEAM'].isin(['DEN', 'UTA'])) & (df['IS_AWAY'] == 1) & (df['B2B'] == 1),
        1.0,
        0.0,
    )
    df['ALTITUDE_FATIGUE_IMPACT'] = altitude_acute_penalty

    # Continuous Circadian Circulatory Fatigue Index
    df['CIRCADIAN_FATIGUE_INDEX'] = (
        (df['TRAVEL_7D'] / 1000.0)
        + 0.8 * df['TZ_CIRCADIAN_PENALTY']
        + 1.0 * df['B2B']
        + 0.6 * df['3_IN_4']
        + 0.5 * df['4_IN_6']
        + 1.5 * df['ALTITUDE_FATIGUE_IMPACT']
    )

    df = df.drop(columns=['HOST_TEAM', 'TRAVEL_DIST_GAME'])
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
    
    fga = df['FGA'] if 'FGA' in df.columns else pd.Series(85.0, index=df.index)
    fta = df['FTA'] if 'FTA' in df.columns else pd.Series(20.0, index=df.index)
    oreb = df['OREB'] if 'OREB' in df.columns else pd.Series(10.0, index=df.index)
    tov = df['TOV'] if 'TOV' in df.columns else pd.Series(14.0, index=df.index)
    pts = df['PTS'] if 'PTS' in df.columns else pd.Series(100.0, index=df.index)
    df['POSSESSIONS'] = fga + POSSESSION_FT_WEIGHT * fta - oreb + tov

    df[f'ROLLING_PTS_{ROLLING_WINDOW}'] = team_groups['PTS'].transform(rolling_sum) if 'PTS' in df.columns else 100.0
    df[f'ROLLING_POSS_{ROLLING_WINDOW}'] = team_groups['POSSESSIONS'].transform(rolling_sum)
    
    df['ROLLING_OFF_RATING'] = (df[f'ROLLING_PTS_{ROLLING_WINDOW}'] / df[f'ROLLING_POSS_{ROLLING_WINDOW}']) * 100
    df['ROLLING_OFF_RATING'] = df['ROLLING_OFF_RATING'].fillna(DEFAULT_OFF_RATING)

    df["OFF_RATING_EWMA_5"] = (
        team_groups["ROLLING_OFF_RATING"]
        .transform(lambda x: ewma(x, span=5))
    )

    df["OFF_RATING_EWMA_10"] = (
        team_groups["ROLLING_OFF_RATING"]
        .transform(lambda x: ewma(x, span=10))
    )

    # Opponent box-score tracking (vectorized 2-team game subtraction)
    has_game_id = 'GAME_ID' in df.columns
    if has_game_id and 'PTS' in df.columns:
        game_pts_sum = df.groupby('GAME_ID')['PTS'].transform('sum')
        game_count = df.groupby('GAME_ID')['PTS'].transform('count')
        df['OPP_PTS'] = np.where(game_count == 2, game_pts_sum - df['PTS'], pts)
    else:
        df['OPP_PTS'] = pts

    stat_defaults = [
        ('FG3M', 0.0), ('FG3A', 0.0), ('FGA', 85.0),
        ('TOV', 14.0), ('FTA', 20.0), ('OREB', 10.0), ('DREB', 32.0),
    ]
    for stat, default_val in stat_defaults:
        if has_game_id and stat in df.columns:
            stat_sum = df.groupby('GAME_ID')[stat].transform('sum')
            game_count = df.groupby('GAME_ID')[stat].transform('count')
            df[f'OPP_{stat}'] = np.where(game_count == 2, stat_sum - df[stat], default_val)
        else:
            df[f'OPP_{stat}'] = df[stat] if stat in df.columns else default_val

    # Opponent box-score tracking (rolling sums use strict shift 1)
    df[f'ROLLING_OPP_PTS_{ROLLING_WINDOW}'] = team_groups['OPP_PTS'].transform(rolling_sum)
    df[f'ROLLING_OPP_FG3M_{ROLLING_WINDOW}'] = team_groups['OPP_FG3M'].transform(rolling_sum)
    df[f'ROLLING_OPP_FG3A_{ROLLING_WINDOW}'] = team_groups['OPP_FG3A'].transform(rolling_sum)
    df[f'ROLLING_OPP_FGA_{ROLLING_WINDOW}'] = team_groups['OPP_FGA'].transform(rolling_sum)
    df[f'ROLLING_OPP_TOV_{ROLLING_WINDOW}'] = team_groups['OPP_TOV'].transform(rolling_sum)
    df[f'ROLLING_OPP_FTA_{ROLLING_WINDOW}'] = team_groups['OPP_FTA'].transform(rolling_sum)
    df[f'ROLLING_OPP_OREB_{ROLLING_WINDOW}'] = team_groups['OPP_OREB'].transform(rolling_sum)

    df['_TMP_DREB'] = df['DREB'] if 'DREB' in df.columns else pd.Series(32.0, index=df.index)
    team_groups = df.groupby('TEAM_ABBREVIATION')
    df[f'ROLLING_DREB_{ROLLING_WINDOW}'] = team_groups['_TMP_DREB'].transform(rolling_sum)
    df = df.drop(columns=['_TMP_DREB'])

    # Defensive style profiles (turnover pressure, foul discipline, rebounding control)
    poss_sum = df[f'ROLLING_POSS_{ROLLING_WINDOW}'].replace(0, np.nan)
    opp_fga_sum = df[f'ROLLING_OPP_FGA_{ROLLING_WINDOW}'].replace(0, np.nan)
    tot_reb_sum = (df[f'ROLLING_DREB_{ROLLING_WINDOW}'] + df[f'ROLLING_OPP_OREB_{ROLLING_WINDOW}']).replace(0, np.nan)

    df['DEF_TOV_RATE'] = (df[f'ROLLING_OPP_TOV_{ROLLING_WINDOW}'] / poss_sum).fillna(0.14)
    df['DEF_FTR'] = (df[f'ROLLING_OPP_FTA_{ROLLING_WINDOW}'] / opp_fga_sum).fillna(0.24)
    df['DEF_REB_RATE'] = (df[f'ROLLING_DREB_{ROLLING_WINDOW}'] / tot_reb_sum).fillna(0.75)

    # Regress opponent 3PT% 50% toward the league average (36.0%)
    # D_3PT_True = 0.5 * D_3PT_Actual + 0.5 * League_Average
    opp_3pt_actual = (
        df[f'ROLLING_OPP_FG3M_{ROLLING_WINDOW}']
        / df[f'ROLLING_OPP_FG3A_{ROLLING_WINDOW}'].replace(0, np.nan)
    ).fillna(LEAGUE_AVERAGE_3PT_PCT)

    df['D_3PT_TRUE'] = (
        OPP_3PT_VARIANCE_REGRESSION_WEIGHT * opp_3pt_actual
        + (1.0 - OPP_3PT_VARIANCE_REGRESSION_WEIGHT) * LEAGUE_AVERAGE_3PT_PCT
    )
    df['D_3PT_ACTUAL'] = opp_3pt_actual

    # Opponent 3PT Attempt Rate (volume allowed - controlled by defensive scheme)
    df['OPP_3PA_RATE'] = (
        df[f'ROLLING_OPP_FG3A_{ROLLING_WINDOW}']
        / df[f'ROLLING_OPP_FGA_{ROLLING_WINDOW}'].replace(0, np.nan)
    ).fillna(0.350)

    # Variance-neutralized defensive points and defensive rating
    expected_opp_fg3m = df[f'ROLLING_OPP_FG3A_{ROLLING_WINDOW}'] * df['D_3PT_TRUE']
    pts_3pt_luck = (df[f'ROLLING_OPP_FG3M_{ROLLING_WINDOW}'] - expected_opp_fg3m) * 3.0
    rolling_opp_pts_adj = df[f'ROLLING_OPP_PTS_{ROLLING_WINDOW}'] - pts_3pt_luck
    df['ROLLING_DEF_RATING'] = (
        rolling_opp_pts_adj / df[f'ROLLING_POSS_{ROLLING_WINDOW}'].replace(0, np.nan)
    ) * 100.0
    df['ROLLING_DEF_RATING'] = df['ROLLING_DEF_RATING'].fillna(DEFAULT_OFF_RATING)

    df['ROLLING_NET_RATING'] = df['ROLLING_OFF_RATING'] - df['ROLLING_DEF_RATING']

    # Single-game luck adjustment for EWMA metrics (strictly shifted inside ewma)
    df['_TMP_OPP_3PT'] = (df['OPP_FG3M'] / df['OPP_FG3A'].replace(0, np.nan)).fillna(LEAGUE_AVERAGE_3PT_PCT)
    df['_TMP_OPP_3PA_RATE'] = (df['OPP_FG3A'] / df['OPP_FGA'].replace(0, np.nan)).fillna(0.350)
    single_pts_luck = (df['OPP_FG3M'] - df['OPP_FG3A'] * LEAGUE_AVERAGE_3PT_PCT) * 3.0
    single_opp_pts_adj = df['OPP_PTS'] - single_pts_luck
    df['_TMP_DEF_RATING'] = ((single_opp_pts_adj / df['POSSESSIONS'].replace(0, np.nan)) * 100.0).fillna(DEFAULT_OFF_RATING)

    team_groups = df.groupby('TEAM_ABBREVIATION')
    for span in EWMA_SPANS:
        ewma_opp_3pt = team_groups['_TMP_OPP_3PT'].transform(lambda x, s=span: ewma(x, span=s)).fillna(LEAGUE_AVERAGE_3PT_PCT)
        df[f'D_3PT_TRUE_EWMA_{span}'] = (
            OPP_3PT_VARIANCE_REGRESSION_WEIGHT * ewma_opp_3pt
            + (1.0 - OPP_3PT_VARIANCE_REGRESSION_WEIGHT) * LEAGUE_AVERAGE_3PT_PCT
        )
        df[f'OPP_3PA_RATE_EWMA_{span}'] = (
            team_groups['_TMP_OPP_3PA_RATE'].transform(lambda x, s=span: ewma(x, span=s)).fillna(0.350)
        )
        df[f'DEF_RATING_EWMA_{span}'] = (
            team_groups['_TMP_DEF_RATING'].transform(lambda x, s=span: ewma(x, span=s)).fillna(DEFAULT_OFF_RATING)
        )
        df[f'NET_RATING_EWMA_{span}'] = df[f'OFF_RATING_EWMA_{span}'] - df[f'DEF_RATING_EWMA_{span}']

    df = df.drop(columns=['_TMP_OPP_3PT', '_TMP_OPP_3PA_RATE', '_TMP_DEF_RATING'])

    # Standardize historical memory for Z-stats
    z_columns = [col for col in df.columns if col.startswith("Z_")]
    if z_columns:
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
    """Simulates a continuous Elo timeline with sample-size dynamic updates."""
    current_elo = {}
    season_games = {}
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
            season_games.clear()
        last_season = season

        if h_team not in current_elo:
            current_elo[h_team] = INITIAL_ELO
        if a_team not in current_elo:
            current_elo[a_team] = INITIAL_ELO

        home_elo_pre = current_elo[h_team]
        away_elo_pre = current_elo[a_team]

        pre_game_elo_records.append((game_id, h_team, home_elo_pre))
        pre_game_elo_records.append((game_id, a_team, away_elo_pre))

        n_h = season_games.get(h_team, 0)
        n_a = season_games.get(a_team, 0)

        elo_diff = (home_elo_pre + HOME_COURT_ADVANTAGE_ELO) - away_elo_pre
        home_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / ELO_DIVISOR))
        home_won = 1 if h_pts > a_pts else 0

        mov = abs(h_pts - a_pts)
        winner_elo = (home_elo_pre + HOME_COURT_ADVANTAGE_ELO) if home_won == 1 else away_elo_pre
        loser_elo = away_elo_pre if home_won == 1 else (home_elo_pre + HOME_COURT_ADVANTAGE_ELO)
        mov_multiplier = np.log(mov + 1) * (2.2 / (((winner_elo - loser_elo) * 0.001) + 2.2))

        k_h = ELO_K_FACTOR * (1.0 + ELO_EARLY_SEASON_BOOST * np.exp(-n_h / ELO_EARLY_SEASON_HALF_LIFE))
        k_a = ELO_K_FACTOR * (1.0 + ELO_EARLY_SEASON_BOOST * np.exp(-n_a / ELO_EARLY_SEASON_HALF_LIFE))

        current_elo[h_team] = home_elo_pre + k_h * mov_multiplier * (home_won - home_prob)
        current_elo[a_team] = away_elo_pre - k_a * mov_multiplier * (home_won - home_prob)

        season_games[h_team] = n_h + 1
        season_games[a_team] = n_a + 1

    elo_df = pd.DataFrame(pre_game_elo_records, columns=["GAME_ID", "TEAM_ABBREVIATION", "PRE_GAME_ELO"])
    df = df.merge(elo_df, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")
    return df




def merge_home_away_games(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combines home and away team game rows into single matchup-level observations."""
    home_df = df[df['MATCHUP'].str.contains(' vs. ')].copy().add_prefix('HOME_')
    away_df = df[df['MATCHUP'].str.contains(' @ ')].copy().add_prefix('AWAY_')
    matchups_df = home_df.merge(away_df, left_on='HOME_GAME_ID', right_on='AWAY_GAME_ID')
    matchups_df['HOME_WIN'] = np.where(matchups_df['HOME_PTS'] > matchups_df['AWAY_PTS'], 1, 0)
    matchups_df['TARGET_MARGIN'] = (matchups_df['HOME_PTS'] - matchups_df['AWAY_PTS']).astype("float32")
    matchups_df['SEASON_YEAR'] = matchups_df['HOME_SEASON_YEAR']
    return matchups_df, home_df


def add_schedule_and_elo_deltas(matchups_df: pd.DataFrame) -> pd.DataFrame:
    """Computes differential fatigue, schedule density, and Elo ratings."""
    matchups_df['REST_ADVANTAGE'] = matchups_df['HOME_REST_DAYS'] - matchups_df['AWAY_REST_DAYS']
    matchups_df['DELTA_ELO'] = matchups_df['HOME_PRE_GAME_ELO'] - matchups_df['AWAY_PRE_GAME_ELO']
    matchups_df['DELTA_ROAD_TRIP_LENGTH'] = matchups_df['HOME_ROAD_TRIP_LENGTH'] - matchups_df['AWAY_ROAD_TRIP_LENGTH']
    matchups_df['DELTA_3_IN_4'] = matchups_df['HOME_3_IN_4'] - matchups_df['AWAY_3_IN_4']
    matchups_df['DELTA_4_IN_5'] = matchups_df['HOME_4_IN_5'] - matchups_df['AWAY_4_IN_5']
    matchups_df[f'DELTA_SOS_ROLLING_{ROLLING_WINDOW}'] = (
        matchups_df[f'HOME_SOS_ROLLING_{ROLLING_WINDOW}'] - matchups_df[f'AWAY_SOS_ROLLING_{ROLLING_WINDOW}']
    )
    matchups_df['DELTA_ROLLING_OFF_RATING'] = (
        matchups_df['HOME_ROLLING_OFF_RATING'] - matchups_df['AWAY_ROLLING_OFF_RATING']
    )
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

    # Opponent 3PT variance neutralization & defensive/net rating deltas
    if 'HOME_D_3PT_TRUE' in matchups_df.columns and 'AWAY_D_3PT_TRUE' in matchups_df.columns:
        matchups_df['DELTA_D_3PT_TRUE'] = matchups_df['HOME_D_3PT_TRUE'] - matchups_df['AWAY_D_3PT_TRUE']
        matchups_df['DELTA_OPP_3PA_RATE'] = matchups_df['HOME_OPP_3PA_RATE'] - matchups_df['AWAY_OPP_3PA_RATE']
        matchups_df['DELTA_ROLLING_DEF_RATING'] = matchups_df['HOME_ROLLING_DEF_RATING'] - matchups_df['AWAY_ROLLING_DEF_RATING']
        matchups_df['DELTA_ROLLING_NET_RATING'] = matchups_df['HOME_ROLLING_NET_RATING'] - matchups_df['AWAY_ROLLING_NET_RATING']
        for span in EWMA_SPANS:
            if f'HOME_D_3PT_TRUE_EWMA_{span}' in matchups_df.columns:
                matchups_df[f'DELTA_D_3PT_TRUE_EWMA_{span}'] = (
                    matchups_df[f'HOME_D_3PT_TRUE_EWMA_{span}'] - matchups_df[f'AWAY_D_3PT_TRUE_EWMA_{span}']
                )
                matchups_df[f'DELTA_OPP_3PA_RATE_EWMA_{span}'] = (
                    matchups_df[f'HOME_OPP_3PA_RATE_EWMA_{span}'] - matchups_df[f'AWAY_OPP_3PA_RATE_EWMA_{span}']
                )
                matchups_df[f'DELTA_DEF_RATING_EWMA_{span}'] = (
                    matchups_df[f'HOME_DEF_RATING_EWMA_{span}'] - matchups_df[f'AWAY_DEF_RATING_EWMA_{span}']
                )
                matchups_df[f'DELTA_NET_RATING_EWMA_{span}'] = (
                    matchups_df[f'HOME_NET_RATING_EWMA_{span}'] - matchups_df[f'AWAY_NET_RATING_EWMA_{span}']
                )
    # Circadian fatigue & travel mileage deltas (positive delta indicates away team is fatigued)
    if 'AWAY_CIRCADIAN_FATIGUE_INDEX' in matchups_df.columns and 'HOME_CIRCADIAN_FATIGUE_INDEX' in matchups_df.columns:
        matchups_df['DELTA_CIRCADIAN_FATIGUE'] = (
            matchups_df['AWAY_CIRCADIAN_FATIGUE_INDEX'] - matchups_df['HOME_CIRCADIAN_FATIGUE_INDEX']
        )
        matchups_df['DELTA_TRAVEL_7D'] = (
            matchups_df['AWAY_TRAVEL_7D'] - matchups_df['HOME_TRAVEL_7D']
        )
        matchups_df['DELTA_TZ_CIRCADIAN_PENALTY'] = (
            matchups_df['AWAY_TZ_CIRCADIAN_PENALTY'] - matchups_df['HOME_TZ_CIRCADIAN_PENALTY']
        )
        if 'HOME_4_IN_6' in matchups_df.columns and 'AWAY_4_IN_6' in matchups_df.columns:
            matchups_df['DELTA_4_IN_6'] = matchups_df['HOME_4_IN_6'] - matchups_df['AWAY_4_IN_6']

    return matchups_df


def add_altitude_matchup_features(matchups_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates home court altitude advantage and back-to-back penalty."""
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
    return matchups_df


def add_four_factors_matchup_deltas(matchups_df: pd.DataFrame) -> pd.DataFrame:
    """Computes differential Four Factors across rolling and EWMA spans."""
    four_factor_features = [
        'FOUR_FACTOR_EFG',
        'FOUR_FACTOR_TOV',
        'FOUR_FACTOR_OREB',
        'FOUR_FACTOR_FTR',
    ]

    for factor in four_factor_features:
        rolling_col = f'{factor}_ROLLING_{ROLLING_WINDOW}'
        matchups_df[f'DELTA_{factor}_ROLLING_{ROLLING_WINDOW}'] = (
            matchups_df[f'HOME_{rolling_col}'] - matchups_df[f'AWAY_{rolling_col}']
        )
        for span in EWMA_SPANS:
            ewma_col = f'{factor}_EWMA_{span}'
            matchups_df[f'DELTA_{factor}_EWMA_{span}'] = (
                matchups_df[f'HOME_{ewma_col}'] - matchups_df[f'AWAY_{ewma_col}']
            )

    matchups_df['DELTA_FOUR_FACTORS_NET_EFG'] = (
        matchups_df['DELTA_FOUR_FACTOR_EFG_ROLLING_8']
    )
    return matchups_df


def add_zscore_matchup_deltas(matchups_df: pd.DataFrame, home_df: pd.DataFrame) -> pd.DataFrame:
    """Computes differentials for era-adjusted Z-score statistics."""
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
    return matchups_df


def add_tactical_clash_matrix(matchups_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes non-transitive tactical matchup clashes:
    1. Turnover Pressure Clash: Defensive turnover creation * opponent ball-security vulnerability.
    2. Crash vs. Leak-out Clash: Offensive rebounding aggression * opponent transition pace.
    3. Free Throw Clash: Offensive FT generation * defensive foul rate conceded.
    4. 3PT Perimeter Exploitation: 3PT volume & efficiency * perimeter attempt rate conceded.
    5. Glass Control Dominance: Offensive glass crashing * opponent defensive rebounding deficit.
    6. Composite Tactical Clash Advantage.
    """
    # 1. Turnover Pressure Clash
    if 'HOME_DEF_TOV_RATE' in matchups_df.columns and 'AWAY_DEF_TOV_RATE' in matchups_df.columns:
        home_tov_vuln = matchups_df.get('AWAY_FOUR_FACTOR_TOV_ROLLING_8', 0.14)
        away_tov_vuln = matchups_df.get('HOME_FOUR_FACTOR_TOV_ROLLING_8', 0.14)
        matchups_df['HOME_TURNOVER_PRESSURE_CLASH'] = matchups_df['HOME_DEF_TOV_RATE'] * home_tov_vuln
        matchups_df['AWAY_TURNOVER_PRESSURE_CLASH'] = matchups_df['AWAY_DEF_TOV_RATE'] * away_tov_vuln
        matchups_df['DELTA_TURNOVER_PRESSURE_CLASH'] = (
            matchups_df['HOME_TURNOVER_PRESSURE_CLASH'] - matchups_df['AWAY_TURNOVER_PRESSURE_CLASH']
        )

    # 2. Crash vs. Leak-out (Offensive Rebounds vs Transition Pace)
    if 'HOME_FOUR_FACTOR_OREB_ROLLING_8' in matchups_df.columns:
        home_oreb = matchups_df['HOME_FOUR_FACTOR_OREB_ROLLING_8']
        away_oreb = matchups_df.get('AWAY_FOUR_FACTOR_OREB_ROLLING_8', 0.25)
        home_pace = matchups_df.get('HOME_ROLLING_PACE_8', 100.0)
        away_pace = matchups_df.get('AWAY_ROLLING_PACE_8', 100.0)

        matchups_df['HOME_REBOUND_PACE_CLASH'] = home_oreb * (away_pace / 100.0)
        matchups_df['AWAY_REBOUND_PACE_CLASH'] = away_oreb * (home_pace / 100.0)
        matchups_df['DELTA_REBOUND_PACE_CLASH'] = (
            matchups_df['HOME_REBOUND_PACE_CLASH'] - matchups_df['AWAY_REBOUND_PACE_CLASH']
        )

    # 3. Free Throw Exploitation Clash
    if 'HOME_FOUR_FACTOR_FTR_ROLLING_8' in matchups_df.columns and 'AWAY_DEF_FTR' in matchups_df.columns:
        matchups_df['HOME_FTR_CLASH'] = matchups_df['HOME_FOUR_FACTOR_FTR_ROLLING_8'] * matchups_df['AWAY_DEF_FTR']
        matchups_df['AWAY_FTR_CLASH'] = matchups_df.get('AWAY_FOUR_FACTOR_FTR_ROLLING_8', 0.25) * matchups_df.get('HOME_DEF_FTR', 0.24)
        matchups_df['DELTA_FTR_CLASH'] = matchups_df['HOME_FTR_CLASH'] - matchups_df['AWAY_FTR_CLASH']

    # 4. 3PT Perimeter Exploitation Clash
    if 'AWAY_OPP_3PA_RATE' in matchups_df.columns and 'HOME_FOUR_FACTOR_EFG_ROLLING_8' in matchups_df.columns:
        matchups_df['HOME_3PT_EXPLOITATION'] = (
            matchups_df['AWAY_OPP_3PA_RATE'] * matchups_df['HOME_FOUR_FACTOR_EFG_ROLLING_8']
        )
        matchups_df['AWAY_3PT_EXPLOITATION'] = (
            matchups_df.get('HOME_OPP_3PA_RATE', 0.35) * matchups_df.get('AWAY_FOUR_FACTOR_EFG_ROLLING_8', 0.50)
        )
        matchups_df['DELTA_3PT_EXPLOITATION'] = (
            matchups_df['HOME_3PT_EXPLOITATION'] - matchups_df['AWAY_3PT_EXPLOITATION']
        )

    # 5. Glass Control Dominance
    if 'HOME_DEF_REB_RATE' in matchups_df.columns and 'AWAY_DEF_REB_RATE' in matchups_df.columns:
        matchups_df['HOME_GLASS_DOMINANCE'] = (
            matchups_df.get('HOME_FOUR_FACTOR_OREB_ROLLING_8', 0.25) * (1.0 - matchups_df['AWAY_DEF_REB_RATE'])
        )
        matchups_df['AWAY_GLASS_DOMINANCE'] = (
            matchups_df.get('AWAY_FOUR_FACTOR_OREB_ROLLING_8', 0.25) * (1.0 - matchups_df['HOME_DEF_REB_RATE'])
        )
        matchups_df['DELTA_GLASS_DOMINANCE'] = (
            matchups_df['HOME_GLASS_DOMINANCE'] - matchups_df['AWAY_GLASS_DOMINANCE']
        )

    # 6. Composite Tactical Clash Advantage
    if (
        'DELTA_TURNOVER_PRESSURE_CLASH' in matchups_df.columns
        and 'DELTA_GLASS_DOMINANCE' in matchups_df.columns
        and 'DELTA_FTR_CLASH' in matchups_df.columns
        and 'DELTA_3PT_EXPLOITATION' in matchups_df.columns
    ):
        matchups_df['DELTA_TACTICAL_CLASH_ADVANTAGE'] = (
            2.0 * matchups_df['DELTA_TURNOVER_PRESSURE_CLASH']
            + 1.5 * matchups_df['DELTA_GLASS_DOMINANCE']
            + 1.0 * matchups_df['DELTA_FTR_CLASH']
            + 1.0 * matchups_df['DELTA_3PT_EXPLOITATION']
        )

    return matchups_df


def drop_postgame_leakage_columns(matchups_df: pd.DataFrame) -> pd.DataFrame:
    """Strips post-game box score statistics to prevent data leakage."""
    raw_box_score_stats = [
        'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
        'BLK', 'TOV', 'PF', 'PLUS_MINUS', 'POSSESSIONS', 'PACE', 'MIN',
        'FOUR_FACTOR_EFG', 'FOUR_FACTOR_TOV', 'FOUR_FACTOR_OREB', 'FOUR_FACTOR_FTR',
        'OPP_PTS', 'OPP_FGM', 'OPP_FGA', 'OPP_FG3M', 'OPP_FG3A', 'OPP_TOV', 'OPP_FTA', 'OPP_OREB', 'OPP_DREB',
        f'ROLLING_OPP_PTS_{ROLLING_WINDOW}', f'ROLLING_OPP_FG3M_{ROLLING_WINDOW}',
        f'ROLLING_OPP_FG3A_{ROLLING_WINDOW}', f'ROLLING_OPP_FGA_{ROLLING_WINDOW}',
        f'ROLLING_OPP_TOV_{ROLLING_WINDOW}', f'ROLLING_OPP_FTA_{ROLLING_WINDOW}',
        f'ROLLING_OPP_OREB_{ROLLING_WINDOW}', f'ROLLING_DREB_{ROLLING_WINDOW}',
    ]
    cols_to_drop = []
    for stat in raw_box_score_stats:
        cols_to_drop.extend([
            f"HOME_{stat}", f"AWAY_{stat}", 
            f"HOME_Z_{stat}", f"AWAY_Z_{stat}",
            stat,
        ])
    return matchups_df.drop(
        columns=[c for c in cols_to_drop if c in matchups_df.columns]
    )


def build_matchups(df: pd.DataFrame) -> pd.DataFrame:
    """Combines home and away rows into single matchup-level observations with engineered deltas."""
    matchups_df, home_df = merge_home_away_games(df)
    matchups_df = add_schedule_and_elo_deltas(matchups_df)
    matchups_df = add_altitude_matchup_features(matchups_df)
    matchups_df = add_four_factors_matchup_deltas(matchups_df)
    matchups_df = add_tactical_clash_matrix(matchups_df)
    matchups_df = add_zscore_matchup_deltas(matchups_df, home_df)
    matchups_df = drop_postgame_leakage_columns(matchups_df)
    return matchups_df


def main() -> None:
    """Executes the full feature engineering pipeline for team matchups."""
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


# ======================================================
# Main Execution
# ======================================================
if __name__ == "__main__":
    main()