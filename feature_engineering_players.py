"""
Creates player-based features for the NBA prediction model.

This script computes leak-free rolling player performance metrics,
estimates expected player impact based on historical playing time,
aggregates those values to the team level, and merges the resulting
features into the matchup dataset.

Input:
    data/raw_player_game_logs.csv
    data/ml_ready_matchups.csv
    data/player_embeddings.csv

Output:
    data/ml_ready_matchups_players.csv
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd

from feature_engineering import calculate_altitude_advantage, load_altitude_map

# ======================================================
# Constants & Configuration
# ======================================================

DATA_DIR = Path(__file__).resolve().parent / "data"

ROLLING_WINDOW = 8
DEFAULT_VALUE = 0.0

GAME_LOGS_FILE = "raw_player_game_logs.csv"
MATCHUPS_FILE = "ml_ready_matchups.csv"
OUTPUT_FILE = "ml_ready_matchups_players.csv"

# ======================================================
# Logging Setup
# ======================================================

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ======================================================
# Helper Functions
# ======================================================

def parse_minutes(minutes: pd.Series) -> pd.Series:
    """Converts NBA API minute strings (MM:SS) into decimal minutes."""
    time_split = minutes.astype(str).str.split(":", expand=True)
    if 1 not in time_split.columns:
        time_split[1] = 0
    return (
        pd.to_numeric(time_split[0], errors="coerce").fillna(DEFAULT_VALUE)
        + pd.to_numeric(time_split[1], errors="coerce").fillna(DEFAULT_VALUE) / 60.0
    )


def calculate_game_score(df: pd.DataFrame) -> pd.Series:
    """Computes John Hollinger's Game Score for each player appearance."""
    return (
        df["PTS"]
        + (0.4 * df["FGM"])
        - (0.7 * df["FGA"])
        - (0.4 * (df["FTA"] - df["FTM"]))
        + (0.7 * df["OREB"])
        + (0.3 * df["DREB"])
        + df["STL"]
        + (0.7 * df["AST"])
        + (0.7 * df["BLK"])
        - (0.4 * df["PF"])
        - df["TOV"]
    )
    
def calculate_obpm_proxy(df: pd.DataFrame) -> pd.Series:
    possessions = (
        df["FGA"]
        + 0.44 * df["FTA"]
        + df["TOV"]
    )

    possessions = possessions.clip(lower=1)

    offensive_value = (
        df["PTS"]
        + 0.7 * df["AST"]
        + 0.7 * df["OREB"]
        - 1.0 * df["TOV"]
    )

    return (offensive_value / possessions) * 100


def calculate_dbpm_proxy(df: pd.DataFrame) -> pd.Series:
    possessions = (
        df["FGA"]
        + 0.44 * df["FTA"]
        + df["TOV"]
    )

    possessions = possessions.clip(lower=1)

    defensive_value = (
        df["STL"]
        + df["BLK"]
        + 0.7 * df["DREB"]
        - 0.5 * df["PF"]
    )

    return (defensive_value / possessions) * 100
    
def calculate_player_four_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-appearance Four Factor metrics for individual players:
    - Effective Field Goal % (eFG%): (FGM + 0.5 * FG3M) / FGA
    - Turnover % (TOV%): TOV / (FGA + 0.44 * FTA + TOV)
    - Offensive Rebounding Rate: OREB / MINUTES_NUM (rebounds per min)
    - Free Throw Rate (FTR): FTA / FGA
    """
    fga = df["FGA"].replace(0, np.nan)
    fgm = df["FGM"]
    fg3m = df["FG3M"] if "FG3M" in df.columns else (df["3PM"] if "3PM" in df.columns else 0.0)
    fta = df["FTA"]
    tov = df["TOV"]
    oreb = df["OREB"]
    mins = df["MINUTES_NUM"].replace(0, np.nan)

    efg = (fgm + 0.5 * fg3m) / fga
    plays = fga + 0.44 * fta + tov
    tov_pct = tov / plays.replace(0, np.nan)
    oreb_rate = oreb / mins
    ft_rate = fta / fga

    return pd.DataFrame({
        "PLAYER_FOUR_FACTOR_EFG": efg.fillna(0.50),
        "PLAYER_FOUR_FACTOR_TOV": tov_pct.fillna(0.12),
        "PLAYER_FOUR_FACTOR_OREB": oreb_rate.fillna(0.03),
        "PLAYER_FOUR_FACTOR_FTR": ft_rate.fillna(0.25),
    }, index=df.index)


HISTORICAL_GAME_SCORE_MEAN = 8.0
HISTORICAL_GAME_SCORE_STD = 5.0
HISTORICAL_ROSTER_FORM_PRIOR = 70.0

def rolling_mean(series: pd.Series, rolling_window: int = ROLLING_WINDOW) -> pd.Series:
    """Computes a rolling mean using only prior observations."""
    return series.shift(1).rolling(rolling_window, min_periods=1).mean()

def rolling_std(series: pd.Series, rolling_window: int = ROLLING_WINDOW) -> pd.Series:
    """Computes a leak-free rolling standard deviation using only prior observations."""
    return series.shift(1).rolling(rolling_window, min_periods=1).std()

def ewma(series: pd.Series, span: int = ROLLING_WINDOW) -> pd.Series:
    """Computes an exponentially weighted moving average using only prior observations."""
    return series.shift(1).ewm(adjust=False, span=span).mean()


def add_volatility_features(
    df: pd.DataFrame,
    prior_mean: float = HISTORICAL_GAME_SCORE_MEAN,
    prior_std: float = HISTORICAL_GAME_SCORE_STD,
) -> pd.DataFrame:
    """
    Computes robust expected player impact with variance penalty and outlier clipping.
    Uses domain-grounded historical priors (8.0 mean, 5.0 std) to maintain strict chronological purity.
    """
    player_groups = df.groupby("PLAYER_ID")
    baseline_mean = df["PLAYER_FORM_ROLLING_5"].fillna(prior_mean)

    df["PLAYER_GAME_SCORE_VOLATILITY"] = (
        player_groups["GAME_SCORE"]
        .transform(rolling_std)
        .fillna(prior_std)
    )

    df["GAME_SCORE_CEILING"] = baseline_mean + 2.5 * df["PLAYER_GAME_SCORE_VOLATILITY"]
    df["GAME_SCORE_FLOOR"] = baseline_mean - 2.5 * df["PLAYER_GAME_SCORE_VOLATILITY"]

    df["CAPPED_GAME_SCORE"] = np.minimum(df["GAME_SCORE"], df["GAME_SCORE_CEILING"])
    df["CAPPED_GAME_SCORE"] = np.maximum(df["CAPPED_GAME_SCORE"], df["GAME_SCORE_FLOOR"])

    robust_mean = (
        player_groups["CAPPED_GAME_SCORE"]
        .transform(rolling_mean)
        .fillna(prior_mean)
    )
    robust_std = (
        player_groups["CAPPED_GAME_SCORE"]
        .transform(rolling_std)
        .fillna(prior_std)
    )

    df["ROBUST_EXPECTED_IMPACT"] = robust_mean - (0.35 * robust_std)

    return df


# ======================================================
# Main Pipeline
# ======================================================

def main() -> None:
    """Generate player-based matchup features."""

    logs_df = pd.read_csv(DATA_DIR / GAME_LOGS_FILE)
    matchups_df = pd.read_csv(DATA_DIR / MATCHUPS_FILE)
    embeddings_df = pd.read_csv(DATA_DIR / "player_embeddings.csv")

    if "ALTITUDE_ADVANTAGE" not in matchups_df.columns and "HOME_TEAM_ABBREVIATION" in matchups_df.columns:
        alt_map = load_altitude_map()
        if alt_map:
            matchups_df["HOME_ALTITUDE"] = matchups_df["HOME_TEAM_ABBREVIATION"].map(alt_map).fillna(0.0)
            matchups_df["AWAY_ALTITUDE"] = matchups_df["AWAY_TEAM_ABBREVIATION"].map(alt_map).fillna(0.0)
            matchups_df["DELTA_ALTITUDE"] = matchups_df["HOME_ALTITUDE"] - matchups_df["AWAY_ALTITUDE"]
            matchups_df["ALTITUDE_ADVANTAGE"] = calculate_altitude_advantage(
                matchups_df["HOME_ALTITUDE"], matchups_df["AWAY_ALTITUDE"]
            )
            b2b = matchups_df["AWAY_B2B"] if "AWAY_B2B" in matchups_df.columns else 0
            matchups_df["ALTITUDE_B2B_PENALTY"] = matchups_df["ALTITUDE_ADVANTAGE"] * b2b

    logger.info(
        f"Loaded {len(logs_df):,} player logs and "
        f"{len(matchups_df):,} matchup rows."
    )

    # ======================================================
    # Player-level feature engineering
    # ======================================================

    logs_df["PLAYER_ID"] = logs_df["PLAYER_ID"].astype(str)
    logs_df["GAME_DATE"] = pd.to_datetime(logs_df["GAME_DATE"])

    embeddings_df["PLAYER_ID"] = embeddings_df["PLAYER_ID"].astype(str)
    embeddings_df["GAME_DATE"] = pd.to_datetime(embeddings_df["GAME_DATE"])

    logs_df = (
        logs_df.sort_values(["PLAYER_ID", "GAME_DATE"])
        .reset_index(drop=True)
    )

    logs_df = logs_df.assign(
        MINUTES_NUM=parse_minutes(logs_df["MIN"]),
        GAME_SCORE=calculate_game_score(logs_df),
        OBPM_PROXY=calculate_obpm_proxy(logs_df),
        DBPM_PROXY=calculate_dbpm_proxy(logs_df)
    )

    player_ff = calculate_player_four_factors(logs_df)
    logs_df["PLAYER_FOUR_FACTOR_EFG"] = player_ff["PLAYER_FOUR_FACTOR_EFG"]
    logs_df["PLAYER_FOUR_FACTOR_TOV"] = player_ff["PLAYER_FOUR_FACTOR_TOV"]
    logs_df["PLAYER_FOUR_FACTOR_OREB"] = player_ff["PLAYER_FOUR_FACTOR_OREB"]
    logs_df["PLAYER_FOUR_FACTOR_FTR"] = player_ff["PLAYER_FOUR_FACTOR_FTR"]

    player_groups = logs_df.groupby("PLAYER_ID")

    logs_df = logs_df.assign(
        PLAYER_FORM_ROLLING_3=player_groups["GAME_SCORE"].transform(rolling_mean, rolling_window=3),
        PLAYER_FORM_ROLLING_5=player_groups["GAME_SCORE"].transform(rolling_mean, rolling_window=5),
        PLAYER_FORM_ROLLING_10=player_groups["GAME_SCORE"].transform(rolling_mean, rolling_window=10),
        
        PLAYER_OBPM_ROLLING_3=player_groups["OBPM_PROXY"].transform(rolling_mean, rolling_window=3).fillna(0.0),
        PLAYER_DBPM_ROLLING_3=player_groups["DBPM_PROXY"].transform(rolling_mean, rolling_window=3).fillna(0.0),
        
        PLAYER_OBPM_ROLLING_5=player_groups["OBPM_PROXY"].transform(rolling_mean, rolling_window=5).fillna(0.0),
        PLAYER_DBPM_ROLLING_5=player_groups["DBPM_PROXY"].transform(rolling_mean, rolling_window=5).fillna(0.0),
        
        PLAYER_OBPM_ROLLING_10=player_groups["OBPM_PROXY"].transform(rolling_mean, rolling_window=10).fillna(0.0),
        PLAYER_DBPM_ROLLING_10=player_groups["DBPM_PROXY"].transform(rolling_mean, rolling_window=10).fillna(0.0),
        
        PLAYER_FOUR_FACTOR_EFG_ROLLING_3=player_groups["PLAYER_FOUR_FACTOR_EFG"].transform(rolling_mean, rolling_window=3).fillna(0.50),
        PLAYER_FOUR_FACTOR_TOV_ROLLING_3=player_groups["PLAYER_FOUR_FACTOR_TOV"].transform(rolling_mean, rolling_window=3).fillna(0.12),
        PLAYER_FOUR_FACTOR_OREB_ROLLING_3=player_groups["PLAYER_FOUR_FACTOR_OREB"].transform(rolling_mean, rolling_window=3).fillna(0.03),
        PLAYER_FOUR_FACTOR_FTR_ROLLING_3=player_groups["PLAYER_FOUR_FACTOR_FTR"].transform(rolling_mean, rolling_window=3).fillna(0.25),

        PLAYER_FOUR_FACTOR_EFG_ROLLING_5=player_groups["PLAYER_FOUR_FACTOR_EFG"].transform(rolling_mean, rolling_window=5).fillna(0.50),
        PLAYER_FOUR_FACTOR_TOV_ROLLING_5=player_groups["PLAYER_FOUR_FACTOR_TOV"].transform(rolling_mean, rolling_window=5).fillna(0.12),
        PLAYER_FOUR_FACTOR_OREB_ROLLING_5=player_groups["PLAYER_FOUR_FACTOR_OREB"].transform(rolling_mean, rolling_window=5).fillna(0.03),
        PLAYER_FOUR_FACTOR_FTR_ROLLING_5=player_groups["PLAYER_FOUR_FACTOR_FTR"].transform(rolling_mean, rolling_window=5).fillna(0.25),
        
        PLAYER_FOUR_FACTOR_EFG_ROLLING_10=player_groups["PLAYER_FOUR_FACTOR_EFG"].transform(rolling_mean, rolling_window=10).fillna(0.50),
        PLAYER_FOUR_FACTOR_TOV_ROLLING_10=player_groups["PLAYER_FOUR_FACTOR_TOV"].transform(rolling_mean, rolling_window=10).fillna(0.12),
        PLAYER_FOUR_FACTOR_OREB_ROLLING_10=player_groups["PLAYER_FOUR_FACTOR_OREB"].transform(rolling_mean, rolling_window=10).fillna(0.03),
        PLAYER_FOUR_FACTOR_FTR_ROLLING_10=player_groups["PLAYER_FOUR_FACTOR_FTR"].transform(rolling_mean, rolling_window=10).fillna(0.25),

        FATIGUE_EWMA_MINUTES_3=player_groups["MINUTES_NUM"].transform(lambda x: ewma(x, span=3)),
        FATIGUE_EWMA_MINUTES_5=player_groups["MINUTES_NUM"].transform(lambda x: ewma(x, span=5)),
        FATIGUE_EWMA_MINUTES_10=player_groups["MINUTES_NUM"].transform(lambda x: ewma(x, span=10)),
    )

    # Acute vs Chronic fatigue surge (short-term minutes spike over baseline)
    logs_df["FATIGUE_SURGE"] = (
        logs_df["FATIGUE_EWMA_MINUTES_3"] - logs_df["FATIGUE_EWMA_MINUTES_10"]
    )

    logs_df = add_volatility_features(logs_df)

    # ======================================================
    # Merge Embeddings & Calculate Expected Impact
    # ======================================================

    logs_df = pd.merge(
        logs_df,
        embeddings_df,
        on=["PLAYER_ID", "GAME_DATE"],
        how="left"
    )

    rolling_cols = [
        "PLAYER_FORM_ROLLING_3",
        "PLAYER_FORM_ROLLING_5",
        "PLAYER_FORM_ROLLING_10",
        "PLAYER_OBPM_ROLLING_3",
        "PLAYER_DBPM_ROLLING_3",
        "PLAYER_OBPM_ROLLING_5",
        "PLAYER_DBPM_ROLLING_5",
        "PLAYER_OBPM_ROLLING_10",
        "PLAYER_DBPM_ROLLING_10",
        "PLAYER_FOUR_FACTOR_EFG_ROLLING_3",
        "PLAYER_FOUR_FACTOR_TOV_ROLLING_3",
        "PLAYER_FOUR_FACTOR_OREB_ROLLING_3",
        "PLAYER_FOUR_FACTOR_FTR_ROLLING_3",
        "PLAYER_FOUR_FACTOR_EFG_ROLLING_5",
        "PLAYER_FOUR_FACTOR_TOV_ROLLING_5",
        "PLAYER_FOUR_FACTOR_OREB_ROLLING_5",
        "PLAYER_FOUR_FACTOR_FTR_ROLLING_5",
        "PLAYER_FOUR_FACTOR_EFG_ROLLING_10",
        "PLAYER_FOUR_FACTOR_TOV_ROLLING_10",
        "PLAYER_FOUR_FACTOR_OREB_ROLLING_10",
        "PLAYER_FOUR_FACTOR_FTR_ROLLING_10",
        "FATIGUE_EWMA_MINUTES_3",
        "FATIGUE_EWMA_MINUTES_5",
        "FATIGUE_EWMA_MINUTES_10",
        "FATIGUE_SURGE",
    ]

    logs_df[rolling_cols] = logs_df[rolling_cols].fillna(0)
    
    embed_cols = sorted([c for c in embeddings_df.columns if c.startswith("EMBED_")])
    logs_df[embed_cols] = logs_df[embed_cols].fillna(0.0)

    logs_df["EXPECTED_IMPACT"] = (
        logs_df["PLAYER_FORM_ROLLING_5"] * logs_df["FATIGUE_EWMA_MINUTES_5"]
    )

    # Minutes-weighted robust expected impact
    logs_df["ROBUST_EXPECTED_IMPACT_MINS"] = (
        logs_df["ROBUST_EXPECTED_IMPACT"] * logs_df["FATIGUE_EWMA_MINUTES_5"]
    )

    # Positive expected impact for star, duo, trio, and bench concentration
    logs_df["POS_EXPECTED_IMPACT"] = np.maximum(logs_df["EXPECTED_IMPACT"], 0.0)

    # Rank players within their active game roster by expected impact descending (1 = top star)
    logs_df["ROSTER_IMPACT_RANK"] = (
        logs_df.groupby(["GAME_ID", "TEAM_ID"])["POS_EXPECTED_IMPACT"]
        .rank(ascending=False, method="first")
    )

    logs_df["STAR_IMPACT"] = np.where(logs_df["ROSTER_IMPACT_RANK"] == 1, logs_df["POS_EXPECTED_IMPACT"], 0.0)
    logs_df["TOP_2_IMPACT"] = np.where(logs_df["ROSTER_IMPACT_RANK"] <= 2, logs_df["POS_EXPECTED_IMPACT"], 0.0)
    logs_df["TOP_3_IMPACT"] = np.where(logs_df["ROSTER_IMPACT_RANK"] <= 3, logs_df["POS_EXPECTED_IMPACT"], 0.0)
    logs_df["BENCH_IMPACT"] = np.where(logs_df["ROSTER_IMPACT_RANK"] > 3, logs_df["POS_EXPECTED_IMPACT"], 0.0)

    logs_df["FATIGUE_IMPORTANCE_3"] = (
        logs_df["PLAYER_FORM_ROLLING_3"] *
        logs_df["FATIGUE_EWMA_MINUTES_3"]
    )

    logs_df["FATIGUE_IMPORTANCE_5"] = (
        logs_df["PLAYER_FORM_ROLLING_5"] *
        logs_df["FATIGUE_EWMA_MINUTES_5"]
    )

    logs_df["FATIGUE_IMPORTANCE_10"] = (
        logs_df["PLAYER_FORM_ROLLING_10"] *
        logs_df["FATIGUE_EWMA_MINUTES_10"]
    )

    for col in embed_cols:
        logs_df[f"EXPECTED_{col}"] = logs_df[col] * logs_df["FATIGUE_EWMA_MINUTES_5"]

    logs_df["EXPECTED_FOUR_FACTOR_EFG"] = (
        logs_df["PLAYER_FOUR_FACTOR_EFG_ROLLING_5"] * logs_df["FATIGUE_EWMA_MINUTES_5"]
    )
    logs_df["EXPECTED_FOUR_FACTOR_TOV"] = (
        logs_df["PLAYER_FOUR_FACTOR_TOV_ROLLING_5"] * logs_df["FATIGUE_EWMA_MINUTES_5"]
    )
    logs_df["EXPECTED_FOUR_FACTOR_OREB"] = (
        logs_df["PLAYER_FOUR_FACTOR_OREB_ROLLING_5"] * logs_df["FATIGUE_EWMA_MINUTES_5"]
    )
    logs_df["EXPECTED_FOUR_FACTOR_FTR"] = (
        logs_df["PLAYER_FOUR_FACTOR_FTR_ROLLING_5"] * logs_df["FATIGUE_EWMA_MINUTES_5"]
    )

    # ======================================================
    # Aggregate to the team level (Named Aggregation)
    # ======================================================

    agg_dict = {
        "GAME_DATE": ("GAME_DATE", "first"),

        "EXPECTED_FOUR_FACTOR_EFG_SUM": ("EXPECTED_FOUR_FACTOR_EFG", "sum"),
        "EXPECTED_FOUR_FACTOR_TOV_SUM": ("EXPECTED_FOUR_FACTOR_TOV", "sum"),
        "EXPECTED_FOUR_FACTOR_OREB_SUM": ("EXPECTED_FOUR_FACTOR_OREB", "sum"),
        "EXPECTED_FOUR_FACTOR_FTR_SUM": ("EXPECTED_FOUR_FACTOR_FTR", "sum"),

        "ACTIVE_ROSTER_FORM_SUM": ("EXPECTED_IMPACT", "sum"),
        "ACTIVE_ROSTER_FORM_STD": ("EXPECTED_IMPACT", "std"),
        "ACTIVE_ROSTER_FORM_MAX": ("EXPECTED_IMPACT", "max"),

        "ACTIVE_ROSTER_POS_FORM_SUM": ("POS_EXPECTED_IMPACT", "sum"),
        "ACTIVE_ROSTER_STAR_IMPACT": ("STAR_IMPACT", "sum"),
        "ACTIVE_ROSTER_TOP_2_IMPACT": ("TOP_2_IMPACT", "sum"),
        "ACTIVE_ROSTER_TOP_3_IMPACT": ("TOP_3_IMPACT", "sum"),
        "ACTIVE_ROSTER_BENCH_IMPACT": ("BENCH_IMPACT", "sum"),

        "ACTIVE_ROSTER_ROBUST_FORM_SUM": ("ROBUST_EXPECTED_IMPACT_MINS", "sum"),
        "ACTIVE_ROSTER_ROBUST_FORM_MAX": ("ROBUST_EXPECTED_IMPACT_MINS", "max"),

        "TOTAL_EXPECTED_MINUTES": ("FATIGUE_EWMA_MINUTES_5", "sum"),

        "FATIGUE_SURGE_SUM": ("FATIGUE_SURGE", "sum"),
        "FATIGUE_SURGE_MAX": ("FATIGUE_SURGE", "max"),

        "FATIGUE_EWMA_MINUTES_3_SUM": ("FATIGUE_EWMA_MINUTES_3", "sum"),
        "FATIGUE_EWMA_MINUTES_3_STD": ("FATIGUE_EWMA_MINUTES_3", "std"),
        "FATIGUE_EWMA_MINUTES_3_MAX": ("FATIGUE_EWMA_MINUTES_3", "max"),
        "FATIGUE_EWMA_MINUTES_3_MEAN": ("FATIGUE_EWMA_MINUTES_3", "mean"),

        "FATIGUE_EWMA_MINUTES_5_SUM": ("FATIGUE_EWMA_MINUTES_5", "sum"),
        "FATIGUE_EWMA_MINUTES_5_STD": ("FATIGUE_EWMA_MINUTES_5", "std"),
        "FATIGUE_EWMA_MINUTES_5_MAX": ("FATIGUE_EWMA_MINUTES_5", "max"),
        "FATIGUE_EWMA_MINUTES_5_MEAN": ("FATIGUE_EWMA_MINUTES_5", "mean"),

        "FATIGUE_EWMA_MINUTES_10_SUM": ("FATIGUE_EWMA_MINUTES_10", "sum"),
        "FATIGUE_EWMA_MINUTES_10_STD": ("FATIGUE_EWMA_MINUTES_10", "std"),
        "FATIGUE_EWMA_MINUTES_10_MAX": ("FATIGUE_EWMA_MINUTES_10", "max"),
        "FATIGUE_EWMA_MINUTES_10_MEAN": ("FATIGUE_EWMA_MINUTES_10", "mean"),

        "FATIGUE_IMPORTANCE_3_SUM": ("FATIGUE_IMPORTANCE_3", "sum"),
        "FATIGUE_IMPORTANCE_3_STD": ("FATIGUE_IMPORTANCE_3", "std"),
        "FATIGUE_IMPORTANCE_3_MAX": ("FATIGUE_IMPORTANCE_3", "max"),

        "FATIGUE_IMPORTANCE_5_SUM": ("FATIGUE_IMPORTANCE_5", "sum"),
        "FATIGUE_IMPORTANCE_5_STD": ("FATIGUE_IMPORTANCE_5", "std"),
        "FATIGUE_IMPORTANCE_5_MAX": ("FATIGUE_IMPORTANCE_5", "max"),

        "FATIGUE_IMPORTANCE_10_SUM": ("FATIGUE_IMPORTANCE_10", "sum"),
        "FATIGUE_IMPORTANCE_10_STD": ("FATIGUE_IMPORTANCE_10", "std"),
        "FATIGUE_IMPORTANCE_10_MAX": ("FATIGUE_IMPORTANCE_10", "max"),
    }
    for col in embed_cols:
        agg_dict[f"EXPECTED_{col}_SUM"] = (f"EXPECTED_{col}", "sum")
        agg_dict[f"EXPECTED_{col}_MAX"] = (f"EXPECTED_{col}", "max")
        agg_dict[f"EXPECTED_{col}_STD"] = (f"EXPECTED_{col}", "std")
        agg_dict[f"{col}_MAX"] = (col, "max")
        agg_dict[f"{col}_STD"] = (col, "std")

    roster_agg = (
        logs_df
        .groupby(["GAME_ID", "TEAM_ID"])
        .agg(**agg_dict)
        .reset_index()
    )

    roster_agg["ACTIVE_ROSTER_FORM_STD"] = (
        roster_agg["ACTIVE_ROSTER_FORM_STD"].fillna(DEFAULT_VALUE)
    )

    fatigue_cols = [
        "FATIGUE_IMPORTANCE_3_STD",
        "FATIGUE_IMPORTANCE_5_STD",
        "FATIGUE_IMPORTANCE_10_STD",
    ]

    roster_agg[fatigue_cols] = roster_agg[fatigue_cols].fillna(DEFAULT_VALUE)

    # Standardized, robust hierarchy for Star, Duo, Trio, and Bench shares
    total_pos = roster_agg["ACTIVE_ROSTER_POS_FORM_SUM"].replace(0, np.nan)
    roster_agg["ACTIVE_ROSTER_STAR_SHARE"] = (
        (roster_agg["ACTIVE_ROSTER_STAR_IMPACT"] / total_pos).fillna(DEFAULT_VALUE)
    )
    roster_agg["ACTIVE_ROSTER_TOP_2_SHARE"] = (
        (roster_agg["ACTIVE_ROSTER_TOP_2_IMPACT"] / total_pos).fillna(DEFAULT_VALUE)
    )
    roster_agg["ACTIVE_ROSTER_TOP_3_SHARE"] = (
        (roster_agg["ACTIVE_ROSTER_TOP_3_IMPACT"] / total_pos).fillna(DEFAULT_VALUE)
    )
    roster_agg["ACTIVE_ROSTER_BENCH_SHARE"] = (
        (roster_agg["ACTIVE_ROSTER_BENCH_IMPACT"] / total_pos).fillna(DEFAULT_VALUE)
    )

    for col in embed_cols:
        roster_agg[f"EXPECTED_{col}_WEIGHTED_MEAN"] = (
            roster_agg[f"EXPECTED_{col}_SUM"] / roster_agg["TOTAL_EXPECTED_MINUTES"]
        ).replace([np.inf, -np.inf], np.nan).fillna(DEFAULT_VALUE)

    total_expected_mins = roster_agg["TOTAL_EXPECTED_MINUTES"].replace(0, np.nan)
    roster_agg["ACTIVE_ROSTER_EXPECTED_EFG"] = (
        roster_agg["EXPECTED_FOUR_FACTOR_EFG_SUM"] / total_expected_mins
    ).replace([np.inf, -np.inf], np.nan).fillna(0.50)

    roster_agg["ACTIVE_ROSTER_EXPECTED_TOV"] = (
        roster_agg["EXPECTED_FOUR_FACTOR_TOV_SUM"] / total_expected_mins
    ).replace([np.inf, -np.inf], np.nan).fillna(0.12)

    roster_agg["ACTIVE_ROSTER_EXPECTED_OREB"] = (
        roster_agg["EXPECTED_FOUR_FACTOR_OREB_SUM"] / total_expected_mins
    ).replace([np.inf, -np.inf], np.nan).fillna(0.03)

    roster_agg["ACTIVE_ROSTER_EXPECTED_FTR"] = (
        roster_agg["EXPECTED_FOUR_FACTOR_FTR_SUM"] / total_expected_mins
    ).replace([np.inf, -np.inf], np.nan).fillna(0.25)

    # ======================================================
    # Team Rolling Lineup & Identity Features (Leak-Free)
    # ======================================================

    roster_agg = roster_agg.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
    team_groups = roster_agg.groupby("TEAM_ID")

    # Team rolling 10-game active roster baseline (strict shift 1 with historical prior)
    roster_agg["ROLLING_ACTIVE_ROSTER_FORM_10"] = (
        team_groups["ACTIVE_ROSTER_FORM_SUM"]
        .transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean())
        .fillna(HISTORICAL_ROSTER_FORM_PRIOR)
    )

    # Lineup availability ratio & missing production deficit (injury/rest detector)
    roster_agg["LINEUP_AVAILABILITY_RATIO"] = (
        roster_agg["ACTIVE_ROSTER_FORM_SUM"]
        / roster_agg["ROLLING_ACTIVE_ROSTER_FORM_10"].replace(0, np.nan)
    ).fillna(1.0).clip(lower=0.2, upper=2.0)

    roster_agg["LINEUP_MISSING_PRODUCTION"] = np.maximum(
        roster_agg["ROLLING_ACTIVE_ROSTER_FORM_10"] - roster_agg["ACTIVE_ROSTER_FORM_SUM"],
        0.0,
    )

    # Rolling team concentration (structural identity: heliocentric vs balanced)
    roster_agg["ROLLING_STAR_SHARE_10"] = (
        team_groups["ACTIVE_ROSTER_STAR_SHARE"]
        .transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean())
        .fillna(0.25)
    )
    roster_agg["ROLLING_TOP_2_SHARE_10"] = (
        team_groups["ACTIVE_ROSTER_TOP_2_SHARE"]
        .transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean())
        .fillna(0.45)
    )

    # Clean intermediate columns before merging with matchups
    roster_agg.drop(
        columns=[
            "GAME_DATE",
            "EXPECTED_FOUR_FACTOR_EFG_SUM",
            "EXPECTED_FOUR_FACTOR_TOV_SUM",
            "EXPECTED_FOUR_FACTOR_OREB_SUM",
            "EXPECTED_FOUR_FACTOR_FTR_SUM",
            "ACTIVE_ROSTER_POS_FORM_SUM",
            "ACTIVE_ROSTER_STAR_IMPACT",
            "ACTIVE_ROSTER_TOP_2_IMPACT",
            "ACTIVE_ROSTER_TOP_3_IMPACT",
        ],
        inplace=True,
        errors="ignore",
    )


    # ======================================================
    # Merge with matchup dataset
    # ======================================================

    matchups_df["HOME_GAME_ID"] = matchups_df["HOME_GAME_ID"].astype(str)
    matchups_df["AWAY_GAME_ID"] = matchups_df["AWAY_GAME_ID"].astype(str)
    matchups_df["HOME_TEAM_ID"] = matchups_df["HOME_TEAM_ID"].astype(str)
    matchups_df["AWAY_TEAM_ID"] = matchups_df["AWAY_TEAM_ID"].astype(str)

    roster_agg["GAME_ID"] = roster_agg["GAME_ID"].astype(str)
    roster_agg["TEAM_ID"] = roster_agg["TEAM_ID"].astype(str)

    matchups_df = pd.merge(
        matchups_df,
        roster_agg.add_prefix("HOME_"),
        left_on=["HOME_GAME_ID", "HOME_TEAM_ID"],
        right_on=["HOME_GAME_ID", "HOME_TEAM_ID"],
        how="left",
    )

    matchups_df = pd.merge(
        matchups_df,
        roster_agg.add_prefix("AWAY_"),
        left_on=["AWAY_GAME_ID", "AWAY_TEAM_ID"],
        right_on=["AWAY_GAME_ID", "AWAY_TEAM_ID"],
        how="left",
    )

    matchups_df = matchups_df.copy()

    # ======================================================
    # Compute matchup-level player feature deltas
    # ======================================================

    matchups_df["DELTA_ACTIVE_ROSTER_EXPECTED_EFG"] = (
        matchups_df["HOME_ACTIVE_ROSTER_EXPECTED_EFG"]
        - matchups_df["AWAY_ACTIVE_ROSTER_EXPECTED_EFG"]
    )

    matchups_df["DELTA_ACTIVE_ROSTER_EXPECTED_TOV"] = (
        matchups_df["HOME_ACTIVE_ROSTER_EXPECTED_TOV"]
        - matchups_df["AWAY_ACTIVE_ROSTER_EXPECTED_TOV"]
    )

    matchups_df["DELTA_ACTIVE_ROSTER_EXPECTED_OREB"] = (
        matchups_df["HOME_ACTIVE_ROSTER_EXPECTED_OREB"]
        - matchups_df["AWAY_ACTIVE_ROSTER_EXPECTED_OREB"]
    )

    matchups_df["DELTA_ACTIVE_ROSTER_EXPECTED_FTR"] = (
        matchups_df["HOME_ACTIVE_ROSTER_EXPECTED_FTR"]
        - matchups_df["AWAY_ACTIVE_ROSTER_EXPECTED_FTR"]
    )

    matchups_df["DELTA_ACTIVE_ROSTER_FORM_SUM"] = (
        matchups_df["HOME_ACTIVE_ROSTER_FORM_SUM"]
        - matchups_df["AWAY_ACTIVE_ROSTER_FORM_SUM"]
    )

    matchups_df["DELTA_ACTIVE_ROSTER_FORM_STD"] = (
        matchups_df["HOME_ACTIVE_ROSTER_FORM_STD"]
        - matchups_df["AWAY_ACTIVE_ROSTER_FORM_STD"]
    )

    matchups_df["DELTA_ACTIVE_ROSTER_FORM_MAX"] = (
        matchups_df["HOME_ACTIVE_ROSTER_FORM_MAX"]
        - matchups_df["AWAY_ACTIVE_ROSTER_FORM_MAX"]
    )

    matchups_df["DELTA_ACTIVE_ROSTER_STAR_SHARE"] = (
        matchups_df["HOME_ACTIVE_ROSTER_STAR_SHARE"]
        - matchups_df["AWAY_ACTIVE_ROSTER_STAR_SHARE"]
    )

    matchups_df["DELTA_ACTIVE_ROSTER_TOP_2_SHARE"] = (
        matchups_df["HOME_ACTIVE_ROSTER_TOP_2_SHARE"]
        - matchups_df["AWAY_ACTIVE_ROSTER_TOP_2_SHARE"]
    )

    matchups_df["DELTA_ACTIVE_ROSTER_TOP_3_SHARE"] = (
        matchups_df["HOME_ACTIVE_ROSTER_TOP_3_SHARE"]
        - matchups_df["AWAY_ACTIVE_ROSTER_TOP_3_SHARE"]
    )

    matchups_df["DELTA_ACTIVE_ROSTER_BENCH_SHARE"] = (
        matchups_df["HOME_ACTIVE_ROSTER_BENCH_SHARE"]
        - matchups_df["AWAY_ACTIVE_ROSTER_BENCH_SHARE"]
    )

    matchups_df["DELTA_ACTIVE_ROSTER_BENCH_IMPACT"] = (
        matchups_df["HOME_ACTIVE_ROSTER_BENCH_IMPACT"]
        - matchups_df["AWAY_ACTIVE_ROSTER_BENCH_IMPACT"]
    )

    matchups_df["DELTA_ACTIVE_ROSTER_ROBUST_FORM_SUM"] = (
        matchups_df["HOME_ACTIVE_ROSTER_ROBUST_FORM_SUM"]
        - matchups_df["AWAY_ACTIVE_ROSTER_ROBUST_FORM_SUM"]
    )

    matchups_df["DELTA_LINEUP_AVAILABILITY_RATIO"] = (
        matchups_df["HOME_LINEUP_AVAILABILITY_RATIO"]
        - matchups_df["AWAY_LINEUP_AVAILABILITY_RATIO"]
    )

    matchups_df["DELTA_LINEUP_MISSING_PRODUCTION"] = (
        matchups_df["HOME_LINEUP_MISSING_PRODUCTION"]
        - matchups_df["AWAY_LINEUP_MISSING_PRODUCTION"]
    )

    matchups_df["DELTA_ROLLING_STAR_SHARE_10"] = (
        matchups_df["HOME_ROLLING_STAR_SHARE_10"]
        - matchups_df["AWAY_ROLLING_STAR_SHARE_10"]
    )

    matchups_df["DELTA_ROLLING_TOP_2_SHARE_10"] = (
        matchups_df["HOME_ROLLING_TOP_2_SHARE_10"]
        - matchups_df["AWAY_ROLLING_TOP_2_SHARE_10"]
    )

    matchups_df["DELTA_FATIGUE_SURGE_SUM"] = (
        matchups_df["HOME_FATIGUE_SURGE_SUM"]
        - matchups_df["AWAY_FATIGUE_SURGE_SUM"]
    )

    matchups_df["DELTA_FATIGUE_SURGE_MAX"] = (
        matchups_df["HOME_FATIGUE_SURGE_MAX"]
        - matchups_df["AWAY_FATIGUE_SURGE_MAX"]
    )


    matchups_df["DELTA_FATIGUE_EWMA_MINUTES_3_SUM"] = (
        matchups_df["HOME_FATIGUE_EWMA_MINUTES_3_SUM"]
        - matchups_df["AWAY_FATIGUE_EWMA_MINUTES_3_SUM"]
    )
    matchups_df["DELTA_FATIGUE_EWMA_MINUTES_3_STD"] = (
        matchups_df["HOME_FATIGUE_EWMA_MINUTES_3_STD"]
        - matchups_df["AWAY_FATIGUE_EWMA_MINUTES_3_STD"]
    )
    matchups_df["DELTA_FATIGUE_EWMA_MINUTES_3_MAX"] = (
        matchups_df["HOME_FATIGUE_EWMA_MINUTES_3_MAX"]
        - matchups_df["AWAY_FATIGUE_EWMA_MINUTES_3_MAX"]
    )
    matchups_df["DELTA_FATIGUE_EWMA_MINUTES_3_MEAN"] = (
        matchups_df["HOME_FATIGUE_EWMA_MINUTES_3_MEAN"]
        - matchups_df["AWAY_FATIGUE_EWMA_MINUTES_3_MEAN"]
    )

    matchups_df["DELTA_FATIGUE_EWMA_MINUTES_5_SUM"] = (
        matchups_df["HOME_FATIGUE_EWMA_MINUTES_5_SUM"]
        - matchups_df["AWAY_FATIGUE_EWMA_MINUTES_5_SUM"]
    )
    matchups_df["DELTA_FATIGUE_EWMA_MINUTES_5_STD"] = (
        matchups_df["HOME_FATIGUE_EWMA_MINUTES_5_STD"]
        - matchups_df["AWAY_FATIGUE_EWMA_MINUTES_5_STD"]
    )
    matchups_df["DELTA_FATIGUE_EWMA_MINUTES_5_MAX"] = (
        matchups_df["HOME_FATIGUE_EWMA_MINUTES_5_MAX"]
        - matchups_df["AWAY_FATIGUE_EWMA_MINUTES_5_MAX"]
    )
    matchups_df["DELTA_FATIGUE_EWMA_MINUTES_5_MEAN"] = (
        matchups_df["HOME_FATIGUE_EWMA_MINUTES_5_MEAN"]
        - matchups_df["AWAY_FATIGUE_EWMA_MINUTES_5_MEAN"]
    )

    matchups_df["DELTA_FATIGUE_EWMA_MINUTES_10_SUM"] = (
        matchups_df["HOME_FATIGUE_EWMA_MINUTES_10_SUM"]
        - matchups_df["AWAY_FATIGUE_EWMA_MINUTES_10_SUM"]
    )
    matchups_df["DELTA_FATIGUE_EWMA_MINUTES_10_STD"] = (
        matchups_df["HOME_FATIGUE_EWMA_MINUTES_10_STD"]
        - matchups_df["AWAY_FATIGUE_EWMA_MINUTES_10_STD"]
    )
    matchups_df["DELTA_FATIGUE_EWMA_MINUTES_10_MAX"] = (
        matchups_df["HOME_FATIGUE_EWMA_MINUTES_10_MAX"]
        - matchups_df["AWAY_FATIGUE_EWMA_MINUTES_10_MAX"]
    )
    matchups_df["DELTA_FATIGUE_EWMA_MINUTES_10_MEAN"] = (
        matchups_df["HOME_FATIGUE_EWMA_MINUTES_10_MEAN"]
        - matchups_df["AWAY_FATIGUE_EWMA_MINUTES_10_MEAN"]
    )

    matchups_df["DELTA_FATIGUE_IMPORTANCE_3_SUM"] = (
        matchups_df["HOME_FATIGUE_IMPORTANCE_3_SUM"]
        - matchups_df["AWAY_FATIGUE_IMPORTANCE_3_SUM"]
    )
    matchups_df["DELTA_FATIGUE_IMPORTANCE_3_STD"] = (
        matchups_df["HOME_FATIGUE_IMPORTANCE_3_STD"]
        - matchups_df["AWAY_FATIGUE_IMPORTANCE_3_STD"]
    )
    matchups_df["DELTA_FATIGUE_IMPORTANCE_3_MAX"] = (
        matchups_df["HOME_FATIGUE_IMPORTANCE_3_MAX"]
        - matchups_df["AWAY_FATIGUE_IMPORTANCE_3_MAX"]
    )

    matchups_df["DELTA_FATIGUE_IMPORTANCE_5_SUM"] = (
        matchups_df["HOME_FATIGUE_IMPORTANCE_5_SUM"]
        - matchups_df["AWAY_FATIGUE_IMPORTANCE_5_SUM"]
    )
    matchups_df["DELTA_FATIGUE_IMPORTANCE_5_STD"] = (
        matchups_df["HOME_FATIGUE_IMPORTANCE_5_STD"]
        - matchups_df["AWAY_FATIGUE_IMPORTANCE_5_STD"]
    )
    matchups_df["DELTA_FATIGUE_IMPORTANCE_5_MAX"] = (
        matchups_df["HOME_FATIGUE_IMPORTANCE_5_MAX"]
        - matchups_df["AWAY_FATIGUE_IMPORTANCE_5_MAX"]
    )

    matchups_df["DELTA_FATIGUE_IMPORTANCE_10_SUM"] = (
        matchups_df["HOME_FATIGUE_IMPORTANCE_10_SUM"]
        - matchups_df["AWAY_FATIGUE_IMPORTANCE_10_SUM"]
    )
    matchups_df["DELTA_FATIGUE_IMPORTANCE_10_STD"] = (
        matchups_df["HOME_FATIGUE_IMPORTANCE_10_STD"]
        - matchups_df["AWAY_FATIGUE_IMPORTANCE_10_STD"]
    )
    matchups_df["DELTA_FATIGUE_IMPORTANCE_10_MAX"] = (
        matchups_df["HOME_FATIGUE_IMPORTANCE_10_MAX"]
        - matchups_df["AWAY_FATIGUE_IMPORTANCE_10_MAX"]
    )
    

    for col in embed_cols:
        dim = col.replace("EMBED_", "")
        matchups_df[f"EMBED_DELTA_{dim}_SUM"] = (
            matchups_df[f"HOME_EXPECTED_{col}_SUM"]
            - matchups_df[f"AWAY_EXPECTED_{col}_SUM"]
        )
        matchups_df[f"EMBED_DELTA_{dim}_MAX"] = (
            matchups_df[f"HOME_EXPECTED_{col}_MAX"]
            - matchups_df[f"AWAY_EXPECTED_{col}_MAX"]
        )
        matchups_df[f"EMBED_DELTA_{dim}_STD"] = (
            matchups_df[f"HOME_EXPECTED_{col}_STD"]
            - matchups_df[f"AWAY_EXPECTED_{col}_STD"]
        )
        matchups_df[f"EMBED_DELTA_{dim}_MEAN"] = (
            matchups_df[f"HOME_EXPECTED_{col}_WEIGHTED_MEAN"]
            - matchups_df[f"AWAY_EXPECTED_{col}_WEIGHTED_MEAN"]
        )

        matchups_df[f"EMBED_RAW_DELTA_{dim}_MAX"] = (
            matchups_df[f"HOME_{col}_MAX"]
            - matchups_df[f"AWAY_{col}_MAX"]
        )
        matchups_df[f"EMBED_RAW_DELTA_{dim}_STD"] = (
            matchups_df[f"HOME_{col}_STD"]
            - matchups_df[f"AWAY_{col}_STD"]
        )

    # ------------------------------------------------------
    # Drop intermediate embedding columns so XGBoost 
    # doesn't accidentally absorb them via 'HOME_' prefixes.
    # ------------------------------------------------------
    intermediate_embed_cols = [
        "HOME_TOTAL_EXPECTED_MINUTES",
        "AWAY_TOTAL_EXPECTED_MINUTES",
    ]
    for col in embed_cols:
        for prefix in ["HOME_", "AWAY_"]:
            intermediate_embed_cols.extend([
                f"{prefix}EXPECTED_{col}_SUM",
                f"{prefix}EXPECTED_{col}_MAX",
                f"{prefix}EXPECTED_{col}_STD",
                f"{prefix}EXPECTED_{col}_WEIGHTED_MEAN",
                f"{prefix}{col}_MAX",
                f"{prefix}{col}_STD",
            ])

    matchups_df.drop(
        columns=intermediate_embed_cols,
        inplace=True,
        errors="ignore",
    )

    num_cols = matchups_df.select_dtypes(include=["number"]).columns
    matchups_df[num_cols] = matchups_df[num_cols].fillna(DEFAULT_VALUE)

    obj_cols = matchups_df.select_dtypes(include=["object", "string"]).columns
    matchups_df[obj_cols] = matchups_df[obj_cols].fillna("").astype(str)

    # ======================================================
    # Save engineered dataset (CSV & Parquet)
    # ======================================================

    output_path = DATA_DIR / OUTPUT_FILE
    matchups_df.to_csv(output_path, index=False)

    parquet_output_path = output_path.with_suffix(".parquet")
    matchups_df.to_parquet(parquet_output_path, index=False)

    logger.info(
        f"Saved engineered matchup dataset to '{OUTPUT_FILE}' and '{parquet_output_path.name}' "
        f"({len(matchups_df):,} rows)."
    )


if __name__ == "__main__":
    main()