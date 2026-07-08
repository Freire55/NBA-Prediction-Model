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

# ======================================================
# Constants & Configuration
# ======================================================

DATA_DIR = Path(__file__).resolve().parent / "data"

ROLLING_WINDOW = 5
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


def rolling_mean(series: pd.Series) -> pd.Series:
    """Computes a rolling mean using only prior observations."""
    return series.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()


# ======================================================
# Main Pipeline
# ======================================================

def main() -> None:
    """Generate player-based matchup features."""

    logs_df = pd.read_csv(DATA_DIR / GAME_LOGS_FILE)
    matchups_df = pd.read_csv(DATA_DIR / MATCHUPS_FILE)
    embeddings_df = pd.read_csv(DATA_DIR / "player_embeddings.csv")

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
    )

    player_groups = logs_df.groupby("PLAYER_ID")

    logs_df = logs_df.assign(
        PLAYER_FORM_ROLLING=player_groups["GAME_SCORE"].transform(rolling_mean),
        MINUTES_ROLLING=player_groups["MINUTES_NUM"].transform(rolling_mean),
    )

    # ======================================================
    # Merge Embeddings & Calculate Expected Impact
    # ======================================================

    logs_df = pd.merge(
        logs_df,
        embeddings_df,
        on=["PLAYER_ID", "GAME_DATE"],
        how="left"
    )

    logs_df["MINUTES_ROLLING"] = logs_df["MINUTES_ROLLING"].fillna(0)
    logs_df["PLAYER_FORM_ROLLING"] = logs_df["PLAYER_FORM_ROLLING"].fillna(0)
    
    logs_df["EMBED_1"] = logs_df["EMBED_1"].fillna(0)
    logs_df["EMBED_2"] = logs_df["EMBED_2"].fillna(0)

    logs_df["EXPECTED_IMPACT"] = (
        logs_df["PLAYER_FORM_ROLLING"] * logs_df["MINUTES_ROLLING"]
    )
    
    logs_df["EXPECTED_EMBED_1"] = logs_df["EMBED_1"] * logs_df["MINUTES_ROLLING"]
    logs_df["EXPECTED_EMBED_2"] = logs_df["EMBED_2"] * logs_df["MINUTES_ROLLING"]

    # ======================================================
    # Aggregate to the team level (Named Aggregation)
    # ======================================================

    roster_agg = (
        logs_df
        .groupby(["GAME_ID", "TEAM_ID"])
        .agg(
            ACTIVE_ROSTER_FORM_SUM=("EXPECTED_IMPACT", "sum"),
            ACTIVE_ROSTER_FORM_STD=("EXPECTED_IMPACT", "std"),
            _TOP_PLAYER_IMPACT=("EXPECTED_IMPACT", "max"),
            EXPECTED_EMBED_1=("EXPECTED_EMBED_1", "sum"),
            EXPECTED_EMBED_2=("EXPECTED_EMBED_2", "sum")
        )
        .reset_index()
    )

    roster_agg["ACTIVE_ROSTER_FORM_STD"] = (
        roster_agg["ACTIVE_ROSTER_FORM_STD"].fillna(DEFAULT_VALUE)
    )

    roster_agg["ACTIVE_ROSTER_STAR_SHARE"] = (
        roster_agg["_TOP_PLAYER_IMPACT"] / roster_agg["ACTIVE_ROSTER_FORM_SUM"]
    ).replace([np.inf, -np.inf], np.nan).fillna(DEFAULT_VALUE)

    roster_agg = roster_agg.drop(columns=["_TOP_PLAYER_IMPACT"])

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

    matchups_df["DELTA_ACTIVE_ROSTER_FORM_SUM"] = (
        matchups_df["HOME_ACTIVE_ROSTER_FORM_SUM"]
        - matchups_df["AWAY_ACTIVE_ROSTER_FORM_SUM"]
    )

    matchups_df["DELTA_ACTIVE_ROSTER_FORM_STD"] = (
        matchups_df["HOME_ACTIVE_ROSTER_FORM_STD"]
        - matchups_df["AWAY_ACTIVE_ROSTER_FORM_STD"]
    )

    matchups_df["DELTA_ACTIVE_ROSTER_STAR_SHARE"] = (
        matchups_df["HOME_ACTIVE_ROSTER_STAR_SHARE"]
        - matchups_df["AWAY_ACTIVE_ROSTER_STAR_SHARE"]
    )

    matchups_df["EMBED_DELTA_1"] = (
        matchups_df["HOME_EXPECTED_EMBED_1"]
        - matchups_df["AWAY_EXPECTED_EMBED_1"]
    )

    matchups_df["EMBED_DELTA_2"] = (
        matchups_df["HOME_EXPECTED_EMBED_2"]
        - matchups_df["AWAY_EXPECTED_EMBED_2"]
    )

    # ------------------------------------------------------
    # Drop intermediate embedding columns so XGBoost 
    # doesn't accidentally absorb them via 'HOME_' prefixes.
    # ------------------------------------------------------
    matchups_df.drop(
        columns=[
            "HOME_EXPECTED_EMBED_1",
            "AWAY_EXPECTED_EMBED_1",
            "HOME_EXPECTED_EMBED_2",
            "AWAY_EXPECTED_EMBED_2",
        ],
        inplace=True,
        errors="ignore"
    )

    matchups_df = matchups_df.fillna(DEFAULT_VALUE)

    # ======================================================
    # Save engineered dataset
    # ======================================================

    output_path = DATA_DIR / OUTPUT_FILE
    matchups_df.to_csv(output_path, index=False)

    logger.info(
        f"Saved engineered matchup dataset to '{OUTPUT_FILE}' "
        f"({len(matchups_df):,} rows)."
    )


if __name__ == "__main__":
    main()