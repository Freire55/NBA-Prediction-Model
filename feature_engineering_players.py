"""
Creates player-based features for the NBA prediction model.

This script computes leak-free rolling player performance metrics,
estimates expected player impact based on historical playing time,
aggregates those values to the team level, and merges the resulting
features into the matchup dataset.

Input:
    data/raw_player_game_logs.csv
    data/ml_ready_matchups.csv

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
    """
    Converts NBA API minute strings (MM:SS) into decimal minutes.

    Handles malformed values safely by coercing invalid values to zero.
    """

    time_split = minutes.astype(str).str.split(":", expand=True)

    # Some API rows may contain only whole minutes (e.g. "38")
    if 1 not in time_split.columns:
        time_split[1] = 0

    return (
        pd.to_numeric(time_split[0], errors="coerce").fillna(DEFAULT_VALUE)
        + pd.to_numeric(time_split[1], errors="coerce").fillna(DEFAULT_VALUE) / 60.0
    )


def calculate_game_score(df: pd.DataFrame) -> pd.Series:
    """
    Computes John Hollinger's Game Score for each player appearance,
    providing a single-game estimate of overall player productivity.
    """

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
    """
    Computes a rolling mean using only prior observations.

    shift(1) prevents target leakage by excluding the current game.
    """

    return (
        series.shift(1)
        .rolling(ROLLING_WINDOW, min_periods=1)
        .mean()
    )


# ======================================================
# Main Pipeline
# ======================================================

def main() -> None:
    """Generate player-based matchup features."""

    # ======================================================
    # Load datasets
    # ======================================================

    logs_df = pd.read_csv(DATA_DIR / GAME_LOGS_FILE)
    matchups_df = pd.read_csv(DATA_DIR / MATCHUPS_FILE)

    logger.info(
        f"Loaded {len(logs_df):,} player logs and "
        f"{len(matchups_df):,} matchup rows."
    )

    # ======================================================
    # Player-level feature engineering
    # ======================================================

    logs_df["PLAYER_ID"] = logs_df["PLAYER_ID"].astype(str)
    logs_df["GAME_DATE"] = pd.to_datetime(logs_df["GAME_DATE"])

    # Ensure rolling statistics follow each player's true career timeline.
    logs_df = (
        logs_df.sort_values(["PLAYER_ID", "GAME_DATE"])
        .reset_index(drop=True)
    )

    logs_df = logs_df.assign(
        MINUTES_NUM=parse_minutes(logs_df["MIN"]),
        GAME_SCORE=calculate_game_score(logs_df),
    )

    # Precompute player groups to avoid repeated groupby operations.
    player_groups = logs_df.groupby("PLAYER_ID")

    logs_df = logs_df.assign(
        PLAYER_FORM_ROLLING=player_groups["GAME_SCORE"].transform(rolling_mean),
        MINUTES_ROLLING=player_groups["MINUTES_NUM"].transform(rolling_mean),
    )

    # ======================================================
    # Aggregate player impact to the team level
    # ======================================================

    # Weight recent player performance by expected playing time to estimate
    # each player's projected contribution before tip-off.
    logs_df["EXPECTED_IMPACT"] = (
        logs_df["PLAYER_FORM_ROLLING"]
        * logs_df["MINUTES_ROLLING"]
    )

    roster_agg = (
        logs_df
        .groupby(["GAME_ID", "TEAM_ID"])["EXPECTED_IMPACT"]
        .agg(["sum", "std"])
        .reset_index()
        .rename(
            columns={
                "sum": "ACTIVE_ROSTER_FORM_SUM",
                "std": "ACTIVE_ROSTER_FORM_STD",
            }
        )
    )

    roster_agg["ACTIVE_ROSTER_FORM_STD"] = (
        roster_agg["ACTIVE_ROSTER_FORM_STD"]
        .fillna(DEFAULT_VALUE)
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

    # Consolidate the dataframe after repeated merge operations to avoid
    # pandas fragmentation and subsequent PerformanceWarnings.
    matchups_df = matchups_df.copy()

    # ======================================================
    # Compute matchup-level player feature deltas
    # ======================================================

    # Compare the projected strength of both active rosters.
    # Positive values indicate an advantage for the home team.
    matchups_df["DELTA_ACTIVE_ROSTER_FORM_SUM"] = (
        matchups_df["HOME_ACTIVE_ROSTER_FORM_SUM"]
        - matchups_df["AWAY_ACTIVE_ROSTER_FORM_SUM"]
    )

    # Compare how concentrated each team's expected production is.
    # Positive values indicate the home team relies more heavily on a
    # small number of players, while negative values indicate the away
    # team has the more top-heavy rotation.
    matchups_df["DELTA_ACTIVE_ROSTER_FORM_STD"] = (
        matchups_df["HOME_ACTIVE_ROSTER_FORM_STD"]
        - matchups_df["AWAY_ACTIVE_ROSTER_FORM_STD"]
    )

    # Fill missing values caused by players or teams without sufficient
    # historical data early in the dataset.
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