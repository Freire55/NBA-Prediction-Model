"""
Data validation schemas and contracts for the NBA prediction pipeline.

Uses Pandera to enforce strict column types, value bounds, nullability,
and data integrity constraints across raw, processed, and ML-ready datasets.
"""

from typing import Optional
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema


# ======================================================
# 1. Raw Player Game Logs Schema
# ======================================================

raw_player_game_logs_schema = DataFrameSchema(
    columns={
        "SEASON_ID": Column(pa.String, nullable=False),
        "PLAYER_ID": Column(pa.String, nullable=False),
        "GAME_ID": Column(pa.String, nullable=False),
        "GAME_DATE": Column(pa.DateTime, nullable=False),
        "MATCHUP": Column(pa.String, nullable=False),
        "WL": Column(pa.String, checks=Check.isin(["W", "L"]), nullable=True),
        "MIN": Column(pa.String, nullable=False),
        "PTS": Column(pa.Float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "FGM": Column(pa.Float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "FGA": Column(pa.Float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "FG3M": Column(pa.Float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "FG3A": Column(pa.Float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "FTM": Column(pa.Float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "FTA": Column(pa.Float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "OREB": Column(pa.Float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "DREB": Column(pa.Float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "REB": Column(pa.Float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "AST": Column(pa.Float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "STL": Column(pa.Float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "BLK": Column(pa.Float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "TOV": Column(pa.Float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "PF": Column(pa.Float, checks=Check.greater_than_or_equal_to(0), nullable=True),
    },
    coerce=True,
    strict=False,
)


# ======================================================
# 2. Raw Historical Team Box Scores Schema
# ======================================================

raw_historical_team_schema = DataFrameSchema(
    columns={
        "SEASON_ID": Column(pa.String, nullable=False),
        "TEAM_ID": Column(pa.String, nullable=False),
        "TEAM_ABBREVIATION": Column(pa.String, nullable=False),
        "GAME_ID": Column(pa.String, nullable=False),
        "GAME_DATE": Column(pa.DateTime, nullable=False),
        "MATCHUP": Column(pa.String, nullable=False),
        "WL": Column(pa.String, checks=Check.isin(["W", "L"]), nullable=True),
        "PTS": Column(pa.Float, checks=Check.greater_than_or_equal_to(40), nullable=False),
        "FGM": Column(pa.Float, checks=Check.greater_than_or_equal_to(10), nullable=False),
        "FGA": Column(pa.Float, checks=Check.greater_than_or_equal_to(20), nullable=False),
        "PLUS_MINUS": Column(pa.Float, nullable=False),
    },
    coerce=True,
    strict=False,
)


# ======================================================
# 3. Final ML-Ready Matchups Schema
# ======================================================

ml_ready_matchups_schema = DataFrameSchema(
    columns={
        "HOME_GAME_ID": Column(pa.String, nullable=False),
        "HOME_SEASON_ID": Column(pa.String, nullable=False),
        "HOME_TEAM_ID": Column(pa.String, nullable=False),
        "AWAY_TEAM_ID": Column(pa.String, nullable=False),
        "HOME_WIN": Column(pa.Int, checks=Check.isin([0, 1]), nullable=False),
        "REST_ADVANTAGE": Column(pa.Float, nullable=False),
        "DELTA_ELO": Column(pa.Float, nullable=False),
        "DELTA_ROLLING_OFF_RATING": Column(pa.Float, nullable=False),
    },
    coerce=True,
    strict=False,
)


# ======================================================
# Validation Helper Functions
# ======================================================

def validate_dataframe(df: pd.DataFrame, schema: DataFrameSchema, name: str = "Dataset") -> pd.DataFrame:
    """Validates a DataFrame against a Pandera schema and raises descriptive errors."""
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        failure_cases = err.failure_cases
        raise ValueError(
            f"Schema validation failed for {name} with {len(failure_cases)} violations:\n"
            f"{failure_cases.head(10)}"
        ) from err
