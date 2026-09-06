"""
Tests for the Pandera data validation schemas.
"""

import pandas as pd
import pytest
from data.schemas import (
    raw_historical_team_schema,
    raw_player_game_logs_schema,
    ml_ready_matchups_schema,
    validate_dataframe,
)


def test_raw_historical_team_schema_valid():
    valid_df = pd.DataFrame({
        "SEASON_ID": ["22023"],
        "TEAM_ID": ["1610612747"],
        "TEAM_ABBREVIATION": ["LAL"],
        "GAME_ID": ["0022300001"],
        "GAME_DATE": pd.to_datetime(["2023-10-24"]),
        "MATCHUP": ["LAL vs. DEN"],
        "WL": ["L"],
        "PTS": [107.0],
        "FGM": [41.0],
        "FGA": [90.0],
        "PLUS_MINUS": [-12.0],
    })
    validated = validate_dataframe(valid_df, raw_historical_team_schema, "Team Box")
    assert len(validated) == 1


def test_raw_historical_team_schema_invalid_points():
    invalid_df = pd.DataFrame({
        "SEASON_ID": ["22023"],
        "TEAM_ID": ["1610612747"],
        "TEAM_ABBREVIATION": ["LAL"],
        "GAME_ID": ["0022300001"],
        "GAME_DATE": pd.to_datetime(["2023-10-24"]),
        "MATCHUP": ["LAL vs. DEN"],
        "WL": ["L"],
        "PTS": [15.0],  # Impossibly low points for an NBA game (< 40)
        "FGM": [41.0],
        "FGA": [90.0],
        "PLUS_MINUS": [-12.0],
    })
    with pytest.raises(ValueError, match="Schema validation failed"):
        validate_dataframe(invalid_df, raw_historical_team_schema, "Team Box")


def test_ml_ready_matchup_schema_valid():
    valid_df = pd.DataFrame({
        "HOME_GAME_ID": ["0022300001"],
        "HOME_SEASON_ID": ["22023"],
        "HOME_TEAM_ID": ["1610612743"],
        "AWAY_TEAM_ID": ["1610612747"],
        "HOME_WIN": [1],
        "REST_ADVANTAGE": [2.0],
        "DELTA_ELO": [45.0],
        "DELTA_ROLLING_OFF_RATING": [3.5],
    })
    validated = validate_dataframe(valid_df, ml_ready_matchups_schema, "ML Matchups")
    assert len(validated) == 1


def test_ml_ready_matchup_schema_invalid_target():
    invalid_df = pd.DataFrame({
        "HOME_GAME_ID": ["0022300001"],
        "HOME_SEASON_ID": ["22023"],
        "HOME_TEAM_ID": ["1610612743"],
        "AWAY_TEAM_ID": ["1610612747"],
        "HOME_WIN": [2],  # Target must be 0 or 1
        "REST_ADVANTAGE": [2.0],
        "DELTA_ELO": [45.0],
        "DELTA_ROLLING_OFF_RATING": [3.5],
    })
    with pytest.raises(ValueError, match="Schema validation failed"):
        validate_dataframe(invalid_df, ml_ready_matchups_schema, "ML Matchups")


def test_ml_ready_matchup_schema_four_factors_and_altitude_contracts():
    """Verifies schema enforcement for Four Factors, Altitude, Spacing and Playmaker metrics."""
    base_valid = {
        "HOME_GAME_ID": ["0022300001"],
        "HOME_SEASON_ID": ["22023"],
        "HOME_TEAM_ID": ["1610612743"],
        "AWAY_TEAM_ID": ["1610612747"],
        "HOME_WIN": [1],
        "REST_ADVANTAGE": [2.0],
        "DELTA_ELO": [45.0],
        "DELTA_ROLLING_OFF_RATING": [3.5],
        "HOME_FOUR_FACTOR_EFG_ROLLING_8": [0.54],
        "AWAY_FOUR_FACTOR_EFG_ROLLING_8": [0.51],
        "HOME_FOUR_FACTOR_TOV_ROLLING_8": [0.13],
        "AWAY_FOUR_FACTOR_TOV_ROLLING_8": [0.14],
        "HOME_ALTITUDE": [5280.0],
        "AWAY_ALTITUDE": [10.0],
        "ALTITUDE_ADVANTAGE": [0.95],
        "HOME_SPACING_GRAVITY_INDEX": [12.5],
        "HOME_PLAYMAKER_CONCENTRATION_RATIO": [0.38],
    }

    # 1. Valid data passes
    df_valid = pd.DataFrame(base_valid)
    validated = validate_dataframe(df_valid, ml_ready_matchups_schema, "ML Matchups")
    assert len(validated) == 1

    # 2. Invalid Four Factor eFG (> 1.0) fails
    df_invalid_efg = pd.DataFrame({**base_valid, "HOME_FOUR_FACTOR_EFG_ROLLING_8": [1.25]})
    with pytest.raises(ValueError, match="Schema validation failed"):
        validate_dataframe(df_invalid_efg, ml_ready_matchups_schema, "ML Matchups")

    # 3. Invalid negative altitude fails
    df_invalid_alt = pd.DataFrame({**base_valid, "HOME_ALTITUDE": [-50.0]})
    with pytest.raises(ValueError, match="Schema validation failed"):
        validate_dataframe(df_invalid_alt, ml_ready_matchups_schema, "ML Matchups")

    # 4. Invalid playmaker concentration (> 1.0) fails
    df_invalid_pm = pd.DataFrame({**base_valid, "HOME_PLAYMAKER_CONCENTRATION_RATIO": [1.4]})
    with pytest.raises(ValueError, match="Schema validation failed"):
        validate_dataframe(df_invalid_pm, ml_ready_matchups_schema, "ML Matchups")

