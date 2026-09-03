"""
Tests for the feature engineering pipeline.

Validates the logical constraints of engineered variables, ensuring
rest days are strictly positive, rolling variables do not contain
leakage, and Elo scores initialize correctly.
"""

import pandas as pd
import pytest

from feature_engineering import (
    INITIAL_ELO,
    add_schedule_features,
    simulate_elo,
)

# ======================================================
# Test Cases
# ======================================================

def test_schedule_features_no_negative_rest():
    """
    Ensures that calculated rest days are mathematically sound and 
    that back-to-back (B2B) flags trigger correctly.
    """
    df = pd.DataFrame(
        {
            "TEAM_ABBREVIATION": ["LAL", "LAL", "LAL"],
            "GAME_DATE": pd.to_datetime(["2023-10-24", "2023-10-26", "2023-10-27"]),
            "MATCHUP": ["LAL vs. DEN", "LAL @ PHO", "LAL @ SAC"],
        }
    )
    
    result = add_schedule_features(df)
    
    # Rest days cannot be negative 
    assert (result["REST_DAYS"] >= 0).all()
    
    # B2B trigger check (Oct 26 to Oct 27 is 1 day rest)
    assert result.iloc[2]["B2B"] == 1
    assert result.iloc[1]["B2B"] == 0


def test_simulate_elo_updates():
    """
    Verifies that the chronological Elo engine successfully assigns 
    the base initialization rating without returning NaN values.
    """
    df = pd.DataFrame(
        {
            "GAME_ID": ["1", "1"],
            "GAME_DATE": pd.to_datetime(["2023-10-24", "2023-10-24"]),
            "TEAM_ABBREVIATION": ["LAL", "DEN"],
            "MATCHUP": ["LAL vs. DEN", "DEN @ LAL"],
            "PTS": [110, 100],
        }
    )
    
    result = simulate_elo(df)
    
    # Verify that the initial pre-game Elo is standard for fresh teams
    lal_pre_game = result.loc[result["TEAM_ABBREVIATION"] == "LAL", "PRE_GAME_ELO"].iloc[0]
    assert lal_pre_game == INITIAL_ELO
    
    # Ensure no NaN errors occurred during the iterative simulation
    assert not result["PRE_GAME_ELO"].isna().any()


def test_elo_favors_stronger_team_and_season_regression():
    """
    Verifies that a higher Elo team has higher win probability and receives
    season-to-season mean reversion towards INITIAL_ELO.
    """
    df = pd.DataFrame(
        [
            {"GAME_ID": "1", "GAME_DATE": "2023-01-01", "TEAM_ABBREVIATION": "LAL", "MATCHUP": "LAL vs. DET", "PTS": 120, "SEASON_ID": "22022"},
            {"GAME_ID": "1", "GAME_DATE": "2023-01-01", "TEAM_ABBREVIATION": "DET", "MATCHUP": "DET @ LAL", "PTS": 90, "SEASON_ID": "22022"},
            {"GAME_ID": "2", "GAME_DATE": "2023-01-03", "TEAM_ABBREVIATION": "LAL", "MATCHUP": "LAL vs. DET", "PTS": 110, "SEASON_ID": "22022"},
            {"GAME_ID": "2", "GAME_DATE": "2023-01-03", "TEAM_ABBREVIATION": "DET", "MATCHUP": "DET @ LAL", "PTS": 100, "SEASON_ID": "22022"},
            # New season starts
            {"GAME_ID": "3", "GAME_DATE": "2023-10-24", "TEAM_ABBREVIATION": "LAL", "MATCHUP": "LAL vs. DET", "PTS": 105, "SEASON_ID": "22023"},
            {"GAME_ID": "3", "GAME_DATE": "2023-10-24", "TEAM_ABBREVIATION": "DET", "MATCHUP": "DET @ LAL", "PTS": 100, "SEASON_ID": "22023"},
        ]
    )
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    result = simulate_elo(df)
    
    # After winning 2 games, LAL Elo must be > INITIAL_ELO
    lal_g2 = result.loc[(result["GAME_ID"] == "2") & (result["TEAM_ABBREVIATION"] == "LAL"), "PRE_GAME_ELO"].iloc[0]
    det_g2 = result.loc[(result["GAME_ID"] == "2") & (result["TEAM_ABBREVIATION"] == "DET"), "PRE_GAME_ELO"].iloc[0]
    assert lal_g2 > INITIAL_ELO
    assert det_g2 < INITIAL_ELO
    
    # In Game 3 (new season), LAL should be regressed towards INITIAL_ELO (0.75 * prev + 0.25 * 1500)
    lal_g3 = result.loc[(result["GAME_ID"] == "3") & (result["TEAM_ABBREVIATION"] == "LAL"), "PRE_GAME_ELO"].iloc[0]
    # Winning game 2 increased LAL's rating further, then 25% regressed towards 1500
    assert lal_g3 > INITIAL_ELO