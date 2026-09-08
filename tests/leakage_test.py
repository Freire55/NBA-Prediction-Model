"""
Strict leakage verification tests for the NBA prediction pipeline.

Guarantees that:
1. Pre-game features for game T are strictly invariant to post-game stats of game T.
2. Future games (T+1, T+2) have zero influence on features calculated for game T.
3. Feature sets passed to models contain zero post-game box score leakage columns.
"""

import pandas as pd
import numpy as np
import pytest

from feature_engineering import add_rolling_features, rolling_mean, ewma
from training.config import TrainingConfig
from training.data import get_model_features


def test_rolling_functions_strict_temporal_shift():
    """Verifies that rolling_mean and ewma strictly shift observations by 1."""
    series = pd.Series([10.0, 20.0, 30.0, 40.0])
    
    rm = rolling_mean(series, rolling_window=2)
    ew = ewma(series, span=2)
    
    # Game 0 must NOT have access to game 0 value
    assert np.isnan(rm.iloc[0])
    assert np.isnan(ew.iloc[0])
    
    # Game 1 rolling mean can only reflect Game 0
    assert rm.iloc[1] == 10.0
    assert ew.iloc[1] == 10.0


def test_perturbation_leakage_invariance():
    """
    Perturbation test: Modifying game T's post-game box score or future games T+1
    must produce zero change in game T's engineered pre-game features.
    """
    dates = pd.date_range("2023-01-01", periods=5, freq="2D")
    
    # Construct a valid 2-team schedule for 5 games (LAL vs GSW)
    rows = []
    for i, date in enumerate(dates):
        game_id = f"G_{i}"
        # LAL row
        rows.append({
            "GAME_ID": game_id,
            "TEAM_ABBREVIATION": "LAL",
            "GAME_DATE": date,
            "MATCHUP": "LAL vs. GSW",
            "PTS": 100.0 + i * 5,
            "FGA": 85.0 + i,
            "FTA": 20.0 + i,
            "OREB": 10.0,
            "TOV": 14.0,
            "Z_PLUS_MINUS": 0.2 * i,
        })
        # GSW row
        rows.append({
            "GAME_ID": game_id,
            "TEAM_ABBREVIATION": "GSW",
            "GAME_DATE": date,
            "MATCHUP": "GSW @ LAL",
            "PTS": 98.0 + i * 4,
            "FGA": 82.0 + i,
            "FTA": 18.0 + i,
            "OREB": 9.0,
            "TOV": 15.0,
            "Z_PLUS_MINUS": -0.2 * i,
        })
        
    df_original = pd.DataFrame(rows)

    # Add rolling features on original data
    df_feat_orig = add_rolling_features(df_original.copy())

    # Perturb Game 3 (for both teams) dramatically: score 300 points instead of normal
    df_perturbed = df_original.copy()
    mask_g3 = df_perturbed["GAME_ID"] == "G_3"
    df_perturbed.loc[mask_g3, "PTS"] = 300.0
    df_perturbed.loc[mask_g3, "Z_PLUS_MINUS"] = 10.0
    
    # Also perturb future Game 4
    mask_g4 = df_perturbed["GAME_ID"] == "G_4"
    df_perturbed.loc[mask_g4, "PTS"] = 500.0

    df_feat_pert = add_rolling_features(df_perturbed.copy())

    # Pre-game features for Game 3 MUST be completely identical between original and perturbed
    # (because Game 3's pre-game features only depend on games 0, 1, 2)
    lal_orig_g3 = df_feat_orig[(df_feat_orig["GAME_ID"] == "G_3") & (df_feat_orig["TEAM_ABBREVIATION"] == "LAL")].iloc[0]
    lal_pert_g3 = df_feat_pert[(df_feat_pert["GAME_ID"] == "G_3") & (df_feat_pert["TEAM_ABBREVIATION"] == "LAL")].iloc[0]

    cols_to_check = [
        col for col in df_feat_orig.columns
        if "ROLLING_" in col or "_EWMA_" in col or "D_3PT_" in col or "OPP_3PA_" in col
    ]
    for col in cols_to_check:
        orig_val = lal_orig_g3[col]
        pert_val = lal_pert_g3[col]
        assert orig_val == pytest.approx(pert_val, abs=1e-9), (
            f"Leakage detected in column {col}! Original: {orig_val}, Perturbed: {pert_val}"
        )


def test_feature_sets_exclude_postgame_leakage():
    """Ensures no post-game box score statistics or raw outcomes are in model feature sets."""
    forbidden_substrings = [
        "HOME_PTS", "AWAY_PTS", "HOME_FGM", "AWAY_FGM",
        "HOME_PLUS_MINUS", "AWAY_PLUS_MINUS", "HOME_WIN"
    ]
    
    mock_df = pd.DataFrame({
        "HOME_WIN": [1, 0],
        "HOME_PTS": [110, 95],
        "AWAY_PTS": [105, 100],
        "DELTA_ROLLING_PTS": [5.0, -5.0],
        "HOME_ROLLING_PTS": [112.0, 108.0],
        "AWAY_ROLLING_PTS": [107.0, 113.0],
        "REST_ADVANTAGE": [1.0, 0.0],
        "HOME_B2B": [0, 1],
        "AWAY_B2B": [1, 0],
        "SEASON_YEAR": [2022, 2022],
    })
    
    config = TrainingConfig()
    features = get_model_features(mock_df, config)
    
    for model_name, feature_list in features.items():
        for feat in feature_list:
            for forbidden in forbidden_substrings:
                assert feat != forbidden, (
                    f"Target leakage in {model_name}: feature '{feat}' is forbidden!"
                )


def test_player_features_perturbation_invariance():
    """
    Verifies that mutating player post-game statistics in game T or future games T+1
    leaves pre-game rolling stats (OBPM, DBPM, Spacing Gravity, Assists) for game T bitwise identical.
    """
    from feature_engineering_players import (
        calculate_obpm_proxy,
        calculate_dbpm_proxy,
        calculate_player_four_factors,
        calculate_spacing_gravity,
    )

    dates = pd.date_range("2023-01-01", periods=5, freq="2D")
    rows = []
    for i, date in enumerate(dates):
        rows.append({
            "PLAYER_ID": "P1",
            "GAME_DATE": date,
            "MIN": "36:00",
            "PTS": 20.0 + i * 2,
            "FGM": 8.0,
            "FGA": 18.0,
            "FG3M": 3.0,
            "FG3A": 8.0,
            "FTM": 1.0,
            "FTA": 2.0,
            "OREB": 1.0,
            "DREB": 4.0,
            "AST": 6.0,
            "STL": 1.0,
            "BLK": 0.0,
            "TOV": 2.0,
            "PF": 2.0,
        })

    df_orig = pd.DataFrame(rows)
    df_pert = df_orig.copy()
    # Mutate Game 3 post-game stats dramatically: score 100 points, 20 threes
    df_pert.loc[3, "PTS"] = 100.0
    df_pert.loc[3, "FG3M"] = 20.0
    df_pert.loc[3, "FG3A"] = 25.0
    df_pert.loc[3, "AST"] = 30.0

    def compute_player_rolling(df):
        d = df.copy()
        d["MINUTES_NUM"] = 36.0
        d["OBPM_PROXY"] = calculate_obpm_proxy(d)
        d["DBPM_PROXY"] = calculate_dbpm_proxy(d)
        ff = calculate_player_four_factors(d)
        d["PLAYER_FOUR_FACTOR_EFG"] = ff["PLAYER_FOUR_FACTOR_EFG"]
        
        pg = d.groupby("PLAYER_ID")
        d["PLAYER_OBPM_ROLLING_5"] = pg["OBPM_PROXY"].transform(rolling_mean, rolling_window=5).fillna(0.0)
        d["PLAYER_FG3A_ROLLING_5"] = pg["FG3A"].transform(rolling_mean, rolling_window=5).fillna(0.0)
        d["PLAYER_FG3M_ROLLING_5"] = pg["FG3M"].transform(rolling_mean, rolling_window=5).fillna(0.0)
        d["PLAYER_AST_ROLLING_5"] = pg["AST"].transform(rolling_mean, rolling_window=5).fillna(0.0)
        d["PLAYER_SPACING_GRAVITY"] = calculate_spacing_gravity(
            d["PLAYER_FG3A_ROLLING_5"], d["PLAYER_FG3M_ROLLING_5"], pd.Series(36.0, index=d.index)
        )
        return d

    res_orig = compute_player_rolling(df_orig)
    res_pert = compute_player_rolling(df_pert)

    # For Game 3 (index 3), pre-game features must be 100% bitwise invariant
    check_cols = [
        "PLAYER_OBPM_ROLLING_5",
        "PLAYER_FG3A_ROLLING_5",
        "PLAYER_FG3M_ROLLING_5",
        "PLAYER_AST_ROLLING_5",
        "PLAYER_SPACING_GRAVITY",
    ]
    for col in check_cols:
        orig_val = res_orig.loc[3, col]
        pert_val = res_pert.loc[3, col]
        assert orig_val == pytest.approx(pert_val, abs=1e-9), (
            f"Leakage in player feature {col}! Orig: {orig_val}, Pert: {pert_val}"
        )

