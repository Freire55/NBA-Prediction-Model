"""
Tests for the feature engineering pipeline.

Validates the logical constraints of engineered variables, ensuring
rest days are strictly positive, rolling variables do not contain
leakage, and Elo scores initialize correctly.
"""

import numpy as np
import pandas as pd
import pytest

from feature_engineering import (
    INITIAL_ELO,
    add_four_factors,
    add_rolling_features,
    add_schedule_features,
    build_matchups,
    calculate_altitude_advantage,
    load_altitude_map,
    simulate_elo,
)
from feature_engineering_players import (
    add_volatility_features,
    calculate_player_four_factors,
    calculate_spacing_gravity,
    calculate_playmaker_concentration,
    calculate_obpm_proxy,
    calculate_dbpm_proxy,
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


def test_player_share_monotonic_hierarchy_and_bounds():
    """
    Verifies that Star, Duo, Trio, and Bench shares satisfy:
    0 <= Star Share <= Duo Share <= Trio Share <= 1.0
    and Trio Share + Bench Share == 1.0.
    """
    import numpy as np

    # Synthetic roster with 5 players
    df = pd.DataFrame({
        "GAME_ID": ["G1"] * 5,
        "TEAM_ID": ["T1"] * 5,
        "PLAYER_ID": ["P1", "P2", "P3", "P4", "P5"],
        "EXPECTED_IMPACT": [35.0, 25.0, 15.0, 8.0, 2.0],
    })

    df["POS_EXPECTED_IMPACT"] = np.maximum(df["EXPECTED_IMPACT"], 0.0)
    df["ROSTER_IMPACT_RANK"] = df.groupby(["GAME_ID", "TEAM_ID"])["POS_EXPECTED_IMPACT"].rank(ascending=False, method="first")

    df["STAR_IMPACT"] = np.where(df["ROSTER_IMPACT_RANK"] == 1, df["POS_EXPECTED_IMPACT"], 0.0)
    df["TOP_2_IMPACT"] = np.where(df["ROSTER_IMPACT_RANK"] <= 2, df["POS_EXPECTED_IMPACT"], 0.0)
    df["TOP_3_IMPACT"] = np.where(df["ROSTER_IMPACT_RANK"] <= 3, df["POS_EXPECTED_IMPACT"], 0.0)
    df["BENCH_IMPACT"] = np.where(df["ROSTER_IMPACT_RANK"] > 3, df["POS_EXPECTED_IMPACT"], 0.0)

    agg = df.groupby(["GAME_ID", "TEAM_ID"]).agg(
        ACTIVE_ROSTER_POS_FORM_SUM=("POS_EXPECTED_IMPACT", "sum"),
        ACTIVE_ROSTER_STAR_IMPACT=("STAR_IMPACT", "sum"),
        ACTIVE_ROSTER_TOP_2_IMPACT=("TOP_2_IMPACT", "sum"),
        ACTIVE_ROSTER_TOP_3_IMPACT=("TOP_3_IMPACT", "sum"),
        ACTIVE_ROSTER_BENCH_IMPACT=("BENCH_IMPACT", "sum"),
    ).reset_index()

    total = agg["ACTIVE_ROSTER_POS_FORM_SUM"]
    star_share = (agg["ACTIVE_ROSTER_STAR_IMPACT"] / total).iloc[0]
    top_2_share = (agg["ACTIVE_ROSTER_TOP_2_IMPACT"] / total).iloc[0]
    top_3_share = (agg["ACTIVE_ROSTER_TOP_3_IMPACT"] / total).iloc[0]
    bench_share = (agg["ACTIVE_ROSTER_BENCH_IMPACT"] / total).iloc[0]

    # Monotonicity: Star <= Top 2 <= Top 3 <= 1.0
    assert 0.0 <= star_share <= top_2_share <= top_3_share <= 1.0
    # Top 3 + Bench = 1.0
    assert np.isclose(top_3_share + bench_share, 1.0)
    # Expected proportions: 35/85, 60/85, 75/85, 10/85
    assert np.isclose(star_share, 35.0 / 85.0)
    assert np.isclose(top_2_share, 60.0 / 85.0)
    assert np.isclose(top_3_share, 75.0 / 85.0)
    assert np.isclose(bench_share, 10.0 / 85.0)


def test_team_four_factors_calculation():
    """Verifies team Four Factors formulas compute correctly and match expected values."""
    df = pd.DataFrame([
        {
            "GAME_ID": "G1",
            "TEAM_ABBREVIATION": "BOS",
            "MATCHUP": "BOS vs. MIA",
            "FGM": 40,
            "FGA": 80,
            "FG3M": 10,
            "FTA": 20,
            "OREB": 10,
            "DREB": 30,
            "TOV": 10,
        },
        {
            "GAME_ID": "G1",
            "TEAM_ABBREVIATION": "MIA",
            "MATCHUP": "MIA @ BOS",
            "FGM": 35,
            "FGA": 75,
            "FG3M": 8,
            "FTA": 15,
            "OREB": 8,
            "DREB": 25,
            "TOV": 12,
        },
    ])
    result = add_four_factors(df)

    # eFG = (40 + 0.5*10) / 80 = 45 / 80 = 0.5625
    bos_efg = result.loc[result["TEAM_ABBREVIATION"] == "BOS", "FOUR_FACTOR_EFG"].iloc[0]
    assert np.isclose(bos_efg, 45.0 / 80.0)

    # TOV% = 10 / (80 + 0.44*20 + 10) = 10 / 98.8
    bos_tov = result.loc[result["TEAM_ABBREVIATION"] == "BOS", "FOUR_FACTOR_TOV"].iloc[0]
    assert np.isclose(bos_tov, 10.0 / (80.0 + 0.44 * 20.0 + 10.0))

    # OREB% = BOS OREB / (BOS OREB + MIA DREB) = 10 / (10 + 25) = 10 / 35
    bos_oreb = result.loc[result["TEAM_ABBREVIATION"] == "BOS", "FOUR_FACTOR_OREB"].iloc[0]
    assert np.isclose(bos_oreb, 10.0 / 35.0)

    # FTR = 20 / 80 = 0.25
    bos_ftr = result.loc[result["TEAM_ABBREVIATION"] == "BOS", "FOUR_FACTOR_FTR"].iloc[0]
    assert np.isclose(bos_ftr, 0.25)


def test_player_four_factors_calculation():
    """Verifies player Four Factors compute per-appearance efficiency correctly."""
    df = pd.DataFrame({
        "MINUTES_NUM": [30.0, 0.0],
        "FGM": [10.0, 0.0],
        "FGA": [20.0, 0.0],
        "FG3M": [4.0, 0.0],
        "FTA": [6.0, 0.0],
        "OREB": [3.0, 0.0],
        "TOV": [2.0, 0.0],
    })
    result = calculate_player_four_factors(df)

    # Player 1: eFG = (10 + 0.5*4) / 20 = 12 / 20 = 0.60
    assert np.isclose(result.iloc[0]["PLAYER_FOUR_FACTOR_EFG"], 0.60)
    # Player 1: TOV% = 2 / (20 + 0.44*6 + 2) = 2 / 24.64
    assert np.isclose(result.iloc[0]["PLAYER_FOUR_FACTOR_TOV"], 2.0 / (20.0 + 0.44 * 6.0 + 2.0))
    # Player 1: OREB rate = 3 / 30 = 0.10
    assert np.isclose(result.iloc[0]["PLAYER_FOUR_FACTOR_OREB"], 3.0 / 30.0)
    # Player 1: FTR = 6 / 20 = 0.30
    assert np.isclose(result.iloc[0]["PLAYER_FOUR_FACTOR_FTR"], 0.30)

    # Player 2 (0 minutes / 0 attempts): defaults without NaNs
    assert not result.isna().any().any()


def test_altitude_database_integrity():
    """Verifies that the team altitudes database covers NBA franchises accurately."""
    alt_map = load_altitude_map()
    assert len(alt_map) >= 30
    assert "DEN" in alt_map
    assert "UTA" in alt_map
    assert "MIA" in alt_map
    assert alt_map["DEN"] == 5280
    assert alt_map["UTA"] == 4226
    assert alt_map["MIA"] == 10
    # All altitudes must be strictly non-negative
    assert all(alt >= 0 for alt in alt_map.values())


def test_altitude_advantage_non_linear_properties():
    """
    Verifies that the altitude advantage formula exhibits the expected
    physiological properties:
    1. Directional asymmetry: Adv(Denver @ Miami) == 0, Adv(Miami @ Denver) > 0.90
    2. Thresholding: Altitude differences below 1000 ft produce 0 advantage.
    3. Acclimation: Denver hosting Utah (1054 ft diff) gets negligible advantage.
    4. Non-linearity: The marginal advantage per foot is not constant.
    """
    home_alt = pd.Series([5280.0, 10.0, 597.0, 4226.0, 5280.0])
    away_alt = pd.Series([10.0, 5280.0, 20.0, 10.0, 4226.0])
    # Matchups:
    # 0: Denver vs Miami (Home Denver, +5270 ft)
    # 1: Miami vs Denver (Home Miami, -5270 ft)
    # 2: Chicago vs Boston (Home Chicago, +577 ft)
    # 3: Utah vs Miami (Home Utah, +4216 ft)
    # 4: Denver vs Utah (Home Denver, +1054 ft)

    adv = calculate_altitude_advantage(home_alt, away_alt)

    # 1. Denver vs Miami has massive advantage (> 0.90)
    assert adv.iloc[0] > 0.90

    # 2. Asymmetry: Miami hosting Denver has zero altitude advantage
    assert adv.iloc[1] == 0.0

    # 3. Threshold: Chicago hosting Boston (577 ft diff) is below 1000 ft threshold -> 0.0
    assert adv.iloc[2] == 0.0

    # 4. Monotonicity: Denver vs Miami > Utah vs Miami
    assert adv.iloc[0] > adv.iloc[3] > 0.75

    # 5. Acclimation: Denver hosting Utah is much lower than Denver hosting Miami
    assert adv.iloc[4] < 0.01
    assert adv.iloc[4] < adv.iloc[0] / 50.0

    # 6. Non-linearity check: compare marginal change (0 -> 1000 ft) vs (3000 -> 4000 ft)
    test_home = pd.Series([1000.0, 2000.0, 4000.0, 5000.0])
    test_away = pd.Series([0.0, 0.0, 0.0, 0.0])
    test_adv = calculate_altitude_advantage(test_home, test_away)
    marginal_low = test_adv.iloc[1] - test_adv.iloc[0]   # from 1000 to 2000 ft
    marginal_high = test_adv.iloc[3] - test_adv.iloc[2]  # from 4000 to 5000 ft
    # Since it is non-linear with threshold, marginal gain at 1000-2000 ft is not equal to high altitudes
    assert not np.isclose(marginal_low, marginal_high)


def test_pace_calculation_and_leak_free_rolling():
    """
    Verifies that Pace is correctly computed at standard NBA scale (~90-105 possessions per 48 min),
    normalizes 240 team minutes properly (avoiding 5x under-estimation), strictly shifts rolling/EWMA
    values by 1 to eliminate target leakage, and drops unshifted single-game PACE and MIN from matchups.
    """
    dates = pd.date_range("2023-01-01", periods=4, freq="2D")
    rows = []
    for i, date in enumerate(dates):
        gid = f"G_{i}"
        # BOS: FGA=85, FTA=20, OREB=10, TOV=14 -> possessions = 85 + 0.44*20 - 10 + 14 = 97.8
        rows.append({
            "GAME_ID": gid, "TEAM_ABBREVIATION": "BOS", "GAME_DATE": date, "MATCHUP": "BOS vs. MIA",
            "PTS": 105.0, "FGM": 40, "FGA": 85.0, "FG3M": 10, "FTA": 20.0, "OREB": 10.0, "DREB": 35.0, "TOV": 14.0,
            "MIN": 240, "Z_PLUS_MINUS": 0.5, "Z_PTS": 0.2, "SEASON_ID": "22022", "SEASON_YEAR": 2022,
        })
        rows.append({
            "GAME_ID": gid, "TEAM_ABBREVIATION": "MIA", "GAME_DATE": date, "MATCHUP": "MIA @ BOS",
            "PTS": 100.0, "FGM": 38, "FGA": 80.0, "FG3M": 9, "FTA": 18.0, "OREB": 8.0, "DREB": 30.0, "TOV": 16.0,
            "MIN": 240, "Z_PLUS_MINUS": -0.5, "Z_PTS": -0.2, "SEASON_ID": "22022", "SEASON_YEAR": 2022,
        })

    df = pd.DataFrame(rows)
    df = add_schedule_features(df)
    df = simulate_elo(df)
    df = add_rolling_features(df)

    # 1. Scale verification: Pace must be ~97.8, NOT ~19.5 (5x error check)
    expected_poss = 85.0 + 0.44 * 20.0 - 10.0 + 14.0
    bos_pace = df.loc[df["TEAM_ABBREVIATION"] == "BOS", "PACE"]
    assert np.isclose(bos_pace.iloc[0], expected_poss)
    assert 90.0 < bos_pace.iloc[0] < 110.0

    # 2. Strict prior shift verification: game 0 has no history (defaults to 100.0)
    bos_rolling = df.loc[df["TEAM_ABBREVIATION"] == "BOS", "ROLLING_PACE_8"].values
    assert np.isclose(bos_rolling[0], 100.0)
    assert np.isclose(bos_rolling[1], expected_poss)

    # 3. Multi-horizon EWMA pace features
    for span in [5, 10]:
        col = f"PACE_EWMA_{span}"
        assert col in df.columns
        assert not df[col].isna().any()
        assert np.isclose(df.loc[df["TEAM_ABBREVIATION"] == "BOS", col].iloc[0], 100.0)

    # 4. Matchup deltas and expected pace
    matchups = build_matchups(df)
    assert "DELTA_ROLLING_PACE" in matchups.columns
    assert "DELTA_ROLLING_PACE_8" in matchups.columns
    assert "MATCHUP_EXPECTED_PACE" in matchups.columns
    for span in [5, 10]:
        assert f"DELTA_PACE_EWMA_{span}" in matchups.columns

    # 5. Target leakage: Single-game unshifted PACE and MIN must be dropped
    assert "HOME_PACE" not in matchups.columns
    assert "AWAY_PACE" not in matchups.columns
    assert "HOME_MIN" not in matchups.columns
    assert "AWAY_MIN" not in matchups.columns


def test_four_factors_abbreviation_mismatch_resilience():
    """
    Verifies that add_four_factors uses robust vectorized GAME_ID pairing to compute
    opponent defensive rebounds, succeeding even when MATCHUP team abbreviations
    differ from TEAM_ABBREVIATION (e.g. historical franchise relocation like NOH vs NO).
    """
    df = pd.DataFrame([
        {
            "GAME_ID": "G_RELOC",
            "TEAM_ABBREVIATION": "NO",
            "MATCHUP": "NO vs. GSW",
            "FGM": 38,
            "FGA": 80,
            "FG3M": 10,
            "FTA": 20,
            "OREB": 10,
            "DREB": 30,
            "TOV": 12,
        },
        {
            "GAME_ID": "G_RELOC",
            "TEAM_ABBREVIATION": "GSW",
            # Mismatch: matchup says NOH (New Orleans Hornets) while row abbreviation is NO
            "MATCHUP": "GSW @ NOH",
            "FGM": 36,
            "FGA": 82,
            "FG3M": 12,
            "FTA": 16,
            "OREB": 8,
            "DREB": 28,
            "TOV": 14,
        },
    ])

    result = add_four_factors(df)

    # For NO: Opponent DREB is GSW's DREB (28)
    # FOUR_FACTOR_OREB = 10 / (10 + 28) = 10 / 38
    no_oreb = result.loc[result["TEAM_ABBREVIATION"] == "NO", "FOUR_FACTOR_OREB"].iloc[0]
    assert np.isclose(no_oreb, 10.0 / 38.0)
    assert not np.isnan(no_oreb)
    assert no_oreb > 0.0

    # For GSW: Opponent DREB is NO's DREB (30)
    # FOUR_FACTOR_OREB = 8 / (8 + 30) = 8 / 38
    gsw_oreb = result.loc[result["TEAM_ABBREVIATION"] == "GSW", "FOUR_FACTOR_OREB"].iloc[0]
    assert np.isclose(gsw_oreb, 8.0 / 38.0)
    assert not np.isnan(gsw_oreb)


def test_add_volatility_features_chronological_purity():
    """
    Verifies that add_volatility_features uses historical domain priors (mean 8.0, std 5.0)
    rather than dataset-wide future averages, ensuring chronological leakage invariance.
    """
    dates = pd.date_range("2023-01-01", periods=5, freq="2D")
    df = pd.DataFrame([
        {
            "PLAYER_ID": "P1",
            "GAME_DATE": date,
            "GAME_SCORE": 10.0 + i * 2,
            "PLAYER_FORM_ROLLING_5": 9.0,
        }
        for i, date in enumerate(dates)
    ])

    result = add_volatility_features(df)
    assert "ROBUST_EXPECTED_IMPACT" in result.columns
    assert "PLAYER_GAME_SCORE_VOLATILITY" in result.columns
    # Prior for first game volatility should equal standard domain prior (5.0)
    assert np.isclose(result["PLAYER_GAME_SCORE_VOLATILITY"].iloc[0], 5.0)
    # No NaNs produced
    assert not result["ROBUST_EXPECTED_IMPACT"].isna().any()


def test_spacing_gravity_calculation_properties():
    """
    Verifies 3-point spacing gravity index:
    1. High 3PA volume and efficiency produces high spacing gravity.
    2. Zero 3PA yields exactly 0.0 spacing gravity.
    3. Monotonicity: higher minutes or higher 3P% increases gravity.
    """
    fg3a = pd.Series([10.0, 0.0, 5.0])
    fg3m = pd.Series([4.0, 0.0, 2.0])  # 40% for shooter 0, 40% for shooter 2
    mins = pd.Series([36.0, 30.0, 18.0])

    gravity = calculate_spacing_gravity(fg3a, fg3m, mins)

    # Shooter 0: 10 * (0.40 / 0.35) * (36 / 36) = 11.428...
    expected_shooter_0 = 10.0 * (0.40 / 0.35) * (36.0 / 36.0)
    assert np.isclose(gravity.iloc[0], expected_shooter_0)

    # Non-shooter: 0.0
    assert gravity.iloc[1] == 0.0

    # Half-minutes, half-attempts shooter: strictly lower than shooter 0
    assert 0.0 < gravity.iloc[2] < gravity.iloc[0]
    assert (gravity >= 0.0).all()


def test_playmaker_concentration_bounds_and_hierarchy():
    """
    Verifies playmaker concentration ratio:
    1. Heliocentric offense (single star with all assists) has ratio = 1.0.
    2. Decentralized offense (4 players with equal assists) has ratio = 0.25.
    3. Handles 0 assists gracefully without division by zero.
    4. Strict bounds [0.0, 1.0].
    """
    max_ast = pd.Series([10.0, 2.5, 0.0])
    sum_ast = pd.Series([10.0, 10.0, 0.0])

    conc = calculate_playmaker_concentration(max_ast, sum_ast)

    # 1. Heliocentric star
    assert conc.iloc[0] == 1.0

    # 2. Balanced 4-player committee
    assert np.isclose(conc.iloc[1], 0.25)

    # 3. Zero assists fallback
    assert conc.iloc[2] == 0.25

    # 4. Strict bounds
    assert ((conc >= 0.0) & (conc <= 1.0)).all()


def test_bpm_proxy_calculation_and_directions():
    """
    Verifies 2-Way Box Plus-Minus (OBPM and DBPM) proxies:
    1. High scoring, playmaking, offensive rebounding yield positive OBPM.
    2. High steals, blocks, defensive rebounding yield positive DBPM.
    3. Turnovers penalize OBPM; fouls penalize DBPM.
    """
    df = pd.DataFrame([
        {
            # Elite offensive creator
            "PTS": 30.0, "AST": 10.0, "OREB": 2.0, "TOV": 2.0,
            "FGA": 20.0, "FTA": 6.0, "STL": 1.0, "BLK": 0.0,
            "DREB": 4.0, "PF": 2.0,
        },
        {
            # Elite defensive anchor
            "PTS": 6.0, "AST": 1.0, "OREB": 1.0, "TOV": 1.0,
            "FGA": 5.0, "FTA": 2.0, "STL": 3.0, "BLK": 4.0,
            "DREB": 12.0, "PF": 2.0,
        },
    ])

    obpm = calculate_obpm_proxy(df)
    dbpm = calculate_dbpm_proxy(df)

    # Offensive creator has higher OBPM than defensive anchor
    assert obpm.iloc[0] > obpm.iloc[1]
    assert obpm.iloc[0] > 100.0

    # Defensive anchor has higher DBPM than offensive creator
    assert dbpm.iloc[1] > dbpm.iloc[0]
    assert dbpm.iloc[1] > 100.0


def test_opponent_3pt_variance_neutralization():
    """
    Verifies opponent 3PT variance neutralization:
    1. Regresses high/low opponent 3PT% 50% toward the league average (36.0%).
    2. Enforces strict chronological lag (game 0 pre-game stats use prior/neutral defaults).
    3. Correctly calculates volume attempt rate (OPP_3PA_RATE).
    """
    dates = pd.date_range("2023-01-01", periods=3, freq="2D")
    rows = []
    # G0: LAL opponent (GSW) shoots 15/30 from three (50.0%)
    rows.append({
        "GAME_ID": "G0", "TEAM_ABBREVIATION": "LAL", "GAME_DATE": dates[0], "MATCHUP": "LAL vs. GSW",
        "PTS": 110.0, "FGM": 40.0, "FGA": 85.0, "FG3M": 10.0, "FG3A": 25.0, "FTA": 20.0, "OREB": 10.0, "TOV": 12.0,
    })
    rows.append({
        "GAME_ID": "G0", "TEAM_ABBREVIATION": "GSW", "GAME_DATE": dates[0], "MATCHUP": "GSW @ LAL",
        "PTS": 115.0, "FGM": 42.0, "FGA": 85.0, "FG3M": 15.0, "FG3A": 30.0, "FTA": 16.0, "OREB": 8.0, "TOV": 14.0,
    })
    # G1: LAL opponent (GSW) shoots 6/30 from three (20.0%)
    rows.append({
        "GAME_ID": "G1", "TEAM_ABBREVIATION": "LAL", "GAME_DATE": dates[1], "MATCHUP": "LAL vs. GSW",
        "PTS": 105.0, "FGM": 39.0, "FGA": 85.0, "FG3M": 8.0, "FG3A": 24.0, "FTA": 19.0, "OREB": 9.0, "TOV": 11.0,
    })
    rows.append({
        "GAME_ID": "G1", "TEAM_ABBREVIATION": "GSW", "GAME_DATE": dates[1], "MATCHUP": "GSW @ LAL",
        "PTS": 95.0, "FGM": 35.0, "FGA": 85.0, "FG3M": 6.0, "FG3A": 30.0, "FTA": 19.0, "OREB": 7.0, "TOV": 15.0,
    })
    # G2
    rows.append({
        "GAME_ID": "G2", "TEAM_ABBREVIATION": "LAL", "GAME_DATE": dates[2], "MATCHUP": "LAL vs. GSW",
        "PTS": 100.0, "FGM": 38.0, "FGA": 85.0, "FG3M": 9.0, "FG3A": 25.0, "FTA": 15.0, "OREB": 10.0, "TOV": 10.0,
    })
    rows.append({
        "GAME_ID": "G2", "TEAM_ABBREVIATION": "GSW", "GAME_DATE": dates[2], "MATCHUP": "GSW @ LAL",
        "PTS": 100.0, "FGM": 38.0, "FGA": 85.0, "FG3M": 9.0, "FG3A": 25.0, "FTA": 15.0, "OREB": 10.0, "TOV": 10.0,
    })

    df = pd.DataFrame(rows)
    df = add_rolling_features(df)

    lal_games = df[df["TEAM_ABBREVIATION"] == "LAL"].sort_values("GAME_DATE").reset_index(drop=True)

    # In Game 1, LAL's prior opponent was GSW in G0 who shot 15/30 = 50.0%
    # Neutralized D_3PT_TRUE = 0.5 * 0.50 + 0.5 * 0.36 = 0.430
    assert lal_games.loc[1, "D_3PT_ACTUAL"] == pytest.approx(0.50, abs=1e-3)
    assert lal_games.loc[1, "D_3PT_TRUE"] == pytest.approx(0.430, abs=1e-3)

    # In Game 2, prior games had 15+6 = 21 makes on 30+30 = 60 attempts = 35.0%
    # Neutralized D_3PT_TRUE = 0.5 * 0.35 + 0.5 * 0.36 = 0.355
    assert lal_games.loc[2, "D_3PT_ACTUAL"] == pytest.approx(0.350, abs=1e-3)
    assert lal_games.loc[2, "D_3PT_TRUE"] == pytest.approx(0.355, abs=1e-3)


def test_circadian_fatigue_and_travel_index():
    """
    Verifies the Circadian Fatigue Index:
    1. Distance calculation for coast-to-coast road trip (MIA to POR > 2,500 miles).
    2. Eastward jet lag penalty when traveling West-to-East (losing hours).
    3. Back-to-back and schedule density compounding fatigue.
    """
    # G0: LAL at home vs GSW (Jan 1)
    # G1: LAL travels to MIA on back-to-back (Jan 2: West to East, crossing 3 time zones)
    df = pd.DataFrame([
        {
            "GAME_ID": "G0", "TEAM_ABBREVIATION": "LAL", "GAME_DATE": pd.to_datetime("2023-01-01"),
            "MATCHUP": "LAL vs. GSW", "PTS": 110.0,
        },
        {
            "GAME_ID": "G1", "TEAM_ABBREVIATION": "LAL", "GAME_DATE": pd.to_datetime("2023-01-02"),
            "MATCHUP": "LAL @ MIA", "PTS": 105.0,
        },
    ])

    result = add_schedule_features(df)
    lal = result[result["TEAM_ABBREVIATION"] == "LAL"].sort_values("GAME_DATE").reset_index(drop=True)

    # In Game 0, LAL is at home (0 travel)
    assert lal.loc[0, "TRAVEL_7D"] == 0.0
    assert lal.loc[0, "TZ_EASTWARD_LOSS"] == 0.0

    # In Game 1, LAL travels from LA to Miami (> 2,300 miles) on B2B
    assert lal.loc[1, "TRAVEL_7D"] > 2300.0
    assert lal.loc[1, "B2B"] == 1
    # West (-8) to East (-5) loses 3 hours
    assert lal.loc[1, "TZ_EASTWARD_LOSS"] == pytest.approx(3.0)
    # Circadian fatigue index should be substantially elevated (> 4.0)
    assert lal.loc[1, "CIRCADIAN_FATIGUE_INDEX"] > 4.0


def test_tactical_clash_matrix_interactions():
    """
    Verifies the non-transitive tactical clash matrix:
    1. Turnover pressure exploits turnover vulnerability.
    2. Free throw pressure exploits high defensive foul rates.
    3. 3PT shooting exploits open perimeter concessions.
    4. Composite tactical clash advantage reflects directional stylistic edge.
    """
    from feature_engineering import add_tactical_clash_matrix

    matchup_df = pd.DataFrame([{
        "HOME_DEF_TOV_RATE": 0.18,
        "AWAY_DEF_TOV_RATE": 0.11,
        "HOME_FOUR_FACTOR_TOV_ROLLING_8": 0.11,
        "AWAY_FOUR_FACTOR_TOV_ROLLING_8": 0.18,
        "HOME_FOUR_FACTOR_OREB_ROLLING_8": 0.32,
        "AWAY_FOUR_FACTOR_OREB_ROLLING_8": 0.20,
        "HOME_ROLLING_PACE_8": 96.0,
        "AWAY_ROLLING_PACE_8": 104.0,
        "HOME_FOUR_FACTOR_FTR_ROLLING_8": 0.28,
        "AWAY_FOUR_FACTOR_FTR_ROLLING_8": 0.18,
        "HOME_DEF_FTR": 0.18,
        "AWAY_DEF_FTR": 0.28,
        "HOME_OPP_3PA_RATE": 0.30,
        "AWAY_OPP_3PA_RATE": 0.42,
        "HOME_FOUR_FACTOR_EFG_ROLLING_8": 0.56,
        "AWAY_FOUR_FACTOR_EFG_ROLLING_8": 0.50,
        "HOME_DEF_REB_RATE": 0.78,
        "AWAY_DEF_REB_RATE": 0.70,
    }])

    result = add_tactical_clash_matrix(matchup_df)

    # Home forces 18% TOV on an 18% TOV prone opponent (0.18 * 0.18 = 0.0324)
    # Away forces 11% TOV on an 11% secure opponent (0.11 * 0.11 = 0.0121)
    assert result.loc[0, "HOME_TURNOVER_PRESSURE_CLASH"] > result.loc[0, "AWAY_TURNOVER_PRESSURE_CLASH"]
    assert result.loc[0, "DELTA_TURNOVER_PRESSURE_CLASH"] > 0.015

    # Home draws fouls against high-fouling Away defense
    assert result.loc[0, "DELTA_FTR_CLASH"] > 0.0

    # Home elite shooting exploits Away high 3PA allowance
    assert result.loc[0, "DELTA_3PT_EXPLOITATION"] > 0.0

    # Composite tactical clash advantage is strongly positive for Home
    assert result.loc[0, "DELTA_TACTICAL_CLASH_ADVANTAGE"] > 0.0