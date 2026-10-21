"""
Unit tests for the production prediction module (modeling/predict_today.py).
"""

from pathlib import Path
import pytest
import numpy as np

from modeling.predict_today import (
    find_latest_model_dir,
    LoadedModelArtifacts,
    extract_and_cache_team_profiles,
    build_matchup_features,
    predict_single_matchup,
)


@pytest.fixture(scope="module")
def artifacts():
    run_dir = find_latest_model_dir()
    return LoadedModelArtifacts(run_dir)


@pytest.fixture(scope="module")
def profiles():
    return extract_and_cache_team_profiles()


def test_model_artifacts_loading(artifacts):
    assert artifacts.mlp_model is not None
    assert artifacts.xgb_model is not None
    assert artifacts.cb_model is not None
    assert artifacts.margin_model is not None
    assert len(artifacts.weights) == 4
    assert np.isclose(np.sum(artifacts.weights), 1.0)
    assert len(artifacts.all_required_features) == 394


def test_team_profiles_content(profiles):
    assert "team_raw_stats" in profiles
    assert "team_roster" in profiles
    assert "alt_map" in profiles
    assert len(profiles["team_raw_stats"]) >= 30
    assert "BOS" in profiles["team_raw_stats"]
    assert "LAL" in profiles["team_raw_stats"]


def test_build_matchup_features(artifacts, profiles):
    feat_df, expected_pace = build_matchup_features(
        home_team="LAL",
        away_team="GSW",
        profiles=profiles,
        all_required_features=artifacts.all_required_features,
    )
    assert feat_df.shape[0] == 1
    assert feat_df.shape[1] == 394
    assert 85.0 <= expected_pace <= 115.0
    assert not feat_df.isna().any().any()


def test_predict_single_matchup(artifacts, profiles):
    pred = predict_single_matchup(
        home_team="LAL",
        away_team="GSW",
        artifacts=artifacts,
        profiles=profiles,
    )
    assert pred["projected_winner"] in ["LAL", "GSW"]
    assert 0.0 <= pred["win_probability"] <= 1.0
    assert 0.0 <= pred["ensemble_home_prob"] <= 1.0
    assert pred["confidence_tier"] in ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]
    assert -30.0 <= pred["projected_home_margin"] <= 30.0
    assert "submodel_breakdown" in pred
    assert len(pred["submodel_breakdown"]) == 5
