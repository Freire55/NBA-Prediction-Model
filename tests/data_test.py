"""
Tests for the data loading and preparation module.

Validates chronological integrity to ensure that the train,
validation, and test splits do not overlap, effectively
preventing target leakage in the time-series forecasting.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from training.config import TrainingConfig
from training.data import load_and_prep_data

# ======================================================
# Test Fixtures
# ======================================================

@pytest.fixture
def mock_matchups_data() -> pd.DataFrame:
    """Provides a minimal dummy dataframe spanning multiple seasons."""
    return pd.DataFrame(
        {
            "HOME_SEASON_ID": ["22017", "22018", "22019", "22020", "22021", "22022"],
            "HOME_WIN": [1, 0, 1, 0, 1, 1],
            "DELTA_PTS": [1, -5, 10, 2, 4, 6],
            "REST_ADVANTAGE": [1, 0, -1, 2, 0, 1],
            "HOME_B2B": [0, 1, 0, 0, 0, 0],
            "AWAY_B2B": [0, 0, 1, 0, 0, 0],
            "SEASON_YEAR": [2017, 2018, 2019, 2020, 2021, 2022],
        }
    )

# ======================================================
# Test Cases
# ======================================================

@patch("training.data.pd.read_csv")
def test_chronological_split(mock_read_csv, mock_matchups_data):
    """
    Verifies that the dataset is split chronologically without overlap
    and that the aggregate counts match the original dataset length.
    """
    mock_read_csv.return_value = mock_matchups_data
    
    config = TrainingConfig(
        train_end="22018", 
        validation_end="22020",
    )
    
    (
        train_X,
        train_y,
        val_X,
        val_y,
        test_X,
        test_y,
        features,
        summary,
    ) = load_and_prep_data(Path("dummy_dir"), config)
    
    # Verify split boundaries using SEASON_YEAR since HOME_SEASON_ID is stripped
    assert train_X["SEASON_YEAR"].max() <= 2018
    assert val_X["SEASON_YEAR"].min() > 2018
    assert val_X["SEASON_YEAR"].max() <= 2020
    assert test_X["SEASON_YEAR"].min() > 2020
    
    # Verify strict non-overlap by checking index intersections
    assert train_X.index.intersection(val_X.index).empty
    assert val_X.index.intersection(test_X.index).empty
    
    # Verify conservation of samples
    total_samples = summary["train_games"] + summary["validation_games"] + summary["test_games"]
    assert total_samples == len(mock_matchups_data)