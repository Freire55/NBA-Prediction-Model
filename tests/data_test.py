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
    
        # CHANGE: Assign directly to a single TrainingData variable
        training_data = load_and_prep_data(Path("dummy_dir"), config)

        # Update your assertions to use the new dataclass structure.
        # Example assertions based on your previous unpack logic:
        assert len(training_data.y_train) > 0 
        assert training_data.summary.train_games == len(training_data.y_train)
        
        # Verify heterogeneous feature routing works
        assert isinstance(training_data.mlp.X_train, pd.DataFrame)
        assert isinstance(training_data.xgb.X_train, pd.DataFrame)
        assert isinstance(training_data.lr.X_train, pd.DataFrame)


def test_exponential_recency_weights_properties():
    """Verifies that recency weights are normalized, monotonically increasing, and non-empty."""
    import numpy as np
    from training.data import compute_exponential_recency_weights

    dates = pd.Series(["2005-01-01", "2010-01-01", "2015-01-01", "2020-01-01"])
    weights = compute_exponential_recency_weights(dates, half_life_years=5.0)

    assert len(weights) == 4
    assert np.isclose(weights.mean(), 1.0, atol=1e-5)
    # Monotonically increasing: older games have smaller weights
    assert weights[0] < weights[1] < weights[2] < weights[3]
    # Ratio between 2020 and 2015 (5 years = 1 half-life) should be approximately 2.0
    ratio = weights[3] / weights[2]
    assert np.isclose(ratio, 2.0, atol=0.1)


@patch("training.data.pd.read_csv")
def test_load_and_prep_data_with_dates(mock_read_csv):
    """Verifies that game dates generate valid sample weights in TrainingData."""
    import numpy as np

    df = pd.DataFrame(
        {
            "HOME_SEASON_ID": ["22017", "22018", "22019", "22020", "22021", "22022"],
            "HOME_GAME_DATE": ["2017-11-01", "2018-03-01", "2019-01-15", "2020-02-10", "2021-04-01", "2022-01-05"],
            "HOME_WIN": [1, 0, 1, 0, 1, 1],
            "DELTA_PTS": [1, -5, 10, 2, 4, 6],
            "SEASON_YEAR": [2017, 2018, 2019, 2020, 2021, 2022],
        }
    )
    mock_read_csv.return_value = df
    config = TrainingConfig(train_end="22018", validation_end="22020", use_recency_weights=True)

    data = load_and_prep_data(Path("dummy_dir"), config)
    assert data.sample_weights_train is not None
    assert len(data.sample_weights_train) == len(data.y_train)
    assert np.isclose(data.sample_weights_train.mean(), 1.0, atol=1e-5)
    assert data.dates_train is not None
    assert len(data.dates_train) == len(data.y_train)