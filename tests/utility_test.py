"""
Tests for the universal utilities module.

Validates generic functionality such as custom JSON serialization 
handling to ensure pipeline artifacts are reliably saved.
"""

import json
from pathlib import Path

import numpy as np
import pytest

# Importing TrainingConfig guarantees our custom NumpyEncoder patch is applied
from training.config import TrainingConfig 
from training.utils import save_json

# ======================================================
# Test Cases
# ======================================================

def test_json_serialization_handles_numpy_types(tmp_path):
    """
    Verifies that the global JSON encoder patch successfully intercepts 
    and sanitizes numpy scalar and array types during dictionary serialization.
    """
    data_with_numpy = {
        "float_val": np.float64(3.14159),
        "array_val": np.array([1, 2, 3]),
        "int_val": np.int64(42),
    }
    
    file_path = tmp_path / "test_artifact.json"
    
    # This should not raise a TypeError
    save_json(data_with_numpy, file_path)
    
    assert file_path.exists()
    
    # Verify the contents were successfully downcast to standard Python types
    with open(file_path, "r") as f:
        loaded_data = json.load(f)
        
    assert isinstance(loaded_data["float_val"], float)
    assert loaded_data["float_val"] == 3.14159
    assert isinstance(loaded_data["array_val"], list)
    assert loaded_data["array_val"] == [1, 2, 3]
    assert isinstance(loaded_data["int_val"], int)
    assert loaded_data["int_val"] == 42