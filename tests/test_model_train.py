import yaml
import pytest
import numpy as np
import pandas as pd

from pathlib import Path
from pandas import DataFrame
from model_train import (
    load_config, 
    load_preprocessed_data,
    train_model,
    evaluate_model,
    log_artifacts
)

@pytest.fixture
def mock_config_file(tmp_path):
    """
    Creates a temporary YAML config file for the test.
    """
    config_content = {
        "data": {
            "train_path": "data/raw/train",
            "test_path": "data/raw/test"
        },
        "model": {
            "model_path": "artifacts/models"
        }
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as file:
        yaml.dump(config_file, file)
    return str(config_file)

def test_load_config_success(mock_config_file):
    """
    Test if the configuration is loaded successfully.
    """
    config = load_config(mock_config_file)
    
    assert isinstance(config, dict)
    
    assert "data" in config
    assert "model" in config
    
    assert config["data"]["model"] == "data/raw/train"
    
def test_load_config_file_not_found(tmp_path):
    """
    Tests if File Not Found Error is handled properly or not.
    """
    file_does_not_exist = tmp_path / "not_a_file.csv"
    
    assert not file_does_not_exist.exists()
    
    with pytest.raises(FileNotFoundError) as e:
        load_config(file_does_not_exist)
        