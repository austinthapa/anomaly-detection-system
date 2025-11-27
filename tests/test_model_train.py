import yaml
import pytest
import numpy as np
import pandas as pd

from pathlib import Path
from pandas import DataFrame
from sklearn.ensemble import IsolationForest
from unittest.mock import MagicMock
from model_train import (
    load_config, 
    load_preprocessed_data,
    train_model,
    evaluate_model,
    log_artifacts
)

@pytest.fixture
def mock_config_file(tmp_path: Path):
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
        yaml.dump(config_content, file, default_flow_style=False)
    return str(config_file)

@pytest.fixture
def mock_processed_data(tmp_path: Path):
    """
    Creates a temporary CSV Loaded file.
    """
    processed_df = pd.DataFrame({
        "feat_1": range(10),
        "feat_2": range(10)
    })
    mock_file = tmp_path / "processed_data.csv"
    processed_df.to_csv(mock_file, index=False)
    return str(mock_file)

@pytest.fixture
def mock_empty_data(tmp_path: Path):
    """
    Creates an empty dataframe to test exceptions.
    """
    empty_df = pd.DataFrame({
        "feat_1": [],
        "feat_2": []
    })
    
    mock_file = tmp_path / "empty_data.csv"
    empty_df.to_csv(mock_file, index=False)
    return str(mock_file)

@pytest.fixture
def mock_model():
    """
    Create a mock model with predict
    """
    model = MagicMock()
    
    model.decision_function.return_value = np.array([-1, 1] * 5)
    return model 

@pytest.fixture
def mock_model_params():
    """
    Create a temporary model parameters
    """
    return {
       "n_estimators": 100,
       "random_state": 42
    }
    
def test_load_config_success(mock_config_file):
    """
    Test if the configuration is loaded successfully.
    """
    config = load_config(mock_config_file)
    
    assert isinstance(config, dict)
    
    assert "data" in config
    assert "model" in config
    
def test_load_config_file_not_found(tmp_path):
    """
    Tests if File Not Found Error is handled properly or not.
    """
    file_does_not_exist = tmp_path / "not_a_file.csv"
    
    assert not file_does_not_exist.exists()
    
    with pytest.raises(FileNotFoundError) as e:
        load_config(file_does_not_exist)

def test_load_preprocessed_data(mock_processed_data):
    """
    Tests if load_preprocessed_data successfully loads the data or not
    """
    result_df = load_preprocessed_data(mock_processed_data)
    
    assert isinstance(result_df, DataFrame)
    assert len(result_df.columns) == 2
    assert len(result_df) == 10

def test_load_preprocessed_data_empty_data(mock_empty_data):
    """
    Test if load_preprocessed_data successfully raises error while reading empty DataFrame.
    """
    assert Path(mock_empty_data).exists()
    
    with pytest.raises(ValueError) as e:
        load_preprocessed_data(mock_empty_data)
        
def test_load_preprocessed_data_not_file_not_found(tmp_path):
    """
    Test if it raises File Not Found Exception if the file is not found.
    """
    non_existent_file = tmp_path / "non_existent_file.csv"
    
    assert not non_existent_file.exists()
    
    with pytest.raises(FileNotFoundError):
        load_preprocessed_data(non_existent_file)

def test_train_model_success(mock_processed_data, mock_model_params):
    """
    Test if train_model successfully trains a model.
    """
    model = train_model(
        pd.read_csv(mock_processed_data),
        "IsolationForest",
        mock_model_params
    )
    
    assert isinstance(model, IsolationForest)
    
    assert model.n_estimators == 100
    assert model.random_state == 42

def test_train_model_empty_data(mock_empty_data, mock_model_params):
    """
    Test if test_train_model for success
    """
    empty_df = pd.read_csv(mock_empty_data)
    with pytest.raises(ValueError) as e:
        train_model(
            empty_df, 
            "IsolationForest",
            mock_model_params   
        )
        
def test_evaluate_model_success(mock_model, mock_processed_data):
    """
    Test if evaluate_model successfully 
    """
    metrics = evaluate_model(mock_model, pd.read_csv(mock_processed_data))
    
    assert isinstance(metrics, dict)

def test_evaluate_model_empty_data(mock_model, mock_empty_data):
    """
    Test if evaluate_model raises error for empty input dataframe.
    """
    with pytest.raises(Exception) as e:
        evaluate_model(mock_model, pd.read_csv(mock_empty_data))