import yaml
import pytest
import pandas as pd

from pathlib import Path
from pandas import DataFrame
from data_preprocessing import (
    load_config,
    load_data,
    drop_metadata_features,
    scale_numeric_features,
    onehot_encode_features,
    encode_ordinal_features,
    encoder_binary_features,
    save_data
)

@pytest.fixture
def mock_config_file(tmp_path):
    """
    Creates a temporary YAML config file for the test.
    """
    config_content = {
        "data": {
            "train_size": 0.8,
            "random_seed": 98
        },
        "model": {
            "n_estimators": 200
        }
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as file:
        yaml.dump(config_content, file)
    return str(config_file)

@pytest.fixture
def mock_valid_csv(tmp_path: Path) -> Path:
    """
    Creates a temporary, valid CSV file for successful loading.
    """
    df = DataFrame({
        "feat_1": range(10),
        "feat_2": range(10),
        "feat_3": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
    })
    mock_file_path = tmp_path / "valid_data.csv"
    df.to_csv(mock_file_path, index=False)
    return mock_file_path

@pytest.fixture
def mock_empty_df(tmp_path: Path) -> Path:
    """
    Creates an empty DataFrame for testing
    """
    file_path = tmp_path / "empty_df.csv"
    pd.DataFrame(columns=["feat_1", "feat_2"]).to_csv(file_path, index=False)
    return file_path

def test_load_config(mock_config_file):
    """
    Test if the function correctly loads and parses the config file.
    """
    config = load_config(mock_config_file)
    
    assert isinstance(config, dict)
    assert config["data"]["train_size"] == 0.8
    assert config["model"]["n_estimators"] == 200
    assert "random_seed" in config["data"]
    
def test_load_config_file_not_found():
    """
    Tests that the function raises FileNotFoundError when the file is missing.
    """
    missing_path = "file_does_not_exist_xyz.yaml"
    
    with pytest.raises(FileNotFoundError):
        load_config(missing_path)

def test_load_data_success(mock_valid_csv: Path):
    """
    Tests if the load_data successfully loads the data from a given path.
    """
    df = load_data(mock_valid_csv)
    
    assert isinstance(df, DataFrame)
    assert df.shape == (10, 2)
    assert "feat_1" in df.columns
    
def test_load_data_file_not_found(tmp_path):
    """
    Tests if the load_data raises FileNotFoundException if the file is not found.
    """
    non_existent_file_path =  tmp_path / "this_file_does_not_exist.csv"
    
    assert not non_existent_file_path.exists()
    
    with pytest.raises(FileNotFoundError):
        load_data(non_existent_file_path)
        
def test_load_data_empty_dataframe(mock_empty_df):
    """
    Tests if it raises ValueError while loading empty DataFrame.
    """
    with pytest.raises(ValueError):
        load_data(mock_empty_df)