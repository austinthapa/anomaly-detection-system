import yaml
import pytest
import numpy as np
import pandas as pd

from pathlib import Path
from pandas import DataFrame
from data_preprocessing import (
    load_config,
    load_data,
    scale_numeric_features,
    drop_metadata_features,
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
        "feat_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "feat_2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "feat_3": ["A", "B", "A", "B", "C", "A", "C", "B", "A", "C"],
        "feat_4": ["x1", "x2", "x3", "x1", "x2", "x3", "x1", "x2", "x3", "x1"],
        "feat_5": ["yes", "no", "yes", "no", "maybe", "yes", "no", "yes", "no", "maybe"],
        "feat_6": ["S", "M", "L", "XL", "S","S", "M", "L", "M","XL"],
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

def test_load_config(mock_config_file: Path):
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
    assert df.shape == (10, 6)
    assert "feat_1" in df.columns
    
def test_load_data_file_not_found(tmp_path: Path):
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

def test_scale_numeric_features(mock_valid_csv, tmp_path):
    """
    Tests if scale_numeric_features correctly scales the numeric features.
    """
    mock_df = pd.read_csv(mock_valid_csv)
    num_cols = ["feat_1", "feat_2"]
    scaler_path = tmp_path / "scaler.joblib"
    
    scaled_df = scale_numeric_features(mock_df, num_cols, tmp_path)
    
    assert np.isclose(scaled_df["feat_1"].mean(), 0.0, atol=1e-6)
    assert np.isclose(scaled_df["feat_1"].std(ddof = 0), 1.0, atol=1e-6)
    
    assert "feat_1" in scaled_df.columns
    
    assert scaler_path.exists()

def test_encode_onehot_features(mock_valid_csv, tmp_path):
    """
    Tests if the onehot_encode_features successfully encodes the onehot features.
    """
    onehot_cols = ["feat_3", "feat_4", "feat_5"]
    mock_df = pd.read_csv(mock_valid_csv)
    encoder_save_path = tmp_path / "onehot_encoder.joblib"
    
    encoded_df = onehot_encode_features(mock_df, onehot_cols, tmp_path)
    
    assert isinstance(encoded_df, DataFrame)
    assert encoder_save_path.exists()
    
    assert len(encoded_df.columns) == 12
    assert "feat_3_A" in encoded_df.columns
    assert "feat_5_no" in encoded_df.columns

def test_encode_ordinal_features(mock_valid_csv, tmp_path):
    """
    Tests if encode_ordinal_features successfully encodes the ordinal features.
    """
    ord_cols = ["feat_6"]
    ord_cat_maps = {
        "feat_6": ["S", "M", "L", "XL"]
    }
    ord_encoder_save_path = tmp_path / "ordinal_encoder.joblib"
    mock_df = pd.read_csv(mock_valid_csv)
    encoded_df = encode_ordinal_features(mock_df, ord_cols, ord_cat_maps, tmp_path)
    
    assert ord_encoder_save_path.exists()
    assert "feat_3" in encoded_df.columns
    assert encoded_df["feat_6"].tolist() == [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.0, 1.0, 3.0]
    
def test_save_data(mock_valid_csv, tmp_path):
    """
    Tests if save_data correctly saves the data or not.
    """
    mock_df = pd.read_csv(mock_valid_csv)
    
    save_dir = tmp_path
    
    save_data(mock_df, save_dir)
    read_df = pd.read_csv(save_dir / "train.csv")
    
    assert save_dir.exists()
    pd.testing.assert_frame_equal(mock_df, read_df)

def test_save_data_empty_dataframe(mock_empty_df, tmp_path):
    """
    Tests if save_data correctly raises ValueError if the DataFrame is empty.
    """
    save_path = tmp_path / "mock.csv"
    mock_df = pd.read_csv(mock_empty_df)
    with pytest.raises(ValueError):
        save_data(mock_df, save_path)