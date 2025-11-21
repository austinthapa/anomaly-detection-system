import pytest
import pandas as pd

from pathlib import Path
from pandas import DataFrame
from data_ingestion import load_data, split_data, save_data

@pytest.fixture
def mock_csv_data(tmp_path: Path) -> Path:
    """
    Creates a temporary, simple CSV file for test to read.
    """
    df = pd.DataFrame({
        "feat_1": range(5),
        "feat_2": ["A", "B", "C", "D", "E",]
    })
    mock_file = tmp_path / "test_raw_data.csv"
    df.to_csv(mock_file, index=False)
    return mock_file

@pytest.fixture
def dummy_df() -> DataFrame:
    """
    Provides a Dummy DataFrame of a known size of 100
    """
    N_ROWS = 100
    return DataFrame({
        "feat_1": range(100),
        "feat_2": [i % 2 for i in range(100)]
    })

@pytest.fixture
def dummy_train_test_data() -> DataFrame:
    """
    Provides 2 sample DataFrame for testing.
    """
    train_df = DataFrame({
        "feat_1": [1, 2, 3, 4, 5],
        "feat_2": [6, 7, 8, 9, 10]
    })
    test_df = DataFrame({
        "feat_1": [10, 20, 30, 40, 50],
        "feat_2": [60, 70, 80, 90, 100]
    })
    return train_df, test_df

def test_load_data(mock_csv_data):
    """
    Test if load_data successfully reads the mock data and returns a DataFrame
    """
    result_df = load_data(mock_csv_data)
    
    assert isinstance(result_df, pd.DataFrame)    
    assert result_df.shape == (5, 2)
    assert "feat_1" in result_df.columns
    
def test_load_data_file_not_found(tmp_path):
    """
    Test error handling when the file does not exist.
    """
    non_existent_path = tmp_path / "file_does_not_exist.csv"
    
    assert not non_existent_path.exists()
    
    with pytest.raises(FileNotFoundError):
        load_data(non_existent_path)
        
def test_split_data_valid_output_types(dummy_df):
    """
    Test if split_data successfully returns a tuple of two DataFrame or not.
    """
    train_df, test_df = split_data(dummy_df, test_size=0.2)
    
    assert isinstance(train_df, DataFrame)
    assert isinstance(test_df, DataFrame)
    
def test_split_data_proportion(dummy_df):
    """
    Test if the resulting dataframe have correct row counts.
    """
    test_size = 0.2
    total_rows = len(dummy_df)
    
    train_df, test_df = split_data(dummy_df, test_size=0.2)
    
    # Calculate the expected rows in train and test df
    expected_train_rows = (1 - test_size) * total_rows
    expected_test_rows = test_size * total_rows
    
    assert len(train_df) == expected_train_rows
    assert len(test_df) == expected_test_rows
    assert len(train_df + test_df) == total_rows
    
def test_split_data_reproducibility(dummy_df):
    """
    Test that the split is identical when using the same random state.
    """
    RANDOM_STATE = 911
    train_df_1, test_df_1 = split_data(dummy_df, test_size=0.2, random_state=RANDOM_STATE)
    train_df_2, test_df_2 = split_data(dummy_df, test_size=0.2, random_state=RANDOM_STATE)
    
    pd.testing.assert_frame_equal(train_df_1, train_df_2, check_names=True)
    pd.testing.assert_frame_equal(test_df_1, test_df_2, check_names=True)

def test_split_data_raises_value_error(dummy_df):
    """
    Test that a ValueError is raised for invalid test_size input.
    """
    invalid_sizes = [1, 0, -1, 2]
    for size in invalid_sizes:
        with pytest.raises(ValueError):
            split_data(dummy_df, test_size=size)

def test_save_data(dummy_train_test_data, tmp_path):
    """
    Test if save_data successfully saves the given DataFrame to a given file path.
    """
    train_df_org, test_df_org = dummy_train_test_data
    
    save_dir = tmp_path / "processed_data"
    save_dir.mkdir()
    
    expected_train_path = save_dir / "train.csv"
    expected_test_path = save_dir / "test.csv"
    
    save_data(train_df_org, test_df_org, save_dir)
    
    assert expected_test_path.is_file()
    assert expected_train_path.is_file()
    
    train_df_saved = pd.read_csv(expected_train_path)
    test_df_saved = pd.read_csv(expected_test_path)
    
    pd.testing.assert_frame_equal(train_df_org, train_df_saved)
    pd.testing.assert_frame_equal(test_df_org, test_df_saved)