import logging
import pandas as pd

from pandas import DataFrame
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s-%(message)s"
)
logger = logging.getLogger(__name__)

"""
load_data -> split_data -> save_data
"""

def load_data(
    data_path: Path
) -> DataFrame:
    """
    Load the data from a given CSV file path.
    It performs validation on the input, checks the input is empty or not, and logs the results.
    
    Args:
        data_path (Path): Relative or absolute path to CSV file.
        
    Returns:
       DataFrame: The loaded pandas DataFrame containing the data.
       
    Raises:
        FileNotFoundError: If the CSV file is not found at given location.
        ValueError: If the file is found, but is empty or unreadable.
        Exception: For any other unexpected errors during loading.
    """
    if not data_path.is_file():
        logger.error(f"File not found at given path: {data_path}")
        raise FileNotFoundError(f"File not found at given path: {data_path}")
    try:
        if data_path.suffix.lower() != ".csv":
            logger.warning(f"File path {data_path} does not have a '.csv' extension")
        
        logger.info("Loading the data...")
        df = pd.read_csv(data_path)
        if df.empty:
            logger.error(f"Read CSV file is empty or contains no valid rows")
            raise ValueError(f"Read CSV file is empty or contains no valid rows")
        
        logger.info(f"Successfully loaded data from: {data_path}\n"
                    f"with {len(df)} rows and {len(df.columns)} columns")
        return df
    except pd.errors.ParserError as e:  
        logger.error(f"Error while parsing the data {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error occured: {e}", exc_info=True)
        raise

def split_data(
    df: DataFrame,
    test_size: float= 0.2,
    random_state: int = 42
) -> Tuple[DataFrame, DataFrame]:
    """
    Split the input DataFrame into training and testing sets.
    
    Args:
        df (DataFrame: The loaded DataFrame containing the input data.
        test_size (float): The proportion of the data to include in test split.
        random_state (int): The seed used by random number generator for reproducibility.
        
    Returns:
        Tuple[DataFrame, DataFrame]: A Tuple containing training and testing DataFrame.

    Raises:
        ValueError: If the DataFrame is empty or test_size is outside (0, 1).
        Exception: For all other unrelated exceptions.
    """
    if df.empty:
        logger.error(f"Loaded DataFrame is empty.")
        raise ValueError(f"Loaded DataFrame is empty.")
    if not 0 < test_size < 1:
        logger.error(f"Invalid test_size: {test_size}. Test size should be between 0 and 1.")
        raise ValueError(f"Test Size should be between 0 and 1\n"
                         f"Got test size of: {test_size}")
    try:
        logger.info(f"Starting train and test split with test_size = {test_size} and random_state={random_state}")
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
        )
        
        logger.info(f"Split Complete...\n"
                    f"Train Split: {len(train_df)} rows\n"
                    f"Test Split: {len(test_df)} rows")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Unexpected Error occured during the split: {e}", exc_info=True)
        raise
    
def save_data(
    train_df: DataFrame,
    test_df: DataFrame,
    data_path: Path
) -> None:
    """
    Saves the training and testing DataFrames to specified directory.
    
    Args:
        train_df (DataFrame): The training DataFrame to save.
        test_df (DataFrame): The testing DataFrame to save.
        data_path (Path): The directory path where the data files will be saved.
    
    Returns:
        None

    Raises:
        ValueError: If either the training or testing DataFrame is empty.
        IOError: If an error occurs during the file writing proces..
        Exception: For all other unrealated exceptions.
    """
    if train_df.empty:
        logger.error(f"Training DataFrame is empty: {len(train_df)}")
        raise ValueError("Training DataFrame is empty.")
    if test_df.empty:
        logger.error(f"Testing DataFrame is empty: {len(test_df)}")
        raise ValueError("Testing DataFrame is empty.")
    try:
        data_path.mkdir(exist_ok=True, parents=True)
        
        extension = ".csv"
        train_path = data_path / f"train{extension}"
        test_path = data_path / f"test{extension}"
        
        logger.info("Saving the data...")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Data saved successfully...\n"
                    f"Train path: {train_path}\n"
                    f"Test path: {test_path}\n")
        return
    except IOError as e:
        logger.error(f"IOError occured while writing the files to {data_path}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected Error occured while saving data: {e}", exc_info=True)
        raise

def main():
    
     # 1. Load the data
    data_path = Path("data/depression_data.csv")
    df = load_data(
        data_path=data_path
    )
    
    # 2. Split the data
    train_df, test_df = split_data(
        df=df,
        test_size=0.2,
        random_state=42
    )
    
    # 3. Save the data
    save_path = Path("data/raw")
    save_data(
        train_df=train_df,
        test_df=test_df,
        data_path=save_path
    )

if __name__ == "__main__":
    main()