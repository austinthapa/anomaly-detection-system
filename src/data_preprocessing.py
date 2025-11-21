import yaml
import logging
import joblib
import pandas as pd

from pandas import DataFrame
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, Binarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s-%(message)s"
)
logger = logging.getLogger(__name__)

"""
load_config -> load_split_data -> scale_num_features -> encode_categorical_features -> save_clean_data
"""

def load_config(
    config_path = "config/columns.yaml"
) -> dict:
    """
    Loads the configuration from a YAML file with error handling.
    
    Args:
        config_path: Configuration path
    
    Returns:
        dict: A dictionary containing configuration data.
        
    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: Error related with parsing YAML file.
        Exceptions: For all other unrelated exceptions.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logging.info(f"Successfully loaded configuration from: {config_path}")
            return config         
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise 
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {config_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading config: {e}")
        raise
    
def load_data(
    data_path: Path
) -> DataFrame:
    """
    Load the data from a given path.
    
    Args:
        data_path (Path): The pathlib object
        
    Returns:
        DataFrame: The loaded pandas DataFrame
        
    Raises:
        ValueError: If the read DataFrame is empty.
        Exceptions: For all other unrelated exceptions.
    """
    if not data_path.exists():
        logger.error(f"File not found at given location: {data_path}")
        raise FileNotFoundError(f"File not found at given location: {data_path}")
    try:
        logger.info(f"Loading the data from {data_path}")
        df = pd.read_csv(data_path)
        
        if df.empty:
            logger.warning(f"Input file loaded successfully but resulted in empty DataFrame")
            raise ValueError(f"Read DataFrame is empty")
        
        logger.info("Data loaded successfully...\n"
                    f"{len(df)} rows and {len(df.columns)} columns")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Error while loading the DataFrame: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.info(f"Unexpected Error occured while loading: {e}", exc_info=True)
        raise
    
def drop_metadata_features(
    df: DataFrame,
    metadata_cols: list
):
    """
    Drop metadata columns from the DataFrame.
    
    Args:
        df (DataFrame): The input pandas DataFrame.
        metadata_cols (list): The list of meta-data columns.
    
    Returns:
        DataFrame: DataFrame with dropped metadata columns.
    
    Raises:
        Exception: If dropping columns fails.
    """
    try:
        missing_cols = [col for col in metadata_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing metadata columns: {missing_cols}")
            raise ValueError(f"Missing metadata columns: {missing_cols}")
        df = df.drop(metadata_cols, axis=1)
        logger.info(f"Successfully droped meta-data columns: {metadata_cols}")
        return df 
    except Exception as e:
        logger.error(f"Error while dropping meta-data features: {e}", exc_info=True)
        raise
    
def scale_numeric_features(
    df: DataFrame,
    num_cols: list,
    scaler_save_path: Path
) -> DataFrame:
    """
    Scale the numeric features
    
    Args:
        df (DataFrame): The input data
        num_cols (list): The list of numeric columns.
    Returns:

    Raises:
        ValueError: If the loaded DataFrame is empty.
        Exception: For all other unrelated exceptions.
    """
    if df.empty:
        logger.warning(f"Loaded DataFrame is empty.\n"
                       f"Columns: {df.columns.tolist()}")
        raise ValueError("Loaded DataFrame is empty.")
    try:
        logger.info("Scaling the numeric features using StandardScaler for following features:\n"
                    f"{num_cols}")

        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        logger.info("Successfully completed numeric features.")
        
        joblib.dump(scaler, scaler_save_path / "scaler.joblib")
        logger.info(f"Scaler successfully saved at: {scaler_save_path}")
        return df
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}", exc_info=True)
        raise

def onehot_encode_features(
    df: DataFrame,
    onehot_cols: list,
    onehot_encoder_save_path: Path
) -> DataFrame:
    """
    Onehot Encode the given categorical columns using Scikit-learn.
    
    Args:
        df (DataFrame): The input DataFrame containing input data.
        onehot_cols (list): The list of categorical columns to onehot encode.
        
    Returns:
        DataFrame: The DataFrame with encoded columns
        
    Raises:
        ValueError: If the loaded DataFrame is empty or encoding columns are not present.
        Exception: For all other unrelated exceptions.
    """
    if df.empty:
        logger.warning(f"The input DataFrame is empty\n"
                       f"{len(df)} rows")
        raise ValueError(f"The DataFrame is empty.")
    
    missing_cols = [col for col in onehot_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Required columns for onehot encoding are missing: {missing_cols}")
        raise ValueError(f"DataFrame is missing required columns")
    
    try:
        logger.info("Starting onehot encoding for following columns:\n"
                    f"{onehot_cols}")
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded_df = DataFrame(
            onehot_encoder.fit_transform(df[onehot_cols]),
            columns=onehot_encoder.get_feature_names_out(onehot_cols),
            index=df.index
        )
        df = pd.concat(
            [df.drop(onehot_cols, axis = 1), encoded_df], axis = 1
        )
        joblib.dump(onehot_encoder, onehot_encoder_save_path / "onehot_encoder.joblib")      #    
        logger.info("Onehot encoding successfully performed...")
        return df
    except Exception as e:
        logger.error(f"Unexpected error while onehot encoding: {e}", exc_info=True)
        raise
    
def encode_ordinal_features(
    df: DataFrame,
    ordinal_cols: list,
    ordinal_categories_map: dict,
    ordinal_encoder_save_path: Path
) -> DataFrame:
    """
    Ordinal encode the given categorical columns using Scikit-learn.
    
    Args:
        df (DataFrame): The input DataFrame containing input data.
        ordinal_cols (list): The list of categorical columns to ordinal encode.
        
    Returns:
        DataFrame: The DataFrame with encoded columns
        
    Raises:
        ValueError: If the loaded DataFrame is empty or encoding columns are not present.
        Exception: For all other unrelated exceptions.
    """
    if df.empty:
        logger.error(f"The Loaded DataFrame is empty."
                     f"{len(df)} rows")
        raise ValueError("The loaded DataFrame is empty.")
    
    missing_cols = [col for col in ordinal_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Required columns for Ordinal Encoding are missing: {missing_cols}")
        raise ValueError(f"Required columns for Ordinal Encoding are missing: {missing_cols}")
    
    try:
        logger.info("Starting ordinal encoding for following columns:\n"
                    f"{ordinal_cols}")
        
        ordinal_categories = []
        for col in ordinal_cols:
            ordinal_categories.append(ordinal_categories_map[col])
            
        ordinal_encoder = OrdinalEncoder(
            categories= ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        ordinal_df = DataFrame(
            ordinal_encoder.fit_transform(df[ordinal_cols]),
            columns=ordinal_cols,
            index= df.index
        )
        df = pd.concat(
            [df.drop(ordinal_cols, axis=1), ordinal_df], axis=1
        )
        joblib.dump(ordinal_encoder, ordinal_encoder_save_path / "ordinal_encoder.joblib")
        logger.info("Ordinal encoding successfully completed...")
        return df
    except Exception as e:
        logger.error(f"Unexpected error while ordinal encoding: {e}", exc_info=True)
        raise

def encoder_binary_features(
    df: DataFrame,
    binary_cols: list,
    binary_mappings: dict
) -> DataFrame:
    """
    Encode the binary features into binary values of 0/1.
     
    Args:
        df (DataFrame): The input DataFrame.
        binary_cols (list): 
        
    Returns:
        DataFrame: The DataFrame with encoded columns.
        
    Raises:
        ValueError: If the loaded DataFrame is empty or encoding columns are not present.
        Exception: For all other unrelated exceptions during processing.
    """
    if df.empty:
        logger.error(f"The Loaded DataFrame is empty."
                     f"{len(df)} rows")
        raise ValueError("The loaded DataFrame is empty.")
    
    missing_cols = [col for col in binary_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Required columns for Binary Encoding are missing: {missing_cols}")
        raise ValueError(f"Required columns for Binary Encoding are missing: {missing_cols}")
    try:
        logger.info("Starting binary encoding for following columns:\n"
                    f"{binary_cols}")
        
        for col in binary_cols:
            mapping = binary_mappings[col]
            if not mapping:
                logger.error(f"No mapping found for column: {col}")
                raise ValueError(f"No mapping defined for: {col}")
            
            df[col] = df[col].map(mapping)
            
            if df[col].isnull().any():
                logger.error(f"NaN values after encoding: {col}")
                raise ValueError(f"Encoding failed for: {col}")
        logger.info("Binary encoding successfully completed...")
        return df
    except Exception as e:
        logger.error(f"Unexpected error while binary encoding: {e}", exc_info=True)
    raise

def save_data(
    df: DataFrame,
    save_path: Path
):
    """
    Save DataFrame to a CSV file.
    
    Args:
        df (DataFrame): The DataFrame to save.
        save_path (Path): Path where CSV file will be saved.
        
    Raises:
        ValueError: If DataFrame is empty or path issues.
        IOError: If file writing fails.
    """
    if df.empty:
        logging.error(f"Input DataFrame is empty: {len(df)} rows")
        raise ValueError(f"Input DataFrame is empty: {len(df)} rows")
    try:
        save_path.mkdir(exist_ok=True, parents=True)
        extension = ".csv"
        
        df.to_csv(save_path / f"train{extension}", index=False)
        logger.info(f"Data successfully saved to: {save_path}")
        logger.info(f"Saved {len(df)} rows and {len(df.columns)} columns")
    except IOError as e:
        logger.error(f"IO error while saving data to {save_path}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error while saving data: {e}", exc_info=True)
        raise

def main():
    
    # Step 0: Load the configuration
    config = load_config()
    
    # Step 1: Load the data
    data_path = Path("data/raw/train.csv")
    df = load_data(data_path=data_path)
    
    metadata_cols = config["features"]["metadata_columns"]
    df = drop_metadata_features(
        df=df,
        metadata_cols=metadata_cols
    )
    
    # Step 2: Scale the numeric features
    num_cols = config["features"]["numeric"]
    scaled_df = scale_numeric_features(
        df=df,
        num_cols=num_cols,
        scaler_save_path=Path("artifacts/")
    )
    
    # Step 3: Onehot Encode Categorical Features
    onehot_cols = config["features"]["onehot"]
    encoded_df = onehot_encode_features(
        df=scaled_df,
        onehot_cols=onehot_cols,
        onehot_encoder_save_path=Path("artifacts/")
    )
    
    # Step 4: Ordinal Encode Categorical Features
    ordinal_cols = config["features"]["ordinal"]
    ordinal_categories = config["ordinal_categories"]
    encoded_df = encode_ordinal_features(
        df=encoded_df,
        ordinal_cols=ordinal_cols,
        ordinal_categories_map=ordinal_categories,
        ordinal_encoder_save_path=Path("artifacts/")
    )
    
    # Step 5: Binary Encode the Categorical features
    binary_cols = config["features"]["binary"]
    binary_mappings = config["binary_mappings"]
    encoded_df = encoder_binary_features(
        df=encoded_df,
        binary_cols=binary_cols,
        binary_mappings=binary_mappings
    )
    
    # Step 4: Save the preprocesed_data
    save_data(
        df=encoded_df,
        save_path=Path("data/clean")
    )

if __name__ == "__main__":
    main()