import logging
import mlflow
import yaml
import joblib
import numpy as np
import pandas as pd

from pandas import DataFrame, Series
from pathlib import Path
from typing import Tuple, Any, Dict
from sklearn.ensemble import IsolationForest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

"""
load_config -> load_preprocessed_data -> train_model -> evaluate_model -> log_artifacts
"""
def load_config(
    config_path: str
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

def load_preprocessed_data(
    data_path: str
) -> DataFrame:
    """
    Load the preprocessed data from a given path
    
    Args:
        data_path (Path): A Path object to the data
    
    Returns:
        DataFrame: A loaded DataFrame Object.
    
    Raises:
        ValueError: If the loaded DataFrame is empty.
        Exception: For all other unrelated exceptions.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        logger.error(f"No file found at given location: {data_path}")
        raise FileNotFoundError(f"No file found at given location: {data_path}")
    try:
        logger.info(f"Reading the file from: {data_path}")
        df = pd.read_csv(data_path)
        
        if df.empty:
            logger.error(f"Loaded DataFrame is empty.\n{len(df)} rows")
            raise ValueError(f"Loaded DataFrame is empty.\n{len(df)} rows")
        logger.info(f"DataFrame successfully loaded with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Unexpected Error Occured: {e}", exc_info=True)
        raise

def train_model(
    df: DataFrame,
    model_name: str,
    model_params: dict
) -> Any:
    """
    Trains the ML model for anomaly detection.
    
    Args:
        df (DataFrame): The input DataFrame.
        model (str): The model we want to train.
        params (dict): A dictionary of model parameters.
        
    Returns:
        Any: Trained Scikit-learn Estimator
    
    Raises:
        ValueError: If the input DataFrame is empty or contains non-numeric data.
        Exceptions: For all other unrelated exceptions.
    """
    if df.empty:
        logger.error("Input DataFrame for training is empty.")
        raise ValueError("Input DataFrame for training cannot be empty.")
    try:
        logger.info(f"Training the model with params: \n{model_params}")
        
        model = IsolationForest(**model_params)
        
        model.fit(df)
        
        joblib.dump(model, "artifacts/model.joblib")
        
        logger.info("Model Trained Successfully")
        return model
    except Exception as e:
        logger.error(f"Unexpected Error Occured: {e}", exc_info=True)
        raise

def evaluate_model(
    model: Any,
    df: DataFrame,
    contamination: float = 0.05
) -> Dict[str, float]:
    """
    Evaluates the unsupervised anomaly detection model.
    
    Since the true labels are unavailable, evaluation focuses on:
        - The distribution of anomaly.
        - The count of predicted anomalies based on contamination threshold.
    
    Args:
        model: The trained anomaly detection model.
        df (DataFrame): The input DataFrame(test/input set)
    Returns:
        Dict[str, float]: A dictionary of evaluation metrics.
    """
    metrics = {}
    try:
        logger.info("Evaluating the anomaly detection model")
        
        anomaly_scores = model.decision_function(df)
        
        metrics['score_mean'] = float(np.mean(anomaly_scores))
        metrics['score_std'] = float(np.std(anomaly_scores))
        metrics['score_min'] = float(np.min(anomaly_scores))
        metrics['score_max'] = float(np.max(anomaly_scores))
        
        threshold = np.quantile(anomaly_scores, contamination)
        metrics["prediction_threshold"] = float(threshold)
        
        predicted_anomalies = (anomaly_scores < threshold)
        predicted_anomaly_count = int(np.sum(predicted_anomalies))
        metrics['predicted_anomaly_count'] = predicted_anomaly_count
        metrics['predicted_anomaly_rate'] = predicted_anomaly_count / len(df)
        
        logger.info(f"Unsupervised Evaluation Complete. Predicted Anomalies: {predicted_anomaly_count}")
        logger.debug(f"Evaluation Metrics: {metrics}")
        
        return metrics
    except Exception as e:
        logger.error(f"Error while evaluating the model: {e}", exc_info=True)
        raise

def log_artifacts(
    model: Any,
    params: Dict[str, Any],
    input_example: DataFrame,
    metrics: Dict[str, float]
) -> None:
    """
    Log the model, parameters & artifacts to the MLflow run.
    
    Args:
        model: The trained Scikit-learn estimator.
        params: The dictionary of model hyperparameters.
        metrics: The dictionary of evaluation metrics.
        
    Raises:
        Exception: Catches any error during MLflow logging.
    """
    try:
        logger.info("Starting artifact logging to MLflow...")
        
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} model parameters")
        
        mlflow.sklearn.log_model(
            sk_model=model,
            name="models",
            registered_model_name="anomaly-detection-mental-health"
        )
        if metrics:
            mlflow.log_metrics(metrics)
            logger.info(f"Logged {len(metrics)} model metrics")
        else:
            logger.info("No metrics provided to log...")
        
        logger.info("Artifacts successfully logged into MLflow")
    except Exception as e:
        logger.error(f"Unexpected Error Occured: {e}", exc_info=True)
        raise
    
def main():
    
    MLFLOW_EXPERIMENT_NAME = "Anomaly_Detection_Pipeline"
    mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        try:
            logger.info(f"Starting MLflow with Run ID: {run.info.run_id}")
            
            # Step 1: Load the data
            data_config = load_config("config/paths.yaml")
            
            data_path = data_config["data"]["clean_data_path"]
            df = load_preprocessed_data(
                data_path=data_path
            )
            
            # Step 2: Train the model
            model_config = load_config("config/models.yaml")
            
            model_name = model_config["active_model_name"]
            model_params = model_config[model_name]
            
            model = train_model(
                df=df,
                model_name=model_name,
                model_params=model_params
            )
            
            # Step 3: Evaluate the model
            metrics = evaluate_model(
                model=model,
                df=df
            )
            
            # Step 4: Log the artifacts
            log_artifacts(
                model=model,
                params=model_params,
                input_example=df.iloc[:, :10],
                metrics=metrics
            )
            
            mlflow.end_run()
            
            logger.info(f"Pipeline completed successfully and MLflow run ended.")
        except Exception as e:
            logger.error(f"Error while training the model: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    main()