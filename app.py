import yaml
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from pydantic import Field, BaseModel
from typing import Dict
from pandas import DataFrame
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s-%(message)s"
)
logger = logging.getLogger(__name__)

scaler = None
onehot_encoder = None
ordinal_encoder = None
model = None
config = None

class PredictionInput(BaseModel):
    """
    Input Schema for prediction
    """
    age: float = Field(..., ge=0, le=120, description="Age in years")
    number_of_children: int = Field(..., ge=0, description="Number of children")
    income: float = Field(..., ge=0, description="Annual income")
    marital_status: str = Field(..., description="Marital status (e.g., 'Married', 'Single', 'Divorced')")
    smoking_status: str = Field(..., description="Smoking status (e.g., 'Non-smoker', 'Ex-smoker', 'Current smoker')")
    physical_activity_level: str = Field(..., description="Physical activity level")
    education_level: str = Field(..., description="Education level")
    alcohol_consumption: str = Field(..., description="Alcohol consumption (e.g., 'None', 'Moderate', 'Heavy')")
    dietary_habits: str = Field(..., description="Dietary habits")
    sleep_patterns: str = Field(..., description="Sleep patterns")
    employment_status: str = Field(..., description="Employment status (e.g., 'Employed', 'Unemployed')")
    history_of_mental_illness: str = Field(..., description="History of mental illness (Yes or No)")
    history_of_substance_abuse: str = Field(..., description="History of substance abuse (Yes or No)")
    family_history_of_depression: str = Field(..., description="Family history of depression (Yes or No)")
    chronic_medical_conditions: str = Field(..., description="Chronic medical condition (Yes or No)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35.0,
                "number_of_children": 2,
                "income": 50000.0,
                "marital_status": "Married",
                "smoking_status": "Non-smoker",
                "physical_activity_level": "Moderate",
                "education_level": "Bachelor's Degree",
                "alcohol_consumption": "Moderate",
                "dietary_habits": "Healthy",
                "sleep_patterns": "Good",
                "employment_status": "Employed",
                "history_of_mental_illness": "No",
                "history_of_substance_abuse": "No",
                "family_history_of_depression": "No",
                "chronic_medical_conditions": "No"
            }
        }
app = FastAPI(
    title="Anomaly Detection API",
    description="API for detecting mentally Ill people.",
    version="1.0.0"
)

@asynccontextmanager
async def load_artifacts(app: FastAPI):
    try:
        artifact_path = Path("artifacts")
        
        # Load the YAML configurations
        with open("config/columns.yaml", "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Column configuration loaded successfully...")
        
        # --- Load the scaler ---
        scaler = joblib.load(artifact_path / "scaler.joblib")
        logger.info("Scaler loaded successfully...")
        
        # --- Load the One Hot Encoder ---
        onehot_encoder = joblib.load(artifact_path / "onehot_encoder.joblib")
        logger.info("OneHot encoder loaded successfully...")
        
        # --- Load the Ordinal Encoder ---
        ordinal_encoder = joblib.load(artifact_path / "ordinal_encoder.joblib")
        logger.info("Ordinal encoder loaded successfully...")
        
        # --- Load the model ---
        model = joblib.load(artifact_path / "model.joblib")
        logger.info("Model loaded successfully...")
        
        app.state.config = config
        app.state.scaler = scaler
        app.state.onehot_encoder = onehot_encoder
        app.state.ordinal_encoder = ordinal_encoder
        app.state.model = model
        
        yield
    except Exception as e:
        logger.error(f"Unexpected error while loading the artifacts: {e}", exc_info=True)
        raise
        
app = FastAPI(lifespan=load_artifacts)

@app.get("/")
def home():
    return {
        "message": "Mental Health Wellness API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }
    
@app.get("/health")
def health_check():
    return {
        "Status": "Okay",
        "Service": "Online"
    }

def preprocess_input(
    request: Request,
    data: PredictionInput
) -> np.ndarray:
    """
    Docstring for preprocess_input
    """
    input_dict = {
        "Age": data.age,
        "Number of Children": data.number_of_children,
        "Income": data.income,
        "Marital Status": data.marital_status,
        "Smoking Status": data.smoking_status,
        "Physical Activity Level": data.physical_activity_level,
        "Education Level": data.education_level,
        "Alcohol Consumption": data.alcohol_consumption,
        "Dietary Habits": data.dietary_habits,
        "Sleep Patterns": data.sleep_patterns,
        "Employment Status": data.employment_status,
        "History of Mental Illness": data.history_of_mental_illness,
        "History of Substance Abuse": data.history_of_substance_abuse,
        "Family History of Depression": data.family_history_of_depression,
        "Chronic Medical Conditions": data.chronic_medical_conditions
    }
    
    df = DataFrame(input_dict, index = [0])

    # Access the configuration
    config = request.app.state.config
    
    # --- Scale the data ---
    scaler = request.app.state.scaler
    num_cols = config["features"]["numeric"]
    if num_cols:
        df[num_cols] = scaler.transform(df[num_cols])
            
    # --- Onehot encode data ---
    onehot_encoder = request.app.state.onehot_encoder
    onehot_cols = config["features"]["onehot"]
    if onehot_cols:
        onehot_encoded = onehot_encoder.transform(df[onehot_cols])
        onehot_feature_names = onehot_encoder.get_feature_names_out(onehot_cols)
        onehot_df = DataFrame(onehot_encoded, columns=onehot_feature_names, index=df.index)
        
        df = df.drop(columns=onehot_cols)
        df = pd.concat([df, onehot_df], axis=1)
        
    # --- Ordinal encode data ---
    ordinal_encoder = request.app.state.ordinal_encoder
    ordinal_cols = config["features"]["ordinal"]
    if ordinal_cols:
        df[ordinal_cols] = ordinal_encoder.transform(df[ordinal_cols])
        
    # --- Binary map the data ---
    binary_mappings = config["binary_mappings"]
    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            if df[col].isna().any():
                raise ValueError(f"Invalid value for {col}. Expected one of {list(mapping.keys())}")
    model = app.state.model
    feature_order = model.feature_names_in_
    return df[feature_order]

@app.post("/predict")
def predict(
    request: Request,
    data: PredictionInput
):
    """
    Make prediction based on input features
    """
    try:
        model = request.app.state.model
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Please check server logs."
            )
        logger.info("Processing prediction request")
        
        processed_data = preprocess_input(
            request=request,
            data=data
        )
        
        prediction = model.predict(processed_data)
        score = model.decision_function(processed_data)
        return {
            "prediction": float(prediction[0]),
            "decision_function": float(score[0])
        }
    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}", exc_info=True)
        raise