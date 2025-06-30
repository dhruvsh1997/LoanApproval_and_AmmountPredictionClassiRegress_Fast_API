from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from typing import Dict
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Loan Risk Prediction API")

# Load trained model artifacts with error handling
MODEL_PATH = Path("loan_risk_model.pkl")
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found")
    model_bundle = joblib.load(MODEL_PATH)
    scaler = model_bundle.get("scaler")
    clf = model_bundle.get("clf")
    reg = model_bundle.get("reg")
    if not all([scaler, clf, reg]):
        raise ValueError("Model bundle missing required keys: scaler, clf, reg")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError(f"Model loading failed: {str(e)}")

# In-memory storage (consider replacing with a database in production)
db: Dict[int, Dict] = {}
applicant_id_counter: int = 0

# Define input and response schemas
class Applicant(BaseModel):
    age: int = Field(..., ge=18, le=120, description="Applicant age")
    income: float = Field(..., ge=0, description="Annual income in USD")
    credit_score: float = Field(..., ge=300, le=850, description="Credit score")
    loan_amount: float = Field(..., ge=0, description="Requested loan amount")
    gender: str = Field(..., min_length=1, description="Gender")
    purpose: str = Field(..., min_length=1, description="Loan purpose")

class ApplicantResponse(BaseModel):
    applicant_id: int
    approved: bool
    default_risk_score: float
    input_data: Applicant

@app.post("/applicants/", response_model=ApplicantResponse)
async def create_applicant(data: Applicant):
    global applicant_id_counter
    try:
        # Convert input to DataFrame and encode
        df = pd.DataFrame([data.dict()])
        df_encoded = pd.get_dummies(df)

        # Align columns with training set
        for col in scaler.feature_names_in_:
            if col not in df_encoded:
                df_encoded[col] = 0
        df_encoded = df_encoded[scaler.feature_names_in_]

        # Scale and predict
        X_scaled = scaler.transform(df_encoded)
        prediction = bool(clf.predict(X_scaled)[0])
        risk_score = float(reg.predict(X_scaled)[0])

        # Store data
        applicant_id_counter += 1
        db[applicant_id_counter] = {
            "input": data.dict(),
            "prediction": prediction,
            "risk_score": risk_score
        }
        logger.info(f"Created applicant ID {applicant_id_counter} with prediction: {prediction}")
        return {
            "applicant_id": applicant_id_counter,
            "approved": prediction,
            "default_risk_score": risk_score,
            "input_data": data
        }
    except Exception as e:
        logger.error(f"Error processing applicant: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/applicants/{applicant_id}")
async def get_applicant(applicant_id: int):
    if applicant_id not in db:
        logger.warning(f"Applicant ID {applicant_id} not found")
        raise HTTPException(status_code=404, detail="Applicant not found")
    logger.info(f"Retrieved applicant ID {applicant_id}")
    return db[applicant_id]

@app.put("/applicants/{applicant_id}", response_model=ApplicantResponse)
async def update_applicant(applicant_id: int, data: Applicant):
    if applicant_id not in db:
        logger.warning(f"Applicant ID {applicant_id} not found")
        raise HTTPException(status_code=404, detail="Applicant not found")
    try:
        # Re-predict with updated data
        df = pd.DataFrame([data.dict()])
        df_encoded = pd.get_dummies(df)
        for col in scaler.feature_names_in_:
            if col not in df_encoded:
                df_encoded[col] = 0
        df_encoded = df_encoded[scaler.feature_names_in_]
        X_scaled = scaler.transform(df_encoded)
        prediction = bool(clf.predict(X_scaled)[0])
        risk_score = float(reg.predict(X_scaled)[0])

        # Update storage
        db[applicant_id] = {
            "input": data.dict(),
            "prediction": prediction,
            "risk_score": risk_score
        }
        logger.info(f"Updated applicant ID {applicant_id} with prediction: {prediction}")
        return {
            "applicant_id": applicant_id,
            "approved": prediction,
            "default_risk_score": risk_score,
            "input_data": data
        }
    except Exception as e:
        logger.error(f"Error updating applicant {applicant_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.delete("/applicants/{applicant_id}")
async def delete_applicant(applicant_id: int):
    if applicant_id not in db:
        logger.warning(f"Applicant ID {applicant_id} not found")
        raise HTTPException(status_code=404, detail="Applicant not found")
    del db[applicant_id]
    logger.info(f"Deleted applicant ID {applicant_id}")
    return {"deleted": True}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}