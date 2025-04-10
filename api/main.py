from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from tensorflow.keras.models import load_model
import logging
import os

# ========== CONFIG ==========

MODEL_PATH = "models/cnn_lstm_anomaly_detector.h5"
EXPECTED_FEATURES = 29  # change this to 30 if needed

# ========== LOGGING SETUP ==========

LOG_FILE = "api/prediction_logs.log"
os.makedirs("api", exist_ok=True)  # Ensure log directory exists

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ========== LOAD MODEL ==========

model = load_model(MODEL_PATH)

# ========== FASTAPI APP ==========

app = FastAPI(title="FinSight AI - Fraud Detection API")

# ========== Pydantic Schemas ==========

class TransactionInput(BaseModel):
    input: List[float]

class PredictionOutput(BaseModel):
    fraud: bool
    confidence: float
    message: str

# ========== ROUTES ==========

@app.get("/")
def root():
    return {"message": "🚀 FinSight AI is live and ready for predictions."}

@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):
    try:
        # Convert input to NumPy array
        input_array = np.array(transaction.input)

        # ✅ Validate shape
        if input_array.shape[0] != EXPECTED_FEATURES:
            raise HTTPException(
                status_code=422,
                detail=f"Expected {EXPECTED_FEATURES} features, but got {input_array.shape[0]}"
            )

        # Reshape to (1, 1, features) for CNN-LSTM
        input_array = input_array.reshape((1, 1, EXPECTED_FEATURES))

        # Run prediction
        prediction = model.predict(input_array)
        confidence = float(prediction[0][0])
        fraud_flag = confidence > 0.5

        # ✅ Log input + result
        logging.info(f"INPUT: {transaction.input}")
        logging.info(f"PREDICTION: confidence={confidence:.4f}, fraud={fraud_flag}")

        return PredictionOutput(
            fraud=fraud_flag,
            confidence=confidence,
            message="✅ Prediction successful"
        )

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
