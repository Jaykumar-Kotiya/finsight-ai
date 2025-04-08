from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from tensorflow.keras.models import load_model
import logging

# Set up logging
logging.basicConfig(
    filename="api/prediction_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load the trained model
model = load_model("models/cnn_lstm_anomaly_detector.h5")
EXPECTED_FEATURES = 29  # Change to 30 if your model uses 30

# Define the FastAPI app
app = FastAPI(title="FinSight AI - Anomaly Detection API")

# Input schema
class TransactionInput(BaseModel):
    input: List[float]

# Response schema
class PredictionOutput(BaseModel):
    fraud: bool
    confidence: float
    message: str

@app.get("/")
def root():
    return {"message": "🚀 Welcome to the FinSight AI Prediction API"}

@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):
    try:
        input_array = np.array(transaction.input)

        # ✅ Input length validation
        if input_array.shape[0] != EXPECTED_FEATURES:
            raise HTTPException(
                status_code=422,
                detail=f"Input must have exactly {EXPECTED_FEATURES} features. Received: {input_array.shape[0]}"
            )

        # Reshape to (1, 1, features)
        input_array = input_array.reshape((1, 1, EXPECTED_FEATURES))

        prediction = model.predict(input_array)
        confidence = float(prediction[0][0])
        fraud_flag = confidence > 0.5

        # ✅ Log input + prediction
        logging.info(f"Input: {transaction.input}")
        logging.info(f"Prediction: {confidence:.4f}, Fraud: {fraud_flag}")

        return PredictionOutput(
            fraud=fraud_flag,
            confidence=confidence,
            message="✅ Prediction successful"
        )

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
