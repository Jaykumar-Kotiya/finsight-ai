# FinSight AI – Real-Time Financial Risk and Anomaly Detection

> A full-stack, end-to-end AI project that detects financial fraud and regulatory risks using deep learning, NLP, and MLOps.

---

##  Project Overview

FinSight AI is a production-grade platform designed to:
- Detect anomalies in financial transactions using CNN + LSTM
- Extract and classify risk information from financial reports using NLP
- Summarize reports using GPT-based generative AI
- Serve predictions via an API (FastAPI)
- Visualize insights through a dashboard (Streamlit or Tableau)
- Maintain versioning and deployment pipelines using MLflow and Docker

---

##  Tech Stack

| Category         | Tools/Frameworks                             |
|------------------|-----------------------------------------------|
| Languages        | Python, SQL                                   |
| ML/DL            | Scikit-learn, TensorFlow, PyTorch, XGBoost    |
| NLP & Gen AI     | Hugging Face Transformers, spaCy, OpenAI GPT  |
| Dashboard        | Streamlit, Tableau                            |
| APIs             | FastAPI, REST                                 |
| Cloud            | AWS, GCP                                      |
| MLOps            | MLflow, Docker, GitHub Actions                |

---

##  Project Structure

```bash
finsight-ai/
├── data/               # Cleaned datasets
├── raw_data/           # Raw data before preprocessing
├── notebooks/          # Jupyter notebooks for EDA and modeling
├── src/                # Core source code (preprocessing, modeling)
├── models/             # Saved model binaries
├── api/                # FastAPI app for serving models
├── dashboards/         # Streamlit or Tableau dashboards
├── deployment/         # Dockerfiles, CI/CD configs
├── mlops/              # MLflow tracking, DVC, pipelines
├── tests/              # Unit and integration tests
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── main.py             # Entrypoint (optional)

---

##🧪 Work in Progress (WIP) Modules

## ✅ GitHub Repo & Folder Structure
## ✅ Module 1: CNN+LSTM Anomaly Detection
## 🔜 Module 2: NLP Risk Classifier
## 🔜 Module 3: GPT-4 Financial Summary
## ✅ Module 4: FastAPI Model Deployment
## 🔜 Module 5: Streamlit / Tableau Dashboard
## 🔜 Module 6: MLflow Logging & MLOps

--------------------------------------------------------------------------

✅ Module 1: CNN+LSTM Anomaly Detection

This module builds a hybrid **Convolutional Neural Network + LSTM model** for detecting fraudulent transactions in highly imbalanced datasets.

###  Highlights

- 📊 Based on the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 📉 Balanced the dataset using undersampling to address class imbalance
- 🧠 Combined **1D CNN** (for feature extraction) + **LSTM** (for temporal/sequential learning)
- 📈 Achieved high accuracy (>99.9%) on test data
- 💾 Model saved as `cnn_lstm_anomaly_detector.h5` in `models/`

---

### 📁 Files Involved

- `notebooks/anomaly_detection_cnn_lstm.ipynb`
- `models/cnn_lstm_anomaly_detector.h5`

---

### ✅ Model Architecture Overview

Input → Conv1D → MaxPooling1D → LSTM → Dense → Output (Sigmoid)

Input Shape: (1, 29) → reshaped for (1, 1, 29)

Output: Binary label (fraud: 1 or not: 0)

Key Metrics

Metric	- Value
Accuracy - 99.91%
Loss - ~0.01
Optimizer - Adam
Loss Function - Binary Crossentropy

--------------------------------------------------------------------------

## ✅ Module 4: FastAPI Model Deployment

This module wraps the trained CNN+LSTM anomaly detection model into a RESTful API using FastAPI. It supports fraud predictions via a /predict endpoint and includes:

-> Input validation using pydantic

-> Input length enforcement (29 features)

-> JSON response with fraud flag and confidence score

-> Request/response logging to prediction_logs.log

-> Postman-tested and Swagger-documented

-> Runs via uvicorn for dev and production readiness

Project Structure
api/
├── main.py               ← FastAPI app code
├── prediction_logs.log   ← Input + prediction log
models/
├── cnn_lstm_anomaly_detector.h5
requirements.txt
README.md

How to Run the API Locally
1. Activate your virtual environment:
source venv/bin/activate

2. Run the FastAPI server:
uvicorn api.main:app --reload

3. Visit Swagger UI:
http://127.0.0.1:8000/docs

## API Endpoints

POST /predict
Make a real-time fraud prediction for a single transaction.

Request Body:

{
  "input": [
    0.1, -1.2, 2.5, 0.3, -0.7, 1.1, 0.0, 0.4, 1.5, 0.3,
    -0.9, 0.1, 0.2, 0.0, -1.2, 0.7, 0.9, 0.3, 0.2, -0.1,
    -1.4, -0.5, 0.6, 0.0, 0.8, -0.6, 0.7, 0.4, -0.2
  ]
}
Response:

{
  "fraud": false,
  "confidence": 0.0421,
  "message": "✅ Prediction successful"
}

POST /predict_batch
Send a batch of transactions (from a CSV) and receive fraud predictions.

Request Body:
{
  "input": [
    [0.1, -1.2, ..., -0.2],
    [0.3, -0.9, ..., 1.2]
  ]
}
Response:
{
  "predictions": [0, 1, 0, 0, 1]
}

## Files Involved
api/
├── main.py               ← FastAPI app code
├── prediction_logs.log   ← Input + prediction log
models/
├── cnn_lstm_anomaly_detector.h5
requirements.txt
README.md

## Logs
All predictions are logged in:
api/prediction_logs.log


Dependencies Used
fastapi

uvicorn

tensorflow

numpy

pydantic

All included in requirements.txt

## ✅ Status
FastAPI deployment is complete, live-tested with Postman & Swagger, and ready for production.

## 🔜 Module 2: NLP Risk Classifier
## 🔜 Module 3: GPT-4 Financial Summary
## 🔜 Module 5: Streamlit / Tableau Dashboard
## 🔜 Module 6: MLflow Logging & MLOps

---

## Author
Jaykumar Kotiya
Data Scientist | Machine Learning & NLP Enthusiast
LinkedIn - https://www.linkedin.com/in/jay-kotiya/
GitHub - https://github.com/Jaykumar-Kotiya

---

## ⭐ Star This Repo
If you find this project helpful or interesting, please consider giving it a ⭐ on GitHub! It helps others discover it.
