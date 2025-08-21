# Claim Fraud Prediction

An end-to-end machine learning project to predict fraudulent insurance claims.
This project covers the entire lifecycle — from **data preprocessing and model training** to **serving predictions via a FastAPI service** and **deployment on AWS ECS** using Docker.

---

## Features

* **Exploratory Data Analysis (EDA):** Data cleaning and preprocessing
* **Modeling:** Built and evaluated multiple ML models (Decision Tree, Random Forest, Bagging, etc.).
* **Experiment Tracking:** MLflow used for experiment tracking and model versioning.
* **Serving API:** FastAPI-based REST API for real-time predictions.
* **Deployment:** Dockerized application deployed on **AWS ECS (Fargate)**.

---

## Project Structure

```
claim-fraud-detection/
├── data/                  # Raw and processed data 
├── mlruns/                # MLflow experiment tracking 
├── models/                # Saved models 
├── src/
│   ├── serving/           # FastAPI service
│   │   ├── predict.py
│   │   ├── schemas.py
│   └── train/             # Training pipeline
│       ├── __init__.py
│       ├── train.py       # Training script
│       ├── main.py        # Training entrypoint
│       ├── preprocess.py  # Data preprocessing
├── Dockerfile             # Docker image for training/serving
├── requirements.txt       # Python dependencies
└── README.md
```

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the repository

```bash
git clone https://github.com/JayendranK/claim_fraud_prediction.git
cd claim_fraud_prediction
```

### 2️⃣ Setup environment

```bash
pip install -r requirements.txt
```

### 3️⃣ Run locally

```bash
uvicorn src.serving.predict:app --reload
```

API will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)


---

## 🛠️ Tech Stack

* **Python**: pandas, numpy, scikit-learn, joblib
* **Experiment Tracking**: MLflow
* **API**: FastAPI, Uvicorn
* **Containerization**: Docker
* **Deployment**: AWS ECS (Fargate)

---
