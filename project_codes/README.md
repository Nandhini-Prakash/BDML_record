# 🚀 AI-Powered Salary Estimation System

A production-ready machine learning system that predicts employee salaries using **XGBoost**, deployed as a **REST API** with **FastAPI** and containerized using **Docker**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Frontend Integration](#frontend-integration)
- [Testing](#testing)

---

## 🎯 Overview

This project demonstrates an end-to-end machine learning pipeline for salary prediction based on employee attributes such as:

- **Years of Experience** (0-50 years)
- **Skill Score** (1-10 scale)
- **Education Level** (Bachelor/Master/PhD)
- **Job Role** (Software Engineer, Data Scientist, ML Engineer, DevOps, Backend Developer)
- **Location Tier** (Tier1/Tier2/Tier3 cities)

The system uses **XGBoost Regression** for accurate predictions and exposes the model through a RESTful API for real-time inference.

---

## 📁 Project Structure

The project is organized into **two main folders**:

### **Backend: `SALARY_PREDICTION/`**

```
SALARY_PREDICTION/
│
├── app/
│   ├── main.py              # FastAPI application entrypoint
│   ├── model.py             # Model loading & prediction logic
│   └── schemas.py           # Pydantic validation schemas
│
├── data/
│   └── salary_data.csv      # Training dataset (from Google Drive)
│
├── models/
│   ├── xgboost_salary_model.pkl   # Trained XGBoost model
│   ├── label_encoders.pkl         # Label encoders for categorical features
│   └── feature_info.json          # Feature metadata
│
├── requirements.txt         # Python dependencies
├── model_train.py          # Model training script
├── Dockerfile              # Docker configuration
└── .dockerignore           # Docker ignore file
```

### **Frontend: `SALARY_PREDICTION_frontend/`**

```
SALARY_PREDICTION_frontend/
│
└── frontend/
    └── index.html          # Web-based user interface
```

---

## ✨ Features

✅ **High-Performance ML Model**: XGBoost with optimized hyperparameters (R² > 0.95)  
✅ **RESTful API**: FastAPI with automatic Swagger/ReDoc documentation  
✅ **Input Validation**: Pydantic schemas for robust data validation  
✅ **Docker Ready**: Containerized for platform-independent deployment  
✅ **Interactive Frontend**: Beautiful web UI for easy testing  
✅ **Health Monitoring**: Built-in health check endpoints  
✅ **Feature Importance**: Model interpretability insights  

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **ML Framework** | XGBoost 2.0.3, Scikit-learn |
| **API Framework** | FastAPI 0.104.1 |
| **Web Server** | Uvicorn |
| **Containerization** | Docker |
| **Data Processing** | Pandas, NumPy |
| **Validation** | Pydantic |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Language** | Python 3.10+ |

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Docker (optional, for containerization)

---

### Step 1: Setup Backend

#### 1.1 Navigate to Backend Folder

```bash
cd SALARY_PREDICTION
```

#### 1.2 Create Virtual Environment

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### 1.3 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 1.4 Train the Model

```bash
python model_train.py
```

---

## 🚀 Usage

### Running the Backend API

#### Option 1: Local Development

```bash
cd SALARY_PREDICTION/app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

### Accessing the API

Once running, the API is available at:

- **Base URL**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## 📡 API Documentation

### Available Endpoints

#### 1️⃣ **Health Check**

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

#### 2️⃣ **Predict Salary**

```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "years_experience": 5.5,
  "skill_score": 8,
  "education_level": "Master",
  "job_role": "Data Scientist",
  "location": "Tier1"
}
```

**Response:**
```json
{
  "predicted_salary": 95000.50,
  "confidence_range": {
    "lower_bound": 85500.45,
    "upper_bound": 104500.55
  },
  "input_summary": {
    "years_experience": 5.5,
    "skill_score": 8,
    "education_level": "Master",
    "job_role": "Data Scientist",
    "location": "Tier1"
  }
}
```

**Validation Rules:**
- `years_experience`: 0-50 (float)
- `skill_score`: 1-10 (integer)
- `education_level`: "Bachelor", "Master", or "PhD"
- `job_role`: "Software Engineer", "Data Scientist", "ML Engineer", "DevOps", "Backend Developer"
- `location`: "Tier1", "Tier2", "Tier3"

---

#### 3️⃣ **Feature Importance**

```http
GET /model/feature-importance
```

**Response:**
```json
{
  "feature_importance": {
    "Years of Experience": 0.35,
    "Job Role": 0.25,
    "Education Level": 0.20,
    "Location": 0.12,
    "Skill Score": 0.08
  },
  "description": "Relative importance of each feature in salary prediction"
}
```

---

#### 4️⃣ **Model Information**

```http
GET /model/info
```

**Response:**
```json
{
  "model_type": "XGBoost Regressor",
  "supported_education_levels": ["Bachelor", "Master", "PhD"],
  "supported_job_roles": [
    "Software Engineer",
    "Data Scientist",
    "ML Engineer",
    "DevOps",
    "Backend Developer"
  ],
  "supported_locations": ["Tier1", "Tier2", "Tier3"]
}
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
cd SALARY_PREDICTION
docker build -t salary-prediction-api:latest .
```

### Run Container

```bash
docker run -d \
  --name salary-api \
  -p 8000:8000 \
  salary-prediction-api:latest
```

### Check Container Status

```bash
docker ps
docker logs salary-api
```

### Stop Container

```bash
docker stop salary-api
docker rm salary-api
```

---

## 🌐 Frontend Integration

### Step 1: Setup Frontend

```bash
cd SALARY_PREDICTION_frontend/frontend
```

### Step 2: Update API URL (if needed)

Open `index.html` and verify the API URL:

```javascript
const API_URL = 'http://localhost:8000';  // Update if using different host/port
```

### Step 3: Run Frontend

**Option A: Direct File Open**
```bash
# Mac
open index.html

# Windows
start index.html

# Linux
xdg-open index.html
```

**Option B: Python HTTP Server (Recommended)**
```bash
python -m http.server 3000
```

Then visit: **http://localhost:3000/index.html**

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **R² Score** | 0.9420 |
| **Mean Absolute Error (MAE)** | ~$5,072.83 |
| **Root Mean Squared Error (RMSE)** | ~$6,207.39 |
| **Training Samples** | 1,000 |
| **Features Used** | 5 |

---

## 🔒 Security Considerations

✅ Input validation with Pydantic  
✅ CORS middleware configured  
✅ No sensitive data in responses  
✅ Health check endpoints  
⚠️ **TODO**: Add authentication (JWT/OAuth2)  
⚠️ **TODO**: Implement rate limiting  

---

## 🚧 Future Enhancements

- [ ] Add user authentication
- [ ] Implement API rate limiting
- [ ] Add database for logging predictions
- [ ] Create batch prediction endpoint
- [ ] Add model versioning
- [ ] Implement A/B testing
- [ ] Add monitoring with Prometheus/Grafana
- [ ] Create CI/CD pipeline

---

## 👨‍💻 Project Information

**Project Name:** AI-Powered Salary Estimation System  
**Version:** 1.0.0  
**Year:** 2025

**Key Technologies:**
- Machine Learning: XGBoost, Scikit-learn
- Backend: FastAPI, Python 3.10
- Frontend: HTML5, CSS3, JavaScript
- Deployment: Docker

---

## 🎯 Quick Start Summary

```bash
# 1. Download dataset from Google Drive to SALARY_PREDICTION/data/

# 2. Setup backend
cd SALARY_PREDICTION
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt
python model_train.py

# 3. Run API
uvicorn app.main:app --reload

# 4. Open frontend
cd ../SALARY_PREDICTION_frontend/frontend
python -m http.server 3000

# 5. Test
Visit: http://localhost:3000/index.html
API Docs: http://localhost:8000/docs
```

---

**🎉 Your AI Salary Prediction System is Ready!**
