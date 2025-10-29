import joblib
import json
import pandas as pd
import numpy as np

# Global predictor and encoders
predictor = None
encoders = None
feature_info = None

def load_model():
    """Load the trained XGBoost model and encoders"""
    global predictor, encoders, feature_info
    predictor = joblib.load('models/xgboost_salary_model.pkl')
    encoders = joblib.load('models/label_encoders.pkl')
    
    with open('models/feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    print("✅ Model and encoders loaded successfully!")
    return predictor

def get_predictor():
    return predictor

def preprocess_input(data: dict) -> pd.DataFrame:
    """Prepare model-ready input from raw data"""
    df = {}
    
    # Encode categorical columns
    for col in feature_info['categorical_columns']:
        df[f'{col}_enc'] = encoders[col].transform([data[col]])[0]
    
    # Add numeric columns
    for col in ['years_experience', 'skill_score']:
        df[col] = data[col]
    
    return pd.DataFrame([df])

def predict(data: dict) -> dict:
    """Make prediction"""
    if predictor is None:
        raise ValueError("Model not loaded")
    
    X = preprocess_input(data)
    y_pred_log = predictor.predict(X)
    y_pred = float(np.expm1(y_pred_log[0]))  # Convert back from log
    
    margin = y_pred * 0.10
    return {
        'predicted_salary': round(y_pred, 2),
        'confidence_range': {
            'lower_bound': round(y_pred - margin, 2),
            'upper_bound': round(y_pred + margin, 2)
        },
        'input_summary': data
    }
