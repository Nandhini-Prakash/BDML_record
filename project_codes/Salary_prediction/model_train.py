import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, median_absolute_error
import xgboost as xgb
import joblib
import json

# ------------------------------
# 1️⃣ Load Dataset
# ------------------------------
print("Loading dataset...")
df = pd.read_csv('data/salary.csv')
print(f"Dataset shape: {df.shape}")

target_col = 'salary_in_usd'
df = df.dropna(subset=[target_col])

# ------------------------------
# 2️⃣ Feature Encoding
# ------------------------------
categorical_cols = ['education_level', 'job_role', 'location']
numeric_cols = ['years_experience', 'skill_score']

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

feature_cols = [f'{col}_enc' for col in categorical_cols] + numeric_cols
X = df[feature_cols]
y = df[target_col].astype(float)

# Optional: log-transform target to reduce skew
y_log = np.log1p(y)

# ------------------------------
# 3️⃣ Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# ------------------------------
# 4️⃣ Train XGBoost
# ------------------------------
model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

# ------------------------------
# 5️⃣ Predictions
# ------------------------------
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

# ------------------------------
# 6️⃣ Evaluation
# ------------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)
explained_var = explained_variance_score(y_test_actual, y_pred)
med_ae = median_absolute_error(y_test_actual, y_pred)
mape = mean_absolute_percentage_error(y_test_actual, y_pred)

print("\n" + "="*50)
print("MODEL EVALUATION METRICS")
print("="*50)
print(f"MAE: ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²: {r2:.4f}")
print(f"Explained Variance: {explained_var:.4f}")
print(f"Median AE: ${med_ae:,.2f}")
print(f"MAPE: {mape:.2f}%")
print("="*50)

# ------------------------------
# 7️⃣ Save Model & Encoders
# ------------------------------
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/xgboost_salary_model.pkl')
joblib.dump(encoders, 'models/label_encoders.pkl')

feature_info = {
    'feature_columns': feature_cols,
    'categorical_columns': categorical_cols
}

with open('models/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print("\n✅ Model training complete and saved!")
