from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.schemas import EmployeeInput, SalaryPrediction, HealthResponse
from app.model import load_model, get_predictor, predict

# Startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading model...")
    load_model()
    yield
    print("🛑 Shutting down...")

app = FastAPI(
    title="Salary Prediction API",
    version="1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", response_model=HealthResponse)
async def root():
    predictor = get_predictor()
    return {
        "status": "online",
        "is_model_loaded": predictor is not None,
        "version": "1.0"
    }

@app.post("/predict", response_model=SalaryPrediction)
async def predict_salary(employee: EmployeeInput):
    try:
        result = predict(employee.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
