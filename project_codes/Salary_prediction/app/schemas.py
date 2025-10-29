from pydantic import BaseModel, Field
from typing import Literal

class EmployeeInput(BaseModel):
    years_experience: float = Field(..., ge=0, le=50)
    skill_score: int = Field(..., ge=1, le=10)
    education_level: Literal['Bachelor', 'Master', 'PhD']
    job_role: Literal['Software Engineer', 'Data Scientist', 'ML Engineer', 'DevOps', 'Backend Developer']
    location: Literal['Tier1', 'Tier2', 'Tier3']

class SalaryPrediction(BaseModel):
    predicted_salary: float
    confidence_range: dict
    input_summary: dict

class HealthResponse(BaseModel):
    status: str
    is_model_loaded: bool
    version: str
