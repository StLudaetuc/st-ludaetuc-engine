from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from engine import TradingSignalEvaluationEngine

app = FastAPI(title="St Ludaetuc Manus Evaluation Engine")
engine = TradingSignalEvaluationEngine()

class AccountInfo(BaseModel):
    balance: float = Field(..., description="Current account balance")
    margin_available: float = Field(..., description="Available margin")

class FullInputPayload(BaseModel):
    payload: Dict[str, Any] = Field(..., description="Canonical St Ludaetuc trading payload")
    account: AccountInfo

@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}

@app.post("/evaluate")
async def evaluate_signal(full_input: FullInputPayload) -> Dict[str, Any]:
    try:
        # Use model_dump() for Pydantic v2
        decision = engine.evaluate(full_input.model_dump())
        return decision
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(exc)}")
