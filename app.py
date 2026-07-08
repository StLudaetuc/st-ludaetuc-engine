"""
app.py

Compatibility wrapper for Render.

The FastAPI application is defined in engine.py.
This file simply re-exports it so existing Render deployments
using "uvicorn app:app" continue to work.
"""

from engine import (
    app,
    engine,
    TradingSignalEvaluationEngine,
    ENGINE_VERSION,
    RULESET_VERSION,
)

__all__ = [
    "app",
    "engine",
    "TradingSignalEvaluationEngine",
    "ENGINE_VERSION",
    "RULESET_VERSION",
]
