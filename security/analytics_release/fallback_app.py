"""Opt-in fallback Uvicorn entrypoint.

Not configured or deployed. If independently accepted and CEO-authorised, use:
    uvicorn security.analytics_release.fallback_app:app --host 0.0.0.0 --port $PORT

Imports the unchanged trading engine application, with an additional outer
unconditional-denial boundary for analytics.
"""
from engine import app as trading_app
from security.analytics_release.containment_fallback import ContainmentOnly

app = ContainmentOnly(trading_app)
