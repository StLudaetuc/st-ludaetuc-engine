"""Mandatory supported R-B entrypoint; containment only, not shared-failure repair."""
from engine import app as trading_app
from security.analytics_release.containment_fallback import ContainmentOnly
app = ContainmentOnly(trading_app)
