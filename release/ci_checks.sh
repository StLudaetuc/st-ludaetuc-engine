#!/bin/sh
set -eu
# No deployment, Render credentials, broker/DB calls or write permissions.
python -I release/runtime_preflight.py --phase build
python -m unittest discover -s security/analytics_access_control -p 'test_*.py' -v
python -m unittest security.analytics_release.test_containment_fallback release.test_candidate release.local_probes operations.analytics_containment_release.test_tools operations.analytics_release_evidence_closure.test_evidence_checks operations.a25_rc2.test_controls -v
python operations/analytics_containment_release/rehearsal.py
python release/verify_release.py "$(git rev-parse HEAD)" release/RELEASE_MANIFEST.json --source-only
python -m pip check
