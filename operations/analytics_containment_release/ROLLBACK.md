# Containment-preserving recovery — DRAFT / NO-GO until assurance

Never roll back to `2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c` or `c035f9d2d74c3eb0d7f2dc1086f2e1095b3bea44`. Do not choose “previous successful deploy” by position. Current historical production artifacts are unprotected. A failed rollout retaining baseline means exposure continues.

## Explicit recovery decision table

| Trigger | Demonstrated local behavior | Permitted future response after specific approval |
|---|---|---|
| Primary registry misconfiguration | Missing/malformed configuration fails closed; trading bypass intact | Keep reviewed corrected runtime; separately approved registry removal only if indicated; never relax scopes or provision access to legacy routes |
| Inner analytics boundary bypass/exception | Unchanged outer fallback returns503 without calling inner analytics | Exact independently approved bundle `5f0d...`, proposed start `uvicorn security.analytics_release.fallback_app:app --host 0.0.0.0 --port $PORT`; start-command/artifact change needs explicit CEO approval and operator verification |
| Audit stdout failure | Primary returns503 audit_unavailable; no legacy data | Preserve containment; operator incident for evidence loss. Fallback has no structured analytics audit and is not an audit repair |
| Engine import/lifespan startup failure | Both normal engine and wrapped engine unavailable | Fallback does not fix it. No safe availability-preserving historical rollback demonstrated; NO-GO absent separately assured response and CEO-approved outage/disclosure trade-off |
| Trading-handler or dependency failure | Fallback reuses same engine/dependencies | Same limitation; reverting middleware alone cannot repair engine. No blind old-version rollback |
| Outer fallback failure | Synthetic request returns500; readiness/trading not restored | Incident; do not claim guaranteed containment/availability or substitute unassured ingress |
| Wrong SHA/new unreviewed main deploy | Invalidates assurance | Stop and hold; only independently reviewed approved artifact may replace it |

## Future fallback operator sequence

1. Validate incident facts without collecting analytics data; notify named CEO/on-call channel through independently authorised communication procedure.
2. Verify independent fallback acceptance, exact source/tree, available built artifact/runtime and explicit approval for start-command/artifact change. No assumption that source in Git implies files are in the currently deployed primary artifact: SHA346 does not contain fallback files, SHA5f0 does.
3. Capture no-secret before-state and live deploy ID. Confirm no concurrent deploy/pending config. Apply only separately approved operation. This task does not execute it.
4. Wait for terminal live state, verify exact source and proposed fallback command, `/health`, all data-route503 responses and passive trading continuity. Fallback failure retains NO-GO; do not automatically fall through to insecure old code.
5. Preserve request/access logs privately because fallback emits no analytics_access event. Record operator decisions and UTC times. Recheck auto-deploy state before authorising further main activity.

## Render artifact rollback caveats — current docs checked October18

Rollback reuses retained build and target start command, health path, instance count and environment values, while current environment-group contents and other configuration may differ. Platform runtime updates are not rolled back. Build retention is plan-dependent and was not verified for this workspace. A new deploy is created; external DB/messages/effects are not reverted. Dashboard rollback disables auto-deploy; API rollback does not. Exact-SHA manual deployment also differs by trigger method. Never use these generic platform features without verifying target containment and resulting auto-deploy state.

No public proxy/WAF or full-service maintenance barrier is independently verified. An operator-approved service outage might stop disclosure but also disrupt trading; it is not pre-authorised by this document and was not rehearsed on Render. If no acceptable safe recovery route closes the incident matrix, **production readiness remains NO-GO**.
