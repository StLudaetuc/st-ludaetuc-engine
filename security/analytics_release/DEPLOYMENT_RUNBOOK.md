# St Ludaetuc Cloud V1.0 — Analytics Containment Release Runbook

**STATUS: DRAFT / NOT AUTHORISED / NOT DEPLOYED.** This file is an operator procedure and decision record, not permission to act. No production change is approved by its existence.

## 1. Immutable release and operating context

- Intended primary release: `3467133b2208a09e1289651b5be0b45342695d52` (Analytics Security v1.1.1).
- Original unprotected baseline: `2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c` — **NEVER use as a rollback target**.
- Earlier v1 candidate: `c035f9d2d74c3eb0d7f2dc1086f2e1095b3bea44` — vulnerable under Uvicorn root_path; **NEVER use as rollback**.
- Render web service `srv-d71v0v7pm1nc73f60520`, workspace `tea-d71u0rf5gffc7383mh5g`, Frankfurt.
- Last read-only verified live deployment `dep-db2b0n0473hc739cejd0` points at unprotected baseline.
- Render branch `main`, auto-deploy enabled on commits. **A merge or upload to main is a production release.**
- Service build command `pip install -r requirements.txt`; configured start command `uvicorn engine:app --host 0.0.0.0 --port $PORT`.
- One instance, no dedicated health-check path configured at last observation.
- Original repository `StLudaetuc/st-ludaetuc-engine` is public; never store real secrets in GitHub, prompts, logs or this runbook.
- Primary security model intentionally denies all legacy analytics data queries even to authenticated callers. Do not provision identities to restore access.
- Independent security reviewer reported `ANALYTICS_SECURITY_V1_1_1_INDEPENDENT_REVIEW_PASS`. Retain its full signed/hashed evidence off the ephemeral reviewer scratchpad.
- Last production dependency/runtime parity, environment-variable names, audited analytics consumer inventory and durable audit retention remain unproved.

## 2. Change approval checkpoints

**Hold / no-go** unless the CEO explicitly approves the exact immutable release SHA, execution window, named operator, bounded post-deploy checks, and fallback procedure in a separate approval. The approval must state that merging `main` triggers Render auto-deploy. An approval to prepare these files is **not** deployment approval.

Before seeking that approval:
1. Preserve exact independent review report, reviewer test archive, verified archive hash, and author handover under private access controls. Do not add sensitive findings to the public repository.
2. Read-only capture: service/owner, current live deploy ID and SHA, configured build/start commands, current Python `python --version` if accessible, and installed dependency versions (`python -m pip freeze`) without exposing credentials. **Do not paste any environment-variable values.**
3. Verify only *names/presence* (not values) of `UVICORN_ROOT_PATH`, other `UVICORN_*` variables and `ANALYTICS_SERVICE_IDENTITIES_JSON`, through approved operator tooling. Absence is not inferred from the start command or this connector's inability to read env vars.
4. Independently establish compatibility with actual runtime (build evidence suggests CPython 3.14; test in a disposable environment or document a version pin/configuration change requiring separate review). `requirements.txt` has unpinned dependencies.
5. Inventory analytics consumers (Make, scheduled jobs, connected research agents, user scripts). Notify owners that data endpoints will fail closed with 401/503. Require explicit owner acknowledgement or CEO-authorised documented exception. **Never interpret 503 as an empty dataset**.
6. Verify log capture and audit retention/export approach; record its limitations. Confirm deployment/incident monitoring owner, and which provider metrics/log views they can read.
7. Independently review and *test* the fallback below in a disposable environment. The fallback is **not yet assured for production**.
8. Record pre-release traffic and error-rate baselines for `/market-data`, `/evaluate`, `/trade-opened`, `/trade-outcome`, and `/health`; request logs do not establish database writes.
9. Freeze a reproducible release artifact and record the actual built runtime. If provider resolves different dependencies, fail the release gate unless compatibility evidence covers those versions.

## 3. Primary release procedure (CEO-approved operator only)

1. Check no unapproved main/branch drift since the frozen candidate; compare exact Git trees and ensure no unrelated code can enter the merge. Confirm service still auto-deploys `main`.
2. Obtain explicit CEO authorisation **before** any main branch push/merge, branch setting change, manual deploy or environment update.
3. Record before-state: UTC time, live SHA, deploy ID, start/build commands, read-only dependency inventory, backup/fallback review SHA and reviewer evidence.
4. Publish only the CEO-approved reviewed release by the separately approved Git workflow; do not use GitHub's browser upload to replace `engine.py` without review. Allow Render's normal auto-deploy to complete. Do not also call trigger_deploy after an auto-deploy.
5. Verify Render shows live **exact approved commit** and successful startup. If not, stop. Do not use older unprotected versions as rollback.
6. Using **non-sensitive, bounded probes only**, assert all three analytics data routes without credentials return 401: `/analytics/export`, `/analytics/strategy-report`, `/analytics/prediction-accuracy`. Include trailing slash / double-slash variants if safe; do not request rows or supply real tokens.
7. With registry intentionally unset, malformed dummy authorization must never yield data; response may be a 401 or fail-closed 503 as specified. Never generate/provision real identities for containment release.
8. Confirm `/health` 200 and `events_db_configured` true, and watch existing trading ingestion HTTP 5xx/latency and independently authorised DB evidence for unchanged ingestion. Do not emit fake production trading signals.
9. Check structured `analytics_access` audit events appear for denial probes; capture their timestamp/request ID, with retention and redaction verified. If any analytics data route responds 200 or invokes a legacy handler, treat as a security incident.
10. Maintain incident watch during and after rollout; retain objective artefacts: UTC timestamps, approved/live SHAs, Render deploy IDs, no-secret probe results, changes in metrics and audit evidence.

## 4. Fallback choices — NO AUTOMATIC ROLLBACK

**Never click Render's “Rollback to previous deploy”** unless the target has been independently verified to contain effective analytics protection. All currently known previous deployments lack protection.

### Fallback A — preferred if only identity configuration causes trouble
If the corrected v1.1.1 service is healthy and only analytics-identity provisioning causes errors, leave/unset `ANALYTICS_SERVICE_IDENTITIES_JSON` under a **separately authorised** configuration change. The protected analytics paths then fail closed, with trading logic unchanged. Do not touch unrelated secrets or env values. This does not repair a trading outage.

### Fallback B — separately reviewed containment-only entrypoint
This release-preparation branch includes:
- `security/analytics_release/containment_fallback.py`: independent outer ASGI wrapper unconditionally returns 503 for resolved analytics routes, 1008 for protected WebSockets.
- `security/analytics_release/fallback_app.py`: imports the existing engine app and applies the outer wrapper.
- `security/analytics_release/test_containment_fallback.py`: standalone synthetic tests.

Proposed **alternative** start command (NOT CONFIGURED):
`uvicorn security.analytics_release.fallback_app:app --host 0.0.0.0 --port $PORT`

This option needs independent security + trading compatibility assurance on a staged/disposable build, an explicitly approved artifact including the fallback files, and a separate CEO-approved Render start-command or release change. The outer wrapper adds a second barrier while reusing the already corrected engine. It is not an independent working replacement for a broken engine: if engine startup or non-analytics routes fail, it may fail too. No promise of recovery for catastrophic runtime failures is made.

### Fallback C — emergency deny at a separately controlled ingress boundary
A verified front-door proxy/WAF rule may deny `/analytics` and all descendant routes while allowing all trading endpoints, **only if** architecture confirms that every access path is covered and encoded/rewritten paths cannot bypass it. No such ingress security mechanism is currently configured or independently verified. Do not assume this exists.

**If all candidate fallback routes are unavailable or unreviewed: release NO-GO.** A documented idea is not a proven rollback. Escalate any incident to CEO/operator and contain disclosure without falsely claiming continued trading availability.

## 5. Failure triggers and stop conditions

- Any unauthenticated 200 or data-bearing response on an analytics endpoint: STOP; security incident; verify boundary and activate only an independently assured containment-preserving response.
- Service fails startup / health, or trading ingress/evaluate degradation above agreed thresholds: stop progression; use *approved* containment-preserving recovery route only.
- Actual resolved Python/package versions outside the independently tested matrix: HOLD pending compatibility check.
- Cannot verify current SHA, safe fallback, access to monitoring, or owner acknowledgement for critical analytics consumers: HOLD.
- Do not relax DB IP allowlist, expose `DATABASE_URL`, disable audit, or change trading credentials to resolve an analytics incident.

## 6. Post-release evidence and sign-off

Operator creates a **private** evidence record containing: CEO approval ID, deployed and fallback approved SHA/tree, actual Render deploy ID, build/runtime versions, all no-secret probes and responses, health, trading traffic/DB continuity evidence, audit-line capture and retention, fallback drill, incident window, open residuals, and operator/CEO sign-off.

Update the programme tracker **only after** independently verifying the deployment and with an appropriately scoped record-change authorisation. This branch and runbook confer none.

## 7. Outstanding security risks (separate workstream)

- `/evaluate`, `/market-data`, `/trade-opened`, `/trade-outcome` have pre-existing unauthenticated ingestion; analytics containment does not repair this.
- Public repo contains source/operational metadata; independent scanner found no active secrets in sampled reachable history, but provider secret-scanning alerts/forks/PR refs were not examined.
- No verified historical account/environment scoping. Legacy analytics are intentionally disabled; restoring them requires a separately authorised scoped reader.
- Audit durability/rate limits and complete analytics consumer inventory remain unproved.
- The fallback on this branch has not been independently accepted or deployed.

## Release gate

**Current: HOLD / NOT DEPLOYED.** Advancing this runbook requires independent fallback/runtime testing and an explicit CEO release decision.
