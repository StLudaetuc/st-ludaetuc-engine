# Runbook v2 — AUTHOR EVIDENCE; NO-GO, proposed operator procedures

This new runbook supersedes the operational guidance in DEPLOYMENT_RUNBOOK.md for this proposed candidate only; that reviewed original is retained unchanged. These procedures confer no deployment authority. Production and platform behaviour are UNPROVED until an authorised operator verifies them. Primary containment (a), fallback containment (b), and operational recovery (c) are distinct.

## Procedure (a): source and first-release gate

1. STOP while A25 or any mandatory acceptance row is unmet. Read release/FIRST_RELEASE_CONTAINMENT.md. Preverified builds alone do not protect an old unprotected instance retained after a failed first deployment.
2. A later named operator requires independent review and approval binding the exact immutable candidate SHA/tree, runtime lock, artifact digest, window, consumer acknowledgements/exceptions and stop criteria. Obtain the external attestation. No source branch name is sufficient authority.
3. Before any ref change, fetch/read approved objects without changing main, verify `git rev-parse APPROVED_SHA^{tree}` equals the attestation, and run `python release/verify_release.py APPROVED_SHA release/RELEASE_MANIFEST.json --attestation EXTERNAL.json --artifact APPLICATION.tar.gz`. Require exit 0. Require current main is an ancestor and unchanged from the approved before-state. Inspect complete diff and protected blobs.
4. A later approved release must be fast-forward-only to that exact SHA. No merge commit, squash, rebase, force push or merge queue substitution. Verify the tree BEFORE moving a ref. Main/ref changes are forbidden under this commission. Concurrent main movement or mismatch: STOP and obtain a fresh exact-source assessment.
5. After a future approved action, verify actual live source SHA/tree, artifact/lock/runtime inventory, command and auto-deploy state before behavioural acceptance. A returned live status is not proof of containment.

## Procedure (b): local fallback activation on one immutable artifact

1. Confirm the SAME frozen SHA/tree holds `engine.py`, primary middleware, `security/analytics_release/containment_fallback.py`, `fallback_app.py`, both package markers, runbook v2, full runtime hash lock and test lock. Do not activate fallback on the old primary-only tree.
2. Use a clean local process, synthetic data and no inherited database/identity/broker environment. Install exact Python 3.14.3 and the selected lock with `pip install --no-index --find-links WHEELHOUSE -r requirements.txt`. Install test lock separately if using TestClient.
3. Start primary with `python -m uvicorn app:app --host 127.0.0.1 --port 8000`; observe /health 200 and unauthenticated analytics 401 with `analytics_access` JSON audit. Stop the local process.
4. Change ONLY start command on the identical artifact to `python -m uvicorn security.analytics_release.fallback_app:app --host 127.0.0.1 --port 8000`. No ref/dependency change is part of fallback activation. The underlying engine must already be healthy. In a future platform environment, command changes have a deployment lifecycle that is UNPROVED here.
5. Probe GET /analytics/strategy-report, /analytics/export, /analytics/prediction-accuracy and /analytics/access-status: each 503, JSON error `analytics_containment_active`, `cache-control: no-store`, correlated x-request-id. HEAD has an empty body. OPTIONS is also denied. WebSocket sends close code 1008 BEFORE accept; no HTTP JSON body applies to WebSocket denial. /health stays 200.
6. For every denial expect exactly one structured `analytics_containment` JSON line on stderr, with generated request_id, fixed route_class, status (503 HTTP / 1008 WS) and UTC at. Request ID correlates to HTTP response. No headers/token/query/body/raw path is recorded. Logging transport errors preserve denial but cannot guarantee event delivery. Burst controls do not prove collector throughput or durable retention.
7. Verify no legacy analytics handler or SELECT execution in the approved matrix and valid positive controls. Supporting author tests here are not substitutes for the unavailable original fallback matrix/PG logs. STOP on any unexpected analytics 2xx, handler execution, SELECT or uncovered bypass. Never store a surprising data-bearing Production response.

## Procedure (c): operational failure and R-B

1. Fallback is containment only. Shared engine-import, dependency, database or trading failures require operational recovery; changing to fallback on the same broken artifact does not repair them.
2. The separate `release/r1-recovery-artefact-v1` composition has its own frozen SHA/tree, manifest and external artifact attestation. Its only supported launcher wraps the engine in mandatory ContainmentOnly. Do not choose bare 2ca3fa49 or c035f9d2 as a recovery target.
3. Independently verify the R-B digest/source/runtime and mandatory entrypoint before any separately authorised switch. Local author drills must inject real core import/dependency/trading/DB-connectivity faults and identify shared failures it cannot survive. No Production recovery is established; no DB/broker state recovery is supplied. A healthy /health response is insufficient proof of persistence/trading correctness.
4. Record failed target, restore target, source/tree, artifact hashes, start/ready time and first successful synthetic trading transaction in disposable evidence. Stop on data exposure, changed trading response or persistence mismatch. Timings here do not predict Production recovery. This candidate's PG acceptance remains incomplete.

## Procedure (c): mandatory checks after every rollback/recovery

Record actual auto-deploy state, pending config, active start command, live SHA/tree/artifact and installed dependency inventory. Compare to approved immutable records; do not assume a historical rollback restores configuration or preserves auto-deploy. Re-enabling auto-deploy may deploy a newer main. Platform behaviour is UNPROVED until separately authorised account-specific observation and staging rehearsal. No rollback to bare baseline/v1 is permitted.

## Procedure (a/b/c): separate operational readiness gate

`/health` is a PROPOSED future health-check path, not an original R1–R5 finding and not configured here. Verify startup/lifespan, passive persistence, trading response, durable audit collection/retention owner, consumers and F-04 disposition separately. Render PYTHON_VERSION would override .python-version; the operator must inspect it and actual Python/pip inventory under later authority. Do not read secret values into evidence.
