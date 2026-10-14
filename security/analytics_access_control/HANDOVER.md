# ST LUDAETUC CLOUD V1.0 — ANALYTICS ACCESS-CONTROL CORRECTION

DRAFT — NOT APPROVED. Independent-review candidate; not deployed.

## Outcome and availability decision

The candidate contains the legacy analytics API at an ASGI boundary. Unauthenticated requests are denied. Valid authenticated identities may obtain scoped access-status metadata, but all three legacy data routes return `503 analytics_scope_unavailable` before any handler or database query runs. This is an intentional containment release, **not a functional scoped analytics reader**.

It is unsafe to enable the shared-table legacy queries merely by adding a bearer token: the historical events and master_trading_log do not establish a verified account/environment security partition. Blocking these data routes is the smallest reliable correction that meets isolation without changing financial calculations, migrating storage, inferring account ownership or altering trading ingestion. There is no environment variable, wildcard permission or administrator bypass that re-enables them. Restoring authorised analytics reads requires a separately reviewed scoped reader and evidence-backed historical ownership mapping. Availability restoration is a named residual, not a hidden success claim.

Baseline: `2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c`.
Branch: `security/analytics-access-control-v1`.
The final exact commit is provided in the engineering handover; review the branch by that immutable SHA.

## Exact findings

| Finding | Evidence / severity | Disposition |
|---|---|---|
| Export exposes predictions without authentication | Confirmed bounded prior public GET, HTTP 200; internal identifiers and decision fields | All analytics paths intercepted before legacy handler |
| All three analytics routes lack route identity/permission checks | Verified source at baseline | Fail-closed service registry and explicit per-route permission |
| Strategy report aggregates all events and reads unscoped master view | Verified SQL | Route contained, no SQL executed |
| Export selects events without account/environment predicate | Verified SQL; legacy environment defaults to production | Route contained; production is never interpreted as live broker scope |
| Accuracy reads shared predictions and market payloads | Verified SQL | Route contained; hypothetical scoring unchanged and inaccessible |
| Caller limits and dates were insufficiently constrained | Export accepts unbounded limit; forgiving timestamps; report no window; accuracy caller bounds | Hard request caps; invalid/missing scope or dates rejected |
| Exception text can disclose database/schema/internal details | Export/report errors use str(exc); accuracy initial error interpolates exception | Legacy error generation cannot execute; middleware responses use fixed codes only |
| No analytics-specific access/denial audit | Verified source | Structured server-generated audit events emitted to stdout; operator must verify log capture/retention |
| Sensitive raw plan/internal identifiers in responses | Existing export final_trade_plan; report SELECT * bridge; accuracy scored details | No legacy data response, status metadata only |
| Complete historical account attribution unavailable | Prior raw DB route blocked; no population/account mapping | Never guess scope or call shared history isolated |

Only `/analytics/export` exposure was verified by the bounded public data probe. The vulnerabilities in the other two handlers are source-confirmed, not separate live exploitation tests. No further production payload was fetched for this correction.

## Production integration inspection and compatibility

Read-only Render inspection reconfirmed live deployment `dep-db2b0n0473hc739cejd0` at the baseline SHA. Service runs `uvicorn engine:app --host 0.0.0.0 --port $PORT`, root directory empty, build `pip install -r requirements.txt`, auto-deploy enabled on main, public web-service ingress. PostgreSQL's protected external allowlist is unchanged.

Baseline repository tree contains engine.py, app.py wrapper, shadow scorer, requirements and seven Pine sources; no AGENTS.md. Engine source declares exactly the three analytics data routes. No analytics call is present in the engine's four trading ingestion/execution routes. The app.py wrapper re-exports the same app, so it receives containment too. Shadow scorer is computation-only; no runtime analytics dependency established. Existing code comments explicitly identify external Claude research use. Previous GPTWork research evidence acquisition also used export/report. Exact inventory of external authorised research jobs, schedules and identities is **not independently complete**. No Make blueprint was available to establish other consumers.

Compatibility: `/evaluate`, `/market-data`, `/trade-opened`, `/trade-outcome`, `/health` and startup code remain unchanged. Trading/model/calculation AST is identical after removing the new import and middleware registration. Analytics callers will break intentionally: public requests become 401; valid scoped data requests become 503. No silent empty/zero response. HEAD/OPTIONS and mutation requests cannot bypass the boundary. CORS is not used as authentication. The OpenAPI schema remains available and discloses route shapes, not records; authenticated status is middleware-owned rather than advertised as a legacy data route.

Research-reader migration: inventory each operator-approved caller first; assign distinct expiring service key IDs and exact scope/permissions; migrate callers to Authorization header and explicit account_alias/system environment/broker environment; use access-status to test identity; stop treating 503 as an empty dataset. Keep legacy research evidence labelled provisional. Authorised readers cannot retrieve legacy data with this containment release. A reviewed scoped acquisition replacement or existing operator-controlled internal export is required before research read availability is restored. Do not share an operator key with an LLM or put credentials in URLs, Pine alerts, spreadsheets, GitHub or reports.

## Configuration and credential provisioning — operator only

No production credential or Render variable was created. Configuration is optional for containment: absent registry still denies everyone and protects all analytics. Registry is read on each protected request; invalid registry fails closed without stopping trading. The sole variable is `ANALYTICS_SERVICE_IDENTITIES_JSON` with strict shape:

```json
{
  "version": 1,
  "identities": [
    {
      "key_id": "research-reader-v1",
      "service_id": "approved-research-service",
      "token_sha256": "REPLACE_WITH_OPERATOR_GENERATED_64_LOWERCASE_HEX_DIGEST",
      "permissions": ["analytics:status", "analytics:export"],
      "scopes": [
        {"account_alias": "approved-practice-alias", "environment": "production", "broker_environment": "practice"}
      ],
      "expires_at": "REPLACE_WITH_EXPLICIT_FUTURE_UTC_TIMESTAMP"
    }
  ]
}
```

This example deliberately fails validation until provisioned. Operators generate at least 32 cryptographically random bytes and encode as unpadded URL-safe base64 (43+ characters), retain token only in the caller's approved secret store, and store SHA-256 of the token in the engine registry. Request header: `Authorization: Bearer <key_id>.<token>`. No cookie, query, forwarded identity or IP is accepted as authentication. Digest comparison uses hmac.compare_digest. IDs/aliases are bounded safe strings; no wildcard permission/scope is permitted. Expired or unknown keys fail 401. Duplicate identity/digest/config keys fail closed. Key rotation uses a separate key ID, approved overlap, then removal/revocation; no shared credential is necessary.

Environment is explicitly one of production/shadow/development/test; broker_environment is practice/live/none. These are separate exact-match labels, never a financial-reality assertion. Scope membership grants metadata access only until a separately reviewed data implementation exists.

Request query always includes account_alias, environment, broker_environment. Export/report additionally require explicit timezone-aware start/end. Dates form a positive half-open window of at most 31 days; end cannot exceed server now by more than five minutes. Row limit 1–1000. Accuracy caps since_hours 744, lookahead_minutes 1440, max_predictions 100. Query bytes max 2048, fields max 20, duplicate/unknown fields rejected. Configuration max 64 KiB, max 32 identities and 16 scopes per identity. These caps govern request acceptance; because data handlers are contained, this release executes zero SQL or row scans. Future data readers must enforce limits again in SQL, not reuse validation as proof of bounded query work.

## Auditing and disclosure

Every analytics decision includes generated request UUID, server timestamp, configured service identity where validated, approved scope where selected, known route or fixed unknown-route label, outcome and HTTP status. No token/digest, raw query, IP, broker account ID, user payload or exception string is logged. Headers enforce no-store, pragma no-cache, Vary Authorization and nosniff; request IDs are generated, never trusted from callers.

Structured audit lines are written/flushed under a lock to stdout, avoiding silent logger-level filtering. A failed audit write suppresses success and returns sanitized 503 audit_unavailable. A completely unavailable sink cannot preserve a durable denial record; operator alerting is required. Render capture, retention, durable export and operator access policy must be verified before production approval. This release does not create an audit database or alter logging configuration, and does not claim tamper-proof durable storage. Any future forwarding middleware must retain this boundary as the outermost analytics control.

## Executed tests

38 tests pass: 29 direct ASGI/security/source checks and 9 real FastAPI/TestClient integration tests. Full names/results are in test_results.txt. Tests exercise authenticated access-status, all legacy route containment, missing/wrong/expired identities, malformed/duplicate/oversized configuration, explicit permissions, account/system/broker scope denials, missing/invalid/timezone/date/future bounds, numeric/row/accuracy bounds, duplicate/unknown/oversized query fields, injection filters, encoded paths, slash variants, method and websocket bypass attempts, secret/error redaction, audit failures and unchanged trading passthrough.

AST equivalence test confirms engine functions/model code/startup remain identical to baseline; real integration tests use a separate FastAPI app with hostile data handlers and prove those handlers never execute. No engine import, startup DDL, production credentials, network request or database connection occurs in the test suite.

Test environment: Python 3.12; isolated fastapi 0.115.12, httpx 0.28.1 and installed dependency versions in test_environment.json. Existing production requirements are unchanged; framework version drift must be checked during approved staging validation. No PostgreSQL runtime test needed for this zero-SQL middleware; database privileges/migrations were not touched. Production load behaviour and durable audit capture are not validated by unit tests.

Reproduce from repository root in an isolated virtual environment:

```
python -m pip install fastapi==0.115.12 httpx==0.28.1
python -m unittest discover -s security/analytics_access_control -p 'test_*.py' -v
```

The AST comparison test's baseline fixture is included in baseline_engine_ast.sha256: it hashes baseline AST independently of comments/line offsets; no frozen branch checkout is required.

## Safe deployment and rollback plan — not executed

1. Independent security reviewer checks exact frozen SHA and availability trade-off; owner approves containment, reader outage/migration and audit retention requirements.
2. In isolated staging, use synthetic records and identities, production-compatible framework versions, verify middleware registration, import and route responses, log capture and normal trading-route behaviour. No actual financial import.
3. Operator performs authorised release/configuration change only after explicit approval. Main auto-deploy makes merge a production action: do not merge from this task. No deployment, key or Render change has been made.
4. Verify unauthenticated analytics 401, authorised status 200 when registry provisioned, authorised legacy reads 503, all trading paths healthy and no sensitive/error body leakage. Verify structured audit arrival and retention.
5. Rollback must preserve containment. Removing the registry disables identity acceptance but cannot reopen data routes; this is safe credential/config rollback. Reverting engine to the baseline reopens the confirmed exposure and is **unsafe** without an independently approved path-level block. Avoid whole-service suspension/network denial: it would disrupt trading ingress. Prefer a reviewed fix-forward containment release if middleware integration fails. Preserve security audit evidence through any operator-controlled rollback.

## Immediate containment options requiring operator approval

Deploy this reviewed deny-all legacy analytics boundary; or install an independently verified path-specific edge/reverse-proxy deny rule for /analytics while preserving trading routes, if the operator actually has a supported edge mechanism. Render metadata did not establish such a mechanism, so this is conditional, not an instruction to invent/configure one. Do not suspend the service, restrict all inbound traffic or weaken PostgreSQL networking. No immediate containment has been enacted; production exposure remains until an approved action.

## Independent review instructions and remaining residuals

Review the exact new SHA, compare only engine.py plus analytics_access_control.py and security/analytics_access_control/* against baseline. Verify no Finance/research/Chief branch or code changes. Attempt bypass through methods, encoded paths, duplicate headers/parameters, routes added beneath /analytics, expired/config-corrupt keys and unrelated trading routes. Confirm that configuration cannot unlock unsafe historical queries. Run all tests with no DATABASE_URL and validate on approved staging before deployment.

Named residuals: R1 legacy analytics availability intentionally blocked; R2 complete research-consumer inventory unavailable; R3 verified historical account/environment mapping unavailable; R4 durable log capture/retention requires operator evidence; R5 production framework/load validation not executed; R6 production remains uncorrected because deployment is prohibited. Rate limiting and sophisticated identity issuer integration are separate later hardening, not claimed implemented. No production credentials, deploy, merge, database migration or network weakening.

ANALYTICS_SECURITY_CORRECTION_READY_FOR_INDEPENDENT_REVIEW
