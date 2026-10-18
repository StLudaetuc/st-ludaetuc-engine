# Monitoring, audit and persistence procedure — PROPOSAL

Live observations and source references are in the separate private bundle. This public document contains reusable connector procedures only. No alerts, subscriptions, credentials, infrastructure or SQL connections were created. No production analytics fetch was made.

## Collection contract

Record UTC start/end, deployID/SHA/tree, entrypoint mode, named collector, source/query, page bounds/cursors, completeness and failures. Use equal-duration comparable trading periods before/after; do not extrapolate a five-minute sample. Empty result or unavailable source is UNKNOWN, never zero. Preserve actual denominator and boundary convention [start,end). Snapshot metrics buckets can include an end timestamp; never equate their sums to application access-line counts without matching populations/bounds. Deduplicate log records by provider logID, not identical message text (two real requests may match).

## Exact future connector procedures — not commands executed by this file

Use already-authorised workspace/service IDs from private deployment record. Do not list environment values or connect to Postgres via the assistant.

1. `get_service(serviceId=APPROVED_SERVICE, workspaceId=APPROVED_WORKSPACE)`; allowlist branch,autoDeploy,trigger,start/build command,region,health path,instance count. Capture `list_deploys(...,limit=5)` and identify actual liveSHA/deployID.
2. `list_logs(resource=[SERVICE],workspaceId=WORKSPACE,startTime=START_UTC,endTime=END_UTC,direction="forward",limit=100)` without type/path filters first. Page with returned nextStartTime/nextEndTime until hasMore=false. Fail incomplete on cap, repeated cursor, error or no-progress. Broader application logs can contain sensitive details: process privately, keep only allowed aggregates for public outputs.
3. Parse only strict Uvicorn access lines for method/route/status. Strip queries; retain no IP/header/body. Provider request-log filters type=request,path,statusCode apply to provider request logs and may miss application-emitted access logs. A filtered empty page does not override a populated broad query.
4. Analytics discovery search: `list_logs(...,text=["/analytics/"],startTime=START,endTime=END,limit=100,direction="forward")`, fully page, then parse exact data routes. These requests prove transport activity, not caller identity, successful research, auth status or response contents. Search literal/normalised variants separately in an approved investigation.
5. Metrics: `get_metrics(resourceId=SERVICE,workspaceId=WORKSPACE,metricTypes=["http_request_count"],aggregateHttpRequestCountsBy="statusCode",httpPath=ROUTE,startTime=START,endTime=END,resolution=60)`. Separately request `metricTypes=["http_latency"],httpPath=ROUTE,httpLatencyQuantile=0.95` and retain units/labels. Do not average per-bucket quantiles into a whole-periodp95. Empty arrays remain unknown.
6. Startup/deploy: `list_events(serviceId=SERVICE,workspaceId=WORKSPACE,startTime=START,endTime=END,eventTypes=["build_ended","deploy_ended","server_failed","server_restarted"],limit=100)`; page. Query build/start logs privately and correlate first error with deployID. A failed deployment may retain unprotected old instances.
7. Primary audit: privately search `text=["analytics_access"]`; parse JSON, allowlist route/status/result/at/request_id, avoid raw scope/service identity in public reports. Correlate an approved denial probe's X-Request-ID with app event. Provider Rndr-Id and app X-Request-ID are distinct unless explicit evidence links them; do not assume equality.

## Measures and decisions

| Measure | Definition / required interpretation |
|---|---|
| Analytics denial | Count primary401 unauthorized, scoped503 analytics_scope_unavailable, configuration/audit503 separately. Fallback all analytics503 analytics_containment_active, including access-status.401/503 never means emptydataset. Denial is not a trading failure. |
| Disclosure | Any2xx on the three data routes triggers incident. Authenticated access-status200 is not data disclosure. Redirect/variant/body findings need separately approved bounded verification. Never read/publish unexpected data bodies. Missing audit cannot prove no handler executed. |
| Trading failures | For each /evaluate,/market-data,/trade-opened,/trade-outcome:5xx / total observed HTTP responses,4xx separately, latency labeled by source. Retries counted as requests, not unique trades. HTTP200 is not a fill or persistence guarantee. |
| TradingView webhook health | /market-data2xx versus4xx/422/5xx with total/window. Literal-placeholder/NaN issues need separately bounded non-secret evidence, not assumptions from status alone. TradingView alert-delivery history remains inaccessible unless owner supplies it. No claim of complete symbol/timeframe coverage. |
| Startup/readiness | Actual liveSHA and no unresolved startup failure; TCP/health200 insufficient for DB persistence. events_db_configured is presence only. |
| Audit continuity | Probe correlation, collection pages, no gaps/export failures; fallback expected loss of primary audit explicitly accepted. No dropped-lines guarantee from stdout. |

Proposed stop criteria: immediate incident for data-route2xx, wrongSHA/tree or sensitive disclosure; hold for untested runtime, missing critical audit coverage, failed startup or unavailable recovery. Numerical trading failure/latency thresholds and allowed observation gaps require CEO/operator selection from representative baseline. No thresholds were silently ratified.

## Durable audit preservation — future authorised operator

Before activation, name collector and approved private destination, retention period, access roster and deletion owner. Start collection before change; capture pre/post windows and first normal trading/EOD interval; use complete pages and explicit gaps. Save raw bounded originals privately after sensitive-data review, plus redacted aggregates, SHA256 manifest, UTC collection times and deploy identity. Read back destination bytes and compare hashes; inspect item/parent ACL inheritance. A local service filesystem is ephemeral. Log streams/alerts must not be assumed configured. Current official docs describe plan-dependent7/14/30-day retention and a6000-line/minute per-instance application-log limit; dropped lines cannot be recovered by merely streaming. Workspace-specific retention/exports remain unproved until operator verifies plan/configuration.

## Persistence check — authorised read-only operator only; NOT EXECUTED

Use an existing approved SELECT-only identity/internal host through its normal secure client; do not grant new privileges, weaken networking, request credentials or use writer Web Shell. Run the supplied `persistence_check.sql` only after validating database/schema/window identity. Start a read-only transaction with bounded statement timeout. Export aggregates only. Table names/columns are source-derived; operator verifies actual schema first. Missing privileges/schema/rows return UNPROVED, not permission to modify production.

Compare event receipt/update counts, type/status and timestamps to passive ingestion evidence. Current events are mutable on conflict; event counts cannot prove immutable receipt populations, exact replay counts or successful fills. A nonzero count shows selected records present, not complete transaction/account coverage. No P&L or account identifiers are exported.
