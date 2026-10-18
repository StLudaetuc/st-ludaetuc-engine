# Monitoring and private audit preservation — DRAFT

## Available capabilities and actual checks

Connector exposes list_logs (time windows, pagination, text/type/path/status filters), list_events (build/deploy/restarts/failures) and get_metrics (HTTP count/status and path latency). Service metadata and live deploy reads succeeded. Historical build logs were returned even when labelled app; an earlier combined text/type filter returned empty. Therefore validate filter behavior rather than assuming a query covers all records.

Read-only queries this task:
- Build October6 08:25–08:27 UTC: recovered Python3.14.3 and successful package versions on retry with broader filter.
- `/market-data` count/status and latency October18 05:00–06:00 UTC,300-second resolution: both empty arrays.
- Recent analytics_access text search: no lines returned; service is still baseline without this instrumentation.
- Request-type seven-hour window October17 23:00 to October18 06:00 UTC for protected/trading paths: no lines returned.
- Text query October16 12:00–13:00 UTC: no matching route lines returned.
- Failure/restart/deploy events October17 to October18 06:10 UTC: empty page.

These are bounded query results, **not proof of no traffic/no failures**. No meaningful pre-release rate/latency baseline obtained. `evidence/monitoring_checks.json` preserves queries and limitations. HTTP metrics availability/plan/labels and durable retention remain UNPROVED. No alerts/settings/sinks created.

## Operator specification

For the approved release capture equal-duration pre/post windows for /evaluate, /market-data, /trade-opened, /trade-outcome: request count by status,5xxrate, p50/p95 latency, received event evidence where independently authorised. Compare same trading-session context. Record denominators and missing buckets. Weekend no activity cannot validate weekday behavior. Request200 is transport success, not order fill or persistence.

Primary structured event fields: event=analytics_access, request_id, at, service_id, route, result, http_status, scope. No raw body/token is emitted by this component. Treat service/scope IDs as private operational metadata. Count denies by status/result/route/window; correlate probe request IDs; unexpected data-route2xx is incident. `/analytics/access-status`200 may be legitimate and is excluded from data-leak count. Audit writes that fail cause audit_unavailable503, but failed writes cannot themselves guarantee durable evidence.

`audit_summary.py` reads supplied JSONL, prints only bounded categorical aggregates and flags unclassified lines/data2xx. Unknown event results require investigation, not silent clean reporting. It does not fetch logs or establish population completeness. Raw Uvicorn lines require separate status/path extraction; do not pass raw mixed logs and interpret unclassified rows as clean. Fallback emits no analytics_access event, so request logs plus bounded denial probes and live-artifact evidence are mandatory there. Streaming stdout availability is not retention.

Startup monitoring: observe build/start first error and deploy event, process exit/restart and `/health`. Empty configured HTTP health path means default TCP checks. `/health` does not test DB; synthetic tripwire errors were non-fatal. Operator must separately establish ingestion continuity through already authorised evidence, not credentials or new SQL access in this task.

## Initial release retention procedure — proposed, requires operator acceptance

Assign a named collector with read-only log/metric/deploy access and approved private encrypted evidence destination. Before window record UTC clock, service ID, source/deploy SHA, query bounds, pagination and collector identity. Capture baseline and continuously observe release plus initial30minutes; repeat at first normal trading and next closure-sync window. Export complete bounded pages, record hasMore/cursors/endpoints and failures. Preserve original logs privately after secret review; publish only aggregate/redacted evidence. Create SHA256 manifest, capture collection timestamps and access controls, have a second operator verify hashes and retention location. Proposed90-day retention is **not policy**; CEO/security owner selects period and access/deletion controls. If export unavailable or collector loses access, HOLD/incident, never assume logs survive instance replacement. Local application filesystem is ephemeral and not an acceptable sink.

No durable retention, recovery of dropped stdout, completeness, alert delivery or accountable collector was verified here. These remain release gates.
