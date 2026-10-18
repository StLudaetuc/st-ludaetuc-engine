# ST LUDAETUC CLOUD V1.0 — Analytics Containment Release Engineering

DRAFT — NOT APPROVED — NOT DEPLOYED. Supporting author engineering evidence, not independent assurance.

## Executive decision

**NO-GO for production; evidence preparation complete with named blockers.** The production-like local rehearsal succeeds, including Python 3.14.3 with the exact 15 packages listed by the October 6 build. This does not certify the running Render process or fallback. Claude's fallback review remains outstanding. No production action, real database connection, notification or credential operation occurred.

Operations branch: `operations/analytics-containment-release-readiness-v1`, sole parent `5f0d07565629fba30d0c2221229fd2483472e701`. The final immutable operations SHA is supplied in the delivery message and ZIP FREEZE.json; the commit cannot contain its own hash. All additions are under `operations/analytics_containment_release/`. The operations branch is **not a deployment target**.

| Artifact | Exact SHA | Role |
|---|---|---|
| Primary security | `3467133b2208a09e1289651b5be0b45342695d52` | Intended primary containment; not CEO-approved by this assignment |
| Frozen preparation/fallback | `5f0d07565629fba30d0c2221229fd2483472e701` | Existing fallback, independently under review; unchanged |
| Production baseline | `2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c` | Live and unprotected; forbidden rollback |
| Old security v1 | `c035f9d2d74c3eb0d7f2dc1086f2e1095b3bea44` | Root-path bypass; forbidden rollback |

## 1. Fresh Render reconciliation

Read-only connector checks on 18 October 2026 confirm:

- `st-ludaetuc-engine`, service `srv-d71v0v7pm1nc73f60520`, workspace `tea-d71u0rf5gffc7383mh5g`, Frankfurt, Python web service, Starter, one instance, not suspended.
- Latest deploy `dep-db2b0n0473hc739cejd0` is `live`, baseline SHA above; created October 6, 08:25:00 UTC, finished 08:25:47 UTC.
- Branch `main`, autoDeploy `yes`, trigger `commit`. **Updating main is a production release**, not a harmless Git operation.
- Build `pip install -r requirements.txt`; start `uvicorn engine:app --host 0.0.0.0 --port $PORT`; HTTP health-check path empty. Web ingress allowlist is `0.0.0.0/0`. This is web ingress metadata, **not PostgreSQL networking**. PostgreSQL was neither queried nor changed.
- Non-sensitive details preserved in `evidence/render_metadata.json`. No environment-variable values requested. `UVICORN_*` names/presence, registry presence, pending configuration, running-process inventory and overlapping-deploy policy remain UNPROVED.

Build logs explicitly state Python 3.14.3 and successful installation of the 15 named versions in `evidence/build_versions.json`. These are stronger than inferring Python from a cp314 wheel, but remain build evidence, not a fresh process `pip freeze`. The next build's unpinned requirements may resolve differently.

## 2. Runtime and dependency evidence

`requirements-observed.in` pins the 15 observed packages plus test-only httpx. `requirements-observed.lock` includes hashes and three httpx transitive dependencies. It is a **local rehearsal lock**, not a production requirements change. Python 3.14.3 was installed into a disposable environment. Existing isolated 3.12.14 and 3.13.16 environments provide additional compatibility checks, with their differing exact inventories recorded.

Every final environment runs the unmodified 42-test security suite, unmodified four-test fallback suite, 12 new operations tests and 12 tooling tests. Each is run twice. See `evidence/results_summary.json`, per-run environment/command/source-hash JSON and full unedited logs. Initial operations fixture failures are retained as development evidence, not relabelled as passes: a noncanonical doubled-root path correctly routed to 404; a synthetic inner app lacked lifespan support. A later repeated3.13 run hit a CLI readiness timeout; its failure is retained. The harness now preserves CLI startup output on failure and uses a bounded15-second readiness window. The entire matrix was rerun after that harness change. Canonical route assertions remain exact 401/503 and zero handler calls. Noncanonical routing rejection is recorded separately.

Local evidence demonstrates startup, real HTTP/Uvicorn roots `''`, `/api`, `/x`, actual CLI environment-root handling, primary 401/503, fallback 503, HEAD/WebSocket containment, audit stdout, audit-write failure, inner middleware exception/bypass and controlled startup failure. The four actual trading handlers produce identical responses and captured log_event arguments against immutable baseline in-memory execution, including two invalid-input cases. Additional real HTTP trading requests pass for primary and fallback. **No SQL effects, OANDA execution, Make delivery, full strategy-input population or Render acceptance was tested.**

The database connection functions are tripwires and inherited connection settings are rejected. Startup's attempted schema check deliberately meets a local tripwire and logs a non-fatal error; this is expected fault-injection evidence, not a real DB connection. `/health` can return 200 with no DB configured. Even `events_db_configured=true` would prove presence, not connectivity, permissions or successful ingestion.

## 3. Consumer impact

See `CONSUMERS.md`. There is documented research use and a historically observed Make relay, but no complete current caller inventory. No messages sent. 401/503 is intentional unavailability; it must not become zero trades, empty data, profit or a successful scheduled report. Authenticated scoped legacy data routes remain unavailable. No real identities should be provisioned to reopen them.

## 4. Monitoring and preservation

See `MONITORING.md`. Connector supports log time windows/pagination, event history, status/path metrics and latency queries. Recent queries returned empty results: **UNPROVED**, not zero traffic. Request-log path filtering and metrics availability need operator validation. No current `analytics_access` is expected from unprotected baseline. Fallback does **not** emit that event; it needs HTTP access-log evidence and active denial probes. No durable sink or retention contract was verified and none was created.

## 5. Rehearsal limits and rollback

See `DEPLOYMENT.md` and `ROLLBACK.md`. Fallback protects against inner analytics bypass/failure, not engine import/startup failure, trading failures, outer-wrapper failure or ingress mistakes. A failed containment rollout can leave the **unprotected old baseline** serving traffic under Render's normal retained-instance behavior; platform recovery is not containment success. There is no safe historical production rollback artifact yet.

The fallback requires independently accepted exact source, production-like build, operator rehearsal and separately approved start-command/artifact change. Its existence alone is not recovery readiness. No independently assured alternate ingress exists. Until a safe response for the approved incident scenarios is evidenced, release remains NO-GO.

## 6. Required CEO/operator evidence

1. Full independent security v1.1.1 report/archive and exact hashes. Preparation runbook reports PASS; this task did not reproduce independent assurance or acquire its full final report.
2. Claude's exact-SHA fallback verdict, test package and residual acceptance. Do not edit the frozen fallback to obtain a pass.
3. Operator confirms process Python/package inventory and **names/presence only** of relevant Uvicorn/registry settings through approved tooling; no credentials or writer-shell output in this public repository.
4. Resolve reproducibility of next build. Current production requirements float. Matching today's local lock cannot force Render's future resolution. Any pin/build-command change needs separate engineering scope and approval; any different resolved version requires fresh compatibility evidence before approval.
5. Current consumer inventory/acknowledgements or explicit CEO exceptions; Make blueprint/schedule/ownership read-only evidence and external agent/report inventories.
6. Monitoring access, actual traffic/error/latency baseline, agreed thresholds and private audit preservation owner/destination/retention.
7. Exact final release artifact and merge-result SHA/tree, fallback artifact/start command, named operator, timing, stop conditions and recovery authority approved separately. If a merge creates a new SHA, review/freeze that exact result **before main moves**.

## 7. Independent verification

Use a fresh full clone; checkout exact operations SHA. Verify sole parent, namespace-only delta, `SHA256SUMS`, runtime/fallback blobs and locked environment. Run commands in `REPRODUCE.md`. Review the initial failure logs as well as final passes. Compare logs with actual commands/source hashes. Confirm no assertions were weakened for canonical data endpoints. Inspect rehearsal isolation and local-only network binds. Review fallback separately; supporting tests are not its independent certification. Run the draft gate and expect HOLD. Reconfirm current Render state read-only immediately before any later approval.

## 8. Recommendation and exact next decision

CEO should commission/receive the frozen fallback assurance and designate a release operator to close runtime, consumer, monitoring and private retention gaps. **Do not approve a merge/deploy yet.** After evidence closes, approve or reject the exact release SHA/tree, controlled main update, execution window, fallback operation and outage-versus-disclosure trade-off. This assignment approves none of those actions.

ANALYTICS_RELEASE_ENGINEERING_BLOCKED_ON_EVIDENCE
