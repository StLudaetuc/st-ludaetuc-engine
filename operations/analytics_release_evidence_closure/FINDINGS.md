# Independent fallback finding disposition — author reconciliation only

Source reviewed:5f0d07565629fba30d0c2221229fd2483472e701. Verdict remains ANALYTICS_FALLBACK_INDEPENDENT_REVIEW_REQUIRES_CORRECTION. No report/source edits; no author self-approval. Independent original evidence publication belongs to its reviewer/publisher, not this workstream.

| Finding | Closure work | Status / remaining gate |
|---|---|---|
| R1 primary SHA lacks fallback | Explicit exactSHA/tree component matrix and pre-activation inventory;3467133 never selected for fallback command | DOCUMENTATION RECONCILED; approved deployable fallback-bearing release still absent |
| R2 merge-generated SHA | Pre-main result SHA/tree/parents procedure, main-drift invalidation, distinct FF/merge/squash/queue handling | DOCUMENTATION RECONCILED; protected-branch/operator mechanism and independent re-review needed |
| R3 activation/status/audit | Ordered procedure below; primary401 versus fallback503; access-status also unavailable; fallback structured-audit gap and log collection; retry impacts | DOCUMENTATION RECONCILED; real operator/Render drill and audit-retention acceptance pending |
| R4 Render rollback/autodeploy | Current official deploy/rollback/health docs retrieved/read and hashed; method-specific side effects documented | PLATFORM DOCUMENTATION VERIFIED; account-specific retained artifacts, settings and actual acceptance unproved |
| R5 floating dependencies | Exact15-runtime-package hash lock proposal; offline wheel/install verification; separate Python/platform proposal | ENGINEERING PROPOSAL; production requirements unchanged and reproducible production build not established |

The original fallback code's security-test success is not a universal rollback verdict. Additional report residuals include namespace-package shadowing, readiness limits, missing fallback audit, and lack of enforced release CI. These are recorded, not patched under this assignment. Reviewer-reported pre-existing market-data NaN/Infinity loss is outside this workstream; no new live test or trading change was made.

## Proposed fallback activation — named CEO-approved operator only, NOT EXECUTED

1. Establish scenario A (engine healthy, analytics boundary concern). Startup/trading/dependency incidents require RECOVERY_MATRIX.md; do not use this procedure as their repair.
2. Verify explicit approval of exact fallback-bearing source/tree, built artifact/runtime and launch command. Confirm files exist in the actual deployed artifact, not merely another branch. Missing files: STOP; do not attempt command change on3467133.
3. Capture no-secret before-state: deploy ID/SHA/tree, primary command, auto-deploy setting, pending config, approved traffic baseline and monitoring operator. Verify engine health and passive trading persistence through the authorised operator.
4. Start private log collection before change. Obtain consumer acknowledgement/CEO exception for all analytics now503 including access-status, and bounded5xx retry policy. Confirm access logs are enabled and retained; primary analytics_access events will cease.
5. Only under separate CEO authority, select the independently accepted fallback-bearing artifact and proposed command `uvicorn security.analytics_release.fallback_app:app --host 0.0.0.0 --port $PORT`. Artifact deployment plus command change may be two consequential actions; both must be approved and their actual Render lifecycle observed. A command change is not a magical switch in the old process. No assumption that all such operations reuse or rebuild dependencies identically.
6. Observe the resulting deploy/build/start to terminal state; compare actualSHA/tree/start command and runtime provenance. Startup errors, wrong artifact or untested packages: STOP/incident. Old retained instance may remain exposed.
7. Bounded no-secret probes: each three data routes and access-status must return503 analytics_containment_active (HEAD empty); approved local/reviewer evidence covers websocket close-before-accept, not a new live websocket operation. No analytics2xx, redirect to data, legacy handler or DB read. Never save a surprising data-bearing body.
8. Verify `/health` and passive trading traffic/error/latency plus separately authorised persistence evidence. TCP orhealth200 alone is insufficient. No synthetic trading payloads sent to production.
9. Confirm Uvicorn/provider access logs and request correlation/private export. Do not expect analytics_access events from fallback and do not label their absence healthy audit. Preserve activation evidence and acknowledged audit limitation.
10. Reversion to primary command is allowed only on the exact independently accepted corrected primary-bearing artifact with valid runtime and CEO authority. Never revert to bare baseline/v1. Confirm final command, SHA/tree and auto-deploy after every operation; re-enabling auto-deploy may cause an immediate new release of currentmain.

This procedure is proposed supporting engineering evidence; independent reviewer must decide whether R1–R3 are resolved operationally. It grants no activation approval.
