# Reproducible future release procedure — DRAFT / NO AUTHORITY

## Approval record and source gates

Primary security source is frozen at `3467133b2208a09e1289651b5be0b45342695d52`; fallback bundle at `5f0d07565629fba30d0c2221229fd2483472e701`. Neither is approved by this package. Decide whether the eventual release artifact contains fallback files (the latter bundle does); independent review must cover that entire artifact and the selected start command. Do not merge this operations branch as the runtime release.

Use a clean disposable clone with full history. Fetch read-only, verify main still equals `2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c`, inspect the entire main-to-release diff and require no unrelated changes. Confirm ancestry with `git merge-base --is-ancestor BASE APPROVED_SHA`; inspect `git status --porcelain`, `git show --format=fuller APPROVED_SHA`, `git diff --name-status BASE APPROVED_SHA`, runtime blobs and full tree ID. Verify independent report references bind the artifact and source hashes. Never approve a moving branch name.

If a fast-forward is selected and allowed by governance, the approved exact SHA can remain the main target. If branch protection requires a merge/squash/rebase commit, prepare the prospective resulting commit in isolation, inspect its full tree and parents, independently review and approve its **new exact SHA before main moves**. Do not rely on “same runtime files” as approval for unrelated contents. A changed main parent invalidates the approval and requires renewed delta review. Operator uses the repository's authorised protected-branch workflow; this package supplies no bypass, push, merge or deploy commands.

`approval_record.template.json` is deliberately UNPROVED. `release_gate.py` checks source identity and evidence-attestation presence offline, returning HOLD until records exist. It cannot authenticate CEO/reviewer signatures or confirm current Render settings. Operator must verify references and identity independently. It is not an authority-granting API.

## Runtime/build gates

1. Preserve the immutable reviewed artifact, tested Python and hash-locked package set privately. Rehearsal lock is under operations only and does not affect Render.
2. Current Render build floats all four direct dependencies. Do not promise a version-locked future build from an exact Git SHA alone. Either obtain separately reviewed/approved deterministic build configuration, or withhold release until actual intended build artifacts/dependencies can be verified against tested versions through an approved staged process. A post-merge surprise is a stop/incident, not permission to test in production.
3. Confirm live process interpreter/package versions and relevant config names/presence without dumping env values; no writer Web Shell or database is needed for this task. Current build evidence is Python3.14.3, fastapi0.142.2, starlette1.7.0, uvicorn0.54.0, pydantic2.13.5 and remaining pinned versions in the lock.
4. Validate start command, `PORT`, Uvicorn root-path behavior and registry absent/present semantics. CLI host/port are explicit; `UVICORN_ROOT_PATH` can affect routing despite no flag in the command. No inference of absence from the visible command.
5. Baseline health and security semantics: `/health` is process information, not DB readiness. No migration is part of this release. Existing startup schema behavior is unchanged; no one may add a migration under this operations assignment.

## Final operator pre-deployment checklist

- [ ] CEO approval references exact release/merge-result SHA, tree, fallback SHA/start command, operator and UTC + Europe/Jersey window.
- [ ] Independent primary and fallback reviews preserved privately with test archives/hashes; residuals accepted by appropriate authority.
- [ ] Clean-tree, ancestry, full-diff and main drift gates pass; concurrent main changes frozen by authorised human procedure.
- [ ] Reproducible actual build and runtime match tested matrix; configuration names/presence checked; no secret copied.
- [ ] Fresh service/owner/live-deploy/build/start/auto-deploy metadata captured. Merging main triggers production. No duplicate manual trigger.
- [ ] Consumer acknowledgements or explicit CEO exceptions recorded, including Make relay; retry storms/false empty reports prevented.
- [ ] Read-only trading baselines, monitoring access, stop thresholds, on-call owner and private audit retention procedure agreed.
- [ ] Safe fallback independently accepted and operator drilled; no historical insecure rollback. Startup-failure response explicitly approved.
- [ ] All public ingress domains/rewrites known; pending configuration and overlapping deploy policy checked.
- [ ] No production probe sends trading fixtures, orders, tokens or SQL. Denial probes are bounded and separately approved.

## Future execution — named CEO-approved operator only

At approved window recheck main/live state and approval freshness. Apply only the approved protected-branch operation. This triggers Render automatically. Record Render deploy ID and commit; monitor build/start and event history to terminal state. An accepted trigger is not live acceptance. Do not call a second deployment trigger. Unexpected runtime versions, wrong SHA or pending drift: stop progression and use only approved containment-preserving incident response.

Normal Render failed rollout behavior may keep the old, unprotected baseline active. Report this as exposure continuing; never call it successful containment. A TCP-ready server or “live” label does not prove denial, DB health or trading continuity.

## Post-deployment validation checklist — future operator, no execution here

- [ ] Actual live deploy ID/SHA equals exact approved result; startup logs and dependency inventory preserved privately.
- [ ] Bounded unauthenticated GET to each three data routes returns 401 with `unauthorized` under primary; approved fallback returns503 `analytics_containment_active`. Do not request raw rows or follow redirects to unknown hosts.
- [ ] Root/path variants covered by approved probe plan; no analytics data-bearing 2xx. `/analytics/access-status` is distinct and may legitimately return200 for authorised identity; do not confuse it with data success.
- [ ] If any protected data route unexpectedly returns2xx, close response without saving data, declare incident, preserve status/headers/request ID only and activate approved containment response.
- [ ] No real identity provisioning needed. Authenticated503 scope behavior is demonstrated locally; production identity probes require separate approved identity handling and must never expose tokens.
- [ ] `/health`200 and expected version/config-presence fields; passive evidence of existing normal ingestion, errors/latency and independently authorised write continuity checked. Do not send synthetic trading payloads to production.
- [ ] Structured primary `analytics_access` lines correlate to denial request IDs. Fallback has no structured audit emission: preserve Uvicorn/provider access logs plus active probe timestamps and deployment evidence.
- [ ] Baseline/post windows same duration/timezone/load context, all four trading routes compared. HTTP200 is not fill or DB-write evidence; cancelled orders are not executed trades.
- [ ] Observe initial30minutes, then first normal trading period and next expected closure-sync window with named operator. Proposed timings require CEO/operator acceptance; weekend silence does not prove weekday compatibility.
- [ ] Sign-off records actual limitations, all incidents and unresolved items. No PG/governance status updates without separate authority.

## Threshold proposal — not ratified

Immediate security stop on any data-route2xx, raw internal response, wrong SHA or missing required audit evidence. Availability stop on startup failure, persistent health failure or loss of trading ingestion. Operator must set numerical 5xx/latency/drop thresholds from a valid comparable baseline **before approval**. Empty connector metrics cannot justify invented thresholds or a clean baseline.
