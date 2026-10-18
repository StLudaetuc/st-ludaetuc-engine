# Operational risks and evidence register — DRAFT

| ID | State / consequence | Evidence or next owner action |
|---|---|---|
| R1 | BLOCKER: fallback independent assurance pending | Claude exact5f0 source review/test package and accepted residuals; this task unchanged fallback support only |
| R2 | BLOCKER: current process and next-build reproducibility unproved | Operator version inventory; separately reviewed deterministic-build approach; unpinned requirements do not lock render build |
| R3 | BLOCKER: consumer completeness/acknowledgements missing | CEO/R&D/Make/report owners; historical relay found, present URL/schedule unknown |
| R4 | BLOCKER: usable traffic baseline and metric access missing | Operator checks query types/labels/plan; empty arrays are unknown, notzero |
| R5 | BLOCKER: durable audit retention and collector absent | Approved private export destination/access/retention; primary stdout and fallback access logs separately |
| R6 | BLOCKER: no independently assured startup-failure recovery | Fallback shares engine; approve safe response including outage trade-off, or NO-GO |
| R7 | BLOCKER: final exact artifact/merge result and CEO release approval absent | Named operator, main delta, timing, stop thresholds and authorised fallback changes |
| R8 | UNPROVED: current Uvicorn env/config names, registry presence, pending settings, ingress paths | Operator metadata-only verification; never dump values |
| R9 | RESIDUAL: baseline remains exposed until a separately approved release | Fresh liveSHA verified; no production changes made |
| R10 | RESIDUAL: no functional legacy analytics after containment | Intentional401/503, never emptydataset; future scoped-reader redesign separate |
| R11 | RESIDUAL: unauthenticated trading ingestion remains pre-existing | Out of scope; no changes to trading logic, DB or broker side effects |
| R12 | LIMITATION: synthetic logging not SQL acceptance | Actual handler differential and routing only; no realDB/OANDA/Make execution |
| R13 | LIMITATION: provider rollout/rollback retention/config drift | Docs checked; no actual Render staging/deployment acceptance; old deployment retention may preserve exposure |
| R14 | LIMITATION: no full final independent primary archive acquired | Runbook reports PASS; preserve original final review/hashes before authorisation; author tests are not that report |

No risk acceptance or governance status was silently granted. All gates remain DRAFT/UNPROVED where evidence is missing. Exact next decision: close independent fallback review and assign operator to complete evidence; then seek a separately scoped CEO exact-SHA go/no-go decision. No routine confirmation needed to finish preparation, but consequential release remains unauthorised.
