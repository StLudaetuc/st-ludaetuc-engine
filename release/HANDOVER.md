# Consolidated R1–R5 handover — AUTHOR EVIDENCE

**RELEASE STATUS: NO-GO**. **BLOCKED_ON_EVIDENCE** for mandatory unavailable tests/build/scan/access. A25 is NOT MET. This is a partial engineering candidate, not engineering build completion or independent acceptance.

No merge, deploy or Production action is requested or authorised by this handover.

## Source and authority

Base 6d17fa31f2b62a9e2c20ffcdea558cde955d0a30, tree 47879f364c39084246d3d379df5fe5e5642204d9, verified directly against GitHub before any candidate commit. All §2 paths were retrieved and byte/tree checked. Only the two commissioned new branches may be published. Final candidate/R-B identities are in external post-freeze attestations; no final commit is patched back into its own tree. Local source ancestry is reconstructed with verified exact Git objects through baseline; local history is shallow at baseline, so full-history scanning remains incomplete.

No Render, Production, credentials, broker, trading strategy, merge, live-money, consumer notification or Chief activation action occurred. Provider/session: OpenAI ChatGPT Work / Codex; session ID NOT_EXPOSED (workspace UUID is not substituted). Single owner account StLudaetuc; author evidence is not independent approval.

## Layer/finding mapping

| Finding | Change | Remaining gate |
|---|---|---|
| R1 / primary (a), fallback (b) | Both components present on one frozen tree; fallback launch only changes command | Independent acceptance and first-release all-path protection |
| R2 / (a) | FF-only exact immutable SHA; tree verified before any release ref change; verifier | Operator approval/protection/current-main recheck |
| R3 / (b) | Ordered runbook, four 503 routes, WS close-before-accept, correlated structured audit | Original reviewer matrix/PG, durable audit owner |
| R4 / (c) | Rollback auto-deploy/command/live SHA/tree checks; UNPROVED platform behaviour | Separately authorised staging/operator rehearsal |
| R5 / (a/b/c) | Full G runtime hash lock; exact Python file; separate test/audit locks; full G/R comparison | Pinned-container build, runtime operator parity, independent review |
| C1 / (a/b) | Two package __init__.py markers; shadowing probe | Fresh independent rerun |
| C2 / (b) | Fixed-field stderr JSON event per denial; audit failure cannot bypass denial | Durable collector/retention and throughput |
| C3 / (b) | Actual fallback entrypoint, newline/%0A/non-string paths, WS/audit tests | Original matrix remains unavailable |

Health-check path /health is a separate proposed operational requirement, not retroactively an original R1–R5 finding. It is not configured on Render.

## Dependency recommendation and evidence

See DEPENDENCY_COMPARISON.md. Both sets pass 42 primary + 4 existing fallback + 5 C3 + 12 prior tools + 12 rehearsal + 16 closure = 91 tests per run, twice per set (364 valid test executions across the two sets). Setup mislaunch logs are retained and INVALID, not counted. A further R confirmation run is supplemental. Original v1.1.1 archive checksum was verified before safe extraction. Original harness source is unchanged; baseline/v1 controls reproduce expected counts. Current R resolution is reconstructed, not an original reviewer lock. G is provisionally recommended for available evidence parity and reduced change; neither meets every mandatory acceptance condition.

The author supplemental 236-call fixture is reconstructed from prior author payloads. It tests only the five trading routes and stubs log_event; all modes match and mock event hashes match. It deliberately does not pretend to be the original reviewer fixture or a PostgreSQL database result. Original fallback archive retrieval returned empty content from GitHub and was absent from the listed Drive fallback folder; no substitute is relabelled original.

Scans return zero known findings through PyPI/OSV, not proof of no vulnerabilities. Database snapshot version/timestamp is not exposed. Wheel filenames/platform tags/markers/hashes are recorded; independent index/attestation access limitations are retained. Hash checks alone do not prove upstream trust.

## Operational recovery (c)

R-B is separately frozen from the consolidated candidate, with exact G lock, same recipe and mandatory supported ContainmentOnly launcher. It is not independently known-good for Production. It shares protected engine/dependencies, so common failures can defeat both. Startup/core corruption restore probes and host packaging evidence are supplemental. PG acceptance remains incomplete. Bare baseline/v1 are prohibited recovery targets. No database/broker-state recovery is provided.

R-B recovery behaviour is AUTHOR EVIDENCE in a disposable environment; it provides no Production recovery claim until independently tested.

## First-release protection

A25 NOT MET. FIRST_RELEASE_CONTAINMENT.md and the raw demonstration show independently modelled denial, plus a live retained-origin bypass control. An immutable artifact or staged R-B cannot protect an already-serving unprotected instance when its first deployment fails. Later operator work must establish an independently controlled boundary and prevent every direct-origin route before release, then test failure semantics. No such platform evidence was collected.

## A1–A25 author acceptance register

| ID | Author status | Evidence / exact limitation |
|---|---|---|
| A1 | AUTHOR PASS (verify external frozen output) | Protected blobs/subtrees match base; engine, app and primary security untouched. |
| A2 | AUTHOR PASS | 42/42 primary ×2 on G, Python 3.14.3; raw clean-process logs/inventories retained. |
| A3 | AUTHOR PASS | Original verified root harness unchanged: candidate 0/20; baseline control 20/20 and reconstructed-v1 control 14/20 on G and R; candidate repeated. |
| A4 | INCOMPLETE / UNAVAILABLE | Original fallback matrix archive unavailable; no disposable PG container. Prior author rehearsal and C3 are supplemental, not substitutes. |
| A5 | AUTHOR PASS, LOCAL ONLY | Actual fallback entrypoint HTTP 503 four routes; /health 200; WS closes 1008 before accept in TestClient/ASGI. Literal real-Uvicorn HTTP transcript retained. |
| A6 | PARTIAL / INCOMPLETE | Literal local start-command switch and HTTP/audit probes reproduced. Full original matrix, PG and platform steps cannot be reproduced here. |
| A7 | AUTHOR PASS, bounded process delivery | Six HTTP denials produce six correlated JSON events; WS and 200-request burst controls pass; primary analytics_access remains. Transport failure preserves denial but delivery/retention is not guaranteed. |
| A8 | AUTHOR PASS (frozen source verification) | One tree contains primary, fallback, runbook v2 and full locks; literal fallback startup succeeded. |
| A9 | AUTHOR PASS, source procedure only | Baseline is an ancestor; FF-only/tree-before-ref checks documented and verified locally. No release ref was moved. |
| A10 | AUTHOR PASS, documentation only | Post-rollback auto-deploy/start-command/live-SHA/tree checks explicit. Platform behaviour UNPROVED. |
| A11 | AUTHOR PASS, selected runtime | G exact 15-wheel lock installed offline; pip check clean; tampered wheel and lock rejected; target Python 3.14.3. Test-only packages and pip identified separately. |
| A12 | INCOMPLETE | No container runtime or verified pinned image digest. Host deterministic packaging twice is supplemental and cannot close the pinned-container criterion. |
| A13 | INCOMPLETE | Supplemental reconstructed 236 trading calls yield zero response differences and equal 216 mock event records/schema hashes on G/R. Not original reviewer fixture or actual PG rows/persistence. |
| A14 | AUTHOR PASS | engine.py diff from baseline is only pre-existing primary import/middleware addition plus its comment/blank lines; unchanged against protected blob. |
| A15 | INCOMPLETE | Recovery candidate staged separately; core-failure/restore probes are supplemental. No PG drill or independently known-good recovery accepted. |
| A16 | INCOMPLETE | Shared dependency/DB failures are expressly not repaired by the wrapper. No original fixture/PG continuity proof. |
| A17 | AUTHOR PASS, limits stated | No DB/broker-state recovery; disposable timings not Production; mandatory no-Production-recovery statement included. |
| A18 | INCOMPLETE | gitleaks absent and official binary download blocked by proxy timeout; manual/pattern privacy screen is not a gitleaks pass. Full history is shallow locally. |
| A19 | AUTHOR PASS | External regular security package on PYTHONPATH does not shadow local primary/fallback imports; no external startup hit. |
| A20 | PENDING EXTERNAL RECEIPT / ACCESS BLOCKED | Archive publication and post-upload byte verification are recorded externally after freeze; Chief/reviewer separate-actor access is not established. Never self-award VERIFIED_PUBLISHED. |
| A21 | DOCUMENTED; ACCESS BLOCKED | Operator evidence OPEN with owner=operator; original restricted operations bundle verified by hash and canonically referenced in the assigned folder. Separate Chief/reviewer access unproved. |
| A22 | PARTIAL / INCOMPLETE | G and R fully inventoried, hashed, clean-suite/root/mock differential/security/offline/tamper checked; PG/original matrix/pinned builds incomplete for both. G recommendation is provisional, neither is release accepted. |
| A23 | PARTIAL / INCOMPLETE PROVENANCE | pip-audit 2.10.1, PyPI and OSV: zero returned findings for both sets. DB snapshot timestamps NOT_EXPOSED; independent current index/attestation requests subject to recorded access limits. No blanket upstream provenance claim. |
| A24 | INCOMPLETE | Separate frozen R-B source/host artifact and mandatory supported barrier entrypoint provided; reproducible pinned-container build and PG recovery drills not completed. |
| A25 | NOT MET — RELEASE NO-GO | Model independent ingress denies throughout failed deploy, but direct retained-origin control leaks. No demonstrated all-path boundary for the existing public origin. |

## Operator-owned OPEN evidence

| OPEN item | Owner | Required evidence |
|---|---|---|
| Actual Render configuration and env-var names/presence | Operator | Build/start commands, PYTHON_VERSION override, UVICORN/identity names, no secret values |
| Deployed package/runtime inventory | Operator | Actual Python, complete pip freeze, wheel/lock/runtime provenance |
| Audit delivery and retention | Operator | Named collector/retention owner, bounded failure/throughput, durable export/readback |
| Consumer sign-off | Operator | Current consumers/owners, acknowledgements or CEO exceptions; no notifications authorised here |
| Persistence and post-deploy probes | Operator | Separately authorised SELECT-only checks and staging/live route probes; no synthetic Production trading |
| F-04 | Operator / assurance owner | Obtain original definition and disposition; meaning remains undefined, never invented |
| First-release all-path boundary and platform rollback | Operator | Default-hostname/custom-domain/bypass coverage, failure rehearsal, auto-deploy/start/live source checks |

## Publication and independent review

Original restricted operations bundle remains canonically at Drive file 1bOOrjAZa3VhqcljVKZjYpeNkKJD2OS-o in the assigned Release Engineering folder, SHA-256 98bf423a3a27b96fcaf9ee6d686d1ce28c797ec29842c43cf2b1832e28a3cc95 (7,693,071 bytes). It was downloaded and independently hashed within this author session; original per-file hashes and canonical access receipt are in the new restricted evidence. Do not duplicate private originals into the public engine repository or public CI artifacts. Chief/reviewer distinct-actor access is unproved: EVIDENCE_PUBLICATION_BLOCKED for that access requirement, even if author upload/readback matches.

Fresh Claude Code review should start from the external frozen SHA/tree and archive hashes, verify §2, rerun all available raw commands, obtain the original fallback archive and PG/pinned-build/gitleaks prerequisites, and reject any substituted fixture or source-only verification presented as full acceptance. Do not approve release with A25 unresolved. CI is READY_NOT_EXECUTED; no Actions run is asserted.

## Exact source delta and justification

The complete changed-file inventory and exact diffs versus 6d17fa31 and 5f0d0756 are emitted as external post-freeze evidence. Protected paths have zero delta. C2 is the only fallback runtime edit; fallback_app and existing fallback tests remain byte-identical. Package markers implement C1. New release files implement C3/probes, source/artifact verification, deterministic packaging recipe, provisional dependency comparison/locks/SBOM, first-release design model, recovery limits and this handover. Runbook v2 is new; old runbook is untouched. requirements.txt/.python-version implement G hash-lock/runtime proposal after A/B comparison. New CI verifies only; no deploy/secrets. No non-protected change modifies trading logic.
