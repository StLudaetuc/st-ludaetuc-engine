# ST LUDAETUC CLOUD V1.0 — HISTORICAL MARKET RECOVERY v1.1

DRAFT — NOT APPROVED. Isolated engineering correction, not an approval to acquire
production data, integrate features, train models or deploy.

## A. Engineering outcome

Successor branch `research/historical-market-recovery-v1-1` starts at exact frozen
v1 `2807fdde9d40d5084f85b9d4d0a3005cc50464f5`. All changes remain under
research/historical_recovery/. The exact new immutable SHA is in the final handover
and must be used for review, never an unverified moving branch.

Known D1/D2/D3/D4/D7 corrections are implemented with reproductions and regressions:

- Canonical aware timestamps use UTC and exactly six fractional digits with Z for
  digest/export serialization. Hash encoding is explicitly v2/incompatible with v1.
  Original source timestamp strings/event identity/raw values remain preserved.
- Producer checksums now establish internal consistency only. Actual file audit
  hashes are checked; optional verifier-supplied TrustedAnchor pins expected
  manifest, audit, record identities/digests/counts/scope/window/acquisition/operator/
  redaction versions. Missing anchors are UNPROVED or fail with --require-anchor;
  a producer-supplied Context cannot independently promote an unanchored export.
- Calendar completeness counts only final validated eligible candles. Observed
  times and missing, invalid, conflicting, unproved and valid slots remain distinct.
- Duplicate/conflict demotion propagates into recovery counts and research readiness;
  original timestamp recovery state, raw observations and indicators remain visible.
- Structured nested secret redaction covers case/format variants of passphrase,
  api_key/key/auth, credentials, DSNs and JWT/bearer strings. Legitimate OHLCV and
  indicator values are preserved. No universal arbitrary-text secret detector is claimed.

Related lower-severity requirements corrected separately: exponent/float ms rejected;
sub-microsecond ISO loss rejected rather than truncated; known source versions must
be bounded strings; explicit Context/manifest/calendar stream restrictions cannot
be silently overridden; source feature naming/provenance is retained; destination
and future dedicated-reader CLI preflights are hardened.

**The full independent review package was not available.** The assignment identifies
D1,D2,D3,D4,D7; exact D5,D6,D8,D9 identities and their original reproductions remain
unavailable. FINDINGS_V11.md explicitly marks those unresolved evidence mappings.
Related requirements are not falsely assigned to an unknown D-label.

## B. Tests and PostgreSQL evidence

The **original 99 tests were reproduced unmodified on Python 3.12.14 and 3.13.16**
before source correction. Original logs are retained under evidence/v11/. Current
final corrected suite: **142 tests** = those original 99 plus **43 new regression/
adversarial controls**, run twice per interpreter in isolated hash-locked environments.
Full unedited logs, commands, exit codes and tested source SHA-256 values are retained.
All four final runs pass, zero failures/errors/skips. See results_summary.json.

Coverage includes 601 synthetic record-hash comparisons (three timestamp patterns) across UTC, Europe/London and
America/New_York datetime renderings; canonical precision; altered exports with
recomputed internal manifests; missing/wrong expected anchors and audits; invalid/
conflict/unproved slots; original-indicator conflicts; source versions/scope; nested
credentials/DSNs/JWT; safe market-value controls; private destinations; writer-role
preflight rejection; original partial/snapshot/pagination/rollback controls.

**PostgreSQL 18 runtime is UNPROVED.** No postgres/initdb/docker executable exists;
only root is mapped and an ordinary switch to an unprivileged UID fails EINVAL.
No root-check bypass, shared/cloud DB, production shell or live credential was used.
The reviewer-reported 38 offline and 51 PostgreSQL 18.4 checks are not reproduced
unchanged: their test package was not located. Mocked transaction/privilege checks
and Python timezone comparisons are not PostgreSQL runtime acceptance.

A development iteration had three reader digest errors after adding timestamp-state
metadata; fixed by recomputing final row digests. Its unedited failure log is kept.
Earlier passing development logs are retained and labelled, not represented as final
source verification. Final runs bind the exact corrected source hashes; final Git
comparison confirms those source bytes are what is frozen.

Reproduce from a full checkout, with no production variables:

```
uv venv --python 3.12 /private/scratch/recovery312
uv pip sync --python /private/scratch/recovery312/bin/python --require-hashes \
  research/historical_recovery/evidence/v11/requirements.lock
python research/historical_recovery/evidence/v11/run_suite.py \
  --python /private/scratch/recovery312/bin/python --label reviewer312
```

Repeat using 3.13 and an independent destination. The runner constructs a clean
child environment and records source hashes. Paths are examples, not permission
to provision a production execution host. Normalisation/auditing have no runtime
external dependencies; JSON Schema testing uses the locked test-only packages.

## C. Data-integrity assessment

D1/D2/D3/D4/D7 logic is corrected and passes authored Python tests. Real database
execution/identity, independent anchor custody and missing D-label mapping remain
unproved. No finding is declared independently cleared by its author.

Reported source population remains exactly the CEO summary: **164,209** rows,
**121,309** ISO payload.time, **42,900** millisecond payload.time_open, **35** streams,
**zero reported missing both**. No real raw rows were acquired/processed in this
assignment. Actual source-time correctness, expected-bar coverage, indicator values,
version/provider lineage and population completeness are still UNPROVED.

The original normalizer/auditor contracts are preserved conceptually, with explicit
v1.1 report and v2 digest version changes. Old v1 hashes cannot be reused as new
trust evidence. PostgreSQL remains authoritative; no ledger, production rewrite,
receipt-time substitution, grid rounding, feature recomputation or financial
arithmetic has been introduced.

Internal consistency, matching independent expectations and source authenticity
are distinct. Anchor verification cannot prove that an anchor file's custodian was
truly independent; an approved out-of-band immutable custody channel is required.
No issuer/key/trust service is provisioned. Offline APIs are not public authenticated
Readers and must not be exposed directly to untrusted source callers.

## D. Research readiness

The tools can process **synthetic or separately authorised private copies** and
produce deterministic record/slot/conflict reports. readiness_summary includes
received/source-time/OHLCV/late/invalid/eligible/research counts, features, calendar
coverage and training blockers. evidence/v11/readiness_demo.json demonstrates valid,
invalid and conflicting windows without pretending the real population was processed.

Valid original timestamps are separate from eligible observations. Invalid/conflicting
slots cannot satisfy COMPLETE_EXPECTED_SLOTS. Late records can support retrospective
descriptions, but cannot become available before receipt for causal linkage. Receipt
snapshots never become complete source-window evidence automatically. Training stays
blocked pending label lineage, temporal split/leakage controls and explicit approval.
All research outputs remain offline/read-only/shadow-only. No Python-derived feature
can affect production; Finance/trading economics are not duplicated here.

## E. Evidence acquisition readiness

**Do not run extraction from the existing Render engine Web Shell:** it reportedly
contains writer credentials. The replacement plan is an explicitly approved isolated
host with a dedicated SELECT-only identity and no execution credential inheritance.
The CLI now requires named identity and isolated-job references and checks role flags,
source-table writes/SELECT, create privileges and writable memberships. Its SQL
preflight is not PG18 runtime validated and is not a comprehensive all-object/function
privilege audit; a future operator security review remains necessary.

Acquisition retains bounded UTC receipt windows, exact source filters, stable event-ID
pages in one repeatable-read read-only snapshot, limits, rollback, private exclusive
output, hashes and redaction. Future deployment needs approved identity/egress/retention,
version/provider/calendar registry and independent expected-anchor custody. No such
resource, credential, grant or environment was created tonight. See OPERATOR_RUNBOOK.

## F. Independent assurance requirements

Fresh reviewer must inspect exact frozen SHA and parent v1; all changed paths must
start research/historical_recovery/. Verify Analytics Security v1.1.1, Finance,
Trading Data, original Historical v1, main and Chief remain untouched. Inspect full
changed-files inventory and SHA-256 manifest, then reproduce original-v1 99 and final
142 cases using the recorded versions; do not mistake author tests for independent
assurance. Check fixture changes: original test files are retained unchanged; added
regressions do not weaken prior expectations.

Obtain the complete D1–D9 review package, map D5/D6/D8/D9 exactly and execute its
38 offline/51 PG checks unchanged where compatible. Document API/version differences.
Use a genuinely disposable non-root local PG18 server with synthetic data only to
validate read-only/session/role enforcement, snapshot/concurrent insert/update behaviour,
C-collation pagination, UTC/London/New_York record hashes, JSONB Decimal fidelity,
microsecond precision, duplicate/malformed rows, partial/resume, errors/timeouts and
rollback. No production/shared DB or writer shell. No runtime PASS is supplied here.

Challenge independent-anchor custody and missing-anchor behaviour; self-consistent
malicious rehashing; invalid/conflicted-slot completeness; feature identity/provenance;
redaction negatives/market positives; CLI scope/destination/error handling. Missing
hashes do not become verified zero and false-PASS guards must fail closed.

## G. Governance and next action

**Single next milestone:** commission independent technical assurance of this exact
isolated correction, supplying the complete historical review package and a safe
non-root disposable PostgreSQL 18 environment. These evidence dependencies must be
closed before declaring a fully assured v1.1 or authorising a real-data pilot.

This task does not authorise extraction, deployment, integration, training activation,
identity provisioning, trading changes, Chief authority or any PG status change.
No production service, Make/TradingView/OANDA resource, financial record or approved
governance document was modified. Implementation stops after freeze.

### Deliverables / outstanding evidence

| Requested output | Artifact |
|---|---|
| Corrected modules / normalisation / audit / extraction / projection | normalizer.py, auditor.py, extract.py, project.py, anchors.py |
| D1–D9 matrix and exact limitations | FINDINGS_V11.md |
| Canonical digest/anchor specification | ANCHOR_SPEC.md |
| Secure operator design | OPERATOR_RUNBOOK.md |
| Original/new/full logs and resolved versions | evidence/v11/ |
| Complete file inventory and hashes | evidence/v11/changed_files.json, SHA256SUMS |
| Synthetic readiness demonstration | evidence/v11/readiness_demo.json |
| No real-data recovery claims | POPULATION_REPORT.md + evidence/population_summary.json |
| PostgreSQL evidence | evidence/v11/postgres_runtime.json — UNPROVED |
| Exact frozen SHA / reviewer handover | final message + this document |

HISTORICAL_MARKET_RECOVERY_V1_1_BLOCKED_ON_EVIDENCE
