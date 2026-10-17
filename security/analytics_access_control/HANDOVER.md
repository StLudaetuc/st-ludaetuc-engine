# ST LUDAETUC CLOUD V1.0 — ANALYTICS SECURITY v1.1.1

DRAFT — NOT APPROVED. Author engineering evidence, not independent assurance.
No merge, deployment, credential provisioning or production action is authorised.

## A. Executive summary

N-01 is corrected in `test_root_path_authenticated_containment`: export and
strategy-report receive valid dates/limit; prediction-accuracy receives its own
bounded since_hours/lookahead_minutes/max_predictions query. Assertions now cover
empty, /api and /x roots for all three routes, require HTTP 503 **and** exact
`analytics_scope_unavailable`, and assert no downstream handler executes.
Nothing was weakened to accept the previous HTTP 400.

**Both runtime files remain byte-identical to v1.1.** This is a fixture, evidence
and documentation correction only. All 42 mandatory tests pass twice in each of
two clean, hash-locked environments: Python 3.12.14 and Python 3.13.16. Additional
author-written 32-case real-engine checks pass on both, and real loopback Uvicorn
probes observe no candidate analytics handler or pool access. The earlier v1
negative control does reach legacy handlers under /api, showing sensitivity.

The complete original reviewer 32-test source was not located. The available
original REPORT.md was read; its findings and the CEO-reported v1.1 delta verdict
are distinguished below from newly executed author results. Our equivalent suite
is **not** an unchanged reproduction of Claude's suite and is not independent.

## B. Exact branch and provenance

- Repository: `StLudaetuc/st-ludaetuc-engine`.
- Production baseline: `2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c`.
- Original v1: `c035f9d2d74c3eb0d7f2dc1086f2e1095b3bea44`.
- Exact starting v1.1: `7aeadda5dab83607355952ffbb21c06f68da8a20`.
- Successor branch: `security/analytics-access-control-v1-1-1`.
- Exact new frozen SHA: supplied in the final handover; it cannot be embedded in
  its own commit without changing that commit. Review the immutable SHA, not a
  moving branch label. Its sole parent must be the exact starting v1.1 SHA.

The repository was cloned with full baseline history because the unchanged v1.1
AST test invokes `git show <production-baseline>:engine.py`. A file-only download
or shallow clone without that object is insufficient. No history replacement or
synthetic commit object was used. No main/frozen branch was advanced.

## Historical finding preservation

| Stage / finding | Evidence and disposition |
|---|---|
| v1 unauthenticated containment | Earlier 38-test author result remains historical; full old result preserved in evidence/v111/historical_v1_test_results.txt and original immutable commit |
| F-01 high root-path bypass | Original v1 checked scope.path but not Starlette's root-path-stripped route. Reviewer reported disclosure under Uvicorn root_path; this task reproduces old-v1 legacy-handler/pool-tripwire access for all three routes in loopback negative controls, without any real database |
| v1.1 runtime fix | Exact v1.1 protects both incoming and prefix-boundary-stripped route paths and authorises using stripped identity. No v1.1.1 runtime change |
| F-02 Python-specific AST hash | v1.1 compares baseline/candidate AST within the running interpreter via git show. It passes on both tested minors; baseline_engine_ast.sha256 remains historical and is no longer the active comparison |
| v1.1 N-01 | Reproduced: 42 tests, 41 pass, sole failure HTTP 400 != 503 on both clean environments. Prediction-accuracy was incorrectly given export query parameters |
| v1.1 stale handover/logs | Original files still described 38 tests. This handover and regenerated logs supersede that current claim while preserving history |
| v1.1.1 | Only endpoint-specific fixture and evidence/docs updated; 42/42 twice per environment, unchanged runtime |

The available original REPORT.md describes F-01 and its scratch proof-of-fix; it
is not the complete v1.1 delta package. The CEO reports Claude's v1.1 32/32 and
trading/database differential passed. Those particular independent runs remain
**reviewer-reported**, not reproduced unchanged here.

## C. Complete changed-file list

Current changes are confined to security/analytics_access_control/:

1. `test_access_control.py` — N-01 fixture and stronger root/status/body/no-handler assertions.
2. `HANDOVER.md` — this updated handover.
3. `test_results.txt` — generated concatenation of four full corrected-suite logs.
4. `test_environment.json` — actual two-environment, four-run metadata.
5. `evidence/v111/*` — self-contained locks, resolved packages, full unedited before/
   after/supplemental logs, their commands/exit metadata, fixture diff, runtime
   blob comparison, reproduction scripts, historical log, summary and hash manifest.

`evidence/v111/changed_files.json` enumerates **every** path, including each evidence
file; use it alongside the Git delta. No production requirements, engine/runtime,
Pine, integration, database, Finance, Chief, historical recovery or other research
file is changed.

## D. Runtime blob comparison

| File | v1.1 Git blob | v1.1.1 Git blob |
|---|---|---|
| analytics_access_control.py | 9f1f84985d92f2cb16df55351f3c2e0bd9dc877a | identical |
| engine.py | 5fbf9ae9eeeba63fa8d2e2be5d61e611a058033e | identical |

Byte equality, SHA-256 and length are recorded in runtime_blob_comparison.json.
The final Git tree check independently verifies those blob IDs. Engine AST equality
with production-minus-middleware registration passes under both Python minors.
No runtime defect requiring a protected-file change was discovered by this task.

## E. Test results and reproducibility

| Environment | Framework stack | Run 1 | Run 2 |
|---|---|---|---|
| Python 3.12.14 | FastAPI 0.115.12 / Starlette 0.46.2 / Uvicorn 0.35.0 | 42 pass, 0 fail/error/skip, exit 0 | same |
| Python 3.13.16 | FastAPI 0.143.3 / Starlette 1.8.0 / Uvicorn 0.54.0 | 42 pass, 0 fail/error/skip, exit 0 | same |

httpx 0.28.1 and pydantic 2.14.1 were resolved in both; every dependency/version is
in environment-py312.json / environment-py313.json and hashed requirements locks.
These are actual tested environments, not a claim of production parity. Python
3.12's Uvicorn version differs from the original reviewer's reported 0.54.0; the
3.13 stack tests 0.54.0. OS: Linux 6.18.55 x86_64, glibc 2.39. Container image digest
is not exposed and remains UNPROVED.

Command from full repository root, for each interpreter, twice:

```
<isolated-venv>/bin/python -W default -m unittest discover \
  -s security/analytics_access_control -p 'test_*.py' -v
```

Tests launch with a freshly constructed environment containing only PATH,
HOME (private scratch) and PYTHONDONTWRITEBYTECODE. No DATABASE_URL, service registry,
OANDA credentials, cloud tokens or production variables are inherited. Network
package downloads occurred during setup; tests use no production or external HTTP
endpoint. The Uvicorn probes bind/connect only to loopback ephemeral sockets.
No real PostgreSQL instance is needed or contacted; engine pool access is a tripwire.

Full unedited logs remain under evidence/v111/, with commands, timestamps and exit
codes in matching *.metadata.json. test_results.txt is a generated concatenation
of the four corrected logs, with source-log headings, not manually edited test output.
Warnings about deprecated FastAPI startup/TestClient APIs are retained, not hidden.
Bootstrap mislaunch logs are also retained: an initial runner followed a virtualenv
symlink to the base interpreter and lacked dependencies. These are labelled setup
failures, not valid 42-case runs; the corrected recorder preserves the venv path.
All required final runs use the actual venv interpreters and pass.

## F. F-01 and compatibility regression evidence

- The corrected 42-test suite retains v1.1 root coverage; authenticated regression
  now explicitly tests all three roots with valid per-route requests.
- equivalent_assurance.py: **32 author-written real engine.app tests**, passed on
  both environments. Tests add handler/pool tripwires in memory, check unauthenticated
  401, scoped authenticated 503/error, real health, prefix boundaries and access-status.
  Trading endpoints use synthetic stand-ins to test routing passthrough. They do
  not repeat the reviewer's 115-request algorithm/database differential.
- uvicorn_probe.py: **33 candidate HTTP probes per environment**, across root_path
  empty, /api and /x. Each has six analytics requests (3 routes x unauth/auth), one
  real health and four synthetic trading requests. Candidate legacy handler and
  pool calls: **zero**. All requests passed their response/effect assertions.
- Negative sensitivity control: the old immutable v1 middleware is loaded in an
  isolated process memory context only; /api requests hit all three real legacy
  handler wrappers and pool tripwires. No source/runtime file is overwritten and
  no real DB connection occurs. It returns the handler's sanitized pool-unavailable
  error, not actual private data; this proves boundary reachability, not new live
  exfiltration. Controls are expected-positive findings on old v1, not current defects.

The original full reviewer 32-test package is unavailable in the checked repository
refs/available Library evidence. A fresh reviewer should obtain and run that suite
unchanged. Equivalent author tests are clearly labelled and never counted as an
independent PASS. Actual algorithm compatibility follows unchanged engine bytes
and same-interpreter AST checks; no new production-trade or database differential
claim is made.

## G. Remaining findings and unproved requirements

Containment intentionally leaves all legacy analytics data unavailable, even to
valid identities. This is not a scoped analytics reader. No configuration grants
access to unsafe legacy queries through the tested routing boundary.

Residuals retained from review, not fixed by this narrow assignment:

- F-03: well-formed unauthenticated tokens can distinguish missing registry (503)
  from configured invalid credentials (401); registry parsing cost remains bounded.
- F-05: production requirements are unpinned; reviewed build parity needs separately
  authorised pinning or exact build-version evidence. No requirements file changed.
- F-06: provisioner must ensure real token entropy; rate limiting and durable audit
  retention/export are not verified or added. Failed stdout audit returns 503, but
  cannot guarantee a durable denial record while its sink is unavailable.
- F-07 pre-existing: trading ingestion/execution routes are unauthenticated in baseline
  and candidate. Out of scope here; no credentials/integration changes were made.
- Consumer inventory, legacy account/environment attribution and operational load
  validation remain incomplete. Containment blocks historical data rather than
  asserting an account boundary it cannot prove.
- Render's **current** live SHA, Python/build versions, start command, environment
  variables including UVICORN_*, root-path/proxy settings and audit retention were
  not inspected in this task. Earlier author metadata checks and reviewer reports
  are historical evidence only. No secret-bearing environment dump was requested.

## H. Secure deployment and rollback prerequisites — not executed

Independent review must approve the exact successor SHA, followed by an explicit
CEO deployment decision. Main auto-deploy means merging can deploy; this task does
neither. Before any approved release, operator verifies actual routing/root-path/
proxy settings, dependency versions, approved fallback target, analytics-consumer
outage acceptance and audit capture/retention. Confirm middleware registration and
root-path checks on the deployed dependency build.

No credential is required for containment: leave ANALYTICS_SERVICE_IDENTITIES_JSON
unset unless separately authorised scoped status access is needed. Never place
secrets in URLs/logs/Pine/Make/spreadsheets/LLM context. Provisioning and service
configuration changes require separate authority; no production key was created.

Approved operator validation should use bounded non-sensitive 401 probes for all
three data routes, including effective root-path forms; health/trading continuity
checks; deployed-SHA confirmation; audit arrival and dependency evidence. If status
identities are provisioned later, valid access-status may return 200; do not alert
on every analytics 200 without distinguishing that route. Legacy data routes must
never return data.

**Never roll back to production baseline or old v1:** baseline exposes analytics
and v1 has F-01. Removing the registry is a safe identity-acceptance rollback; it
does not reopen data. Prepare an independently approved root-path-safe containment
fallback or fix forward. Do not suspend trading service, block all ingress or weaken
PostgreSQL networking to compensate. Preserve audit/deploy evidence through rollback.

## I. Fresh independent delta review instructions

1. Fetch exact new SHA and `7aeadda5...`; verify sole parent and strict changed-path
   list. Compare both protected blobs to table D. Main and prior candidates must
   remain unchanged; do not merge/deploy during review.
2. Read fixture.diff and the corrected method: validate each endpoint's real query
   allowlist, all 9 auth/root/route cases, required 503/error and zero downstream calls.
3. Verify evidence hashes, then create clean Python 3.12/3.13 venvs using locks and
   full baseline Git history. Repeat the complete 42-test command twice per stack.
4. Run author equivalent/real Uvicorn scripts, inspect tripwires/stand-in limitations,
   and obtain the original reviewer suite for an unchanged 32-test reproduction.
5. Check old-v1 sensitivity controls reach handlers, current candidate does not;
   review parser/routing consistency with each tested Starlette/Uvicorn version.
6. Inspect environment metadata, unedited logs, warnings and setup-failure labels.
   Author results do not establish independent assurance. Report any new runtime
   defect without modifying protected files under this assignment.

Full commands are in evidence/v111/REPRODUCE.md. Exact Git delta can be obtained
with `git diff 7aeadda5dab83607355952ffbb21c06f68da8a20 <new-sha>`; fixture.diff is the
complete behavioural source change. New evidence/docs are included as files with
hashes rather than recursively embedding a patch of itself.

## J. Exact next CEO decision

Commission a fresh independent Claude Code delta assurance review of the exact
v1.1.1 SHA, supplying the original reviewer suite if available. After a satisfactory
independent verdict and operator verification of deployment prerequisites, the CEO
may make a separate, explicit containment-release decision. This handover grants
no production deployment, merge, Stage 1 activation or Effective Authority.

Once frozen, implementation stops pending independent review.

ANALYTICS_SECURITY_V1_1_1_READY_FOR_INDEPENDENT_REVIEW
