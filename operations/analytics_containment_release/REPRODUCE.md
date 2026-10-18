# Independent operations reproduction

Use a full clone, exact frozen operations SHA and no production credentials. Do not alter main or frozen candidates. Local loopback sockets only; no Render deploy or DB needed.

1. `git show --no-patch --format='%H %P %T' OPERATIONS_SHA`; sole parent must be5f0d07565629fba30d0c2221229fd2483472e701. `git diff --name-only 5f0d07565629fba30d0c2221229fd2483472e701 OPERATIONS_SHA` must be exclusively operations/analytics_containment_release/.
2. `sha256sum -c operations/analytics_containment_release/evidence/SHA256SUMS` from repository root. Verify runtime and fallback blobs with `git rev-parse OPERATIONS_SHA:path`, expected in evidence/source_integrity.json. `git diff --exit-code 5f0d07565629fba30d0c2221229fd2483472e701 OPERATIONS_SHA -- engine.py analytics_access_control.py requirements.txt security/` must be empty.
3. Create disposable environments with Python3.14.3,3.12.14,3.13.16. For3.14.3 use:

```
uv venv --python 3.14.3 /private/scratch/ops314
uv pip sync --python /private/scratch/ops314/bin/python --require-hashes operations/analytics_containment_release/requirements-observed.lock
```

For additional versions use existing candidate's hash locks `security/analytics_access_control/evidence/v111/requirements-py312.lock` and `requirements-py313.lock`. This is a compatibility matrix, not one common resolved environment. Capture actual installed versions. Production-like314 lock has15 observed runtime packages plus4 test-client packages; production requirements remain untouched.

4. In an independently writable evidence checkout run twice per interpreter, labels unique:

```
/private/scratch/ops314/bin/python operations/analytics_containment_release/run_tests.py --python /private/scratch/ops314/bin/python --label reviewer314a
```

Recorder itself starts children with only PATH and PYTHONDONTWRITEBYTECODE; no inherited DB/broker/identity variables. Interpreter paths use absolute, not symlink-resolved paths, preserving virtualenv packages. Four commands: tooling unittest12, original security unittest42, unchanged fallback unittest4, operations rehearsal12. All must exit0. Full logs contain intentional startup/audit fault injections. Inspect unittest totals, not just ERROR text inside a passing fault-injection test. Never delete original author logs or change old evidence; reviewer owns a separate output location/branch/checkout.

5. Rehearsal validates real Uvicorn roots, local CLI UVICORN_ROOT_PATH, actual trading handlers versus baseline with in-memory logging, zero legacy analytics handler calls, audit failure and fallback limitations. First failed author operations fixture run remains in evidence/initial314_*; pre-final runs may bind older rehearsal source. Only validated314a/b, validated312a/b and validated313a/b are final matrix runs. A pre-final313b CLI readiness timeout is retained; the runner now preserves CLI startup logs even on failure and permits a bounded15-second readiness window. Compare source hashes for these to final candidate.
6. `release_gate.py --repo . --record operations/analytics_containment_release/approval_record.template.json` must return HOLD/exit1. It does not authenticate attestations or grant authority. Do not fill approval fields merely to make it pass.
7. `verify_http.py` defaults to local origins. Rehearsal executes it only on loopback; remote-default refusal and no-body-read behavior are tested. The operator-approved flag is only an assertion by its invoker, not approval issued by software. Do not invoke against production without separate CEO scope.
8. Read private independent security/fallback reports separately and confirm exactSHA/source coverage. These operations tests do not commission or replace independent assurance. Fresh read-only Render checks are required before any CEO release decision.

## Source/reference evidence

Current official Render Markdown docs deploys, health-checks and rollbacks were fetched/read October18. URLs, content hashes and retrieval notes are in evidence/documentation_references.json. The full upstream documents are not copied here. Runtime metadata is allowlisted; build evidence stores only Python/package lines. Mail/Drive body text and sensitive operational records are not committed. Evidence/source_search.json records bounded source inventory and gaps.
