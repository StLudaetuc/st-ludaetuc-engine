# v1.1.1 author evidence reproduction

Use a fresh full clone and the exact immutable v1.1.1 SHA from the final handover.
Never check out or advance main for these tests. No production configuration,
credentials or database is needed. Do not import engine with a real DATABASE_URL.

## Two environments, two mandatory runs each

From the cloned candidate root, create venvs with Python 3.12 and 3.13 respectively.
Install with uv (or equivalent pip --require-hashes) using these test-only locks:

```
uv venv --python 3.12 /private/scratch/v111-venv312
uv pip sync --python /private/scratch/v111-venv312/bin/python --require-hashes \
  security/analytics_access_control/evidence/v111/requirements-py312.lock
uv venv --python 3.13 /private/scratch/v111-venv313
uv pip sync --python /private/scratch/v111-venv313/bin/python --require-hashes \
  security/analytics_access_control/evidence/v111/requirements-py313.lock
```

The paths are examples, not existing approved destinations. In a clean process
with no inherited production variables, run for each venv twice:

```
env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 \
  /private/scratch/v111-venv312/bin/python -W default -m unittest discover \
  -s security/analytics_access_control -p 'test_*.py' -v
```

Repeat using v111-venv313. Expected: 42 tests, 42 pass, no errors/failures/skips,
exit 0 on every run. Git must be on PATH; `git show 2ca3fa49...:engine.py` must work
from the checkout. The baseline AST fixture's old hash is historical only.

## Supplemental author probes (not the original reviewer suite)

In each of the same clean environments, from candidate root:

```
<venv>/bin/python -W default security/analytics_access_control/evidence/v111/equivalent_assurance.py
<venv>/bin/python -W default security/analytics_access_control/evidence/v111/uvicorn_probe.py
```

Expected: 32 author-written equivalent tests pass; Uvicorn script exits zero with
33 candidate HTTP cases, 3 expected-positive old-v1 negative controls, zero candidate
legacy-handler/pool calls. The negative control uses `git show c035f9d2...` in memory;
it never edits source files. Original reviewer suite is **not** in this package.
Trading probe handlers are synthetic stand-ins; actual algorithm/DB differential
was not rerun. Real engine registration, health and analytics tripwires are exercised.

No network egress is needed during testing. Uvicorn requires loopback socket binds.
Dependency installation needs package access. Actual production routing, dependency
versions and durable audit capture remain separately unverified.

## Logs and recorder

All *.log files are full unedited process stdout/stderr, with matching exit/command
metadata. Corrected_* are the four mandatory final runs. before_correction_* are
the real 42-test N-01 reproductions. bootstrap_mislaunch_* are setup diagnostics,
not valid suite results. equivalent_* and uvicorn_* are separately labelled author
supplemental results. Do not confuse any of these with independent assurance.

run_record.py is the exact author harness. To use its unchanged directory layout,
copy it to a new scratch directory containing checkout `repo/`, environments
`venv312/`, `venv313/`, and private `runner-home/`; invoke it there with executable,
label and Python arguments. It records child exit codes even for negative controls;
the caller must inspect each metadata exit_code, not merely the harness exit code.

## Integrity and exact scope

From candidate root:

```
sha256sum -c security/analytics_access_control/evidence/v111/SHA256SUMS
# Replace NEW_SHA with the frozen SHA supplied in the final handover.
git diff --name-status 7aeadda5dab83607355952ffbb21c06f68da8a20 NEW_SHA
git rev-parse NEW_SHA:analytics_access_control.py NEW_SHA:engine.py
git diff --exit-code 7aeadda5dab83607355952ffbb21c06f68da8a20 NEW_SHA -- analytics_access_control.py engine.py
```

All changed paths must be under security/analytics_access_control/. Runtime Git
blobs must remain 9f1f84985d92f2cb16df55351f3c2e0bd9dc877a and
5fbf9ae9eeeba63fa8d2e2be5d61e611a058033e. fixture.diff is the exact full test-code
delta; the full candidate delta includes docs/evidence files too and is generated
by Git, avoiding a self-referential patch-of-itself.

SHA256SUMS covers all changed candidate files except itself; changed_files.json
includes itself and SHA256SUMS. The final Git commit binds the manifest's bytes.
Verify hashes and provenance before execution. This is author evidence, not approval.
