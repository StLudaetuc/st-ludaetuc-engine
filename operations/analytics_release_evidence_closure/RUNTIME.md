# Python3.14 and reproducibility assessment — PROPOSAL

The previous647c0f6 evidence demonstrated70 tests twice on Python3.14.3 with the fifteen runtime versions listed in runtime-observed.in plus separate test-client packages. This task's runtime-only lock removes test-only packages and locks all15 observed runtime distributions with hashes. It does not edit root requirements.txt or any runtime setting.

## Exact proposed runtime

Python3.14.3; Linuxx86_64 with CPython314 ABI. Observed runtime package versions:

```
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.15.1
click==8.5.0
fastapi==0.142.2
h11==0.16.0
idna==3.20
opentelemetry-api==1.45.0
psycopg2-binary==2.9.13
pydantic==2.13.5
pydantic-core==2.46.5
starlette==1.7.0
typing-extensions==4.16.0
typing-inspection==0.4.4
uvicorn==0.54.0
```

These are October6 build-observed versions and the previous production-like rehearsal set, not a claim of current process inventory or the independent reviewer's later package resolution. Do not mix fastapi0.143.3/pydantic2.14.1/Starlette1.8.0 from reviewer runs into this lock without separate testing.

`runtime-observed.lock` is generated with hashes. A complete15-wheel Linux/CPython314 selection was acquired and hash-recorded locally. A network-disabled-index install (`--no-index --find-links ... --require-hashes`) into a disposable3.14.3 environment was executed. The verification evidence records selected wheel names/SHA256, Python/build/platform/ABI and installer versions. A changed wheel is rejected by hash verification. This is local package reproducibility, not a deployable Render build.

## Proposed future commands — separate approved implementation required

```
python -m pip install --no-index --find-links APPROVED_WHEELHOUSE \
  --require-hashes -r APPROVED_RUNTIME_LOCK
python -m pip check
```

Operator must verify wheelhouse and lock hashes before invocation, use an approved pinned interpreter/base runtime and retain installer/build provenance. Wheel selection is platform-specific; it is not valid to assume Linuxx86_64 wheels cover another architecture, a free-threaded ABI or another Python minor version. Hash lock includes permitted distributions for multiple platforms; the selected wheel manifest narrows the actual artifact. No sdist compilation was used. Existing root command remains `pip install -r requirements.txt` and can re-resolve a different dependency set at the next build.

Potential native-Render proposal: separately reviewed root Python-version declaration/build configuration installs the approved lock from a retained wheelhouse or controlled index. Potential stronger proposal: reviewed immutable container image digest bundling runtime and wheels. Neither was implemented/configured; both require explicit engineering scope, compatibility tests and CEO approval. Pinning package versions alone does not freeze OS/runtime/platform configuration.

## Still requires a running-process or provider attestation

Current process Python version/executable/build, current installed distributions and changes since build, actual ABI/libc, process launch arguments, Uvicorn environment variable names/presence, registry presence, inherited Python paths/site-packages, current configuration values and pending changes, and actual next-build artifact provenance remain UNPROVED. Do not dump environment values or obtain credentials. Startup health output cannot supply these facts.

A fallback-bearing tree can still fail import if another regular `security` package shadows the namespace package. This report residual was not patched. Verify the exact package graph/search path in disposable acceptance and obtain a separately scoped hardening decision if needed.
