# Finance foundation — independent-review correction candidate

Offline, undeployed. Only `finance/foundation-v1` is writable. No imports into
engine/app, no trading dependencies changed, no runtime/Chief grants, no broker
calls, no Make/TradingView changes. See `docs/HANDOVER.md` and `docs/TEST_RESULTS.json`.
The original v1 report/code remains at `189587ce3e211b22782dc221af9afdf6a8cc123c`.

```sh
python -m pip install -r finance/requirements-test.txt
python -m unittest discover -s finance/tests -v
```

`serializer.serialize()` is the sole implemented egress boundary for all six
Reader products and weekly inputs. It checks Draft 2020-12 shape plus explicit
scope, current server-ordered coverage/reconciliation, trusted expected metric
values/classification, counts, recognition and non-cash Practice rules. Missing
its dependency fails closed. It is not a deployed API. The trusted producer
context/journal must never be exposed as an LLM tool or populated from model
claims. A future publisher must build its manifest from verified deterministic
SQL/Python outputs and evidence, not copy an untrusted input payload.

`policy.py` defines scale-12, half-away-from-zero conversion/quantisation. Source
money is Decimal/NUMERIC, finite, bounded and currency-registry validated. Convert
and quantise each cash component once, then sum. Recognition uses differences
of cumulative quantised service allocations to preserve period additivity.
Source timestamps describe business events; they do not rank authoritative state.

`core.py`: append-only expense revisions; separate trusted verification decisions;
bounded atomic unverified CSV staging; cash/economic/run-rate separation;
explicit evidence coverage and project-level economics. Low-level `spend()` is
an observed subtotal, not an authoritative completeness or zero-spend claim.
Import rows cannot self-verify. Actual broker data has explicit `Scope` with no
money/environment defaults. `project_economics()` aggregates eligible live
accounts before deducting project costs once. Practice units are hypothetical.

`reconciliation.py`: six canonical states, partial slices, exact replay dedupe,
malformed-record quarantine and immutable, reviewed resolution/withdrawal history.
`oanda.py`: conservative offline profiles, top-level cash only, nested attribution
checks. Unknown profiles remain exceptions; no unclassified cash becomes zero.

Migration `001_foundation.sql` is a replacement of an **unapplied v1 candidate**,
not an upgrade to any deployed Finance schema. It must run only in a fresh
isolated PG18 database. There are no service grants and PUBLIC permissions are
revoked. Migration owners/superusers remain privileged; triggers do not constrain
DB owners. A staging writer must never receive verification/coverage authority.

A local-only opt-in validation runner is supplied:

```sh
python -m finance.tests.postgres_local_validation --socket /absolute/task-scratch/finance-pg18/socket --user local_test_owner
```

It requires existing `finance_validation_isolated`, an actual local scratch Unix
socket, PG18 and psql. It refuses remote/TCP/URI/shared-target fallback, scrubs
libpq environment settings, and performs migration, acceptance, arithmetic parity
and concurrent revision/order checks. **NOT RUN here:** no local PG18 server/client
binaries; package setup also encountered an OS identity restriction. PG18 SQL and
PL/pgSQL grammar parsing passed; that is not runtime, privilege or parity proof.

The currency whitelist and recognition/FX/allocation policy choices are explicit
candidate proposals for ratification. An expense with no recognition policy or
paid-at evidence returns unavailable for that measure. No actual costs/P&L have
been imported; no agent, gate or production readiness has been asserted.
