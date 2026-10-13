# Finance & Broker Reconciliation Foundation v1

Review candidate, offline only. No engine import, route, credential, deployment,
startup DDL or Chief/Reader integration. Start with `docs/HANDOVER.md`.

Run pure Python checks from repository root:

```sh
python -m unittest discover -s finance/tests -v
```

PostgreSQL checks are supplied, but have **not** run in this environment:

```sh
psql "$FINANCE_ISOLATED_TEST_DB_URL" -X -v ON_ERROR_STOP=1 -f finance/migrations/001_foundation.sql
psql "$FINANCE_ISOLATED_TEST_DB_URL" -X -v ON_ERROR_STOP=1 -f finance/tests/postgres_acceptance.sql
```

Use only a fresh disposable PostgreSQL 18 Finance test database. The migration
intentionally fails if schema `finance` already exists. Apply once via a reviewed
migration runner. Never run against Chief/Reader or current trading databases.
Migration ownership is privileged; append-only triggers do not constrain a DB
owner/superuser. Future importer must have INSERT-only scoped functions/roles;
future Reader must have an allowlisted aggregate interface. No grants provided.

`core.py` is the decimal arithmetic and CSV validation reference. `oanda.py`
accepts already-exported transactions and has no network or broker credential
capability. Persistence adapters, full transaction classification, complete
broker coverage and production access roles are intentionally pending review.

Do not certify profitability from empty fixtures, estimated expense records,
incomplete exports, or OANDA Practice profit. No real financial totals imported.
