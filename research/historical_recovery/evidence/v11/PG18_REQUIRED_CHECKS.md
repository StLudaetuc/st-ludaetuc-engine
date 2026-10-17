# Disposable PostgreSQL 18 validation still required — NOT EXECUTED

No PG18 server/client binary or runnable unprivileged identity was available here.
Do not interpret mock/Python tests or the reviewer's reported 51 checks as authored
PG acceptance. Obtain the complete original review package before final assurance.

In a separate allowed non-root local environment, use only a fresh disposable
synthetic database. Verify localhost target, PG major 18, empty source namespace
and non-production identity before creating test fixtures. Never use Render or the
writer Web Shell. Retain exact version/OS/setup/commands and full results.

Required matrix:

1. Source fixture columns matching SELECT_FIELDS, 601 rows plus duplicates/malformed
   payloads; original raw payload and timestamptz values unchanged.
2. SET TIME ZONE UTC / Europe/London / America/New_York on separate connections:
   extracted source-record digests identical for the same rows, microseconds exact,
   Decimal JSONB non-float fidelity, original payload strings not rewritten.
3. Read-only transaction and dedicated SELECT-only login; attempted INSERT/UPDATE/
   DELETE/TRUNCATE/DDL denied; preflight rejects owner/superuser/writer/role membership.
4. Concurrent insert/update while extractor holds repeatable snapshot: count/pages
   remain same snapshot; subsequent fresh capture sees changes; rollback on exceptions.
5. Stable C/event-ID pages, boundary counts, empty/partial/max cap, new-snapshot resume;
   no fake complete claim; timeout/lock failure and oversize errors fail closed.
6. Parameter-bound filters and wrong stream/environment rejection. Market scopes do
   not assert broker-account isolation or introduce financial/accounting truth.
7. v2 manifest, actual export_audit file hashes, independent expected anchor and
   original/redacted record lineage; wrong/missing anchors/tampering tested.
8. No writer credentials, raw secret-bearing outputs, production actions or cloud
   networking changes. Tear down only the explicitly disposable local fixtures.

These are planned acceptance requirements, not a delivered passing SQL transcript.
